#!/usr/bin/env python3
"""Stockfish 19 official WDL rating dataset.

Position-level Win/Draw/Loss from `UCI_ShowWDL`. That is SF19's fishtest-LTC
self-play model (eval + remaining material). It is not FIDE/Lichess Elo and
it is not `data_loader.compute_wdl`'s sigmoid.

Stored `wdl` is White-absolute: [P(White wins), P(draw), P(White loses)].
`wdl_raw` is the same triple as UCI per-mille integers (sum 1000).
Rows without official WDL are dropped. No sigmoid fallback.

Usage:
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u scripts/sf19_wdl_dataset.py smoke --out-dir outputs/sf19_wdl/smoke
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u scripts/sf19_wdl_dataset.py generate --go --mode eco \\
      --out-dir outputs/sf19_wdl/eco --target 1000000 --workers 16
  MOVE_VOCAB_VERSION=compact python -u scripts/sf19_wdl_dataset.py push \\
    --out-dir outputs/sf19_wdl/eco --repo avewright/local-wdl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import chess
import chess.engine
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import compute_wdl  # noqa: E402
from move_vocab import UCI_TO_IDX, VOCAB_SIZE  # noqa: E402
from scripts.lichess_openings import (  # noqa: E402
    load_openings as load_eco_openings,
    openings_summary,
    start_positions as eco_start_positions,
)
from scripts.sf19_soft_dataset import (  # noqa: E402
    DEFAULT_TAU,
    ORIGIN_RELABEL,
    ORIGIN_SELFPLAY,
    SOFT_K,
    SOURCE_SF19,
    SeenDB,
    _hf_token,
    _load_json,
    _upload_with_retry,
    assert_resume_compatible,
    build_eco_game_spec,
    compact_key_bytes,
    encode_board,
    inbox_state,
    log,
    next_shard_dir,
    parse_multipv,
    phase_from_board,
    pick_play_move,
    require_sf19,
    resolve_sf,
    row_key,
    sample_seed_fens,
    stack_rows,
    stream_multipv_infos,
    teacher_fingerprint,
    to_white_abs,
    write_manifest,
    write_shard,
)

WDL_SOURCE_UCI = 1
WDL_SOURCE_TERMINAL = 2
WDL_FINGERPRINT_KEYS = (
    "uci_name", "binary_sha256", "eval_file",
    "nodes", "multipv", "wdl_required", "source_id",
)


@dataclass
class WdlConfig:
    nodes: int = 25_000
    play_nodes: int = 2_000
    multipv: int = 1
    tau: float = DEFAULT_TAU
    epsilon: float = 0.20
    ply_stride: int = 2
    ply_skip_open: int = 0
    ply_cap: int = 180
    book_noise: int = 2
    watchdog_s: float = 12.0
    hash_mb: int = 64


def wdl_run_fingerprint(sf_fp: dict, cfg: WdlConfig) -> dict:
    body = {
        "uci_name": sf_fp["uci_name"],
        "binary_sha256": sf_fp["binary_sha256"],
        "eval_file": sf_fp.get("eval_file") or "",
        "nodes": int(cfg.nodes),
        "multipv": 1,
        "wdl_required": 1,
        "source_id": int(SOURCE_SF19),
    }
    payload = json.dumps({k: body[k] for k in WDL_FINGERPRINT_KEYS}, sort_keys=True, separators=(",", ":"))
    body["fingerprint_id"] = hashlib.sha256(payload.encode()).hexdigest()
    body["kind"] = "sf19_wdl"
    return body


def parse_uci_wdl(wdl_obj, *, turn_black: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """Official UCI WDL → White-absolute per-mille and probabilities.

    UCI reports side-to-move (wins, draws, losses). We flip for Black.
    """
    if wdl_obj is None:
        return None
    try:
        rel = wdl_obj.relative if hasattr(wdl_obj, "relative") else wdl_obj
        raw = np.array([int(rel.wins), int(rel.draws), int(rel.losses)], dtype=np.int16)
    except Exception:
        return None
    if int(raw.sum()) <= 0:
        return None
    if turn_black:
        raw = raw[[2, 1, 0]]
    probs = raw.astype(np.float32) / float(raw.sum())
    return raw, probs


def terminal_wdl(board: chess.Board) -> tuple[np.ndarray, np.ndarray] | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is chess.WHITE:
        raw = np.array([1000, 0, 0], dtype=np.int16)
    elif outcome.winner is chess.BLACK:
        raw = np.array([0, 0, 1000], dtype=np.int16)
    else:
        raw = np.array([0, 1000, 0], dtype=np.int16)
    return raw, raw.astype(np.float32) / 1000.0


def sigmoid_wdl_white(cp: int, mate: int) -> np.ndarray:
    """Project fallback only — never stored. Used to audit the gap."""
    return compute_wdl(
        torch.tensor([int(cp)], dtype=torch.int32),
        torch.tensor([int(mate)], dtype=torch.int32),
    )[0].numpy().astype(np.float32)


def _best_move_slots(uci: str, stm_cp: int, stm_mate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    soft_i = np.full(SOFT_K, -1, dtype=np.int64)
    soft_p = np.zeros(SOFT_K, dtype=np.float32)
    soft_c = np.zeros(SOFT_K, dtype=np.int32)
    soft_m = np.zeros(SOFT_K, dtype=np.int32)
    if uci in UCI_TO_IDX:
        soft_i[0] = UCI_TO_IDX[uci]
        soft_p[0] = 1.0
        soft_c[0] = int(stm_cp)
        soft_m[0] = int(stm_mate)
    return soft_i, soft_p, soft_c, soft_m


def label_wdl_row(
    board: chess.Board,
    parsed: dict | None,
    *,
    nodes_budget: int,
    tau: float,
) -> dict | None:
    enc = encode_board(board)
    if enc is None:
        return None
    arr, turn, castling, ep = enc
    n_legal = board.legal_moves.count()
    if n_legal == 0 or (parsed or {}).get("terminal"):
        tw = terminal_wdl(board)
        if tw is None:
            return None
        raw, probs = tw
        outcome = board.outcome(claim_draw=True)
        mate = 0
        cp = 0
        if outcome and outcome.winner is chess.WHITE:
            mate = 1
        elif outcome and outcome.winner is chess.BLACK:
            mate = -1
        return _finish_row(
            arr, turn, castling, ep,
            move_idx=-1, cp=cp, mate=mate,
            soft_i=np.full(SOFT_K, -1, dtype=np.int64),
            soft_p=np.zeros(SOFT_K, dtype=np.float32),
            soft_c=np.zeros(SOFT_K, dtype=np.int32),
            soft_m=np.zeros(SOFT_K, dtype=np.int32),
            wdl=probs, wdl_raw=raw, wdl_source=WDL_SOURCE_TERMINAL,
            depth=0, nodes=0, nodes_budget=nodes_budget, tau=tau,
            policy_mask=0, bound_skipped=0, n_soft=0,
        )
    if parsed is None or not parsed.get("items"):
        return None
    item = parsed["items"][0]
    official = item.get("wdl")
    raw = item.get("wdl_raw")
    if official is None or raw is None:
        return None
    official = np.asarray(official, dtype=np.float32).reshape(-1)[:3]
    raw = np.asarray(raw, dtype=np.int16).reshape(-1)[:3]
    if official.shape != (3,) or raw.shape != (3,) or int(raw.sum()) <= 0:
        return None
    if not np.isfinite(official).all() or abs(float(official.sum()) - 1.0) > 1e-3:
        return None
    turn_black = turn == 1
    w_cp, w_mate = to_white_abs(item["stm_cp"], item["stm_mate"], turn_black)
    soft_i, soft_p, soft_c, soft_m = _best_move_slots(item["uci"], item["stm_cp"], item["stm_mate"])
    if int(soft_i[0]) < 0:
        return None
    return _finish_row(
        arr, turn, castling, ep,
        move_idx=int(soft_i[0]), cp=w_cp, mate=w_mate,
        soft_i=soft_i, soft_p=soft_p, soft_c=soft_c, soft_m=soft_m,
        wdl=official.astype(np.float32), wdl_raw=raw.astype(np.int16),
        wdl_source=WDL_SOURCE_UCI,
        depth=int(parsed.get("depth") or 0),
        nodes=int(parsed.get("nodes") or 0),
        nodes_budget=nodes_budget, tau=tau,
        policy_mask=1,
        bound_skipped=int(parsed.get("bound_skipped") or 0),
        n_soft=1,
    )


def _finish_row(
    arr, turn, castling, ep, *,
    move_idx, cp, mate, soft_i, soft_p, soft_c, soft_m,
    wdl, wdl_raw, wdl_source, depth, nodes, nodes_budget, tau,
    policy_mask, bound_skipped, n_soft,
) -> dict:
    return {
        "board_array": arr,
        "turn": np.int8(turn),
        "castling": np.int8(castling),
        "ep_square": np.int8(ep),
        "move_idx": np.int64(move_idx),
        "cp": np.int32(cp),
        "mate": np.int32(mate),
        "soft_indices": soft_i,
        "soft_probs": soft_p,
        "soft_cps": soft_c,
        "soft_mates": soft_m,
        "label_depth": np.int16(depth),
        "phase": np.int8(phase_from_board(arr)),
        "source": np.int8(SOURCE_SF19),
        "wdl": np.asarray(wdl, dtype=np.float32).reshape(3),
        "wdl_raw": np.asarray(wdl_raw, dtype=np.int16).reshape(3),
        "wdl_source": np.int8(wdl_source),
        "nodes": np.int32(nodes),
        "policy_mask": np.int8(policy_mask),
        "tau": np.float32(tau),
        "nodes_budget": np.int32(nodes_budget),
        "game_id": np.int64(-1),
        "ply": np.int16(-1),
        "split": np.int8(0),
        "origin": np.int8(ORIGIN_RELABEL),
        "flags": np.int16(0),
        "bound_skipped": np.int16(bound_skipped),
        "n_pieces": np.int8(int(np.count_nonzero(arr))),
        "n_soft": np.int8(n_soft),
    }


def attach_official_wdl(parsed: dict, infos: list[dict], board: chess.Board) -> dict | None:
    """Copy official WDL onto the chosen MultiPV-1 line. Drop the row if missing."""
    if parsed is None or parsed.get("terminal") or not parsed.get("items"):
        return parsed
    depth = int(parsed.get("depth") or 0)
    uci = parsed["items"][0]["uci"]
    turn_black = board.turn == chess.BLACK
    found = None
    for info in infos:
        if int(info.get("depth") or -1) != depth:
            continue
        if info.get("upperbound") or info.get("lowerbound"):
            continue
        pv = info.get("pv") or []
        if not pv or pv[0].uci() != uci:
            continue
        found = parse_uci_wdl(info.get("wdl"), turn_black=turn_black)
        if found is not None:
            break
    if found is None:
        return None
    raw, probs = found
    parsed["items"][0]["wdl"] = probs
    parsed["items"][0]["wdl_raw"] = raw
    parsed["best_wdl"] = probs
    return parsed


def analyze_wdl(engine, board: chess.Board, *, nodes: int, tau: float, watchdog_s: float) -> dict | None:
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return label_wdl_row(board, {"terminal": True}, nodes_budget=nodes, tau=tau)
    try:
        infos = stream_multipv_infos(engine, board, nodes=nodes, multipv=1, watchdog_s=watchdog_s)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
        if _W_ENGINE is engine:
            try:
                _start_engine()
            except Exception:
                pass
        return None
    parsed = parse_multipv(infos, board, k=1, tau=tau)
    parsed = attach_official_wdl(parsed, infos, board)
    if parsed is None:
        return None
    return label_wdl_row(board, parsed, nodes_budget=nodes, tau=tau)


def stack_wdl_rows(rows: list[dict]) -> dict:
    data = stack_rows(rows)
    data["wdl_raw"] = torch.from_numpy(np.stack([r["wdl_raw"] for r in rows]))
    data["wdl_source"] = torch.tensor([int(r["wdl_source"]) for r in rows], dtype=torch.int8)
    return data


_W_ENGINE = None
_W_SF = None
_W_CFG: WdlConfig | None = None
_W_SEEN = None


def _close_worker():
    global _W_ENGINE
    if _W_ENGINE is not None:
        try:
            _W_ENGINE.quit()
        except Exception:
            pass
        _W_ENGINE = None


def _start_engine() -> None:
    global _W_ENGINE
    _close_worker()
    assert _W_SF is not None and _W_CFG is not None
    _W_ENGINE = chess.engine.SimpleEngine.popen_uci(_W_SF, timeout=60.0)
    _W_ENGINE.configure({
        "Threads": 1,
        "Hash": int(_W_CFG.hash_mb),
        "UCI_ShowWDL": True,
    })


def _init_worker(sf_path: str, cfg: WdlConfig, seen_path: str | None = None):
    global _W_SF, _W_CFG, _W_SEEN
    import atexit
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _W_SF = sf_path
    _W_CFG = cfg
    _W_SEEN = SeenDB(Path(seen_path), readonly=True) if seen_path else None
    time.sleep(random.random() * 2.0)
    _start_engine()
    atexit.register(_close_worker)


def play_wdl_game(spec: dict) -> dict:
    assert _W_ENGINE is not None and _W_CFG is not None
    cfg = _W_CFG
    rng = random.Random(int(spec["seed"]))
    eps = float(spec.get("epsilon", cfg.epsilon))
    wild = float(spec.get("wild", 0.0))
    start_fen = spec.get("start_fen")
    if start_fen:
        try:
            board = chess.Board(start_fen)
        except ValueError:
            board = chess.Board()
    else:
        board = chess.Board()
        for uci in spec.get("opening") or []:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
    for _ in range(int(spec.get("book_noise") or cfg.book_noise)):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(list(board.legal_moves)))
    rows = []
    rejects = {"seen_skip": 0, "stride": 0, "analyze_fail": 0, "no_wdl": 0}
    ply = 0
    last_kept = -999
    while not board.is_game_over(claim_draw=True) and ply < cfg.ply_cap:
        keep = (
            ply >= cfg.ply_skip_open
            and (ply - cfg.ply_skip_open) % cfg.ply_stride == 0
            and (ply - last_kept) >= cfg.ply_stride
        )
        parsed_play = None
        if keep and _W_SEEN is not None:
            enc = encode_board(board)
            if enc is not None:
                arr, turn, castling, ep = enc
                if _W_SEEN.has(compact_key_bytes(arr, turn, castling, ep)):
                    rejects["seen_skip"] += 1
                    keep = False
        if keep:
            row = analyze_wdl(
                _W_ENGINE, board,
                nodes=cfg.nodes, tau=cfg.tau, watchdog_s=cfg.watchdog_s,
            )
            if row is None:
                rejects["analyze_fail"] += 1
            elif int(row["wdl_source"]) not in (WDL_SOURCE_UCI, WDL_SOURCE_TERMINAL):
                rejects["no_wdl"] += 1
            else:
                row["game_id"] = np.int64(spec["game_id"])
                row["ply"] = np.int16(ply)
                row["split"] = np.int8(spec.get("split") or 0)
                row["origin"] = np.int8(ORIGIN_SELFPLAY)
                rows.append(row)
                last_kept = ply
                if int(row["move_idx"]) >= 0:
                    for cand in board.legal_moves:
                        if UCI_TO_IDX.get(cand.uci()) == int(row["move_idx"]):
                            parsed_play = {"items": [{"uci": cand.uci()}], "probs": [1.0]}
                            break
        else:
            rejects["stride"] += 1
        if board.is_game_over(claim_draw=True):
            break
        if parsed_play is None:
            play = analyze_wdl(
                _W_ENGINE, board,
                nodes=cfg.play_nodes, tau=cfg.tau,
                watchdog_s=min(cfg.watchdog_s, 3.0),
            )
            if play is not None and int(play["move_idx"]) >= 0:
                for cand in board.legal_moves:
                    if UCI_TO_IDX.get(cand.uci()) == int(play["move_idx"]):
                        parsed_play = {"items": [{"uci": cand.uci()}], "probs": [1.0]}
                        break
        board.push(pick_play_move(parsed_play, board, rng, epsilon=eps, wild=wild))
        ply += 1
    return {"rows": rows, "rejects": rejects, "game_id": spec["game_id"], "plies": ply}


def _run_job(spec: dict) -> dict:
    empty = {"seen_skip": 0, "stride": 0, "analyze_fail": 0, "no_wdl": 0}
    if spec.get("relabel"):
        assert _W_CFG is not None
        board = chess.Board(str(spec["fen"]))
        row = analyze_wdl(
            _W_ENGINE, board,
            nodes=_W_CFG.nodes, tau=_W_CFG.tau, watchdog_s=_W_CFG.watchdog_s,
        )
        if row is None:
            empty["analyze_fail"] = 1
            return {"rows": [], "rejects": empty, "game_id": spec["game_id"], "plies": 0}
        row["game_id"] = np.int64(spec["game_id"])
        row["ply"] = np.int16(0)
        row["split"] = np.int8(spec.get("split") or 0)
        row["origin"] = np.int8(ORIGIN_RELABEL)
        return {"rows": [row], "rejects": empty, "game_id": spec["game_id"], "plies": 0}
    return play_wdl_game(spec)


def generate(args) -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(f"need compact vocab, got {VOCAB_SIZE}")
    sf = resolve_sf()
    fp = teacher_fingerprint(sf)
    require_sf19(fp)
    out = Path(args.out_dir)
    inbox = out / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    cfg = WdlConfig(
        nodes=int(args.nodes),
        play_nodes=int(args.play_nodes),
        epsilon=float(args.epsilon),
        ply_stride=int(args.ply_stride),
        ply_skip_open=int(args.ply_skip_open),
        ply_cap=int(args.ply_cap),
        book_noise=int(args.book_noise),
        watchdog_s=float(args.watchdog_s),
        hash_mb=int(args.hash_mb),
    )
    run_fp = wdl_run_fingerprint(fp, cfg)
    assert_resume_compatible(out, run_fp)
    teacher_path = out / "teacher.json"
    if not teacher_path.exists():
        teacher_path.write_text(json.dumps({
            "teacher": fp,
            "config": asdict(cfg),
            "run": run_fp,
            "fingerprint_id": run_fp["fingerprint_id"],
            "wdl": {
                "model": "stockfish19_uci_show_wdl",
                "calibration": "fishtest_ltc_selfplay",
                "perspective": "white_absolute",
                "required": True,
                "sigmoid_fallback": False,
            },
        }, indent=2), encoding="utf-8")
    log(
        f"teacher={fp['uci_name']} fp={run_fp['fingerprint_id'][:12]} "
        f"nodes={cfg.nodes} multipv=1 wdl=uci workers={args.workers}",
        log_path,
    )
    seen = SeenDB(out / "seen.sqlite")
    committed, next_game = inbox_state(inbox)
    known = seen.known_shards()
    for sh in sorted(inbox.glob("shard_*")):
        if not (sh / "READY").exists() or not (sh / "soft_cache.pt").exists():
            continue
        if sh.name in known:
            continue
        seen.ingest_shard_keys(sh)
    log(f"resume seen={len(seen):,} committed={committed:,} next_game={next_game}", log_path)
    for raw in getattr(args, "exclude_caches", None) or []:
        p = Path(raw)
        caches = sorted(p.rglob("soft_cache.pt")) if p.is_dir() else ([p] if p.exists() else [])
        for cache in caches:
            added = seen.ingest_cache_keys(cache, f"exclude:{cache}")
            log(f"exclude {cache} +{added:,} keys seen={len(seen):,}", log_path)
    eco_starts: list[dict] = []
    seed_fens: list[str] = []
    if args.mode == "eco":
        openings = load_eco_openings()
        eco_starts = eco_start_positions(openings, include_prefixes=not args.no_prefixes)
        summary = openings_summary(openings, eco_starts)
        (out / "openings.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(
            f"eco openings={summary['n_rows']} unique={summary['n_unique']} "
            f"starts={len(eco_starts)} volumes={summary['volumes']}",
            log_path,
        )
    else:
        seed_paths = [Path(p) for p in (args.seed_caches or []) if Path(p).exists()]
        seed_fens = sample_seed_fens(seed_paths, int(args.seed_fens_n), random.Random(args.seed + 3))
        log(f"relabel fens={len(seed_fens)} from {[str(p) for p in seed_paths]}", log_path)
        if not seed_fens:
            raise SystemExit("relabel mode needs --seed-caches with existing boards")
    if committed >= args.target:
        write_manifest(out, {"teacher": fp, "config": asdict(cfg), "already_complete": True})
        log(f"already complete committed={committed:,} target={args.target:,}", log_path)
        return
    ctx = get_context("spawn")
    pool = ctx.Pool(max(1, args.workers), initializer=_init_worker, initargs=(sf, cfg, str(out / "seen.sqlite")))
    t0 = time.time()
    new_rows = 0
    rejected = {"dup": 0, "stride": 0, "analyze_fail": 0, "no_wdl": 0, "seen_skip": 0}
    pending: list[dict] = []
    pending_keys: set[bytes] = set()
    game_i = args.game_start if args.game_start else next_game
    relabel_i = 0

    def flush(force: bool = False) -> None:
        nonlocal pending, pending_keys
        if not pending or (len(pending) < args.shard_size and not force):
            return
        take = pending[: args.shard_size] if not force else pending
        pending = pending[len(take):] if not force else []
        keys = [row_key(r) for r in take]
        data = stack_wdl_rows(take)
        sh = next_shard_dir(inbox)
        write_shard(data, sh, {
            "teacher": fp["uci_name"],
            "binary_sha256": fp["binary_sha256"],
            "fingerprint_id": run_fp["fingerprint_id"],
            "nodes": cfg.nodes,
            "multipv": 1,
            "wdl_required": 1,
            "max_game_id": int(data["game_id"].max()),
        })
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        seen.forget_hot(keys)
        pending_keys.difference_update(keys)
        log(f"wrote {sh} n={len(take)} total={committed + new_rows:,}", log_path)
        write_manifest(out, {
            "teacher": fp["uci_name"],
            "fingerprint_id": run_fp["fingerprint_id"],
            "nodes": cfg.nodes,
            "kind": "sf19_wdl",
        })

    try:
        while committed + new_rows < args.target:
            left = args.target - (committed + new_rows)
            if args.mode == "relabel":
                n_jobs = max(1, min(args.workers * 2, left))
            else:
                # ECO games yield several labels; do not queue a long leftover tail.
                n_jobs = max(1, min(args.workers, max(1, (left + 3) // 4)))
            jobs = []
            for _ in range(n_jobs):
                if args.mode == "eco":
                    jobs.append(build_eco_game_spec(
                        game_i, eco_starts, seed=args.seed, holdout_frac=args.holdout_frac,
                    ))
                    game_i += 1
                else:
                    fen = seed_fens[relabel_i % len(seed_fens)]
                    relabel_i += 1
                    jobs.append({
                        "relabel": True,
                        "fen": fen,
                        "game_id": game_i,
                        "split": 1 if random.Random(game_i + 17).random() < args.holdout_frac else 0,
                    })
                    game_i += 1
            for result in pool.imap_unordered(_run_job, jobs):
                for k, v in (result.get("rejects") or {}).items():
                    rejected[k] = rejected.get(k, 0) + int(v)
                for row in result.get("rows") or []:
                    key = row_key(row)
                    if key in pending_keys or seen.has(key):
                        rejected["dup"] += 1
                        continue
                    pending.append(row)
                    pending_keys.add(key)
                    seen.remember_hot([key])
                    new_rows += 1
                    if committed + new_rows >= args.target:
                        break
                flush(False)
                if committed + new_rows >= args.target:
                    break
            elapsed = max(time.time() - t0, 1e-6)
            log(
                f"progress committed={committed + new_rows:,}/{args.target:,} "
                f"rate={(new_rows / elapsed):.2f}/s pending={len(pending)} "
                f"rejects={rejected}",
                log_path,
            )
        flush(True)
    finally:
        if committed + new_rows >= args.target:
            pool.terminate()
        else:
            pool.close()
        pool.join()
        _close_worker()
    write_manifest(out, {
        "teacher": fp,
        "config": asdict(cfg),
        "run": run_fp,
        "rejected": rejected,
        "seconds": time.time() - t0,
    })
    log(f"done committed={committed + new_rows:,} rejects={rejected}", log_path)


def bench(args) -> None:
    sf = resolve_sf()
    fp = teacher_fingerprint(sf)
    require_sf19(fp)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": 16, "UCI_ShowWDL": True})
    boards = [chess.Board()]
    b = chess.Board()
    b.push_uci("e2e4")
    boards.append(b)
    boards.append(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1"))
    boards.append(chess.Board("8/8/8/8/8/5k2/8/4K3 w - - 0 1"))
    rows = []
    t0 = time.time()
    for board in boards:
        row = analyze_wdl(eng, board, nodes=int(args.nodes), tau=DEFAULT_TAU, watchdog_s=8.0)
        if row is None:
            raise SystemExit(f"bench failed to label {board.fen()}")
        rows.append(row)
    elapsed = max(time.time() - t0, 1e-6)
    gaps = []
    for row in rows:
        if int(row["wdl_source"]) != WDL_SOURCE_UCI:
            continue
        sig = sigmoid_wdl_white(int(row["cp"]), int(row["mate"]))
        gaps.append(float(np.abs(row["wdl"] - sig).sum()))
    report = {
        "teacher": fp["uci_name"],
        "n": len(rows),
        "seconds": elapsed,
        "labels_per_s": len(rows) / elapsed,
        "official_vs_sigmoid_l1": {
            "n": len(gaps),
            "mean": float(np.mean(gaps)) if gaps else None,
            "max": float(np.max(gaps)) if gaps else None,
            "note": "large L1 is expected: official WDL is much more drawish",
        },
        "rows": [
            {
                "n_pieces": int(r["n_pieces"]),
                "cp": int(r["cp"]),
                "mate": int(r["mate"]),
                "wdl": [float(x) for x in r["wdl"]],
                "wdl_raw": [int(x) for x in r["wdl_raw"]],
                "wdl_source": int(r["wdl_source"]),
            }
            for r in rows
        ],
    }
    (out / "bench.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps(report, indent=2))
    eng.quit()


def wdl_chunk_table(d: dict, name: str, start: int, end: int):
    from scripts.export_soft_caches_to_hf import _fixed_list, sf19_chunk_table
    import pyarrow as pa

    table = sf19_chunk_table(d, name, start, end)
    n = end - start
    if "wdl_raw" in d:
        raw = d["wdl_raw"][start:end].numpy().astype(np.int16, copy=False)
    else:
        raw = np.zeros((n, 3), dtype=np.int16)
    if "wdl_source" in d:
        src = d["wdl_source"][start:end].numpy().astype(np.int8, copy=False)
    else:
        src = np.ones(n, dtype=np.int8)
    table = table.append_column("wdl_raw", _fixed_list(raw, pa.int16(), 3))
    table = table.append_column("wdl_source", pa.array(src))
    return table


def _local_wdl_readme(repo: str, n_total: int, openings: dict, teacher: dict, stats: dict) -> str:
    cfg = (teacher.get("config") or {})
    run = (teacher.get("run") or {})
    fp = run.get("binary_sha256") or ""
    vols = openings.get("volumes") or {}
    rejected = stats.get("rejected") or {}
    analyze_fail = int(rejected.get("analyze_fail") or 0)
    no_wdl = int(rejected.get("no_wdl") or 0)
    nodes = int(cfg.get("nodes") or run.get("nodes") or 25_000)
    play_nodes = int(cfg.get("play_nodes") or 2_000)
    return f"""---
license: mit
pretty_name: Local WDL
task_categories:
- other
size_categories:
- 1M<n<10M
tags:
- chess
- stockfish-19
- wdl
- value
- official-wdl
configs:
- config_name: default
  data_files: data/*.parquet
---

# Local WDL

Value-only chess dataset. Each row is a unique board labeled with **official
Stockfish 19 `UCI_ShowWDL`**. Use `wdl` as the value target. Do not treat this
as MultiPV policy data.

**{n_total:,}** rows. Source id `4`. Compact move vocab (1968).

This is SF19's fishtest-LTC self-play WDL model (eval + remaining material).
It is **not** FIDE/Lichess Elo and **not** a sigmoid of `cp`. Official WDL is
much more drawish: a start-like +27cp is about 5% White win / 94% draw.

## Load

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")
train = ds.filter(lambda r: int(r["split"]) == 0)
holdout = ds.filter(lambda r: int(r["split"]) == 1)

# White-absolute value target: [P(White wins), P(draw), P(White loses)]
wdl = train[0]["wdl"]
```

Honor `split`. `split=1` is a 5% holdout. Do not invent a new hash holdout.

For value training, use `wdl` with KL / cross-entropy against a 3-class head
ordered win/draw/loss. Drop or keep `wdl_source==2` terminals explicitly.

```python
# Skip anything that is not official UCI WDL (should be none in this pack)
uci = train.filter(lambda r: int(r["wdl_source"]) == 1)
```

`soft_indices` / `soft_probs` are a single best-move slot (`n_soft=1`) so the
row stacks with the project's soft-cache schema. This pack is **not** MultiPV.
Do not train a policy from these probs as if they were an 8-move distribution.

## Columns

| column | type | meaning |
|---|---|---|
| `board_array` | list[int8] 64 | mailbox, a1=0. 0 empty, 1-6 White P,N,B,R,Q,K, 7-12 Black P,N,B,R,Q,K |
| `turn` | int8 | 0 White, 1 Black |
| `castling` | int8 | bits `K=8 Q=4 k=2 q=1` |
| `ep_square` | int8 | 0-63 or -1 |
| `wdl` | list[float32] 3 | **White-absolute** `[P(White wins), P(draw), P(White loses)]`, sums to 1 |
| `wdl_raw` | list[int16] 3 | same triple as UCI per-mille, sums to 1000 |
| `wdl_source` | int8 | `1` official UCI, `2` terminal mate/draw. Never sigmoid |
| `cp` | int32 | White-absolute centipawns. 0 if mate |
| `mate` | int32 | White-absolute mate distance. 0 if cp |
| `n_pieces` | int8 | pieces on the board (WDL is material-dependent) |
| `move_idx` | int64 | compact-vocab best move, or -1 |
| `soft_indices` | list[int64] 8 | best move in slot 0, pad `-1` |
| `soft_probs` | list[float32] 8 | `1.0` in slot 0, pad `0` |
| `n_soft` | int8 | `1` for labeled searches |
| `split` | int8 | `0` train, `1` holdout |
| `source` | int8 | `4` (SF19) |
| `phase` | int8 | `0` opening (≥26 pcs), `1` mid (≥14), `2` end |
| `ply` / `game_id` | int16 / int64 | harvest game |
| `nodes` / `nodes_budget` | int32 | achieved / requested search |
| `label_depth` | int16 | last complete PV depth |
| `policy_mask` | int8 | 1 if a best move is stored |

## Reconstruct a FEN

```python
import chess

ID_TO_SYMBOL = {{
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}}
CASTLE = ((8, "K"), (4, "Q"), (2, "k"), (1, "q"))

def row_to_fen(row) -> str:
    ranks = []
    ba = row["board_array"]
    for rank in range(7, -1, -1):
        empty = 0
        cells = []
        for file in range(8):
            pid = int(ba[rank * 8 + file])
            if pid <= 0:
                empty += 1
                continue
            if empty:
                cells.append(str(empty))
                empty = 0
            cells.append(ID_TO_SYMBOL[pid])
        if empty:
            cells.append(str(empty))
        ranks.append("".join(cells))
    castle = "".join(ch for bit, ch in CASTLE if int(row["castling"]) & bit) or "-"
    ep_i = int(row["ep_square"])
    ep = chess.square_name(ep_i) if 0 <= ep_i <= 63 else "-"
    stm = "b" if int(row["turn"]) else "w"
    return f"{{'/'.join(ranks)}} {{stm}} {{castle}} {{ep}} 0 1"
```

Clocks and repetition are unknown. The identity key is the 4-field position
(board, side, castling, ep).

## Teacher

- Stockfish 19, EvalFile `nn-1a298aa575a0.nnue`
- Binary SHA-256 `{fp}`
- Full strength, `Threads=1`, `Hash=64`, `UCI_ShowWDL=true`
- Label: **{nodes:,} nodes / MultiPV=1**
- Play on unlabeled plies: {play_nodes:,} nodes
- Rows without official WDL were dropped. `analyze_fail={analyze_fail}`, `no_wdl={no_wdl}`

ECO starts from Lichess openings: {int(openings.get('n_starts') or 0):,} unique
(A {int(vols.get('A') or 0):,} / B {int(vols.get('B') or 0):,} /
C {int(vols.get('C') or 0):,} / D {int(vols.get('D') or 0):,} /
E {int(vols.get('E') or 0):,}). Then SF19 vs SF19 with epsilon noise.

## Do not

- Convert `cp` through a sigmoid and call it WDL
- Mix these shards with MultiPV-8 soft-target out-dirs
- Treat `UCI_Elo` on the teacher dump as a human rating (LimitStrength was off)
- Train policy as if `soft_probs` were an 8-move teacher
- Ignore `split`

## Files

- `data/shard_XXXXXX.parquet` — 5,000 rows each
- `teacher.json`, `openings.json`, `manifest.json`
"""


def push_hf(args) -> None:
    from huggingface_hub import HfApi, create_repo
    import pyarrow.parquet as pq

    token = _hf_token()
    repo = args.repo
    out = Path(args.out_dir)
    inbox = out / "inbox"
    wait_s = float(getattr(args, "wait_s", 0) or 0)
    if wait_s > 0:
        log(f"waiting {wait_s:.0f}s for HF commit rate limit")
        time.sleep(wait_s)
    api = HfApi(token=token)
    last_create = None
    for i in range(5):
        try:
            create_repo(repo, repo_type="dataset", private=False, exist_ok=True, token=token)
            last_create = None
            break
        except Exception as exc:
            last_create = exc
            time.sleep(min(90, 8 * (2 ** i)))
    if last_create is not None:
        raise last_create
    staging = out / "hf_staging"
    data_dir = staging / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = out / "hf_upload.json"
    state = _load_json(state_path)
    uploaded = set(state.get("uploaded") or [])
    n_new = 0
    pending: list[str] = []
    shards = sorted(p for p in inbox.glob("shard_*") if (p / "READY").exists())
    for sh in shards:
        cache = sh / "soft_cache.pt"
        if not cache.exists():
            continue
        try:
            local_i = int(sh.name.split("_", 1)[1])
        except (IndexError, ValueError):
            local_i = 0
        remote_name = f"data/shard_{local_i:06d}.parquet"
        dest = data_dir / Path(remote_name).name
        if dest.exists() and dest.stat().st_size > 0:
            n = int(pq.read_metadata(dest).num_rows)
        else:
            d = torch.load(cache, map_location="cpu", weights_only=False)
            n = int(d["move_idx"].shape[0])
            pq.write_table(wdl_chunk_table(d, sh.name, 0, n), dest, compression="zstd")
            del d
        n_new += n
        if remote_name in uploaded:
            log(f"skip {remote_name} n={n} new={n_new:,}")
            continue
        pending.append(remote_name)
        log(f"staged {remote_name} n={n} pending={len(pending)}")
    extras = []
    for name in ("teacher.json", "manifest.json", "openings.json", "stats.json"):
        src = out / name
        if src.exists():
            dest = staging / name
            dest.write_bytes(src.read_bytes())
            extras.append(name)
    openings = _load_json(out / "openings.json")
    teacher = _load_json(out / "teacher.json")
    stats = _load_json(out / "stats.json")
    if not stats:
        stats = {"accepted": n_new, "rejected": {}}
    readme = staging / "README.md"
    readme.write_text(_local_wdl_readme(repo, n_new, openings, teacher, stats), encoding="utf-8")
    extras.append("README.md")
    allow = pending + extras
    if allow:
        log(f"folder upload files={len(allow)} rows={n_new:,}")
        last = None
        for i in range(5):
            try:
                api.upload_folder(
                    folder_path=str(staging),
                    repo_id=repo,
                    repo_type="dataset",
                    token=token,
                    allow_patterns=allow,
                    commit_message=f"Local WDL {n_new:,} rows ({len(pending)} shards)",
                )
                last = None
                break
            except Exception as exc:
                last = exc
                wait = min(600, 30 * (2 ** i))
                log(f"folder upload retry {i + 1}/5 wait={wait}s err={type(exc).__name__}: {exc}")
                time.sleep(wait)
        if last is not None:
            raise last
    uploaded.update(pending)
    state_path.write_text(json.dumps({
        "repo": repo, "uploaded": sorted(uploaded), "rows": n_new, "done": True,
    }, indent=2), encoding="utf-8")
    log(f"https://huggingface.co/datasets/{repo} rows={n_new:,}")


def apply_generate_defaults(args) -> None:
    if args.cmd != "generate":
        return
    if args.smoke:
        args.go = True
        args.target = min(int(args.target), 24)
        args.workers = min(int(args.workers), 2)
        args.nodes = min(int(args.nodes), 3_000)
        args.play_nodes = min(int(args.play_nodes), 400)
        args.ply_cap = min(int(args.ply_cap), 16)
        args.shard_size = min(int(args.shard_size), 16)
        args.watchdog_s = min(float(args.watchdog_s), 6.0)
    if args.pilot:
        args.target = min(int(args.target), 10_000)
    if args.workers == 8:
        args.workers = max(1, (os.cpu_count() or 8) - 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add_shared(p):
        p.add_argument("--out-dir", default="outputs/sf19_wdl/pilot")
        p.add_argument("--seed", type=int, default=19)
    b = sub.add_parser("bench")
    add_shared(b)
    b.add_argument("--nodes", type=int, default=8_000)
    g = sub.add_parser("generate")
    add_shared(g)
    g.add_argument("--go", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--target", type=int, default=1_000_000)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--nodes", type=int, default=25_000)
    g.add_argument("--play-nodes", type=int, default=2_000)
    g.add_argument("--epsilon", type=float, default=0.20)
    g.add_argument("--ply-stride", type=int, default=2)
    g.add_argument("--ply-skip-open", type=int, default=0)
    g.add_argument("--ply-cap", type=int, default=180)
    g.add_argument("--book-noise", type=int, default=2)
    g.add_argument("--watchdog-s", type=float, default=12.0)
    g.add_argument("--hash-mb", type=int, default=64)
    g.add_argument("--shard-size", type=int, default=5_000)
    g.add_argument("--holdout-frac", type=float, default=0.05)
    g.add_argument("--game-start", type=int, default=0)
    g.add_argument("--seed-caches", nargs="*", default=None)
    g.add_argument("--seed-fens-n", type=int, default=2048)
    g.add_argument("--exclude-caches", nargs="*", default=None)
    g.add_argument("--no-prefixes", action="store_true")
    g.add_argument("--mode", choices=("eco", "relabel"), default="eco")
    s = sub.add_parser("smoke")
    add_shared(s)
    s.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("push")
    add_shared(p)
    p.add_argument("--repo", default="avewright/local-wdl")
    p.add_argument("--wait-s", type=float, default=0,
                   help="Sleep first (HF commit rate limit). Remaining files go in one folder upload.")
    args = ap.parse_args()
    if args.cmd == "smoke":
        args.cmd = "generate"
        args.go = True
        args.smoke = True
        args.pilot = False
        args.mode = "eco"
        args.target = 24
        args.nodes = 3_000
        args.play_nodes = 400
        args.ply_cap = 16
        args.shard_size = 16
        args.holdout_frac = 0.0
        args.epsilon = 0.20
        args.ply_stride = 2
        args.ply_skip_open = 0
        args.book_noise = 1
        args.watchdog_s = 6.0
        args.hash_mb = 16
        args.game_start = 0
        args.seed_caches = None
        args.seed_fens_n = 0
        args.exclude_caches = None
        args.no_prefixes = False
        generate(args)
        return
    if args.cmd == "bench":
        bench(args)
        return
    if args.cmd == "push":
        push_hf(args)
        return
    if not args.go:
        raise SystemExit("pass --go")
    apply_generate_defaults(args)
    generate(args)


if __name__ == "__main__":
    main()
