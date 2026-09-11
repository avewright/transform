#!/usr/bin/env python3
"""Stockfish 19 soft-target dataset: SF vs SF games + optional local FEN seeds.

Policy softmax is side-to-move. Stored `cp`/`mate`/`wdl` are White-absolute so
`data_loader.compute_wdl(cp, mate)` (no turn) matches the training loader.

FEN-only inputs drop repetition history and (unless present) halfmove/fullmove.
We keep 4-field position keys for dedupe: board, side, castling, ep.

Usage:
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u scripts/sf19_soft_dataset.py bench --out-dir outputs/sf19_soft/pilot
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u scripts/sf19_soft_dataset.py generate --go --pilot \\
      --out-dir outputs/sf19_soft/pilot
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u scripts/sf19_soft_dataset.py generate --go --mode eco \\
      --out-dir outputs/sf19_soft/eco_1m --target 1000000 --workers 16
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u scripts/sf19_soft_dataset.py generate --go --mode piece_curve \\
      --out-dir outputs/sf19_soft/piece_curve --target 300000 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
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
import chess.polyglot
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import CASTLING_MAP, _fast_parse_fen, compute_wdl  # noqa: E402
from move_vocab import UCI_TO_IDX, VOCAB_SIZE  # noqa: E402
from scripts.lichess_openings import (  # noqa: E402
    load_openings as load_eco_openings,
    openings_summary,
    start_positions as eco_start_positions,
)
from scripts.piece_curve import (  # noqa: E402
    LABEL_WHEN_PIECES_LE,
    STREAMS,
    curve_summary,
    empty_counts,
    label_stride,
    local_copy_paths,
    local_fen_paths,
    most_deficit_n,
    n_pieces_from_row,
    recount_inbox_pieces,
    should_keep as curve_should_keep,
    target_pmf,
)

SOFT_K = 8
SOURCE_SF19 = 4
MATE_BASE = 100_000
DEFAULT_TAU = 120.0
ORIGIN_RELABEL = 0
ORIGIN_SELFPLAY = 1
FLAG_SHALLOW = 1 << 0
FLAG_BOUNDS = 1 << 1
FLAG_DECIDED = 1 << 2
SHALLOW_DEPTH = 8
DECIDED_CP = 400
OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4"],
    ["g1f3"],
    ["e2e4", "c7c6"],
    ["e2e4", "g8f6"],
    ["d2d4", "f7f5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["e2e4", "e7e5", "f2f4"],
    ["e2e4", "d7d5"],
    ["e2e4", "g7g6"],
    ["c2c4", "c7c5"],
    ["g1f3", "d7d5"],
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    ["e2e4", "c7c5", "g1f3", "b8c6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    ["b2b3"],
    ["g2g3"],
    ["e2e3"],
    ["f2f4"],
    ["b1c3"],
    ["a2a4"],
]


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def resolve_sf() -> str:
    for p in (
        os.environ.get("STOCKFISH_PATH", ""),
        str(Path.home() / ".local/bin/stockfish-19"),
        str(Path.home() / ".local/bin/stockfish"),
    ):
        if p and Path(p).exists():
            return str(Path(p).resolve())
    raise FileNotFoundError("Stockfish 19 binary not found")


def file_sha256(path: str) -> str:
    """Full-file SHA-256. Do not truncate the read; resume keys depend on it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def teacher_fingerprint(sf_path: str) -> dict:
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        name = str(eng.id.get("name") or "")
        author = str(eng.id.get("author") or "")
        opts = {k: str(v.default) for k, v in eng.options.items()}
    finally:
        eng.quit()
    src = Path.home() / ".local/src/Stockfish"
    rev = ""
    if (src / ".git").exists():
        head = (src / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        rev = head
        if head.startswith("ref:"):
            ref = src / ".git" / head.split(" ", 1)[1]
            if ref.exists():
                rev = ref.read_text(encoding="utf-8").strip()[:40]
    return {
        "uci_name": name,
        "author": author,
        "binary": sf_path,
        "binary_sha256": file_sha256(sf_path),
        "source_revision": rev,
        "eval_file": str(opts.get("EvalFile") or ""),
        "options_sample": {k: str(opts[k]) for k in list(opts)[:24]},
    }


LABEL_FINGERPRINT_KEYS = (
    "uci_name", "binary_sha256", "source_revision", "eval_file",
    "nodes", "multipv", "tau", "vocab", "soft_k", "source_id",
)


def run_fingerprint(sf_fp: dict, cfg: "GenConfig") -> dict:
    body = {
        "uci_name": sf_fp["uci_name"],
        "binary_sha256": sf_fp["binary_sha256"],
        "source_revision": sf_fp.get("source_revision") or "",
        "eval_file": sf_fp.get("eval_file") or "",
        "nodes": int(cfg.nodes),
        "multipv": int(cfg.multipv),
        "tau": float(cfg.tau),
        "vocab": int(VOCAB_SIZE),
        "soft_k": int(SOFT_K),
        "source_id": int(SOURCE_SF19),
    }
    payload = json.dumps({k: body[k] for k in LABEL_FINGERPRINT_KEYS}, sort_keys=True, separators=(",", ":"))
    body["fingerprint_id"] = hashlib.sha256(payload.encode()).hexdigest()
    return body


def assert_resume_compatible(out: Path, fp: dict) -> None:
    path = out / "teacher.json"
    if not path.exists():
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    old_id = old.get("fingerprint_id") or (old.get("run") or {}).get("fingerprint_id")
    if old_id and old_id != fp["fingerprint_id"]:
        raise SystemExit(
            f"teacher fingerprint mismatch in {path}: stored={old_id} now={fp['fingerprint_id']}. "
            "Use a new --out-dir rather than mixing label configs."
        )
    inbox = out / "inbox"
    for meta_p in inbox.glob("shard_*/meta.json"):
        if not (meta_p.parent / "READY").exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        sid = meta.get("fingerprint_id")
        if sid and sid != fp["fingerprint_id"]:
            raise SystemExit(f"shard {meta_p.parent.name} fingerprint {sid} != {fp['fingerprint_id']}")


def require_sf19(fp: dict) -> None:
    if "Stockfish 19" not in str(fp.get("uci_name") or ""):
        raise SystemExit(f"need Stockfish 19, got {fp.get('uci_name')!r}")


def mate_rank_score(mate: int) -> int:
    """STM score for a mate line: faster wins beat slower; slower losses beat faster."""
    m = int(mate)
    if m == 0:
        return 0
    return (1 if m > 0 else -1) * (MATE_BASE - min(abs(m), 1000))


def stm_rank_score(cp: int, mate: int) -> int:
    if int(mate) != 0:
        return mate_rank_score(mate)
    return int(cp)


def to_white_abs(cp: int, mate: int, turn_black: bool) -> tuple[int, int]:
    if not turn_black:
        return int(cp), int(mate)
    return -int(cp), -int(mate)


def softmax_from_scores(scores: list[float], tau: float) -> list[float]:
    if not scores:
        return []
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and > 0")
    mx = max(scores)
    exps = [math.exp((s - mx) / tau) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def phase_from_board(arr: np.ndarray) -> int:
    n = int(np.count_nonzero(arr))
    if n >= 26:
        return 0
    if n >= 14:
        return 1
    return 2


def position_key(board: chess.Board) -> str:
    return chess.polyglot.zobrist_hash(board).to_bytes(8, "little").hex()


def encode_board(board: chess.Board) -> tuple[np.ndarray, int, int, int] | None:
    arr = np.zeros(64, dtype=np.int8)
    try:
        turn, castling, ep = _fast_parse_fen(board.fen(), arr)
    except Exception:
        return None
    return arr, int(turn), int(castling), int(ep)


def wdl_white(info: dict, turn_black: bool) -> np.ndarray | None:
    wdl = info.get("wdl")
    if wdl is None:
        return None
    try:
        rel = wdl.relative if hasattr(wdl, "relative") else wdl
        trip = np.array([rel.wins, rel.draws, rel.losses], dtype=np.float32)
        trip = trip / max(float(trip.sum()), 1.0)
    except Exception:
        return None
    if turn_black:
        trip = trip[[2, 1, 0]]
    return trip


def _info_line(info: dict, board: chess.Board) -> dict | None:
    if info.get("upperbound") or info.get("lowerbound"):
        return None
    pv = info.get("pv") or []
    sc = info.get("score")
    if not pv or sc is None:
        return None
    pov = sc.pov(board.turn)
    if pov.is_mate():
        mate = int(pov.mate() or 0)
        cp = 0
    else:
        mate = 0
        raw = pov.score(mate_score=None)
        if raw is None:
            return None
        cp = int(raw)
    return {
        "uci": pv[0].uci(),
        "stm_cp": cp,
        "stm_mate": mate,
        "rank": stm_rank_score(cp, mate),
        "depth": int(info.get("depth") or 0),
        "nodes": int(info.get("nodes") or 0),
        "multipv": int(info.get("multipv") or 1),
        "wdl": wdl_white(info, board.turn == chess.BLACK),
    }


def last_complete_iteration(infos, board: chess.Board, *, k: int) -> tuple[list[dict], int, int] | None:
    """Keep the last depth that has K unbound MultiPV lines. Drop incomplete cuts."""
    if not isinstance(infos, list):
        infos = [infos]
    wanted = min(max(int(k), 1), SOFT_K, max(board.legal_moves.count(), 1))
    by_depth: dict[int, dict[int, dict]] = {}
    bound_skipped = 0
    for info in infos:
        if info.get("upperbound") or info.get("lowerbound"):
            bound_skipped += 1
            continue
        if info.get("depth") is None:
            continue
        rec = _info_line(info, board)
        if rec is None:
            bound_skipped += 1
            continue
        depth = int(rec["depth"])
        bucket = by_depth.setdefault(depth, {})
        mpv = int(rec["multipv"])
        if info.get("multipv") is None:
            mpv = len(bucket) + 1
            rec["multipv"] = mpv
        bucket[mpv] = rec
    complete = [d for d, m in by_depth.items() if len(m) >= wanted]
    if not complete:
        return None
    depth = max(complete)
    items = list(by_depth[depth].values())
    return items, depth, bound_skipped


def parse_multipv(infos, board: chess.Board, *, k: int, tau: float) -> dict | None:
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return {"terminal": True, "n_legal": 0}
    selected = last_complete_iteration(infos, board, k=k)
    if selected is None:
        return None
    raw_items, depth, bound_skipped = selected
    best: dict[str, dict] = {}
    nodes = 0
    for rec in raw_items:
        uci = rec["uci"]
        if uci not in best or rec["rank"] > best[uci]["rank"]:
            best[uci] = rec
        nodes = max(nodes, int(rec.get("nodes") or 0))
    if not best:
        return None
    items = sorted(best.values(), key=lambda r: -r["rank"])[: min(k, SOFT_K, n_legal)]
    probs = softmax_from_scores([float(it["rank"]) for it in items], tau)
    return {
        "terminal": False,
        "items": items,
        "probs": probs,
        "depth": depth,
        "nodes": nodes,
        "n_legal": n_legal,
        "bound_skipped": bound_skipped,
        "best_wdl": items[0].get("wdl"),
        "complete_iteration": True,
    }


def stream_multipv_infos(engine, board: chess.Board, *, nodes: int, multipv: int, watchdog_s: float) -> list[dict]:
    k = min(int(multipv), max(board.legal_moves.count(), 1), SOFT_K)
    limit = chess.engine.Limit(nodes=max(1, nodes), time=max(0.5, watchdog_s))
    out: list[dict] = []
    with engine.analysis(board, limit, multipv=k) as analysis:
        for info in analysis:
            out.append({
                "pv": list(info.get("pv") or []),
                "score": info.get("score"),
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
                "multipv": info.get("multipv"),
                "wdl": info.get("wdl"),
                "upperbound": info.get("upperbound"),
                "lowerbound": info.get("lowerbound"),
            })
    return out


def label_to_row(board: chess.Board, parsed: dict, *, tau: float, nodes_budget: int) -> dict | None:
    if parsed.get("terminal"):
        enc = encode_board(board)
        if enc is None:
            return None
        arr, turn, castling, ep = enc
        outcome = board.outcome(claim_draw=True)
        mate = 0
        cp = 0
        if outcome and outcome.winner is not None:
            mate = 1 if outcome.winner == chess.WHITE else -1
            cp = 0
        wdl = compute_wdl(
            torch.tensor([cp], dtype=torch.int32),
            torch.tensor([mate], dtype=torch.int32),
        )[0].numpy().astype(np.float32)
        return _with_row_meta({
            "board_array": arr,
            "turn": np.int8(turn),
            "castling": np.int8(castling),
            "ep_square": np.int8(ep),
            "move_idx": np.int64(-1),
            "cp": np.int32(cp),
            "mate": np.int32(mate),
            "soft_indices": np.full(SOFT_K, -1, dtype=np.int64),
            "soft_probs": np.zeros(SOFT_K, dtype=np.float32),
            "soft_cps": np.zeros(SOFT_K, dtype=np.int32),
            "soft_mates": np.zeros(SOFT_K, dtype=np.int32),
            "label_depth": np.int16(0),
            "phase": np.int8(phase_from_board(arr)),
            "source": np.int8(SOURCE_SF19),
            "wdl": wdl,
            "nodes": np.int32(0),
            "policy_mask": np.int8(0),
            "tau": np.float32(tau),
            "nodes_budget": np.int32(nodes_budget),
        }, parsed)
    enc = encode_board(board)
    if enc is None:
        return None
    arr, turn, castling, ep = enc
    turn_black = turn == 1
    items = parsed["items"]
    probs = parsed["probs"]
    soft_i = np.full(SOFT_K, -1, dtype=np.int64)
    soft_p = np.zeros(SOFT_K, dtype=np.float32)
    soft_c = np.zeros(SOFT_K, dtype=np.int32)
    soft_m = np.zeros(SOFT_K, dtype=np.int32)
    for i, (it, pr) in enumerate(zip(items, probs)):
        uci = it["uci"]
        if uci not in UCI_TO_IDX:
            continue
        soft_i[i] = UCI_TO_IDX[uci]
        soft_p[i] = float(pr)
        soft_c[i] = int(it["stm_cp"])
        soft_m[i] = int(it["stm_mate"])
    if soft_i[0] < 0:
        return None
    z = float(soft_p.sum()) or 1.0
    soft_p = (soft_p / z).astype(np.float32)
    w_cp, w_mate = to_white_abs(items[0]["stm_cp"], items[0]["stm_mate"], turn_black)
    wdl = items[0].get("wdl")
    if wdl is None:
        wdl = compute_wdl(
            torch.tensor([w_cp], dtype=torch.int32),
            torch.tensor([w_mate], dtype=torch.int32),
        )[0].numpy().astype(np.float32)
    return _with_row_meta({
        "board_array": arr,
        "turn": np.int8(turn),
        "castling": np.int8(castling),
        "ep_square": np.int8(ep),
        "move_idx": np.int64(soft_i[0]),
        "cp": np.int32(w_cp),
        "mate": np.int32(w_mate),
        "soft_indices": soft_i,
        "soft_probs": soft_p,
        "soft_cps": soft_c,
        "soft_mates": soft_m,
        "label_depth": np.int16(parsed["depth"]),
        "phase": np.int8(phase_from_board(arr)),
        "source": np.int8(SOURCE_SF19),
        "wdl": np.asarray(wdl, dtype=np.float32),
        "nodes": np.int32(parsed.get("nodes") or 0),
        "policy_mask": np.int8(1),
        "tau": np.float32(tau),
        "nodes_budget": np.int32(nodes_budget),
    }, parsed)


def compute_flags(row: dict, parsed: dict | None = None) -> int:
    flags = 0
    if int(row.get("policy_mask", 1)) == 0 or int(row.get("label_depth", 0)) < SHALLOW_DEPTH:
        flags |= FLAG_SHALLOW
    if int((parsed or {}).get("bound_skipped") or row.get("bound_skipped") or 0) > 0:
        flags |= FLAG_BOUNDS
    if int(row.get("mate") or 0) != 0 or abs(int(row.get("cp") or 0)) >= DECIDED_CP:
        flags |= FLAG_DECIDED
    return flags


def _with_row_meta(row: dict, parsed: dict | None = None) -> dict:
    row.setdefault("game_id", np.int64(-1))
    row.setdefault("ply", np.int16(-1))
    row.setdefault("split", np.int8(0))
    row.setdefault("origin", np.int8(ORIGIN_RELABEL))
    row.setdefault("bound_skipped", np.int16(int((parsed or {}).get("bound_skipped") or 0)))
    row.setdefault("flags", np.int16(compute_flags(row, parsed)))
    return row


def analyze_board(engine, board: chess.Board, *, nodes: int, multipv: int, tau: float, watchdog_s: float):
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return label_to_row(board, {"terminal": True}, tau=tau, nodes_budget=nodes)
    try:
        infos = stream_multipv_infos(engine, board, nodes=nodes, multipv=multipv, watchdog_s=watchdog_s)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
        if _W_ENGINE is engine:
            try:
                _start_engine()
            except Exception:
                pass
        return None
    parsed = parse_multipv(infos, board, k=min(int(multipv), n_legal, SOFT_K), tau=tau)
    if parsed is None:
        return None
    return label_to_row(board, parsed, tau=tau, nodes_budget=nodes)


def row_move_ranks(row: dict) -> dict[int, int]:
    ranks = {}
    for idx, cp, mate in zip(row["soft_indices"], row["soft_cps"], row["soft_mates"]):
        if int(idx) < 0:
            continue
        ranks[int(idx)] = stm_rank_score(int(cp), int(mate))
    return ranks


def ref_move_regret(ref: dict, cand: dict) -> tuple[float | None, bool]:
    """Reference STM rank of its best move minus its rank of the candidate's pick."""
    ranks = row_move_ranks(ref)
    if not ranks:
        return None, True
    ref_best = max(ranks.values())
    chosen = int(cand["move_idx"])
    if chosen not in ranks:
        return None, True
    return float(ref_best - ranks[chosen]), False


def union_kl_and_coverage(ref: dict, cand: dict, *, eps: float = 1e-8) -> tuple[float, float, float]:
    """KL(ref || cand) on the union of move supports. Missing mass is not dropped."""
    rmap = {int(i): float(p) for i, p in zip(ref["soft_indices"], ref["soft_probs"]) if int(i) >= 0 and p > 0}
    qmap = {int(i): float(p) for i, p in zip(cand["soft_indices"], cand["soft_probs"]) if int(i) >= 0 and p > 0}
    keys = sorted(set(rmap) | set(qmap))
    if not keys:
        return float("nan"), 0.0, 0.0
    r = np.array([rmap.get(k, 0.0) for k in keys], dtype=np.float64)
    q = np.array([qmap.get(k, 0.0) for k in keys], dtype=np.float64)
    r = r + eps
    q = q + eps
    r = r / r.sum()
    q = q / q.sum()
    kl = float((r * np.log(r / q)).sum())
    ref_mass_in_cand = float(sum(rmap[k] for k in rmap if k in qmap))
    cand_mass_in_ref = float(sum(qmap[k] for k in qmap if k in rmap))
    return kl, ref_mass_in_cand, cand_mass_in_ref


ECO_EPSILONS = (0.10, 0.18, 0.28, 0.40)
ECO_WILDS = (0.00, 0.03, 0.06, 0.10)


def pick_play_move(
    parsed: dict | None,
    board: chess.Board,
    rng: random.Random,
    *,
    epsilon: float,
    wild: float = 0.0,
) -> chess.Move:
    legal = list(board.legal_moves)
    if not legal:
        raise RuntimeError("no legal moves")
    if wild > 0 and rng.random() < wild:
        return rng.choice(legal)
    if parsed and not parsed.get("terminal") and parsed.get("items"):
        best = chess.Move.from_uci(parsed["items"][0]["uci"])
        if best not in board.legal_moves:
            best = legal[0]
        if rng.random() >= epsilon:
            return best
        k = min(4, len(parsed["items"]))
        ucis = [it["uci"] for it in parsed["items"][:k]]
        weights = parsed["probs"][:k]
        pick = rng.choices(ucis, weights=weights, k=1)[0]
        mv = chess.Move.from_uci(pick)
        return mv if mv in board.legal_moves else best
    if rng.random() < epsilon:
        return rng.choice(legal)
    return legal[0]


def build_eco_game_spec(
    game_i: int,
    starts: list[dict],
    *,
    seed: int,
    holdout_frac: float,
) -> dict:
    start = starts[game_i % len(starts)]
    split = 1 if random.Random(game_i + 17).random() < holdout_frac else 0
    return {
        "game_id": game_i,
        "seed": seed + game_i * 10007,
        "start_fen": start["fen"],
        "opening": [],
        "book_noise": game_i % 5,
        "epsilon": ECO_EPSILONS[game_i % len(ECO_EPSILONS)],
        "wild": ECO_WILDS[(game_i // 3) % len(ECO_WILDS)],
        "split": split,
        "eco": start.get("eco"),
        "opening_name": start.get("name"),
    }


def build_piece_curve_game_spec(
    game_i: int,
    starts: list[dict],
    *,
    seed: int,
    holdout_frac: float,
) -> dict:
    """Full ECO game. Rotate start depth so the N(17,6) piece curve can fill."""
    spec = build_eco_game_spec(game_i, starts, seed=seed, holdout_frac=holdout_frac)
    stream = STREAMS[game_i % len(STREAMS)]
    spec["piece_curve"] = True
    spec["stream"] = stream
    spec["label_when_pieces_le"] = LABEL_WHEN_PIECES_LE[stream]
    if stream == "endgame":
        spec["wild"] = max(float(spec["wild"]), 0.12)
        spec["epsilon"] = max(float(spec["epsilon"]), 0.28)
    return spec


ROW_STACK_KEYS = (
    "board_array", "turn", "castling", "ep_square", "move_idx",
    "cp", "mate", "soft_indices", "soft_probs", "soft_cps", "soft_mates",
    "label_depth", "phase", "source", "wdl", "nodes", "policy_mask",
    "tau", "nodes_budget", "game_id", "ply", "split",
    "origin", "flags", "bound_skipped",
)


def _as_numpy(val, dtype=None):
    if torch.is_tensor(val):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def normalize_harvest_row(row: dict) -> dict | None:
    """Pad older caches so they stack with eco/prod shards."""
    if "board_array" not in row or "move_idx" not in row:
        return None
    if "soft_indices" not in row or "soft_probs" not in row:
        return None
    out = dict(row)
    out["board_array"] = _as_numpy(out["board_array"], np.int8).reshape(64)
    out["soft_indices"] = _as_numpy(out["soft_indices"], np.int64).reshape(-1)
    out["soft_probs"] = _as_numpy(out["soft_probs"], np.float32).reshape(-1)
    if out["soft_indices"].shape[0] < SOFT_K:
        return None
    out["soft_indices"] = out["soft_indices"][:SOFT_K]
    out["soft_probs"] = out["soft_probs"][:SOFT_K]
    if "soft_cps" not in out:
        out["soft_cps"] = np.zeros(SOFT_K, dtype=np.int32)
    if "soft_mates" not in out:
        out["soft_mates"] = np.zeros(SOFT_K, dtype=np.int32)
    defaults = {
        "turn": (np.int8, 0),
        "castling": (np.int8, 0),
        "ep_square": (np.int8, -1),
        "move_idx": (np.int64, -1),
        "cp": (np.int32, 0),
        "mate": (np.int32, 0),
        "label_depth": (np.int16, 12),
        "phase": (np.int8, 1),
        "source": (np.int8, SOURCE_SF19),
        "nodes": (np.int32, 0),
        "policy_mask": (np.int8, 1),
        "tau": (np.float32, DEFAULT_TAU),
        "nodes_budget": (np.int32, 100_000),
        "game_id": (np.int64, -1),
        "ply": (np.int16, 0),
        "split": (np.int8, 0),
        "origin": (np.int8, ORIGIN_RELABEL),
        "flags": (np.int16, 0),
        "bound_skipped": (np.int16, 0),
    }
    for key, (dtype, fill) in defaults.items():
        out[key] = _as_numpy(out[key], dtype) if key in out else np.asarray(fill, dtype=dtype)
    if "wdl" not in out:
        out["wdl"] = np.array([0.33, 0.34, 0.33], dtype=np.float32)
    else:
        out["wdl"] = _as_numpy(out["wdl"], np.float32).reshape(-1)[:3]
    if int(out["policy_mask"]) == 0 or int(out["move_idx"]) < 0:
        return None
    return out


def iter_cache_rows(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    n = int(data["move_idx"].shape[0])
    keys = [k for k in data if hasattr(data[k], "__getitem__") and getattr(data[k], "shape", None) is not None and int(data[k].shape[0]) == n]
    for i in range(n):
        yield {k: data[k][i] for k in keys}
    del data


def collect_fen_buckets(paths: list[Path], rng: random.Random, *, per_n: int = 2500) -> dict[int, list[str]]:
    from scripts.harvest_exp201_lapses import board_array_to_fen

    buckets: dict[int, list[str]] = {n: [] for n in range(2, 33)}
    for path in paths:
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        n = int(data["move_idx"].shape[0])
        take = min(n, 80_000)
        idx = rng.sample(range(n), k=take) if take < n else list(range(n))
        for i in idx:
            pcs = int((data["board_array"][i] != 0).sum())
            if pcs < 2 or pcs > 32 or len(buckets[pcs]) >= per_n:
                continue
            fen = board_array_to_fen(
                data["board_array"][i].numpy(),
                int(data["turn"][i]),
                int(data["castling"][i]),
                int(data["ep_square"][i]),
            )
            buckets[pcs].append(fen)
        del data
    master = ROOT / "outputs/chess_master_v1/positions"
    if master.is_dir():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pq = None
        if pq is not None:
            for fp in sorted(master.glob("mix-*.parquet"))[:12]:
                table = pq.read_table(fp, columns=["fen_4"])
                fens = table.column("fen_4").to_pylist()
                for fen4 in rng.sample(fens, k=min(len(fens), 4000)):
                    if not fen4:
                        continue
                    try:
                        board = chess.Board(str(fen4) + " 0 1")
                    except ValueError:
                        continue
                    pcs = int(board.occupied.bit_count())
                    if 2 <= pcs <= 32 and len(buckets[pcs]) < per_n:
                        buckets[pcs].append(board.fen())
    for n in buckets:
        rng.shuffle(buckets[n])
    return buckets


def ingest_existing_curve(
    *,
    inbox: Path,
    seen,
    target: int,
    shard_size: int,
    pmf,
    have,
    teacher_meta: dict,
    log_path: Path,
) -> int:
    """Copy local SF19 rows into the bell. No new Stockfish search."""
    pending: list[dict] = []
    pending_keys: set[bytes] = set()
    kept = 0
    scanned = 0
    skipped = {"dup": 0, "curve": 0, "bad": 0}

    def flush() -> None:
        nonlocal pending, pending_keys
        if len(pending) < shard_size:
            return
        take = pending[:shard_size]
        pending = pending[shard_size:]
        keys = [row_key(r) for r in take]
        data = stack_rows(take)
        sh = next_shard_dir(inbox)
        write_shard(data, sh, {**teacher_meta, "origin": "ingest", "max_game_id": -1})
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        seen.forget_hot(keys)
        pending_keys.difference_update(keys)
        log(f"ingest wrote {sh} n={len(take)} copied={kept:,}", log_path)

    sources = local_copy_paths(ROOT)
    log(f"ingest sources={len(sources)} target={target:,}", log_path)
    for path in sources:
        if kept >= target:
            break
        try:
            rows = iter_cache_rows(path)
        except Exception as exc:
            log(f"ingest skip {path}: {type(exc).__name__}: {exc}", log_path)
            continue
        for raw in rows:
            scanned += 1
            row = normalize_harvest_row(raw)
            if row is None:
                skipped["bad"] += 1
                continue
            k = row_key(row)
            if seen.has(k) or k in pending_keys:
                skipped["dup"] += 1
                continue
            n_pcs = n_pieces_from_row(row)
            if not curve_should_keep(n_pcs, have, int(have.sum()), pmf):
                skipped["curve"] += 1
                continue
            have[n_pcs] += 1
            pending.append(row)
            pending_keys.add(k)
            seen.remember_hot([k])
            kept += 1
            if len(pending) >= shard_size:
                flush()
            if kept >= target:
                break
        if scanned and scanned % 200_000 < 5000:
            log(
                f"ingest scanned={scanned:,} kept={kept:,} {curve_summary(have)} "
                f"skip={skipped}",
                log_path,
            )
    if pending:
        take = pending
        data = stack_rows(take)
        sh = next_shard_dir(inbox)
        write_shard(data, sh, {**teacher_meta, "origin": "ingest", "max_game_id": -1})
        keys = [row_key(r) for r in take]
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        seen.forget_hot(keys)
        log(f"ingest wrote {sh} n={len(take)} copied={kept:,}", log_path)
    log(f"ingest done kept={kept:,} scanned={scanned:,} skip={skipped} {curve_summary(have)}", log_path)
    return kept


@dataclass
class GenConfig:
    nodes: int = 30_000
    play_nodes: int = 4_000
    multipv: int = 8
    tau: float = DEFAULT_TAU
    epsilon: float = 0.20
    ply_stride: int = 2
    ply_skip_open: int = 4
    ply_cap: int = 160
    book_noise: int = 2
    watchdog_s: float = 8.0
    hash_mb: int = 64
    clear_hash_every: int = 0


_W_ENGINE = None
_W_SF = None
_W_CFG: GenConfig | None = None
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
    _W_ENGINE = chess.engine.SimpleEngine.popen_uci(_W_SF)
    _W_ENGINE.configure({
        "Threads": 1,
        "Hash": int(_W_CFG.hash_mb),
        "UCI_ShowWDL": True,
    })


def _init_worker(sf_path: str, cfg: GenConfig, seen_path: str | None = None):
    global _W_SF, _W_CFG, _W_SEEN
    import atexit
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _W_SF = sf_path
    _W_CFG = cfg
    _W_SEEN = SeenDB(Path(seen_path), readonly=True) if seen_path else None
    _start_engine()
    atexit.register(_close_worker)


def _label_fen(fen: str) -> dict | None:
    assert _W_ENGINE is not None and _W_CFG is not None
    board = chess.Board(fen)
    return analyze_board(
        _W_ENGINE, board,
        nodes=_W_CFG.nodes, multipv=_W_CFG.multipv,
        tau=_W_CFG.tau, watchdog_s=_W_CFG.watchdog_s,
    )


def _play_one_game(spec: dict) -> dict:
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
    rejects = {"seen_skip": 0, "stride": 0, "analyze_fail": 0, "adjacent": 0}
    ply = 0
    last_kept_ply = -999
    n_search = 0
    piece_curve = bool(spec.get("piece_curve"))
    label_le = int(spec.get("label_when_pieces_le") or 32)
    while not board.is_game_over(claim_draw=True) and ply < cfg.ply_cap:
        n_pcs = int(board.occupied.bit_count())
        stride = label_stride(n_pcs, default=cfg.ply_stride) if piece_curve else cfg.ply_stride
        keep = (
            ply >= cfg.ply_skip_open
            and n_pcs <= label_le
            and (ply - cfg.ply_skip_open) % stride == 0
            and (ply - last_kept_ply) >= stride
        )
        parsed = None
        if keep and _W_SEEN is not None:
            enc = encode_board(board)
            if enc is not None:
                arr, turn, castling, ep = enc
                if _W_SEEN.has(compact_key_bytes(arr, turn, castling, ep)):
                    rejects["seen_skip"] += 1
                    keep = False
        if keep:
            n_search += 1
            if cfg.clear_hash_every and n_search % cfg.clear_hash_every == 0:
                try:
                    _W_ENGINE.configure({"Clear Hash": True})
                except Exception:
                    pass
            row = analyze_board(
                _W_ENGINE, board,
                nodes=cfg.nodes, multipv=cfg.multipv,
                tau=cfg.tau, watchdog_s=cfg.watchdog_s,
            )
            if row is None:
                rejects["analyze_fail"] += 1
            elif int(row["policy_mask"]) == 0:
                pass
            else:
                row["game_id"] = np.int64(spec["game_id"])
                row["ply"] = np.int16(ply)
                row["split"] = np.int8(spec.get("split") or 0)
                rows.append(row)
                last_kept_ply = ply
                ucis, weights = [], []
                for idx, pr in zip(row["soft_indices"].tolist(), row["soft_probs"].tolist()):
                    if idx < 0 or pr <= 0:
                        continue
                    for mv in board.legal_moves:
                        if UCI_TO_IDX.get(mv.uci()) == idx:
                            ucis.append(mv.uci())
                            weights.append(pr)
                            break
                if ucis:
                    parsed = {"terminal": False, "items": [{"uci": u} for u in ucis], "probs": weights}
        else:
            rejects["stride"] += 1
        if board.is_game_over(claim_draw=True):
            break
        if parsed and parsed.get("items"):
            mv = pick_play_move(parsed, board, rng, epsilon=eps, wild=wild)
        else:
            play = analyze_board(
                _W_ENGINE, board,
                nodes=cfg.play_nodes, multipv=1,
                tau=cfg.tau, watchdog_s=min(cfg.watchdog_s, 3.0),
            )
            parsed_play = None
            if play is not None and int(play["move_idx"]) >= 0:
                for cand in board.legal_moves:
                    if UCI_TO_IDX.get(cand.uci()) == int(play["move_idx"]):
                        parsed_play = {"items": [{"uci": cand.uci()}], "probs": [1.0]}
                        break
            mv = pick_play_move(parsed_play, board, rng, epsilon=eps, wild=wild)
        board.push(mv)
        ply += 1
    return {"rows": rows, "rejects": rejects, "game_id": spec["game_id"], "plies": ply}


def _run_job(spec: dict) -> dict:
    if spec.get("relabel"):
        empty = {"seen_skip": 0, "stride": 0, "analyze_fail": 0, "adjacent": 0}
        row = _label_fen(str(spec["fen"]))
        if row is None:
            empty["analyze_fail"] = 1
            return {"rows": [], "rejects": empty, "game_id": spec["game_id"], "plies": 0}
        if int(row.get("policy_mask", 0)) == 0:
            return {"rows": [], "rejects": empty, "game_id": spec["game_id"], "plies": 0}
        row["game_id"] = np.int64(spec["game_id"])
        row["ply"] = np.int16(0)
        row["split"] = np.int8(spec.get("split") or 0)
        row["origin"] = np.int8(ORIGIN_RELABEL)
        return {"rows": [row], "rejects": empty, "game_id": spec["game_id"], "plies": 0}
    return _play_one_game(spec)


def stack_rows(rows: list[dict]) -> dict:
    keys = [
        "board_array", "turn", "castling", "ep_square", "move_idx",
        "cp", "mate", "soft_indices", "soft_probs", "soft_cps", "soft_mates",
        "label_depth", "phase", "source", "wdl", "nodes", "policy_mask",
        "tau", "nodes_budget", "game_id", "ply", "split",
        "origin", "flags", "bound_skipped",
    ]
    out = {}
    for k in keys:
        out[k] = torch.from_numpy(np.stack([r[k] for r in rows]))
    return out


def write_shard(data: dict, dest: Path, meta: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / "soft_cache.pt"
    tmp = cache.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, cache)
    n = int(data["move_idx"].shape[0])
    payload = {"n": n, **meta}
    (dest / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (dest / "READY").write_text(f"n={n}\n", encoding="utf-8")


def compact_key_bytes(board_array, turn, castling, ep_square) -> bytes:
    raw = np.asarray(board_array, dtype=np.int8).reshape(64).tobytes()
    tail = bytes([int(turn) & 0xFF, int(castling) & 0xFF, (int(ep_square) + 1) & 0xFF])
    return hashlib.blake2s(raw + tail, digest_size=8).digest()


def compact_key_from_board(board: chess.Board) -> bytes | None:
    enc = encode_board(board)
    if enc is None:
        return None
    arr, turn, castling, ep = enc
    return compact_key_bytes(arr, turn, castling, ep)


class SeenDB:
    """Disk-backed 8-byte keys. No full in-memory copy of the key set."""

    def __init__(self, path: Path, *, readonly: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if readonly:
            if not path.exists():
                self.con = sqlite3.connect(":memory:")
            else:
                self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        else:
            self.con = sqlite3.connect(str(path))
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA synchronous=NORMAL")
            self.con.execute("CREATE TABLE IF NOT EXISTS seen (k BLOB PRIMARY KEY)")
            self.con.execute("CREATE TABLE IF NOT EXISTS shards (name TEXT PRIMARY KEY, n INTEGER)")
            self.con.commit()
        self.readonly = readonly
        self._hot: set[bytes] = set()

    def __len__(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM seen").fetchone()[0])

    def has(self, key: bytes) -> bool:
        if key in self._hot:
            return True
        row = self.con.execute("SELECT 1 FROM seen WHERE k = ?", (key,)).fetchone()
        return row is not None

    def has_many(self, keys: list[bytes]) -> set[bytes]:
        found: set[bytes] = set()
        unknown: list[bytes] = []
        for k in keys:
            if k in self._hot:
                found.add(k)
            else:
                unknown.append(k)
        for i in range(0, len(unknown), 400):
            chunk = unknown[i:i + 400]
            q = f"SELECT k FROM seen WHERE k IN ({','.join('?' * len(chunk))})"
            found.update(r[0] for r in self.con.execute(q, chunk))
        return found

    def add_many(self, keys: list[bytes]) -> int:
        if self.readonly or not keys:
            return 0
        before = len(self)
        self.con.executemany("INSERT OR IGNORE INTO seen(k) VALUES (?)", [(k,) for k in keys])
        self.con.commit()
        return len(self) - before

    def remember_hot(self, keys: list[bytes]) -> None:
        self._hot.update(keys)

    def forget_hot(self, keys: list[bytes]) -> None:
        self._hot.difference_update(keys)

    def mark_shard(self, name: str, n: int) -> None:
        if self.readonly:
            return
        self.con.execute("INSERT OR REPLACE INTO shards(name, n) VALUES (?, ?)", (name, n))
        self.con.commit()

    def known_shards(self) -> set[str]:
        try:
            return {r[0] for r in self.con.execute("SELECT name FROM shards")}
        except sqlite3.OperationalError:
            return set()

    def ingest_cache_keys(self, cache: Path, name: str | None = None) -> int:
        cache = Path(cache)
        data = torch.load(cache, map_location="cpu", weights_only=False)
        n_rows = int(data["move_idx"].shape[0])
        keys = [
            compact_key_bytes(data["board_array"][i], data["turn"][i], data["castling"][i], data["ep_square"][i])
            for i in range(n_rows)
        ]
        n = self.add_many(keys)
        self.mark_shard(name or cache.name, n_rows)
        del data
        return n

    def ingest_shard_keys(self, shard_dir: Path) -> int:
        return self.ingest_cache_keys(shard_dir / "soft_cache.pt", shard_dir.name)


def next_shard_dir(inbox: Path) -> Path:
    n = 0
    for p in inbox.glob("shard_*"):
        try:
            n = max(n, int(p.name.split("_", 1)[1]) + 1)
        except ValueError:
            continue
    return inbox / f"shard_{n:06d}"


def inbox_state(inbox: Path) -> tuple[int, int]:
    """Committed rows and next game_id from READY shard metadata when possible."""
    n_rows = 0
    next_game = 0
    for sh in sorted(inbox.glob("shard_*")):
        if not (sh / "READY").exists() or not (sh / "soft_cache.pt").exists():
            continue
        meta = {}
        if (sh / "meta.json").exists():
            try:
                meta = json.loads((sh / "meta.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        n = int(meta.get("n") or 0)
        mg = meta.get("max_game_id")
        if n <= 0 or mg is None:
            data = torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)
            n = int(data["move_idx"].shape[0])
            if "game_id" in data:
                mg = int(data["game_id"].max())
            del data
        n_rows += n
        if mg is not None:
            next_game = max(next_game, int(mg) + 1)
    return n_rows, next_game


def write_manifest(out: Path, extra: dict | None = None) -> None:
    inbox = out / "inbox"
    shards = []
    bytes_total = 0
    n_total = 0
    for sh in sorted(inbox.glob("shard_*")):
        cache = sh / "soft_cache.pt"
        if not (sh / "READY").exists() or not cache.exists():
            continue
        n = 0
        meta = {}
        if (sh / "meta.json").exists():
            meta = json.loads((sh / "meta.json").read_text(encoding="utf-8"))
            n = int(meta.get("n") or 0)
        sz = cache.stat().st_size
        bytes_total += sz
        n_total += n
        shards.append({"name": sh.name, "n": n, "bytes": sz, **{k: meta[k] for k in ("nodes", "multipv", "tau") if k in meta}})
    payload = {
        "shards": shards,
        "n_total": n_total,
        "bytes_total": bytes_total,
        "bytes_per_position": (bytes_total / n_total) if n_total else 0,
        **(extra or {}),
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def row_key(row: dict) -> bytes:
    return compact_key_bytes(row["board_array"], row["turn"], row["castling"], row["ep_square"])


def sample_seed_fens(paths: list[Path], n: int, rng: random.Random) -> list[str]:
    """Phase-stratified start FENs so self-play is not opening-heavy."""
    from scripts.harvest_exp201_lapses import board_array_to_fen

    buckets: dict[int, list[str]] = {0: [], 1: [], 2: []}
    per_path = max(n, 512)
    for p in paths:
        if not p.exists():
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        n_rows = int(data["move_idx"].shape[0])
        take = min(n_rows, per_path)
        idx = rng.sample(range(n_rows), k=take)
        phase = data["phase"].view(-1).numpy() if "phase" in data else None
        for i in idx:
            fen = board_array_to_fen(
                data["board_array"][i].numpy(),
                int(data["turn"][i]),
                int(data["castling"][i]),
                int(data["ep_square"][i]),
            )
            ph = int(phase[i]) if phase is not None else 1
            buckets.setdefault(ph if ph in buckets else 1, []).append(fen)
        del data
    out: list[str] = []
    while len(out) < n and any(buckets.values()):
        for ph in (0, 1, 2):
            if buckets[ph]:
                out.append(buckets[ph].pop())
            if len(out) >= n:
                break
    rng.shuffle(out)
    return out[:n]


def bench(args) -> dict:
    sf = resolve_sf()
    fp = teacher_fingerprint(sf)
    require_sf19(fp)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "bench.log"
    log(f"teacher {json.dumps(fp)}", log_path)
    rng = random.Random(args.seed)
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
    # diverse starts via short random walks
    for i in range(max(32, args.bench_n) - 1):
        b = chess.Board()
        for uci in OPENINGS[i % len(OPENINGS)]:
            m = chess.Move.from_uci(uci)
            if m in b.legal_moves:
                b.push(m)
        for _ in range(rng.randint(4, 18)):
            if b.is_game_over(claim_draw=True):
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over(claim_draw=True):
            fens.append(b.fen())
    fens = fens[: args.bench_n]
    ref_nodes = args.ref_nodes
    configs = [
        (4, 10_000), (8, 10_000),
        (4, 30_000), (8, 30_000),
        (4, 100_000), (8, 100_000),
    ]
    def _fresh_engine():
        eng = chess.engine.SimpleEngine.popen_uci(sf)
        eng.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
        return eng

    def _clear(eng):
        try:
            eng.configure({"Clear Hash": True})
        except Exception:
            pass

    cand_engine = _fresh_engine()
    ref_engine = _fresh_engine()
    results = []
    # Candidates first on a clean engine so they cannot reuse reference hash.
    cand_rows: dict[tuple[int, int], list] = {}
    cand_time: dict[tuple[int, int], float] = {}
    for mpv, nodes in configs:
        rows = []
        t1 = time.time()
        for fen in fens:
            _clear(cand_engine)
            rows.append(analyze_board(cand_engine, chess.Board(fen), nodes=nodes, multipv=mpv, tau=args.tau, watchdog_s=12.0))
        cand_time[(mpv, nodes)] = max(time.time() - t1, 1e-6)
        cand_rows[(mpv, nodes)] = rows
        log(f"cand multipv={mpv} nodes={nodes} {len(fens)/cand_time[(mpv, nodes)]:.2f} pos/s", log_path)
    refs = []
    t0 = time.time()
    for fen in fens:
        _clear(ref_engine)
        refs.append(analyze_board(ref_engine, chess.Board(fen), nodes=ref_nodes, multipv=8, tau=args.tau, watchdog_s=20.0))
    log(f"reference n={len(fens)} nodes={ref_nodes} {len(fens)/max(time.time()-t0,1e-6):.2f} pos/s", log_path)
    for mpv, nodes in configs:
        agree = regret_sum = kl_sum =         n_ok = val = 0.0
        missing = 0
        cov_ref = cov_cand = 0.0
        n = 0
        for ref, row in zip(refs, cand_rows[(mpv, nodes)]):
            if ref is None or int(ref["policy_mask"]) == 0:
                continue
            n += 1
            if row is None:
                continue
            n_ok += 1
            agree += float(int(row["move_idx"]) == int(ref["move_idx"]))
            reg, miss = ref_move_regret(ref, row)
            if miss or reg is None:
                missing += 1
            else:
                regret_sum += reg
            val += float(np.abs(row["wdl"].astype(np.float64) - ref["wdl"].astype(np.float64)).sum())
            kl, ref_in_q, q_in_ref = union_kl_and_coverage(ref, row)
            if math.isfinite(kl):
                kl_sum += kl
            cov_ref += ref_in_q
            cov_cand += q_in_ref
        dt = cand_time[(mpv, nodes)]
        scored = n_ok - missing
        rec = {
            "multipv": mpv,
            "nodes": nodes,
            "n": n,
            "ok": n_ok,
            "top1_agree": agree / max(n_ok, 1),
            "mean_ref_regret": (regret_sum / scored) if scored else float("inf"),
            "missing_from_ref_frac": missing / max(n_ok, 1),
            "mean_wdl_l1": val / max(n_ok, 1),
            "mean_kl_union": kl_sum / max(n_ok, 1),
            "mean_ref_mass_in_cand": cov_ref / max(n_ok, 1),
            "mean_cand_mass_in_ref": cov_cand / max(n_ok, 1),
            "pos_per_s": n / max(dt, 1e-6),
            "cpu_hours_per_m": (1e6 / max(n / max(dt, 1e-6), 1e-9)) / 3600.0,
        }
        results.append(rec)
        log(json.dumps(rec), log_path)
    # unique labels / sec: relabel existing FENs vs self-play
    relabel_n = min(16, len(fens))
    t_rel = time.time()
    relabel_ok = 0
    for fen in fens[:relabel_n]:
        _clear(cand_engine)
        if analyze_board(cand_engine, chess.Board(fen), nodes=10_000, multipv=4, tau=args.tau, watchdog_s=8.0) is not None:
            relabel_ok += 1
    relabel_s = max(time.time() - t_rel, 1e-6)
    t_sp = time.time()
    sp_ok = 0
    rng = random.Random(args.seed + 99)
    while sp_ok < relabel_n and time.time() - t_sp < 30:
        b = chess.Board()
        for uci in OPENINGS[sp_ok % len(OPENINGS)]:
            m = chess.Move.from_uci(uci)
            if m in b.legal_moves:
                b.push(m)
        for _ in range(rng.randint(2, 8)):
            if b.is_game_over(claim_draw=True):
                break
            b.push(rng.choice(list(b.legal_moves)))
        if b.is_game_over(claim_draw=True):
            continue
        _clear(cand_engine)
        if analyze_board(cand_engine, b, nodes=10_000, multipv=4, tau=args.tau, watchdog_s=8.0) is not None:
            sp_ok += 1
    sp_s = max(time.time() - t_sp, 1e-6)
    source_compare = {
        "relabel_labels_per_s": relabel_ok / relabel_s,
        "selfplay_labels_per_s": sp_ok / sp_s,
        "n": relabel_n,
        "note": "same 10k/MultiPV=4 label cost; self-play also spends play-search plies in generate",
    }
    log(f"source_compare {json.dumps(source_compare)}", log_path)
    cand_engine.quit()
    ref_engine.quit()
    chosen = None
    for rec in sorted(results, key=lambda r: (r["nodes"], r["multipv"])):
        if rec["top1_agree"] >= 0.80 and rec["mean_ref_regret"] <= 20.0 and rec["missing_from_ref_frac"] <= 0.25:
            chosen = rec
            break
    if chosen is None:
        chosen = max(results, key=lambda r: r["top1_agree"])
    report = {
        "teacher": fp,
        "ref_nodes": ref_nodes,
        "tau": args.tau,
        "n_positions": len(fens),
        "results": results,
        "chosen": chosen,
        "source_compare": source_compare,
        "threshold": {
            "top1_agree": 0.80,
            "mean_ref_regret": 20.0,
            "missing_from_ref_frac": 0.25,
            "note": "regret is ref(best) - ref(cand_pick) on a clean engine; hash cleared per search",
        },
        "hash_policy": "Clear Hash before every bench search; candidate engine isolated from reference",
        "fen_limitations": "4-field key only; no repetition/halfmove in cache rows",
    }
    (out / "bench.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"CHOSEN {json.dumps(chosen)}", log_path)
    return report


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
    cfg = GenConfig(
        nodes=args.nodes,
        play_nodes=args.play_nodes,
        multipv=args.multipv,
        tau=args.tau,
        epsilon=args.epsilon,
        ply_stride=args.ply_stride,
        ply_skip_open=args.ply_skip_open,
        ply_cap=args.ply_cap,
        book_noise=args.book_noise,
        hash_mb=args.hash_mb,
        clear_hash_every=args.clear_hash_every,
        watchdog_s=float(getattr(args, "watchdog_s", 8.0) or 8.0),
    )
    run_fp = run_fingerprint(fp, cfg)
    assert_resume_compatible(out, run_fp)
    teacher_path = out / "teacher.json"
    if not teacher_path.exists():
        teacher_path.write_text(json.dumps({"teacher": fp, "config": asdict(cfg), "run": run_fp, "fingerprint_id": run_fp["fingerprint_id"]}, indent=2), encoding="utf-8")
    log(f"teacher={fp['uci_name']} fp={run_fp['fingerprint_id'][:12]} nodes={cfg.nodes} multipv={cfg.multipv} workers={args.workers}", log_path)
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
    target = args.target
    workers = max(1, args.workers)
    for raw in getattr(args, "exclude_caches", None) or []:
        p = Path(raw)
        if not p.exists():
            continue
        added = seen.ingest_cache_keys(p, f"exclude:{p}")
        log(f"exclude {p} +{added:,} keys seen={len(seen):,}", log_path)
    eco_starts: list[dict] = []
    piece_curve = getattr(args, "mode", "") == "piece_curve"
    curve_pmf = target_pmf() if piece_curve else None
    curve_have = empty_counts()
    fen_buckets: dict[int, list[str]] = {}
    if getattr(args, "mode", "") in ("eco", "piece_curve"):
        openings = load_eco_openings()
        eco_starts = eco_start_positions(
            openings,
            include_prefixes=not getattr(args, "no_prefixes", False),
        )
        summary = openings_summary(openings, eco_starts)
        (out / "openings.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(
            f"{'piece_curve' if piece_curve else 'eco'} openings={summary['n_rows']} "
            f"unique={summary['n_unique']} starts={len(eco_starts)} "
            f"prefixes={not getattr(args, 'no_prefixes', False)} "
            f"volumes={summary['volumes']}",
            log_path,
        )
        if piece_curve:
            curve_have = recount_inbox_pieces(inbox)
            log(f"piece_curve resume {curve_summary(curve_have)}", log_path)
            if committed < target:
                ingest_existing_curve(
                    inbox=inbox,
                    seen=seen,
                    target=target,
                    shard_size=args.shard_size,
                    pmf=curve_pmf,
                    have=curve_have,
                    teacher_meta={
                        "teacher": fp["uci_name"],
                        "binary_sha256": fp["binary_sha256"],
                        "fingerprint_id": run_fp["fingerprint_id"],
                        "nodes": cfg.nodes,
                        "multipv": cfg.multipv,
                        "tau": cfg.tau,
                    },
                    log_path=log_path,
                )
                committed, next_game = inbox_state(inbox)
                curve_have = recount_inbox_pieces(inbox)
            log(f"piece_curve after ingest committed={committed:,} {curve_summary(curve_have)}", log_path)
            fen_buckets = collect_fen_buckets(
                local_fen_paths(ROOT), random.Random(args.seed + 9),
            )
            log(
                f"fen_seeds={sum(len(v) for v in fen_buckets.values()):,}",
                log_path,
            )
        seed_fens = []
        seed_paths = []
    else:
        seed_paths = [Path(p) for p in (getattr(args, "seed_caches", None) or [])]
        if not getattr(args, "no_seed_caches", False) and not seed_paths:
            seed_paths = [
                ROOT / "outputs/autoresearch_8gb/soft_cache_200k.pt",
            ]
        seed_n = int(getattr(args, "seed_fens_n", 2048) or 2048)
        seed_paths = [p for p in seed_paths if p.exists()]
        seed_fens = [] if args.no_seed_caches else sample_seed_fens(seed_paths, seed_n, random.Random(args.seed + 3))
        if seed_fens:
            ok = []
            for fen in seed_fens:
                try:
                    b = chess.Board(fen)
                except ValueError:
                    continue
                if b.is_valid() and not b.is_game_over(claim_draw=True):
                    ok.append(fen)
            seed_fens = ok
        log(f"seed_fens={len(seed_fens)} from {[str(p) for p in seed_paths]}", log_path)
    if committed >= target:
        write_manifest(out, {"teacher": fp, "config": asdict(cfg), "already_complete": True})
        log(f"already complete committed={committed:,} target={target:,}", log_path)
        return
    ctx = get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=(sf, cfg, str(out / "seen.sqlite")))
    t0 = time.time()
    new_rows = 0
    rejected = {"dup": 0, "stride": 0, "analyze_fail": 0, "curve": 0}
    pending: list[dict] = []
    pending_keys: set[str] = set()
    game_i = args.game_start if args.game_start else next_game
    stats_path = out / "stats.json"

    def flush(force: bool = False) -> None:
        nonlocal pending, pending_keys
        if not pending or (len(pending) < args.shard_size and not force):
            return
        take = pending[: args.shard_size] if not force else pending
        pending = pending[len(take):] if not force else []
        keys = [row_key(r) for r in take]
        data = stack_rows(take)
        sh = next_shard_dir(inbox)
        write_shard(data, sh, {
            "teacher": fp["uci_name"],
            "binary_sha256": fp["binary_sha256"],
            "fingerprint_id": run_fp["fingerprint_id"],
            "nodes": cfg.nodes,
            "multipv": cfg.multipv,
            "tau": cfg.tau,
            "max_game_id": int(data["game_id"].max()),
        })
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        seen.forget_hot(keys)
        pending_keys.difference_update(keys)
        total = committed + new_rows
        log(f"wrote {sh} n={len(take)} total={total:,}", log_path)
        write_manifest(out, {"teacher": fp["uci_name"], "fingerprint_id": run_fp["fingerprint_id"], "nodes": cfg.nodes, "multipv": cfg.multipv})

    try:
        while committed + new_rows < target:
            jobs = []
            for _ in range(max(workers * 2, 8)):
                if committed + new_rows >= target:
                    break
                if piece_curve:
                    n_need = most_deficit_n(curve_have, curve_pmf)
                    if n_need is not None and fen_buckets.get(n_need):
                        spec = {
                            "relabel": True,
                            "fen": fen_buckets[n_need].pop(),
                            "game_id": game_i,
                            "seed": args.seed + game_i * 10007,
                            "split": 1 if random.Random(game_i + 17).random() < args.holdout_frac else 0,
                        }
                    elif eco_starts:
                        spec = build_piece_curve_game_spec(
                            game_i, eco_starts, seed=args.seed, holdout_frac=args.holdout_frac,
                        )
                    else:
                        spec = {
                            "game_id": game_i,
                            "seed": args.seed + game_i * 10007,
                            "opening": list(OPENINGS[game_i % len(OPENINGS)]),
                            "split": 0,
                        }
                elif eco_starts:
                    spec = build_eco_game_spec(
                        game_i, eco_starts, seed=args.seed, holdout_frac=args.holdout_frac,
                    )
                else:
                    split = 1 if random.Random(game_i + 17).random() < args.holdout_frac else 0
                    spec = {
                        "game_id": game_i,
                        "seed": args.seed + game_i * 10007,
                        "opening": list(OPENINGS[game_i % len(OPENINGS)]),
                        "book_noise": int(cfg.book_noise) + (game_i % 7),
                        "split": split,
                    }
                    if seed_fens and game_i % 2 == 1:
                        spec["start_fen"] = seed_fens[game_i % len(seed_fens)]
                jobs.append(spec)
                game_i += 1
            if not jobs:
                break
            for result in pool.imap_unordered(_run_job, jobs, chunksize=1):
                rejected["stride"] += int(result["rejects"].get("stride") or 0)
                rejected["analyze_fail"] += int(result["rejects"].get("analyze_fail") or 0)
                keys = []
                kept = []
                for row in result["rows"]:
                    k = row_key(row)
                    if seen.has(k) or k in pending_keys:
                        rejected["dup"] += 1
                        continue
                    if curve_pmf is not None:
                        n_pcs = n_pieces_from_row(row)
                        if not curve_should_keep(n_pcs, curve_have, int(curve_have.sum()), curve_pmf):
                            rejected["curve"] += 1
                            continue
                        curve_have[n_pcs] += 1
                    keys.append(k)
                    kept.append(row)
                    pending_keys.add(k)
                seen.remember_hot(keys)
                pending.extend(kept)
                new_rows += len(kept)
                accepted = committed + new_rows
                elapsed = max(time.time() - t0, 1e-6)
                if new_rows and new_rows % 64 < max(len(kept), 1):
                    rate = new_rows / elapsed
                    eta = (target - accepted) / max(rate, 1e-9)
                    extra = ""
                    stats = {
                        "accepted": accepted,
                        "new_rows": new_rows,
                        "rejected": rejected,
                        "seen": len(seen),
                        "pos_per_s": rate,
                        "elapsed_s": elapsed,
                    }
                    if piece_curve:
                        summ = curve_summary(curve_have)
                        extra = (
                            f" curve={rejected['curve']} "
                            f"n̄={summ['mean']} σ={summ['std']} peak={summ['peak']}"
                        )
                        stats["piece_curve"] = summ
                        stats["piece_hist"] = [int(x) for x in curve_have.tolist()]
                    log(
                        f"accepted={accepted:,} pending={len(pending)} seen={len(seen):,} "
                        f"dup={rejected['dup']} fail={rejected['analyze_fail']} "
                        f"{rate:.1f}/s eta={eta/60:.1f}m{extra}",
                        log_path,
                    )
                    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
                flush(False)
                if accepted >= target:
                    break
    finally:
        flush(True)
        pool.terminate()
        pool.join()
    elapsed = max(time.time() - t0, 1e-6)
    accepted = committed + new_rows
    rate = new_rows / elapsed if new_rows else (accepted / max(elapsed, 1e-6))
    summary = {
        "accepted": accepted,
        "new_rows": new_rows,
        "rejected": rejected,
        "seen": len(seen),
        "pos_per_s": rate,
        "elapsed_s": elapsed,
        "teacher": fp,
        "config": asdict(cfg),
        "hours_per_1m": (1e6 / max(rate, 1e-9)) / 3600.0,
        "hours_per_10m": (1e7 / max(rate, 1e-9)) / 3600.0,
        "hours_per_100m": (1e8 / max(rate, 1e-9)) / 3600.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(out, {"summary": {k: summary[k] for k in ("accepted", "pos_per_s", "hours_per_1m", "hours_per_10m", "hours_per_100m")}})
    log(json.dumps(summary), log_path)


def verify_loader(args) -> None:
    from scripts.autoresearch_8gb.pipeline import attach_static_targets, prepare_soft_batch, soft_policy_loss

    inbox = Path(args.out_dir) / "inbox"
    shards = sorted(p for p in inbox.glob("shard_*/soft_cache.pt"))
    if not shards:
        raise SystemExit("no shards to verify")
    data = attach_static_targets(torch.load(shards[0], map_location="cpu", weights_only=False))
    n = min(32, int(data["move_idx"].shape[0]))
    device = torch.device("cpu")
    bi, hard, wdl, si, sp = prepare_soft_batch(data, torch.arange(n), device, hflip_p=0.0)
    assert bi["fused_ids"].shape[0] == n
    assert wdl.shape == (n, 3)
    assert si.shape[1] == SOFT_K
    # dummy logits to exercise loss
    logits = torch.zeros(n, VOCAB_SIZE)
    for i in range(n):
        if int(si[i, 0]) >= 0:
            logits[i, int(si[i, 0])] = 4.0
    loss = soft_policy_loss(logits, si, sp)
    if not torch.isfinite(loss):
        raise SystemExit("soft loss not finite")
    # score orientation: White-abs cp should flip WDL with turn when converted
    for i in range(n):
        if int(data["mate"][i]) != 0:
            continue
        cp = int(data["cp"][i])
        turn = int(data["turn"][i])
        w = float(wdl[i, 0] - wdl[i, 2])
        if turn == 0 and cp > 80 and w <= 0:
            raise SystemExit(f"white-to-move plus score has non-positive WDL i={i}")
        if turn == 1 and cp < -80 and w >= 0:
            raise SystemExit(f"black-to-move (white-abs negative) has non-negative WDL i={i}")
    from scripts.export_soft_caches_to_hf import sf19_chunk_table, sf19_table_to_cache
    table = sf19_chunk_table(data, "verify", 0, n)
    back = sf19_table_to_cache(table)
    for key in ("soft_indices", "soft_probs", "wdl", "soft_cps", "game_id", "split", "policy_mask"):
        if key in data and key in back:
            if not torch.allclose(data[key][:n].float(), back[key].float(), atol=1e-5):
                raise SystemExit(f"parquet round-trip mismatch {key}")
    log(f"loader ok n={n} soft_ce={float(loss):.4f} wdl_mean={wdl.mean(0).tolist()} parquet_roundtrip=ok")
    ckpt_path = Path(getattr(args, "ckpt", "") or "outputs/hf100m_lapse_ft_30m/latest.pt")
    if ckpt_path.exists():
        from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64
        from scripts.autoresearch_8gb.pipeline import load_model_state
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_squares64(DEFAULT_100M_SQUARES64_CONFIG)
        model.load_state_dict(load_model_state(ckpt), strict=False)
        model.eval()
        with torch.no_grad():
            out = model(bi)
            pol = soft_policy_loss(out["policy_logits"], si, sp)
            val = torch.nn.functional.cross_entropy(out["value_logits"].float(), wdl)
        if not torch.isfinite(pol) or not torch.isfinite(val):
            raise SystemExit("model forward loss not finite")
        log(f"model forward ok ckpt={ckpt_path} soft_ce={float(pol):.4f} value_ce={float(val):.4f}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    for path in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if path.exists():
            val = path.read_text(encoding="utf-8").strip()
            if val:
                os.environ["HF_TOKEN"] = val
                return val
    raise SystemExit("HF_TOKEN missing")


def should_push_rows(ready: int, uploaded: int, every: int, finished: bool) -> bool:
    """Push on 50k landmarks, or flush leftovers when the harvest is done."""
    if ready <= uploaded or every <= 0:
        return False
    if finished:
        return True
    return (ready // every) > (uploaded // every)


def _upload_with_retry(api, *, path: str, path_in_repo: str, repo: str, token: str, message: str, attempts: int = 5) -> None:
    last = None
    for i in range(attempts):
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=repo,
                repo_type="dataset",
                commit_message=message,
                token=token,
            )
            return
        except Exception as exc:
            last = exc
            wait = min(90, 8 * (2 ** i))
            log(f"upload retry {i + 1}/{attempts} {path_in_repo} wait={wait}s err={type(exc).__name__}")
            time.sleep(wait)
    raise last


def _sf19_readme(repo: str, n_total: int, tau: float, summary: dict, audit: dict, sampling: dict) -> str:
    origin = summary.get("origin") or sampling.get("origin") or {}
    n_relabel = int(origin.get("0") or origin.get(0) or 0)
    n_selfplay = int(origin.get("1") or origin.get(1) or 0)
    phase = summary.get("phase") or {}
    ev = summary.get("eval_bucket") or {}
    top1 = audit.get("top1")
    regret = audit.get("regret") or {}
    miss = audit.get("missing_ref_mass") or {}
    by_o = audit.get("by_origin") or {}
    fp = (audit.get("teacher") or {}).get("source_revision") or "edb0d9db6731067ec50ce619ff372b463bc4dd5d"
    return f"""---
license: mit
tags:
- chess
- stockfish-19
- soft-labels
- multipv
pretty_name: Stockfish 19 soft targets
---

# {repo}

Official **Stockfish 19** MultiPV soft targets. This release supersedes the 25k
pilot. It is **not** a filter of `chess-soft-multipv-lichess` or
`chess-soft-100m-disagreements`.

**{n_total:,}** rows. Source id `4`. Vocab `compact` (1968).

## Mix (as generated)

| origin | rows | note |
|---|---:|---|
| self-play (`origin=1`) | {n_selfplay:,} | SF19 vs SF19, ε=0.20, book + 4 random legal |
| relabel (`origin=0`) | {n_relabel:,} | existing local boards, new SF19 labels |
| frozen eval | 10,000 | `split=1` in `data/shard_000000.parquet` |

Relabel was intended to be 80%. The existing-board iterator stopped at ~84k
accepted labels; self-play filled the 1M train remainder. `split=0` is train.

Phase (train+eval new rows): opening {int(phase.get('0', 0)):,} / middlegame {int(phase.get('1', 0)):,} / endgame {int(phase.get('2', 0)):,}.
Eval bucket: equal {int(ev.get('equal', 0)):,} / winning {int(ev.get('winning', 0)):,} / losing {int(ev.get('losing', 0)):,} / mate {int(ev.get('mate', 0)):,}.

## Teacher

- Stockfish 19 tag `sf_19` (`{fp[:8]}`), EvalFile `nn-1a298aa575a0.nnue`
- Full file SHA-256 in `teacher.json`
- Full strength, `Threads=1`, `Hash=32`, `UCI_ShowWDL=true`
- `Ponder` left to python-chess (do not set it via `configure`)
- Hash is **not** cleared between production searches
- Label budget: **100k nodes / MultiPV=8** / `tau=120`
- Play (self-play only): 4k-node cheap search on unlabeled plies, label every 3 plies after ply 4

## Targets

- Policy: STM softmax(`tau={tau:g}`) over the last **complete** MultiPV-8 iteration
- Unsearched legal moves are absent, not proven bad
- Mate rank is `sign * (100000 - min(|mate|, 1000))`, not mate-as-cp
- Bound scores are dropped
- `cp` / `mate` / `wdl`: **White-absolute** (training loader contract)
- `soft_indices` / `soft_probs`: width 8, pad `-1` / `0`
- `soft_cps` / `soft_mates` are stored so softmax can be rebuilt without SF
- FEN-only relabel rows drop repetition history and may omit EP if no legal capture

Honor `split`. `shard_000000` is the frozen eval set (`saved_split_v1`). Do not
invent a new position-hash holdout.

## Quality audit (2,000 isolated positions, 100k/8 vs 1M/8)

- top-1 vs 1M: **{top1}** (self-play {by_o.get('selfplay', {}).get('top1', '?')}, relabel {by_o.get('relabel', {}).get('top1', '?')}; endgame worst)
- regret p50={regret.get('p50', 0):.0f} / p90={regret.get('p90', 0):.0f} / mean={regret.get('mean', 0):.0f}
- missing 1M-ref mass p50={100 * float(miss.get('p50') or 0):.1f}% / p90={100 * float(miss.get('p90') or 0):.1f}%

See `audit.json`. Flags did not predict disagreement; no adaptive extra search.

## Files

- `data/shard_XXXXXX.parquet` — one inbox shard per file
- `teacher.json`, `summary.json`, `sampling.json`, `audit.json`, `eval_manifest.json`
"""


def _sf19_eco_readme(repo: str, n_total: int, tau: float, openings: dict, teacher: dict, stats: dict) -> str:
    cfg = (teacher.get("config") or {})
    run = (teacher.get("run") or {})
    fp = run.get("source_revision") or (teacher.get("teacher") or {}).get("source_revision") or "edb0d9db6731067ec50ce619ff372b463bc4dd5d"
    vols = openings.get("volumes") or {}
    return f"""---
license: mit
tags:
- chess
- stockfish-19
- soft-labels
- multipv
- eco
- openings
pretty_name: Stockfish 19 soft targets
---

# {repo}

Official **Stockfish 19** MultiPV soft targets, mined from the Lichess ECO
opening set. In-progress snapshot toward 1M unique positions.

**{n_total:,}** rows in this upload. Source id `4`. Vocab `compact` (1968).

## How positions are chosen

Games start from the Lichess Chess Openings dataset
(`lichess-org/chess-openings`): {int(openings.get('n_unique') or 0):,} named
leaves (HF card still lists {int(openings.get('hf_card_n') or 3704):,}) plus
book prefixes, **{int(openings.get('n_starts') or 0):,}** unique starts.

ECO volumes: A {int(vols.get('A') or 0):,} / B {int(vols.get('B') or 0):,} /
C {int(vols.get('C') or 0):,} / D {int(vols.get('D') or 0):,} /
E {int(vols.get('E') or 0):,}.

From each start: 0–4 random legal noise, then SF19 vs SF19 with rotating
ε ∈ {{0.10, 0.18, 0.28, 0.40}} and a small uniform-legal wild chance. Labels
every 2 plies from the start through ply {int(cfg.get('ply_cap') or 180)}.
Deduped by 4-field board key against this run and existing local mixes.

`split=1` is a ~5% holdout. Honor `split`. Do not invent a new position-hash holdout.

## Teacher

- Stockfish 19 tag `sf_19` (`{str(fp)[:8]}`), EvalFile `nn-1a298aa575a0.nnue`
- Full file SHA-256 in `teacher.json`
- Full strength, `Threads=1`, `Hash=64`, `UCI_ShowWDL=true`
- Label budget: **{int(cfg.get('nodes') or 100000):,} nodes / MultiPV={int(cfg.get('multipv') or 8)}** / `tau={tau:g}`
- Play on unlabeled plies: {int(cfg.get('play_nodes') or 4000):,} nodes

## Targets

- Policy: STM softmax(`tau={tau:g}`) over the last **complete** MultiPV-8 iteration
- Unsearched legal moves are absent, not proven bad
- Mate rank is `sign * (100000 - min(|mate|, 1000))`, not mate-as-cp
- Bound scores are dropped
- `cp` / `mate` / `wdl`: **White-absolute** (training loader contract)
- `soft_indices` / `soft_probs`: width 8, pad `-1` / `0`
- `soft_cps` / `soft_mates` are stored so softmax can be rebuilt without SF

## Files

- `data/shard_XXXXXX.parquet` — 5,000 rows each
- `teacher.json`, `openings.json`, `manifest.json`, `stats.json`
"""


def push_hf(args) -> None:
    from huggingface_hub import HfApi, create_repo
    from scripts.export_soft_caches_to_hf import sf19_chunk_table
    import pyarrow.parquet as pq

    token = _hf_token()
    repo = args.repo
    out = Path(args.out_dir)
    inbox = out / "inbox"
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
    shard_offset = int(getattr(args, "shard_offset", 0) or 0)
    base_rows = int(getattr(args, "base_rows", 0) or 0)
    n_new = 0
    shards = sorted(inbox.glob("shard_*"))
    for sh in shards:
        cache = sh / "soft_cache.pt"
        if not cache.exists():
            continue
        try:
            local_i = int(sh.name.split("_", 1)[1])
        except (IndexError, ValueError):
            local_i = 0
        remote_name = f"data/shard_{local_i + shard_offset:06d}.parquet"
        dest = data_dir / Path(remote_name).name
        if dest.exists() and dest.stat().st_size > 0:
            n = int(pq.read_metadata(dest).num_rows)
        else:
            d = torch.load(cache, map_location="cpu", weights_only=False)
            n = int(d["move_idx"].shape[0])
            pq.write_table(sf19_chunk_table(d, sh.name, 0, n), dest, compression="zstd")
        n_new += n
        if remote_name in uploaded:
            log(f"skip {remote_name} n={n} new={n_new:,}")
            continue
        _upload_with_retry(
            api, path=str(dest), path_in_repo=remote_name, repo=repo, token=token,
            message=f"add {remote_name} n={n:,}",
        )
        uploaded.add(remote_name)
        state = {"repo": repo, "uploaded": sorted(uploaded), "rows": n_new, "shard_offset": shard_offset}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        log(f"uploaded {remote_name} n={n} new={n_new:,}")
    n_total = base_rows + n_new
    extras = (
        "teacher.json", "bench.json", "manifest.json", "summary.json",
        "sampling.json", "audit.json", "eval_manifest.json",
        "openings.json", "stats.json",
    )
    for name in extras:
        src = out / name
        if src.exists():
            _upload_with_retry(
                api, path=str(src), path_in_repo=name, repo=repo, token=token,
                message=f"add {name}",
            )
    summary = _load_json(out / "summary.json")
    audit = _load_json(out / "audit.json")
    sampling = _load_json(out / "sampling.json")
    openings = _load_json(out / "openings.json")
    teacher = _load_json(out / "teacher.json")
    stats = _load_json(out / "stats.json")
    readme = staging / "README.md"
    if openings:
        readme.write_text(
            _sf19_eco_readme(repo, n_total, float(args.tau), openings, teacher, stats),
            encoding="utf-8",
        )
    else:
        readme.write_text(
            _sf19_readme(repo, n_total, float(args.tau), summary, audit, sampling),
            encoding="utf-8",
        )
    _upload_with_retry(
        api, path=str(readme), path_in_repo="README.md", repo=repo, token=token,
        message=f"card: {n_total:,} SF19 rows",
    )
    state_path.write_text(json.dumps({
        "repo": repo, "uploaded": sorted(uploaded), "rows": n_total, "done": True,
    }, indent=2), encoding="utf-8")
    log(f"https://huggingface.co/datasets/{repo} rows={n_total:,}")


def watch_push(args) -> None:
    """Poll READY shards and push each time we cross another --every rows."""
    out = Path(args.out_dir)
    inbox = out / "inbox"
    log_path = out / "push_watch.log"
    halt = out / "HALT_PUSH"
    lock = out / "push.lock"
    every = max(1, int(args.every))
    poll = max(5.0, float(args.poll))
    _hf_token()
    log(
        f"watch-push out={out} repo={args.repo} every={every} poll={poll}s",
        log_path,
    )
    while True:
        if halt.exists():
            log("HALT_PUSH", log_path)
            return
        ready, _ = inbox_state(inbox)
        uploaded = int(_load_json(out / "hf_upload.json").get("rows") or 0)
        finished = (out / "summary.json").exists()
        if should_push_rows(ready, uploaded, every, finished):
            if lock.exists():
                age = time.time() - lock.stat().st_mtime
                if age < 30 * 60:
                    log(f"skip push lock age={age:.0f}s ready={ready:,} uploaded={uploaded:,}", log_path)
                    time.sleep(poll)
                    continue
            lock.write_text(str(os.getpid()), encoding="utf-8")
            try:
                log(f"push ready={ready:,} uploaded={uploaded:,} finished={finished}", log_path)
                push_hf(args)
            finally:
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
            uploaded = int(_load_json(out / "hf_upload.json").get("rows") or 0)
            if finished and ready <= uploaded:
                log(f"watch-push done uploaded={uploaded:,}", log_path)
                return
        elif finished and ready <= uploaded:
            log(f"watch-push done uploaded={uploaded:,}", log_path)
            return
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add_shared(p):
        p.add_argument("--out-dir", default="outputs/sf19_soft/pilot")
        p.add_argument("--tau", type=float, default=DEFAULT_TAU)
        p.add_argument("--seed", type=int, default=19)
    b = sub.add_parser("bench")
    add_shared(b)
    b.add_argument("--bench-n", type=int, default=48)
    b.add_argument("--ref-nodes", type=int, default=200_000)
    g = sub.add_parser("generate")
    add_shared(g)
    g.add_argument("--go", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--target", type=int, default=25_000)
    g.add_argument("--workers", type=int, default=14)
    g.add_argument("--nodes", type=int, default=100_000)
    g.add_argument("--play-nodes", type=int, default=4_000)
    g.add_argument("--multipv", type=int, default=8)
    g.add_argument("--epsilon", type=float, default=0.20)
    g.add_argument("--ply-stride", type=int, default=2)
    g.add_argument("--ply-skip-open", type=int, default=4)
    g.add_argument("--ply-cap", type=int, default=140)
    g.add_argument("--book-noise", type=int, default=2)
    g.add_argument("--watchdog-s", type=float, default=8.0)
    g.add_argument("--hash-mb", type=int, default=64)
    g.add_argument("--clear-hash-every", type=int, default=0)
    g.add_argument("--shard-size", type=int, default=5_000)
    g.add_argument("--holdout-frac", type=float, default=0.1)
    g.add_argument("--game-start", type=int, default=0)
    g.add_argument("--seed-caches", nargs="*", default=None)
    g.add_argument("--seed-fens-n", type=int, default=2048)
    g.add_argument("--exclude-caches", nargs="*", default=None,
                   help="Caches whose position keys are blocked (already in the public set).")
    g.add_argument("--no-seed-caches", action="store_true")
    g.add_argument("--no-prefixes", action="store_true",
                   help="ECO mode: start only from named leaves, not book prefixes.")
    g.add_argument("--mode", choices=("selfplay", "mix", "eco", "piece_curve"), default="mix")
    g.add_argument("--relabel-frac", type=float, default=0.8)
    a = sub.add_parser("audit")
    add_shared(a)
    a.add_argument("--audit-n", type=int, default=2000)
    a.add_argument("--ref-nodes", type=int, default=1_000_000)
    a.add_argument("--deep-n", type=int, default=200)
    a.add_argument("--deep-nodes", type=int, default=5_000_000)
    a.add_argument("--nodes", type=int, default=100_000)
    a.add_argument("--multipv", type=int, default=8)
    a.add_argument("--workers", type=int, default=14)
    a.add_argument("--hash-mb", type=int, default=32)
    a.add_argument("--seed-caches", nargs="*", default=None)
    e = sub.add_parser("freeze-eval")
    add_shared(e)
    e.add_argument("--eval-n", type=int, default=10_000)
    e.add_argument("--nodes", type=int, default=100_000)
    e.add_argument("--multipv", type=int, default=8)
    e.add_argument("--workers", type=int, default=14)
    e.add_argument("--hash-mb", type=int, default=32)
    e.add_argument("--seed-caches", nargs="*", default=None)
    c = sub.add_parser("compare-ft")
    add_shared(c)
    c.add_argument("--ckpt", default="outputs/hf100m_lapse_ft_30m/latest.pt")
    c.add_argument("--baseline-cache", default="outputs/hf_soft_mix/soft_cache.pt")
    c.add_argument("--ft-steps", type=int, default=400)
    c.add_argument("--ft-minutes", type=float, default=50.0)
    v = sub.add_parser("verify")
    add_shared(v)
    v.add_argument("--ckpt", default="outputs/hf100m_lapse_ft_30m/latest.pt")
    p = sub.add_parser("push")
    add_shared(p)
    p.add_argument("--repo", default="avewright/chess-soft-sf19")
    p.add_argument("--shard-offset", type=int, default=0,
                   help="Remote parquet index = local shard index + offset (append without clobber).")
    p.add_argument("--base-rows", type=int, default=0,
                   help="Existing remote rows to include in the card total when appending.")
    w = sub.add_parser("watch-push")
    add_shared(w)
    w.add_argument("--repo", default="avewright/stockfish-19-soft-targets")
    w.add_argument("--every", type=int, default=50_000,
                   help="Upload when READY rows cross another multiple of this.")
    w.add_argument("--poll", type=float, default=30.0)
    w.add_argument("--shard-offset", type=int, default=0)
    w.add_argument("--base-rows", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "bench":
        bench(args)
    elif args.cmd == "generate":
        if not args.go:
            raise SystemExit("pass --go")
        if args.mode in ("eco", "piece_curve"):
            args.no_seed_caches = True
            if args.ply_skip_open == 4:
                args.ply_skip_open = 0
            if args.ply_cap == 140:
                args.ply_cap = 220 if args.mode == "piece_curve" else 180
            if args.watchdog_s == 8.0:
                args.watchdog_s = 20.0
            if args.target == 25_000 and not args.pilot and not args.smoke:
                args.target = 300_000 if args.mode == "piece_curve" else 1_000_000
            if args.workers == 14:
                args.workers = max(1, (os.cpu_count() or 8) - 2)
        if args.smoke:
            args.target = min(args.target, 48)
            args.workers = min(args.workers, 2)
            args.nodes = min(args.nodes, 4_000)
            args.play_nodes = min(args.play_nodes, 800)
            args.ply_cap = min(args.ply_cap, 24)
            args.shard_size = min(args.shard_size, 32)
            args.watchdog_s = min(args.watchdog_s, 6.0)
        if args.pilot:
            args.target = min(args.target, 25_000)
        if getattr(args, "mode", "mix") == "mix":
            from scripts.sf19_soft_prod import generate_mix
            generate_mix(args)
        else:
            generate(args)
    elif args.cmd == "audit":
        from scripts.sf19_soft_prod import run_quality_audit
        run_quality_audit(args)
    elif args.cmd == "freeze-eval":
        from scripts.sf19_soft_prod import freeze_eval_set
        freeze_eval_set(args)
    elif args.cmd == "compare-ft":
        from scripts.sf19_soft_prod import compare_ft
        compare_ft(args)
    elif args.cmd == "verify":
        verify_loader(args)
    elif args.cmd == "push":
        push_hf(args)
    elif args.cmd == "watch-push":
        watch_push(args)


if __name__ == "__main__":
    main()
