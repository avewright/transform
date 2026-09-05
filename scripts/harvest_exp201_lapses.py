#!/usr/bin/env python3
"""Mine exp201 lapses vs full-strength Stockfish and pack a local soft cache.

CPU-only. Does not touch the training GPU.

Each model ply: MultiPV teacher labels on the current board, then measure
value lost if the greedy policy move is not the teacher best. Keep hard
slips, conversion failures, and positions from lost games.

Usage:
  CUDA_VISIBLE_DEVICES= MOVE_VOCAB_VERSION=compact python -u \\
    scripts/harvest_exp201_lapses.py --go \\
    --ckpt outputs/exp201_elo_max/baseline/live_step57295.pt \\
    --games 240 --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
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
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
from chess_inference import load_checkpoint  # noqa: E402
from data_loader import CASTLING_MAP, _fast_parse_fen  # noqa: E402
from move_vocab import UCI_TO_IDX, index_to_move, legal_move_mask  # noqa: E402

SOFT_K = 8
SOURCE_HARVEST = 3
OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "c7c6"],
    ["e2e4", "g8f6"],
    ["d2d4", "f7f5"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["e2e4", "c7c5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["e2e4", "e7e5", "f2f4"],
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    ["c2c4", "c7c5"],
    ["g1f3", "g8f6"],
    ["e2e4", "d7d6"],
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    ["e2e4", "e7e5", "b1c3"],
    ["e2e4", "g7g6"],
]


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def resolve_sf() -> str:
    for p in (
        os.environ.get("STOCKFISH_PATH", ""),
        shutil.which("stockfish") or "",
        "/usr/games/stockfish",
    ):
        if p and Path(p).exists():
            return str(p)
    raise FileNotFoundError("Stockfish not found")


def phase_id(board: chess.Board) -> int:
    n = len(board.piece_map())
    if n >= 26:
        return 0
    if n >= 14:
        return 1
    return 2


def score_cp_mate(score: chess.engine.PovScore, turn: chess.Color) -> tuple[int, int]:
    pov = score.pov(turn)
    if pov.is_mate():
        m = pov.mate()
        assert m is not None
        cp = (100000 - min(abs(m), 1000)) * (1 if m > 0 else -1)
        return int(cp), int(m)
    cp = pov.score(mate_score=100000)
    return int(cp if cp is not None else 0), 0


@torch.no_grad()
def model_move(model, board, device) -> chess.Move:
    x = batch_boards_to_fused_token_ids([board], device)
    logits = model(x)["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits = logits.masked_fill(~mask, float("-inf"))
    mv = index_to_move(int(logits.argmax().item()))
    if mv not in board.legal_moves:
        mv = next(iter(board.legal_moves))
    return mv


def analyze_multipv(engine, board, *, nodes: int, movetime: float, tau: float) -> dict | None:
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return None
    if movetime > 0:
        limit = chess.engine.Limit(time=movetime)
    elif nodes > 0:
        limit = chess.engine.Limit(nodes=nodes)
    else:
        limit = chess.engine.Limit(depth=16)
    try:
        infos = engine.analyse(board, limit, multipv=min(SOFT_K, n_legal))
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return None
    if not isinstance(infos, list):
        infos = [infos]
    best: dict[str, tuple[int, int]] = {}
    used_depth = 0
    for info in infos:
        pv = info.get("pv") or []
        sc = info.get("score")
        if not pv or sc is None:
            continue
        uci = pv[0].uci()
        cp, mate = score_cp_mate(sc, board.turn)
        if uci not in best or cp > best[uci][0]:
            best[uci] = (cp, mate)
        if "depth" in info:
            used_depth = max(used_depth, int(info["depth"]))
    if not best:
        return None
    items = sorted(best.items(), key=lambda x: -x[1][0])[:SOFT_K]
    cps = [c for _, (c, _) in items]
    mx = max(cps)
    exps = [math.exp((c - mx) / tau) for c in cps]
    z = sum(exps) or 1.0
    return {
        "ucis": [u for u, _ in items],
        "cps": cps,
        "mates": [m for _, (_, m) in items],
        "probs": [e / z for e in exps],
        "best_cp": cps[0],
        "best_mate": items[0][1][1],
        "best_uci": items[0][0],
        "depth": used_depth,
    }


def _teacher_limit(*, nodes: int, movetime: float) -> chess.engine.Limit:
    """Same budget as MultiPV teacher labels. Do not silently downscale."""
    if movetime > 0:
        return chess.engine.Limit(time=movetime)
    if nodes > 0:
        return chess.engine.Limit(nodes=nodes)
    return chess.engine.Limit(depth=16)


def eval_move_score(
    engine,
    board,
    move: chess.Move,
    *,
    nodes: int,
    movetime: float,
    retries: int = 2,
) -> tuple[int, int] | None:
    """Score the played move with the teacher. None after bounded retries (no fake drop)."""
    if move not in board.legal_moves:
        return None
    for _ in range(max(1, int(retries) + 1)):
        board.push(move)
        try:
            info = engine.analyse(board, _teacher_limit(nodes=nodes, movetime=movetime), multipv=1)
            if isinstance(info, list):
                info = info[0]
            sc = info.get("score")
            if sc is None:
                continue
            opp_cp, opp_mate = score_cp_mate(sc, board.turn)
            return -int(opp_cp), -int(opp_mate)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            continue
        finally:
            board.pop()
    return None


def eval_move_cp(engine, board, move: chess.Move, *, nodes: int, movetime: float) -> int | None:
    scored = eval_move_score(engine, board, move, nodes=nodes, movetime=movetime, retries=2)
    return None if scored is None else scored[0]


def board_row(board: chess.Board, soft: dict) -> dict | None:
    """Canonical encoding: CASTLING_MAP K=8 Q=4 k=2 q=1, ep=-1 if none."""
    arr = np.zeros(64, dtype=np.int8)
    try:
        turn, castling, ep = _fast_parse_fen(board.fen(), arr)
    except Exception:
        return None
    soft_i = [-1] * SOFT_K
    soft_p = [0.0] * SOFT_K
    for i, (uci, pr) in enumerate(zip(soft["ucis"], soft["probs"])):
        if uci not in UCI_TO_IDX:
            continue
        soft_i[i] = UCI_TO_IDX[uci]
        soft_p[i] = float(pr)
    if soft_i[0] < 0 or soft["best_uci"] not in UCI_TO_IDX:
        return None
    s = sum(soft_p) or 1.0
    return {
        "board_array": arr.copy(),
        "turn": int(turn),
        "castling": int(castling),
        "ep_square": int(ep),
        "move_idx": UCI_TO_IDX[soft["best_uci"]],
        "cp": int(soft["best_cp"]),
        "mate": int(soft.get("best_mate") or 0),
        "soft_indices": soft_i,
        "soft_probs": [p / s for p in soft_p],
        "label_depth": int(soft["depth"]),
        "phase": phase_id(board),
        "source": SOURCE_HARVEST,
    }


def classify_lapse(
    *,
    best_cp: int,
    best_mate: int,
    model_cp: int,
    model_mate: int,
    model_in_pv: bool,
) -> dict:
    """Mate transitions stay off the centipawn drop used for stats / severity."""
    best_mate = int(best_mate or 0)
    model_mate = int(model_mate or 0)
    if best_mate > 0 and model_mate <= 0:
        return {"tag": "major", "kind": "missed_mate", "drop_cp": None, "mate_delta": best_mate}
    if model_mate < 0 and best_mate >= 0:
        return {"tag": "major", "kind": "allowed_mate", "drop_cp": None, "mate_delta": model_mate}
    if best_mate > 0 and model_mate > best_mate:
        return {
            "tag": "inaccuracy",
            "kind": "slower_mate",
            "drop_cp": None,
            "mate_delta": model_mate - best_mate,
        }
    drop = int(best_cp) - int(model_cp)
    if drop >= 300:
        tag = "major"
    elif int(best_cp) >= 200 and int(model_cp) < 50 and drop >= 150:
        tag = "conversion"
    elif drop >= 150:
        tag = "blunder"
    elif drop >= 75:
        tag = "inaccuracy"
    elif not model_in_pv and drop >= 50:
        tag = "off_pv"
    else:
        tag = "ok"
    return {"tag": tag, "kind": "cp", "drop_cp": drop, "mate_delta": 0}


def tag_lapse(
    *,
    drop: int,
    best_cp: int,
    model_cp: int,
    best_mate: int,
    model_in_pv: bool,
    model_mate: int = 0,
) -> str:
    info = classify_lapse(
        best_cp=best_cp,
        best_mate=best_mate,
        model_cp=model_cp,
        model_mate=model_mate,
        model_in_pv=model_in_pv,
    )
    if info["kind"] == "cp" and info["drop_cp"] != drop and int(best_mate or 0) == 0:
        # Tests pass an explicit drop; keep the computed tag from cp fields.
        pass
    return info["tag"]


def summarize_lapse_audit(items: list[dict]) -> dict:
    cp_drops = [int(a["drop_cp"]) for a in items if a.get("kind", "cp") == "cp" and a.get("drop_cp") is not None]
    mate_n = sum(1 for a in items if a.get("kind") in ("missed_mate", "allowed_mate"))
    tags: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for a in items:
        tags[a.get("tag", "?")] = tags.get(a.get("tag", "?"), 0) + 1
        kinds[a.get("kind", "cp")] = kinds.get(a.get("kind", "cp"), 0) + 1
    return {
        "n": len(items),
        "tags": tags,
        "kinds": kinds,
        "mean_drop_cp": (sum(cp_drops) / len(cp_drops)) if cp_drops else 0.0,
        "median_drop_cp": float(sorted(cp_drops)[len(cp_drops) // 2]) if cp_drops else 0.0,
        "n_cp": len(cp_drops),
        "n_mate_fail": mate_n,
        "n_label_fail": sum(1 for a in items if a.get("label_fail")),
        "major_rate": tags.get("major", 0) / max(1, len(items)),
        "blunder_rate": tags.get("blunder", 0) / max(1, len(items)),
        "conversion_rate": tags.get("conversion", 0) / max(1, len(items)),
    }


def play_one(model, device, opp, teacher, *, model_color, opening, opp_label,
             nodes, movetime, tau, ply_cap, sf_movetime, unlimited_opp):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    kept = []
    meta = {
        "model_color": "white" if model_color == chess.WHITE else "black",
        "opening": " ".join(opening) if opening else "startpos",
        "opp": opp_label,
        "n_model_plies": 0,
        "n_major": 0,
        "n_blunder": 0,
        "n_inacc": 0,
        "n_conversion": 0,
        "sum_drop": 0,
        "n_label_fail": 0,
        "n_analyze_fail": 0,
        "n_missed_mate": 0,
        "n_borderline_confirm": 0,
    }
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            meta["n_model_plies"] += 1
            mv = model_move(model, board, device)
            soft = analyze_multipv(teacher, board, nodes=nodes, movetime=movetime, tau=tau)
            if soft is None:
                meta["n_analyze_fail"] += 1
            else:
                in_pv = mv.uci() in soft["ucis"]
                model_mate = 0
                label_fail = False
                if in_pv:
                    i = soft["ucis"].index(mv.uci())
                    model_cp = int(soft["cps"][i])
                    model_mate = int(soft["mates"][i])
                else:
                    probed = eval_move_score(
                        teacher, board, mv, nodes=nodes, movetime=movetime, retries=2,
                    )
                    if probed is None:
                        meta["n_label_fail"] += 1
                        label_fail = True
                        model_cp = 0
                    else:
                        model_cp, model_mate = probed
                        info0 = classify_lapse(
                            best_cp=int(soft["best_cp"]),
                            best_mate=int(soft.get("best_mate") or 0),
                            model_cp=model_cp,
                            model_mate=model_mate,
                            model_in_pv=False,
                        )
                        drop0 = info0.get("drop_cp")
                        if info0["kind"] == "cp" and drop0 is not None and 50 <= drop0 < 150:
                            meta["n_borderline_confirm"] += 1
                            probed2 = eval_move_score(
                                teacher, board, mv, nodes=nodes, movetime=movetime, retries=1,
                            )
                            if probed2 is not None:
                                model_cp, model_mate = probed2
                if not label_fail:
                    info = classify_lapse(
                        best_cp=int(soft["best_cp"]),
                        best_mate=int(soft.get("best_mate") or 0),
                        model_cp=int(model_cp),
                        model_mate=int(model_mate),
                        model_in_pv=in_pv,
                    )
                    tag = info["tag"]
                    drop_cp = info["drop_cp"]
                    if tag == "major":
                        meta["n_major"] += 1
                    elif tag == "blunder":
                        meta["n_blunder"] += 1
                    elif tag == "inaccuracy":
                        meta["n_inacc"] += 1
                    elif tag == "conversion":
                        meta["n_conversion"] += 1
                    if info["kind"] in ("missed_mate", "allowed_mate"):
                        meta["n_missed_mate"] += 1
                    if drop_cp is not None:
                        meta["sum_drop"] += max(int(drop_cp), 0)
                    row = board_row(board, soft)
                    if row is not None:
                        row["_tag"] = tag
                        row["_kind"] = info["kind"]
                        row["_fen"] = board.fen()
                        row["_drop"] = int(drop_cp) if drop_cp is not None else 0
                        row["_drop_cp"] = drop_cp
                        row["_mate_delta"] = info["mate_delta"]
                        row["_model_uci"] = mv.uci()
                        row["_best_uci"] = soft["best_uci"]
                        row["_model_cp"] = int(model_cp)
                        row["_best_cp"] = int(soft["best_cp"])
                        row["_model_mate"] = int(model_mate)
                        row["_best_mate"] = int(soft.get("best_mate") or 0)
                        kept.append(row)
            if mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            board.push(mv)
        else:
            limit = chess.engine.Limit(time=sf_movetime)
            mv = opp.play(board, limit).move
            if mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            board.push(mv)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == model_color:
        score = 1.0
    else:
        score = 0.0
    meta.update({
        "score": score,
        "plies": len(board.move_stack),
        "termination": outcome.termination.name if outcome else "PLY_CAP",
        "unlimited_opp": unlimited_opp,
    })
    return kept, meta


MISTAKE_TAGS = frozenset({"major", "blunder", "inaccuracy", "conversion"})


def rows_to_data(rows: list[dict]) -> dict:
    return {
        "board_array": torch.from_numpy(np.stack([r["board_array"] for r in rows])),
        "turn": torch.tensor([r["turn"] for r in rows], dtype=torch.int8),
        "castling": torch.tensor([r["castling"] for r in rows], dtype=torch.int8),
        "ep_square": torch.tensor([r["ep_square"] for r in rows], dtype=torch.int8),
        "move_idx": torch.tensor([r["move_idx"] for r in rows], dtype=torch.int64),
        "cp": torch.tensor([r["cp"] for r in rows], dtype=torch.int32),
        "mate": torch.tensor([r["mate"] for r in rows], dtype=torch.int32),
        "soft_indices": torch.tensor([r["soft_indices"] for r in rows], dtype=torch.int64),
        "soft_probs": torch.tensor([r["soft_probs"] for r in rows], dtype=torch.float32),
        "label_depth": torch.tensor([r["label_depth"] for r in rows], dtype=torch.int16),
        "phase": torch.tensor([r["phase"] for r in rows], dtype=torch.int8),
        "source": torch.tensor([r["source"] for r in rows], dtype=torch.int8),
    }


def pack_cache(rows: list[dict], out_pt: Path) -> int:
    if not rows:
        return 0
    data = rows_to_data(rows)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_pt)
    return len(rows)


def next_inbox_shard(inbox: Path) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in inbox.glob("shard_*"):
        try:
            n = max(n, int(p.name.split("_", 1)[1]) + 1)
        except (IndexError, ValueError):
            continue
    return inbox / f"shard_{n:06d}"


def write_inbox_shard(
    rows: list[dict],
    inbox: Path,
    seen_h: np.ndarray | None,
    extra_meta: dict | None = None,
):
    """Write a READY mistakes shard; drop hashes already in ``seen_h``."""
    from autoresearch_8gb.pipeline import filter_disjoint

    if not rows:
        return None, seen_h, {"n_in": 0, "n_out": 0, "internal_dups": 0, "vs_seen": 0}
    data, hs, stats = filter_disjoint(rows_to_data(rows), seen_h)
    if int(stats["n_out"]) == 0:
        return None, seen_h, stats
    sh = next_inbox_shard(inbox)
    sh.mkdir(parents=True, exist_ok=True)
    cache = sh / "soft_cache.pt"
    tmp = cache.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, cache)
    meta = {"n": int(stats["n_out"]), **stats}
    if extra_meta:
        meta.update(extra_meta)
    (sh / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (sh / "READY").write_text(f"n={stats['n_out']}\n", encoding="utf-8")
    if seen_h is None or not getattr(seen_h, "size", 0):
        new_seen = np.unique(hs.astype(np.uint64, copy=False))
    else:
        new_seen = np.unique(np.concatenate([seen_h.astype(np.uint64, copy=False), hs.astype(np.uint64, copy=False)]))
    return sh, new_seen, stats


def newest_stable_step(learner_dir: Path, *, min_age_s: float = 45.0, min_step: int = 0) -> Path | None:
    """Newest step_XXXXX.pt whose mtime is old enough (not being overwritten)."""
    best: Path | None = None
    best_step = int(min_step)
    root = Path(learner_dir)
    if not root.is_dir():
        return None
    for p in root.glob("step_*.pt"):
        try:
            step = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if step <= best_step:
            continue
        try:
            age = time.time() - p.stat().st_mtime
        except OSError:
            continue
        if age < min_age_s:
            continue
        best = p
        best_step = step
    return best


def parse_ckpt_step(path: Path) -> int | None:
    import re
    m = re.search(r"(\d{4,})", Path(path).stem)
    return int(m.group(1)) if m else None


def row_key(row: dict) -> bytes:
    return row["board_array"].tobytes() + bytes([
        int(row["turn"]) & 0xFF,
        int(row["castling"]) & 0xFF,
        (int(row["ep_square"]) + 1) & 0xFF,
    ])


_W_MODEL = None
_W_DEVICE = None
_W_TEACHER = None
_W_SF = None


def _init_worker(checkpoint: str, sf_path: str):
    global _W_MODEL, _W_DEVICE, _W_TEACHER, _W_SF
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _W_DEVICE = torch.device("cpu")
    _W_SF = sf_path
    _W_MODEL = load_checkpoint(checkpoint, _W_DEVICE)
    _W_MODEL.eval()
    _W_TEACHER = chess.engine.SimpleEngine.popen_uci(sf_path)
    _W_TEACHER.configure({"Threads": 1, "Hash": 64})


def _configure_opp(sf_path: str, spec: dict):
    opp = chess.engine.SimpleEngine.popen_uci(sf_path)
    cfg = {"Threads": 1, "Hash": 32}
    if not spec.get("unlimited"):
        cfg["UCI_LimitStrength"] = True
        cfg["UCI_Elo"] = int(spec["sf_elo"])
    opp.configure(cfg)
    return opp


def _worker_game(spec_and_cfg: tuple) -> dict:
    spec, cfg = spec_and_cfg
    opp = _configure_opp(_W_SF, spec)
    try:
        kept, meta = play_one(
            _W_MODEL, _W_DEVICE, opp, _W_TEACHER,
            model_color=chess.BLACK if spec["as_black"] else chess.WHITE,
            opening=spec["opening"],
            opp_label=("unlimited" if spec.get("unlimited") else f"sf{spec['sf_elo']}"),
            nodes=int(cfg["teacher_nodes"]),
            movetime=float(cfg["teacher_movetime"]),
            tau=float(cfg["tau"]),
            ply_cap=int(cfg["ply_cap"]),
            sf_movetime=float(cfg["sf_movetime"]),
            unlimited_opp=bool(spec.get("unlimited")),
        )
    finally:
        opp.quit()
    lost = meta["score"] == 0.0
    holdout = bool(spec.get("holdout"))
    keep_tags = {"major", "blunder", "conversion", "inaccuracy"}
    rows, audit = [], []
    seen: set[str] = set()
    for r in kept:
        fen = r.pop("_fen")
        tag = r.pop("_tag")
        kind = r.pop("_kind", "cp")
        drop = int(r.pop("_drop"))
        drop_cp = r.pop("_drop_cp")
        mate_delta = r.pop("_mate_delta", 0)
        model_uci = r.pop("_model_uci")
        best_uci = r.pop("_best_uci")
        model_cp = int(r.pop("_model_cp"))
        best_cp = int(r.pop("_best_cp"))
        model_mate = int(r.pop("_model_mate", 0))
        best_mate = int(r.pop("_best_mate", 0))
        keep = holdout or tag in keep_tags or (lost and cfg["keep_all_from_losses"])
        if not keep:
            continue
        key = " ".join(fen.split()[:4])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
        audit.append({
            "fen": fen,
            "tag": tag,
            "kind": kind,
            "drop": drop,
            "drop_cp": drop_cp,
            "mate_delta": mate_delta,
            "model_uci": model_uci,
            "best_uci": best_uci,
            "model_cp": model_cp,
            "best_cp": best_cp,
            "model_mate": model_mate,
            "best_mate": best_mate,
            "lost_game": lost,
            "holdout": holdout,
            "label_fail": False,
            "ckpt": cfg.get("ckpt"),
        })
    meta["added"] = len(rows)
    meta["lost"] = lost
    meta["holdout"] = holdout
    meta["game_idx"] = spec["game_idx"]
    meta["ckpt"] = cfg.get("ckpt")
    line = (
        f"g{spec['game_idx']:03d} {meta['opp']:<11} "
        f"{'b' if spec['as_black'] else 'w'} {meta['opening'][:14]:<14} "
        f"{'H' if holdout else ('L' if lost else ('W' if meta['score'] == 1 else 'D'))} "
        f"+{len(rows):3d} maj={meta['n_major']} bl={meta['n_blunder']} "
        f"cv={meta['n_conversion']} drop_cp={meta['sum_drop']} "
        f"fail={meta.get('n_label_fail', 0)} matef={meta.get('n_missed_mate', 0)}"
    )
    return {"rows": rows, "metas": [meta], "logs": [line], "audit": audit}


def split_test(rows: list[dict], audit: list[dict], *, frac: float, seed: int, min_n: int):
    hard_idx = [
        i for i, a in enumerate(audit)
        if a.get("tag") in ("major", "blunder", "conversion")
    ]
    rng = random.Random(seed)
    rng.shuffle(hard_idx)
    n_test = min(len(hard_idx), max(min_n, int(len(hard_idx) * frac)))
    test_i = set(hard_idx[:n_test])
    train_rows, test_rows = [], []
    train_audit, test_audit = [], []
    for i, (r, a) in enumerate(zip(rows, audit)):
        if i in test_i:
            test_rows.append(r)
            test_audit.append(a)
        else:
            train_rows.append(r)
            train_audit.append(a)
    return train_rows, test_rows, train_audit, test_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--ckpt", default="outputs/exp201_elo_max/baseline/live_step57295.pt")
    ap.add_argument("--out-dir", default="outputs/exp201_lapses")
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--black-frac", type=float, default=0.5)
    ap.add_argument("--sf-elos", type=int, nargs="+", default=[1600, 1750, 1900, 2200])
    ap.add_argument("--unlimited-frac", type=float, default=0.2)
    ap.add_argument("--teacher-nodes", type=int, default=600_000)
    ap.add_argument("--teacher-movetime", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=120.0)
    ap.add_argument("--sf-movetime", type=float, default=0.06)
    ap.add_argument("--ply-cap", type=int, default=160)
    ap.add_argument("--keep-all-from-losses", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--test-min", type=int, default=256)
    ap.add_argument("--seed", type=int, default=201)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--inbox",
        default=None,
        help="Write READY mistake shards here as games finish (live trainer ingest).",
    )
    ap.add_argument("--flush-every", type=int, default=8, help="Games between inbox shard writes")
    ap.add_argument(
        "--exclude-cache",
        nargs="*",
        default=None,
        help="Caches whose positions must not enter inbox shards (holdout + current bonus).",
    )
    ap.add_argument("--loop", action="store_true", help="Keep hunting waves until STOP")
    ap.add_argument("--max-rounds", type=int, default=10**9)
    ap.add_argument(
        "--holdout-game-frac",
        type=float,
        default=0.0,
        help="Reserve this fraction of games entirely for holdout (never inbox).",
    )
    ap.add_argument(
        "--refresh-from-dir",
        default=None,
        help="If set, after each wave copy a newer stable step_*.pt (never latest.pt).",
    )
    ap.add_argument("--refresh-min-age-s", type=float, default=45.0)
    ap.add_argument("--frozen-dir", default="outputs/exp201_lapses_frozen")
    args = ap.parse_args()
    if not args.go and not args.smoke:
        raise SystemExit("pass --go or --smoke")
    if args.smoke:
        args.games = 2
        args.workers = 1
        args.max_rounds = 1
        args.loop = False

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inbox = Path(args.inbox) if args.inbox else None
    if inbox is not None:
        inbox.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    if not args.smoke and log_path.exists() and not args.loop:
        log_path.unlink()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    sf_path = resolve_sf()
    n_workers = max(1, min(args.workers, args.games))

    cfg = {
        "teacher_nodes": args.teacher_nodes,
        "teacher_movetime": args.teacher_movetime,
        "tau": args.tau,
        "ply_cap": args.ply_cap,
        "sf_movetime": args.sf_movetime,
        "keep_all_from_losses": args.keep_all_from_losses,
        "ckpt": str(ckpt),
    }

    from autoresearch_8gb.pipeline import load_position_hashes

    seen_h = None
    for raw in args.exclude_cache or []:
        p = Path(raw)
        if not p.exists():
            continue
        extra = load_position_hashes(p)
        seen_h = extra if seen_h is None else np.unique(
            np.concatenate([seen_h.astype(np.uint64, copy=False), extra.astype(np.uint64, copy=False)])
        )
    if inbox is not None:
        for sh in sorted(inbox.glob("shard_*")):
            cache = sh / "soft_cache.pt"
            if not cache.exists():
                continue
            extra = load_position_hashes(cache)
            seen_h = extra if seen_h is None else np.unique(
                np.concatenate([seen_h.astype(np.uint64, copy=False), extra.astype(np.uint64, copy=False)])
            )
    holdout_pt = out / "holdout.pt"
    if holdout_pt.exists():
        extra = load_position_hashes(holdout_pt)
        seen_h = extra if seen_h is None else np.unique(
            np.concatenate([seen_h.astype(np.uint64, copy=False), extra.astype(np.uint64, copy=False)])
        )

    def _stop() -> bool:
        return (out / "STOP").exists() or (inbox is not None and (inbox / "STOP").exists())

    def _counts(items):
        c: dict[str, int] = {}
        for a in items:
            c[a["tag"]] = c.get(a["tag"], 0) + 1
        return c

    def _make_jobs(rng: random.Random, n: int, start_idx: int):
        jobs = []
        for gi in range(n):
            unlimited = rng.random() < args.unlimited_frac
            jobs.append((
                {
                    "game_idx": start_idx + gi,
                    "sf_elo": args.sf_elos[(start_idx + gi) % len(args.sf_elos)],
                    "unlimited": unlimited,
                    "as_black": rng.random() < args.black_frac,
                    "opening": list(OPENINGS[rng.randrange(len(OPENINGS))]),
                    "holdout": rng.random() < float(args.holdout_game_frac),
                },
                dict(cfg),
            ))
        return jobs

    log(f"ckpt={ckpt}", log_path)
    log(
        f"CPU harvest workers={n_workers} games/wave={args.games} "
        f"nodes={args.teacher_nodes} sf={sf_path} device=cpu "
        f"inbox={inbox} loop={args.loop}",
        log_path,
    )
    log(f"castling canonical K={CASTLING_MAP['K']} Q={CASTLING_MAP['Q']}", log_path)

    t0 = time.time()
    rows: list[dict] = []
    audit_all: list[dict] = []
    holdout_rows: list[dict] = []
    holdout_audit: list[dict] = []
    game_metas: list[dict] = []
    seen: set[bytes] = set()
    pending: list[dict] = []
    n_inbox = 0
    flushed_games = 0
    wave = 0
    n_label_fail = 0
    current_ckpt = ckpt
    current_step = parse_ckpt_step(ckpt) or 0
    refresh_dir = Path(args.refresh_from_dir) if args.refresh_from_dir else None
    frozen_dir = Path(args.frozen_dir)

    def _flush_inbox(force: bool = False) -> None:
        nonlocal pending, seen_h, n_inbox, flushed_games
        if inbox is None or not pending:
            return
        if not force and (len(game_metas) - flushed_games) < max(1, args.flush_every):
            return
        mistakes = [r for r in pending]
        pending = []
        flushed_games = len(game_metas)
        sh, seen_h, stats = write_inbox_shard(
            mistakes, inbox, seen_h,
            extra_meta={"ckpt": cfg.get("ckpt"), "wave": wave, "ckpt_step": current_step},
        )
        n_inbox += int(stats["n_out"])
        if sh is not None:
            log(
                f"inbox {sh.name} in={stats['n_in']} out={stats['n_out']} "
                f"dups={stats['internal_dups']} vs_seen={stats['vs_seen']} total={n_inbox}",
                log_path,
            )

    def _absorb_holdout(new_rows: list[dict]) -> None:
        nonlocal seen_h
        if not new_rows:
            return
        from autoresearch_8gb.pipeline import position_hashes
        hs = position_hashes(rows_to_data(new_rows)).astype(np.uint64, copy=False)
        seen_h = hs if seen_h is None else np.unique(
            np.concatenate([seen_h.astype(np.uint64, copy=False), hs])
        )
        pack_cache(holdout_rows, holdout_pt)
        with open(out / "holdout.jsonl", "a", encoding="utf-8") as hf:
            for a in holdout_audit[-len(new_rows):]:
                hf.write(json.dumps(a) + "\n")

    ctx = get_context("spawn")
    pool = None

    def _open_pool(path: Path) -> None:
        nonlocal pool
        if pool is not None:
            pool.close()
            pool.join()
        cfg["ckpt"] = str(path)
        pool = ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(str(path), sf_path))
        log(f"workers loaded ckpt={path}", log_path)

    def _maybe_refresh() -> bool:
        nonlocal current_ckpt, current_step
        if refresh_dir is None:
            return False
        newer = newest_stable_step(
            refresh_dir, min_age_s=float(args.refresh_min_age_s), min_step=current_step,
        )
        if newer is None:
            return False
        step = parse_ckpt_step(newer) or (current_step + 1)
        frozen_dir.mkdir(parents=True, exist_ok=True)
        dest = frozen_dir / f"live_step{step}.pt"
        if not dest.exists():
            shutil.copy2(newer, dest)
        current_ckpt = dest
        current_step = step
        _open_pool(dest)
        log(f"refresh harvest ckpt step={step} src={newer} dest={dest}", log_path)
        return True

    try:
        _maybe_refresh()
        if pool is None:
            _open_pool(current_ckpt)
        while wave < args.max_rounds and not _stop():
            rng = random.Random(args.seed + wave * 10007)
            jobs = _make_jobs(rng, args.games, start_idx=len(game_metas))
            wave_t0 = time.time()
            log(
                f"wave {wave} games={args.games} start_idx={len(game_metas)} "
                f"ckpt={current_ckpt} step={current_step}",
                log_path,
            )
            for result in pool.imap_unordered(_worker_game, jobs, chunksize=1):
                if _stop():
                    log("STOP seen; finishing current result then exiting", log_path)
                    break
                for line in result["logs"]:
                    log(line, log_path)
                game_metas.extend(result["metas"])
                n_label_fail += sum(int(m.get("n_label_fail") or 0) for m in result["metas"])
                new_hold = []
                for r, a in zip(result["rows"], result["audit"]):
                    key = row_key(r)
                    if key in seen:
                        continue
                    seen.add(key)
                    if a.get("holdout"):
                        holdout_rows.append(r)
                        holdout_audit.append(a)
                        new_hold.append(r)
                        continue
                    rows.append(r)
                    audit_all.append(a)
                    if a.get("tag") in MISTAKE_TAGS:
                        pending.append(r)
                if new_hold:
                    _absorb_holdout(new_hold)
                if len(game_metas) % 8 == 0:
                    pack_cache(rows, out / "soft_cache_raw.pt")
                    (out / "games_metas.json").write_text(json.dumps(game_metas, indent=2))
                    sc = sum(m["score"] for m in game_metas) / max(1, len(game_metas))
                    mins = max((time.time() - t0) / 60.0, 1e-6)
                    log(
                        f"progress games={len(game_metas)} cache={len(rows):,} "
                        f"inbox={n_inbox} holdout={len(holdout_rows)} "
                        f"losses={sum(1 for m in game_metas if m.get('lost'))} "
                        f"rate={sc:.2f} uniq/min={n_inbox / mins:.1f} "
                        f"label_fail={n_label_fail} [{time.time() - t0:.0f}s]",
                        log_path,
                    )
                    _flush_inbox(False)
            _flush_inbox(True)
            pack_cache(rows, out / "soft_cache_raw.pt")
            (out / "games_metas.json").write_text(json.dumps(game_metas, indent=2))
            log(f"wave {wave} done [{time.time() - wave_t0:.0f}s]", log_path)
            wave += 1
            if _stop() or not args.loop:
                break
            _maybe_refresh()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if inbox is None:
        train_rows, test_rows, train_audit, test_audit = split_test(
            rows, audit_all, frac=args.test_frac, seed=args.seed, min_n=args.test_min,
        )
        n_train = pack_cache(train_rows, out / "soft_cache.pt")
        n_test = pack_cache(test_rows, out / "test_set.pt")
        (out / "lapses_train.jsonl").write_text(
            "".join(json.dumps(a) + "\n" for a in train_audit), encoding="utf-8",
        )
        (out / "lapses_test.jsonl").write_text(
            "".join(json.dumps(a) + "\n" for a in test_audit), encoding="utf-8",
        )
    else:
        train_rows = [r for r, a in zip(rows, audit_all) if a.get("tag") in MISTAKE_TAGS]
        train_audit = [a for a in audit_all if a.get("tag") in MISTAKE_TAGS]
        test_rows, test_audit = holdout_rows, holdout_audit
        n_train = pack_cache(train_rows, out / "soft_cache.pt")
        n_test = pack_cache(test_rows, holdout_pt) if test_rows else 0
        with open(out / "lapses_train.jsonl", "a", encoding="utf-8") as f:
            for a in train_audit:
                f.write(json.dumps(a) + "\n")

    train_sum = summarize_lapse_audit(train_audit)
    test_sum = summarize_lapse_audit(test_audit)
    report = {
        "n_train": n_train,
        "n_test": n_test,
        "n_raw": len(rows),
        "n_inbox": n_inbox,
        "n_holdout": len(holdout_rows),
        "games": len(game_metas),
        "holdout_games": sum(1 for m in game_metas if m.get("holdout")),
        "waves": wave,
        "loss_games": sum(1 for m in game_metas if m.get("lost")),
        "score_rate": sum(m["score"] for m in game_metas) / max(1, len(game_metas)),
        "train_tags": train_sum["tags"],
        "test_tags": test_sum["tags"],
        "train_kinds": train_sum["kinds"],
        "test_kinds": test_sum["kinds"],
        "mean_drop_cp_train": train_sum["mean_drop_cp"],
        "mean_drop_cp_test": test_sum["mean_drop_cp"],
        "n_mate_fail_train": train_sum["n_mate_fail"],
        "n_mate_fail_test": test_sum["n_mate_fail"],
        "n_label_fail": n_label_fail,
        "workers": n_workers,
        "teacher_nodes": args.teacher_nodes,
        "checkpoint": str(current_ckpt),
        "ckpt_step": current_step,
        "inbox": str(inbox) if inbox else None,
        "rating_note": "Teacher is full-strength Stockfish (no UCI_LimitStrength). Opponent may be limited.",
        "elapsed_s": round(time.time() - t0, 1),
        "uniq_per_min": round(n_inbox / max((time.time() - t0) / 60.0, 1e-6), 2),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if not args.loop:
        (out / "DONE").write_text(json.dumps(report, indent=2))
    log(f"DONE {json.dumps(report)}", log_path)


if __name__ == "__main__":
    main()
