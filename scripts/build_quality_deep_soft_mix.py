#!/usr/bin/env python3
"""Build a phase-balanced quality deep soft cache for max-Elo FT.

Sources (later wins on dedupe key):
  1. Lichess cloud evals soft (depth/knodes filtered) — volume + thoroughness
  2. Local harvests: exp190 phase-deep, exp192 edge, exp193 puzzles, exp095 endgame
  3. Syzygy-perfect soft for ≤5-piece positions (capped — must not drown mid/open)

Final phase quotas (approx): opening 22% / middlegame 48% / endgame 30%.
Syzygy rows are capped at --syzygy-frac of the *final* cache (default 8%).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.syzygy
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import _fast_parse_fen  # noqa: E402
from move_vocab import UCI_TO_IDX  # noqa: E402

SOFT_K = 8
PHASE_TARGETS = {"opening": 0.22, "middlegame": 0.48, "endgame": 0.30}
PHASE_ID = {"opening": 0, "middlegame": 1, "endgame": 2}
CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)


def row_key(data, i):
    return (
        bytes(data["board_array"][i].tolist()),
        int(data["turn"][i]),
        int(data["castling"][i]),
        int(data["ep_square"][i]),
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cache(path: Path) -> dict | None:
    if not path.exists():
        log(f"  miss {path}")
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = int(d["board_array"].shape[0])
    log(f"  load {path}: {n:,}")
    return d


def ensure_phase(data: dict) -> torch.Tensor:
    if "phase" in data:
        return data["phase"].to(torch.int8)
    # fallback from piece count
    ba = data["board_array"].numpy()
    n = ba.shape[0]
    ph = np.zeros(n, dtype=np.int8)
    for i in range(n):
        pcs = int(np.count_nonzero(ba[i]))
        ph[i] = 0 if pcs >= 26 else (1 if pcs >= 14 else 2)
    return torch.from_numpy(ph)


def subsample_phase_balanced(
    data: dict,
    n_target: int,
    targets: dict[str, float],
    rng: random.Random,
) -> dict:
    """Take up to n_target rows respecting phase fractions."""
    ph = ensure_phase(data)
    n = ph.shape[0]
    want = {k: int(round(n_target * f)) for k, f in targets.items()}
    # fix rounding
    while sum(want.values()) > n_target:
        want["middlegame"] -= 1
    while sum(want.values()) < n_target:
        want["middlegame"] += 1

    chosen: list[int] = []
    for name, pid in PHASE_ID.items():
        idx = (ph == pid).nonzero(as_tuple=True)[0].tolist()
        rng.shuffle(idx)
        take = min(want[name], len(idx))
        chosen.extend(idx[:take])
        if take < want[name]:
            log(f"  phase {name}: only {take:,}/{want[name]:,} available")

    rng.shuffle(chosen)
    keep = list(CORE)
    if "phase" in data:
        keep.append("phase")
    if "label_depth" in data:
        keep.append("label_depth")
    out = {k: data[k][chosen].contiguous() for k in keep if k in data}
    if "phase" not in out:
        out["phase"] = ph[chosen].contiguous()
    return out


def merge_dedupe(chunks: list[dict]) -> dict:
    """Later chunks win on board key."""
    last: dict = {}
    key_sets = []
    for ci, data in enumerate(chunks):
        n = int(data["board_array"].shape[0])
        key_sets.append(set(data.keys()))
        for i in range(n):
            last[row_key(data, i)] = (ci, i)
    final = list(last.values())
    opt = set.intersection(*key_sets) - set(CORE)
    keep = list(CORE) + sorted(opt)
    out = {k: torch.stack([chunks[ci][k][ii] for ci, ii in final]) for k in keep}
    log(f"  merge unique={len(final):,}")
    return out


def wdl_to_score(wdl: int) -> float:
    # STM-centric: higher better
    return {2: 400.0, 1: 200.0, 0: 0.0, -1: -200.0, -2: -400.0}.get(int(wdl), 0.0)


def gen_syzygy_soft(n: int, syzygy_dir: Path, seed: int, tau: float) -> dict | None:
    if n <= 0 or not syzygy_dir.exists():
        return None
    try:
        tb = chess.syzygy.open_tablebase(str(syzygy_dir))
    except Exception as e:
        log(f"  syzygy open failed: {e}")
        return None

    rng = random.Random(seed)
    # Prefer templates with 3–5 pieces
    pieces_pool = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
    ]

    boards_out = []
    turns, castles, eps, midxs, cps, mates = [], [], [], [], [], []
    soft_i, soft_p, depths, phases = [], [], [], []
    arr_buf = np.zeros(64, dtype=np.int8)
    seen = set()
    tries = 0
    t0 = time.time()

    while len(boards_out) < n and tries < n * 40:
        tries += 1
        board = chess.Board(None)
        board.clear()
        # place kings
        wk = rng.randrange(64)
        bk = rng.randrange(64)
        if chess.square_distance(wk, bk) <= 1:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        n_extra = rng.randint(1, 3)  # total pieces 3–5
        sqs = [i for i in range(64) if i not in (wk, bk)]
        rng.shuffle(sqs)
        ok = True
        for j in range(n_extra):
            pt = rng.choice(pieces_pool)
            color = chess.WHITE if rng.random() < 0.5 else chess.BLACK
            sq = sqs[j]
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(sq, chess.Piece(pt, color))
        if not ok or not board.is_valid() or board.is_checkmate() or board.is_stalemate():
            continue
        if len(board.piece_map()) > 5:
            continue
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        if board.is_checkmate() or board.is_stalemate():
            continue
        fen = board.fen()
        key = " ".join(fen.split(" ")[:4])
        if key in seen:
            continue
        try:
            # score each legal move via TB
            scored = []
            for mv in board.legal_moves:
                uci = mv.uci()
                if uci not in UCI_TO_IDX:
                    continue
                board.push(mv)
                try:
                    wdl = -tb.probe_wdl(board)  # STM after push → negate to pre-move STM
                    scored.append((uci, wdl_to_score(wdl)))
                except Exception:
                    board.pop()
                    continue
                board.pop()
            if not scored:
                continue
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:SOFT_K]
            scores = [s for _, s in scored]
            mx = max(scores)
            exps = [math.exp((s - mx) / tau) for s in scores]
            z = sum(exps) or 1.0
            probs = [e / z for e in exps]
            t, c, e = _fast_parse_fen(" ".join(fen.split(" ")[:4]), arr_buf)
            boards_out.append(arr_buf.copy())
            turns.append(t)
            castles.append(c)
            eps.append(e)
            midxs.append(UCI_TO_IDX[scored[0][0]])
            cps.append(int(scores[0]))
            mates.append(0)
            si = [-1] * SOFT_K
            sp = [0.0] * SOFT_K
            for k, (uci, _) in enumerate(scored):
                si[k] = UCI_TO_IDX[uci]
                sp[k] = probs[k]
            soft_i.append(si)
            soft_p.append(sp)
            depths.append(999)
            phases.append(2)
            seen.add(key)
        except Exception:
            continue
        if len(boards_out) % 5000 == 0 and boards_out:
            log(f"  syzygy labeled {len(boards_out):,}/{n:,}")

    if not boards_out:
        log("  syzygy produced 0 rows")
        return None
    log(f"  syzygy done {len(boards_out):,} in {time.time()-t0:.1f}s tries={tries}")
    return {
        "board_array": torch.tensor(np.stack(boards_out), dtype=torch.int8),
        "turn": torch.tensor(turns, dtype=torch.int8),
        "castling": torch.tensor(castles, dtype=torch.int8),
        "ep_square": torch.tensor(eps, dtype=torch.int8),
        "move_idx": torch.tensor(midxs, dtype=torch.int64),
        "cp": torch.tensor(cps, dtype=torch.int32),
        "mate": torch.tensor(mates, dtype=torch.int32),
        "soft_indices": torch.tensor(soft_i, dtype=torch.int64),
        "soft_probs": torch.tensor(soft_p, dtype=torch.float32),
        "label_depth": torch.tensor(depths, dtype=torch.int16),
        "phase": torch.tensor(phases, dtype=torch.int8),
    }


def summarize(data: dict, label: str) -> None:
    ph = ensure_phase(data)
    n = ph.shape[0]
    parts = []
    for name, pid in PHASE_ID.items():
        c = int((ph == pid).sum())
        parts.append(f"{name[0]}={c/n*100:.1f}%")
    depth = data.get("label_depth")
    dmean = float(depth.float().mean()) if depth is not None else float("nan")
    sw = float((data["soft_indices"] >= 0).sum(1).float().mean())
    log(f"{label}: n={n:,} phases[{', '.join(parts)}] depth_mean={dmean:.1f} soft_w={sw:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/lichess_evals_soft/quality_deep_mix.pt")
    ap.add_argument("--lichess", default="outputs/lichess_evals_soft/soft_cache_8m.pt")
    ap.add_argument("--lichess-fallback", default="outputs/lichess_evals_soft/soft_cache.pt")
    ap.add_argument("--target", type=int, default=4_000_000)
    ap.add_argument("--syzygy-frac", type=float, default=0.08, help="Max fraction of final cache from Syzygy")
    ap.add_argument("--syzygy-dir", default="syzygy")
    ap.add_argument("--tau", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-syzygy", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    lichess = load_cache(Path(args.lichess)) or load_cache(Path(args.lichess_fallback))
    if lichess is None:
        raise SystemExit("no lichess soft cache")

    harvest_paths = [
        Path("outputs/exp190_phase_deep/soft_cache.pt"),
        Path("outputs/exp190_phase_deep_continue/soft_cache.pt"),
        Path("outputs/exp192_edge_soft/soft_cache.pt"),
        Path("outputs/exp193_puzzle_soft/soft_cache.pt"),
        Path("outputs/exp095_endgame_deep/soft_cache.pt"),
    ]
    harvests = []
    for p in harvest_paths:
        d = load_cache(p)
        if d is not None:
            if "phase" not in d:
                d["phase"] = ensure_phase(d)
            harvests.append(d)

    # Budget: 75% phase-balanced Lichess + 17% harvests + ≤8% syzygy (by default)
    n_syz = 0 if args.skip_syzygy else int(args.target * args.syzygy_frac)
    n_harv = int(args.target * 0.17)
    n_lich = args.target - n_syz - n_harv

    log(f"budgets: lichess={n_lich:,} harvest={n_harv:,} syzygy={n_syz:,}")

    lich_bal = subsample_phase_balanced(lichess, n_lich, PHASE_TARGETS, rng)
    summarize(lich_bal, "lichess_balanced")

    chunks = [lich_bal]
    if harvests:
        harv_merged = merge_dedupe(harvests)
        # Prefer edge/puzzle: already in merge (later files win). Phase-balance harvest slice.
        harv_bal = subsample_phase_balanced(
            harv_merged,
            min(n_harv, harv_merged["board_array"].shape[0]),
            # slightly mid/edge heavy for harvests
            {"opening": 0.15, "middlegame": 0.50, "endgame": 0.35},
            rng,
        )
        summarize(harv_bal, "harvest_balanced")
        chunks.append(harv_bal)

    if n_syz > 0:
        syz = gen_syzygy_soft(n_syz, Path(args.syzygy_dir), args.seed, args.tau)
        if syz is not None:
            summarize(syz, "syzygy")
            chunks.append(syz)

    mixed = merge_dedupe(chunks)
    # Final phase rebalance to exact quotas (drop surplus randomly)
    final = subsample_phase_balanced(mixed, min(args.target, mixed["board_array"].shape[0]), PHASE_TARGETS, rng)
    summarize(final, "FINAL")

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".pt.tmp")
    torch.save(final, tmp)
    os.replace(tmp, outp)
    log(f"wrote {outp} ({outp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
