#!/usr/bin/env python3
"""Generate MultiPV soft data from standard openings + low-depth SF sampling.

Pipeline:
  1. Build a pool of ~N opening positions from book mainlines (+ SF branching).
  2. From each opening, play a short game by sampling SF's low-depth MultiPV.
  3. Label visited positions with MultiPV soft targets → soft_cache.pt.

Usage:
  python scripts/generate_opening_sf_soft.py --go --smoke
  python scripts/generate_opening_sf_soft.py --go --n-openings 3000 --target 80000 --workers 14
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
from multiprocessing import Event, Process, Queue
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from opening_book import _BOOK_LINES  # noqa: E402
from harvest_local_multipv import (  # noqa: E402
    analyze,
    build_cache,
    resolve_sf,
    score_to_cp,
)

LABEL_TAU = 120.0

# Broad ECO-ish first moves to seed diversity beyond the hardcoded book.
SEED_FIRST = [
    ["e2e4"], ["d2d4"], ["c2c4"], ["g1f3"], ["b1c3"], ["f2f4"], ["b2b3"], ["g2g3"],
    ["e2e4", "e7e5"], ["e2e4", "c7c5"], ["e2e4", "e7e6"], ["e2e4", "c7c6"],
    ["e2e4", "d7d5"], ["e2e4", "d7d6"], ["e2e4", "g8f6"], ["e2e4", "g7g6"],
    ["d2d4", "d7d5"], ["d2d4", "g8f6"], ["d2d4", "e7e6"], ["d2d4", "f7f5"],
    ["d2d4", "c7c5"], ["d2d4", "g7g6"], ["c2c4", "e7e5"], ["c2c4", "c7c5"],
    ["c2c4", "g8f6"], ["g1f3", "d7d5"], ["g1f3", "g8f6"], ["g1f3", "c7c5"],
]


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def apply_uci(board: chess.Board, ucis: list[str]) -> bool:
    for u in ucis:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            return False
        if mv not in board.legal_moves:
            return False
        board.push(mv)
    return True


def book_opening_fens() -> list[str]:
    """All prefix positions from hardcoded book lines."""
    out, seen = [], set()
    for line in _BOOK_LINES:
        b = chess.Board()
        fen = b.fen()
        k = fen_key(fen)
        if k not in seen:
            seen.add(k)
            out.append(fen)
        for u in line:
            try:
                mv = chess.Move.from_uci(u)
            except ValueError:
                break
            if mv not in b.legal_moves:
                break
            b.push(mv)
            fen = b.fen()
            k = fen_key(fen)
            if k not in seen:
                seen.add(k)
                out.append(fen)
    for line in SEED_FIRST:
        b = chess.Board()
        if apply_uci(b, line):
            fen = b.fen()
            k = fen_key(fen)
            if k not in seen:
                seen.add(k)
                out.append(fen)
    return out


def sample_move_from_multipv(engine, board, depth: int, multipv: int, tau: float, rng: random.Random):
    """Sample a next move from SF soft MultiPV at low depth."""
    n = board.legal_moves.count()
    if n == 0:
        return None
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=min(multipv, n))
    if not isinstance(infos, list):
        infos = [infos]
    moves, cps = [], []
    for info in infos:
        mv = info.get("pv", [None])[0]
        if mv is None or mv not in board.legal_moves:
            continue
        cp, _ = score_to_cp(info["score"], board.turn)
        moves.append(mv)
        cps.append(cp)
    if not moves:
        return None
    t = torch.tensor(cps, dtype=torch.float32)
    probs = F.softmax(t / max(tau, 1e-6), dim=0).tolist()
    return rng.choices(moves, weights=probs, k=1)[0]


def expand_openings_with_sf(
    seeds: list[str],
    n_openings: int,
    *,
    sf_path: Path,
    branch_depth: int = 3,
    branch_multipv: int = 5,
    branch_plies: int = 6,
    hash_mb: int = 32,
) -> list[str]:
    """Grow opening pool by SF-sampling a few plies from each seed."""
    eng = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    eng.configure({"Threads": 1, "Hash": hash_mb})
    rng = random.Random(42)
    seen = {fen_key(f) for f in seeds}
    out = list(seeds)
    try:
        i = 0
        while len(out) < n_openings and i < n_openings * 8:
            base = seeds[i % len(seeds)]
            i += 1
            try:
                board = chess.Board(base)
            except Exception:
                continue
            for _ in range(branch_plies):
                if board.is_game_over() or len(out) >= n_openings:
                    break
                mv = sample_move_from_multipv(
                    eng, board, branch_depth, branch_multipv, LABEL_TAU, rng,
                )
                if mv is None:
                    break
                board.push(mv)
                fen = board.fen()
                k = fen_key(fen)
                if k not in seen:
                    seen.add(k)
                    out.append(fen)
    finally:
        eng.quit()
    return out[:n_openings]


def playout_worker(
    wid: int,
    task_q: Queue,
    result_q: Queue,
    stop_ev: Event,
    *,
    sample_depth: int,
    label_depth_min: int,
    label_depth_max: int,
    multipv: int,
    tau: float,
    plies_min: int,
    plies_max: int,
    hash_mb: int,
    sf_path: Path,
):
    eng = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    eng.configure({"Threads": 1, "Hash": hash_mb})
    rng = random.Random(2000 + wid)
    try:
        while not stop_ev.is_set():
            try:
                item = task_q.get(timeout=0.5)
            except Exception:
                continue
            if item is None:
                break
            opening_fen, games = item
            try:
                for _g in range(games):
                    board = chess.Board(opening_fen)
                    n_plies = rng.randint(plies_min, plies_max)
                    for _ in range(n_plies):
                        if board.is_game_over() or stop_ev.is_set():
                            break
                        # Label current position
                        lab_depth = rng.randint(label_depth_min, label_depth_max)
                        rec = analyze(eng, board, lab_depth, multipv, tau)
                        if rec is not None:
                            rec["source"] = "opening_sf_sample"
                            result_q.put(rec)
                        # Advance with low-depth SF sample
                        mv = sample_move_from_multipv(
                            eng, board, sample_depth, multipv, tau, rng,
                        )
                        if mv is None:
                            break
                        board.push(mv)
            except Exception:
                continue
    finally:
        try:
            eng.quit()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-openings", type=int, default=3000)
    ap.add_argument("--games-per-opening", type=int, default=2)
    ap.add_argument("--target", type=int, default=80000, help="Max labeled positions")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample-depth", type=int, default=3, help="SF depth for next-move sample")
    ap.add_argument("--label-depth-min", type=int, default=3)
    ap.add_argument("--label-depth-max", type=int, default=6)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--plies-min", type=int, default=12)
    ap.add_argument("--plies-max", type=int, default=36)
    ap.add_argument("--branch-depth", type=int, default=3)
    ap.add_argument("--tau", type=float, default=LABEL_TAU)
    ap.add_argument("--hash-mb", type=int, default=32)
    ap.add_argument("--out", type=str, default="outputs/autoresearch_8gb/soft_cache_openings.pt")
    ap.add_argument("--shard-name", type=str, default="opening_sf_positions.jsonl")
    args = ap.parse_args()

    if not args.go:
        print("Pass --go")
        return
    if args.smoke:
        args.n_openings = 64
        args.games_per_opening = 1
        args.target = 400
        args.workers = min(4, args.workers)
        args.plies_min, args.plies_max = 6, 12

    sf = resolve_sf()
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / args.shard_name

    print(f"building opening pool from book + SF branch (target {args.n_openings})...")
    seeds = book_opening_fens()
    print(f"  book/seed prefixes: {len(seeds)}")
    openings = expand_openings_with_sf(
        seeds,
        args.n_openings,
        sf_path=sf,
        branch_depth=args.branch_depth,
        branch_multipv=args.multipv,
        branch_plies=6,
        hash_mb=args.hash_mb,
    )
    random.Random(42).shuffle(openings)
    print(f"  openings: {len(openings)}  SF={sf}")

    stop_ev = Event()

    def _stop(*_):
        stop_ev.set()
        print("STOP")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    task_q: Queue = Queue(maxsize=args.workers * 32)
    result_q: Queue = Queue(maxsize=args.workers * 64)
    procs = []
    for wid in range(args.workers):
        p = Process(
            target=playout_worker,
            kwargs=dict(
                wid=wid,
                task_q=task_q,
                result_q=result_q,
                stop_ev=stop_ev,
                sample_depth=args.sample_depth,
                label_depth_min=args.label_depth_min,
                label_depth_max=args.label_depth_max,
                multipv=args.multipv,
                tau=args.tau,
                plies_min=args.plies_min,
                plies_max=args.plies_max,
                hash_mb=args.hash_mb,
                sf_path=sf,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    written = 0
    if shard.exists():
        with open(shard, encoding="utf-8") as f:
            written = sum(1 for _ in f)
        print(f"resume written={written:,}")

    t0 = time.time()
    oi = 0
    with open(shard, "a", encoding="utf-8") as out_f:
        while written < args.target and not stop_ev.is_set():
            while oi < len(openings) and not task_q.full() and not stop_ev.is_set():
                task_q.put((openings[oi], args.games_per_opening))
                oi += 1
            # If we exhausted openings but need more rows, recycle
            if oi >= len(openings) and task_q.empty():
                oi = 0
                random.Random(int(time.time())).shuffle(openings)
            try:
                rec = result_q.get(timeout=0.5)
            except Exception:
                if written >= args.target:
                    break
                continue
            out_f.write(json.dumps(rec) + "\n")
            written += 1
            if written % 500 == 0:
                out_f.flush()
                rate = written / max(time.time() - t0, 1e-6)
                print(f"labeled {written:,}/{args.target:,} ({rate:.1f}/s)", flush=True)

    stop_ev.set()
    for _ in procs:
        try:
            task_q.put_nowait(None)
        except Exception:
            pass
    for p in procs:
        p.join(timeout=5)

    n = build_cache(shard, out_path, max_rows=args.target)
    print(f"done n={n} → {out_path} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
