"""Relabel existing positions with deeper SF analysis (more PVs).
Picks random positions from existing JSONL files and relabels with SF depth 12, 10 PVs.
Run on CPU while GPU trains.

Usage: CUDA_VISIBLE_DEVICES="" python3 -u relabel_deep.py [--n 50000] [--depth 12] [--pvs 10]
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from move_vocab import UCI_TO_IDX

SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
OUTPUT_DIR = Path("outputs/deep_labeled")


def relabel_positions(fens, depth, n_pvs, threads):
    """Relabel positions with SF at given depth and number of PVs."""
    from stockfish import Stockfish

    sf = Stockfish(path=SF_PATH, depth=depth,
                   parameters={"Threads": threads, "Hash": 512})

    results = []
    t0 = time.time()

    for i, (fen, source, phase) in enumerate(fens):
        try:
            board = chess.Board(fen)
            sf.set_fen_position(fen)
            top_moves = sf.get_top_moves(n_pvs)
            if not top_moves:
                continue

            best = top_moves[0]
            best_move = best["Move"]

            if best_move not in UCI_TO_IDX:
                continue
            move_obj = chess.Move.from_uci(best_move)
            if move_obj not in board.legal_moves:
                continue

            eval_type = "mate" if best.get("Mate") is not None else "cp"
            eval_value = best["Mate"] if eval_type == "mate" else best.get("Centipawn", 0)

            if eval_type == "mate":
                wdl = [1.0, 0.0, 0.0] if eval_value > 0 else [0.0, 0.0, 1.0]
            else:
                k = 1.0 / 111.7
                win = 1.0 / (1.0 + math.exp(-k * eval_value))
                loss_p = 1.0 - win
                draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
                total = win + draw + loss_p
                wdl = [win / total, draw / total, loss_p / total]

            top_moves_data = []
            for m in top_moves:
                uci = m["Move"]
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves or uci not in UCI_TO_IDX:
                    continue
                entry = {"uci": uci}
                if m.get("Mate") is not None:
                    entry["mate"] = m["Mate"]
                else:
                    entry["cp"] = m.get("Centipawn", 0)
                top_moves_data.append(entry)

            if len(top_moves_data) < 2:
                continue

            results.append({
                "fen": fen,
                "best_move": best_move,
                "eval_type": eval_type,
                "eval_value": eval_value,
                "wdl": wdl,
                "phase": phase,
                "source": source,
                "top_moves": top_moves_data,
                "sf_depth": depth,
                "n_pvs": len(top_moves_data),
            })

        except Exception:
            try:
                sf = Stockfish(path=SF_PATH, depth=depth,
                               parameters={"Threads": threads, "Hash": 512})
            except Exception:
                pass
            continue

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(fens) - i - 1) / rate / 60
            print(f"  {i+1:,}/{len(fens):,} | {len(results):,} labeled | "
                  f"{rate:.1f}/s | ETA {eta:.1f}m", flush=True)

    return results


def load_existing_fens(max_n):
    """Load FENs from existing generated data."""
    all_positions = []

    # Load from generated batches
    for path in sorted(glob.glob("outputs/generated_data/batch_*.jsonl")):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                all_positions.append((d["fen"], d.get("source", "unknown"), d.get("phase", "unknown")))

    # Load from exp059
    exp059_path = Path("outputs/exp059_data_scaling/generated_200k.jsonl")
    if exp059_path.exists():
        with open(exp059_path) as f:
            for line in f:
                d = json.loads(line)
                all_positions.append((d["fen"], d.get("source", "unknown"), d.get("phase", "unknown")))

    print(f"  Total available: {len(all_positions):,}")

    # Deduplicate by FEN
    seen = set()
    unique = []
    for fen, source, phase in all_positions:
        if fen not in seen:
            seen.add(fen)
            unique.append((fen, source, phase))

    print(f"  After dedup: {len(unique):,}")

    # Sample if needed
    if len(unique) > max_n:
        random.seed(42)
        random.shuffle(unique)
        unique = unique[:max_n]

    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--pvs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Loading existing positions...")
    fens = load_existing_fens(args.n)
    print(f"  Selected: {len(fens):,} for relabeling")

    print(f"\n[2/2] Relabeling with SF depth {args.depth}, {args.pvs} PVs ({args.threads} threads)...")
    t0 = time.time()
    results = relabel_positions(fens, args.depth, args.pvs, args.threads)
    elapsed = time.time() - t0

    # Save
    out_path = OUTPUT_DIR / f"deep_d{args.depth}_pv{args.pvs}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n  Done: {len(results):,} positions in {elapsed/60:.1f}m "
          f"({len(results)/elapsed:.1f}/s)")
    print(f"  Saved: {out_path}")
    print(f"  Avg PVs per position: {sum(r['n_pvs'] for r in results)/len(results):.1f}")


if __name__ == "__main__":
    main()
