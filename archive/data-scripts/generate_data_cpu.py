"""CPU-only data generation + SF labeling pipeline.
Run this while GPU trains. Generates positions and labels with Stockfish.
Output: outputs/generated_data/batch_*.jsonl files.

Usage: CUDA_VISIBLE_DEVICES="" python3 -u generate_data_cpu.py [--n 500000] [--depth 6] [--threads 8] [--seed 123]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import generate_positions
from move_vocab import UCI_TO_IDX

SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
OUTPUT_DIR = Path("outputs/generated_data")


def label_batch(positions, depth, threads, batch_id, output_dir):
    """Label a batch of positions with SF and write to JSONL."""
    from stockfish import Stockfish

    sf = Stockfish(path=SF_PATH, depth=depth,
                   parameters={"Threads": threads, "Hash": 256})

    out_path = output_dir / f"batch_{batch_id:03d}.jsonl"
    results = []
    t0 = time.time()

    for i, (board, source) in enumerate(positions):
        try:
            fen = board.fen()
            sf.set_fen_position(fen)
            top_moves = sf.get_top_moves(5)
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

            # Phase
            material = 0
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.piece_type != chess.KING:
                    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                            chess.ROOK: 5, chess.QUEEN: 9}
                    material += vals.get(piece.piece_type, 0)
            if material >= 50 and board.fullmove_number <= 12:
                phase = "opening"
            elif material <= 26:
                phase = "endgame"
            else:
                phase = "middlegame"

            top_moves_data = []
            for m in top_moves:
                entry = {"uci": m["Move"]}
                if m.get("Mate") is not None:
                    entry["mate"] = m["Mate"]
                else:
                    entry["cp"] = m.get("Centipawn", 0)
                top_moves_data.append(entry)

            results.append({
                "fen": fen,
                "best_move": best_move,
                "eval_type": eval_type,
                "eval_value": eval_value,
                "wdl": wdl,
                "phase": phase,
                "source": source,
                "top_moves": top_moves_data,
            })

        except Exception:
            try:
                sf = Stockfish(path=SF_PATH, depth=depth,
                               parameters={"Threads": threads, "Hash": 256})
            except Exception:
                pass
            continue

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(positions) - i - 1) / rate
            print(f"  Batch {batch_id}: {i+1:,}/{len(positions):,} | "
                  f"{len(results):,} labeled | {rate:.0f}/s | ETA {eta/60:.1f}m")

    # Write results
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    print(f"  Batch {batch_id} done: {len(results):,}/{len(positions):,} "
          f"in {elapsed:.0f}s ({len(results)/max(elapsed,1):.0f}/s) → {out_path}")
    return len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500000, help="Total positions to generate")
    parser.add_argument("--depth", type=int, default=8, help="SF search depth")
    parser.add_argument("--threads", type=int, default=8, help="SF threads")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (different from exp059's 42)")
    parser.add_argument("--batch-size", type=int, default=100000, help="Positions per batch file")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== CPU Data Generation Pipeline ===")
    print(f"  Target: {args.n:,} positions")
    print(f"  SF depth: {args.depth}, threads: {args.threads}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    # Phase 1: Generate positions
    print(f"[1/2] Generating {args.n:,} positions (seed={args.seed})...")
    t0 = time.time()
    positions = generate_positions(args.n, seed=args.seed)
    gen_time = time.time() - t0
    print(f"  Generated {len(positions):,} in {gen_time:.0f}s")

    # Phase 2: Label in batches
    print(f"\n[2/2] Labeling with SF depth {args.depth} ({args.threads} threads)...")
    total_labeled = 0
    t_label = time.time()
    
    n_batches = (len(positions) + args.batch_size - 1) // args.batch_size
    for batch_id in range(n_batches):
        start = batch_id * args.batch_size
        end = min(start + args.batch_size, len(positions))
        batch = positions[start:end]
        n_labeled = label_batch(batch, args.depth, args.threads, batch_id, OUTPUT_DIR)
        total_labeled += n_labeled

    total_time = time.time() - t0
    label_time = time.time() - t_label

    # Write manifest
    manifest = {
        "total_generated": len(positions),
        "total_labeled": total_labeled,
        "sf_depth": args.depth,
        "sf_threads": args.threads,
        "seed": args.seed,
        "n_batches": n_batches,
        "batch_size": args.batch_size,
        "gen_time_s": round(gen_time),
        "label_time_s": round(label_time),
        "total_time_s": round(total_time),
        "rate_per_s": round(total_labeled / max(label_time, 1)),
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f" DONE: {total_labeled:,} positions labeled")
    print(f" Generation: {gen_time:.0f}s | Labeling: {label_time:.0f}s")
    print(f" Total: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f" Output: {OUTPUT_DIR}/ ({n_batches} batch files)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
