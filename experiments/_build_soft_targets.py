"""Build soft policy targets from Stockfish multi-PV analysis.

Reads existing shard .pt files, converts board_array → FEN, runs Stockfish
multi-PV=5 at depth 6, and saves auxiliary soft target files alongside shards.

This is CPU-only work — run it while GPU trains.

Usage:
  python experiments/_build_soft_targets.py                          # all shards
  python experiments/_build_soft_targets.py --shard 0                # single shard
  python experiments/_build_soft_targets.py --depth 8 --pvs 5       # deeper analysis
  python experiments/_build_soft_targets.py --max-positions 10000    # quick test
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from move_vocab import UCI_TO_IDX

SF_PATH = str(ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe")
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"

PIECE_CHARS = ".PNBRQKpnbrqk"


def board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
    """Convert shard board_array representation back to FEN string."""
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file_idx in range(8):
            sq = rank * 8 + file_idx
            p = int(ba_row[sq])
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += PIECE_CHARS[p]
        if empty > 0:
            row += str(empty)
        fen_rows.append(row)
    board_str = "/".join(fen_rows)
    turn_str = "w" if int(turn_val) == 0 else "b"
    cv = int(castling_val)
    castle_str = ""
    if cv & 8: castle_str += "K"
    if cv & 4: castle_str += "Q"
    if cv & 2: castle_str += "k"
    if cv & 1: castle_str += "q"
    if not castle_str:
        castle_str = "-"
    ev = int(ep_val)
    if 0 <= ev < 64:
        ep_str = chr(ord('a') + ev % 8) + str(ev // 8 + 1)
    else:
        ep_str = "-"
    return f"{board_str} {turn_str} {castle_str} {ep_str} 0 1"


def analyze_chunk(chunk_data, depth, pvs, sf_path):
    """Analyze a chunk of FENs with Stockfish using chess.engine (reliable)."""
    import chess
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 1, "Hash": 64})

    results = []
    for idx, fen in chunk_data:
        try:
            board = chess.Board(fen)
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=pvs,
            )
            if not infos:
                results.append((idx, [], []))
                continue

            indices = []
            cp_vals = []
            for info in infos:
                pv = info.get("pv")
                if not pv:
                    continue
                uci = pv[0].uci()
                if uci not in UCI_TO_IDX:
                    continue
                move_idx = UCI_TO_IDX[uci]
                score = info.get("score")
                if score is not None:
                    cp_score = score.white().score(mate_score=30000)
                    if cp_score is None:
                        cp_score = 0
                else:
                    cp_score = 0
                indices.append(move_idx)
                cp_vals.append(cp_score)

            results.append((idx, indices, cp_vals))
        except Exception:
            results.append((idx, [], []))

    engine.quit()
    return results


def process_shard(shard_path, depth, pvs, workers, max_positions=None):
    """Process one shard file: load, analyze with SF, save soft targets."""
    shard_name = shard_path.stem
    out_path = shard_path.parent / f"{shard_name}_soft.pt"

    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return out_path

    print(f"  Loading {shard_path.name}...")
    data = torch.load(shard_path, map_location="cpu", weights_only=False)
    n = data["board_array"].shape[0]
    if max_positions:
        n = min(n, max_positions)

    print(f"  Converting {n:,} positions to FEN...")
    t0 = time.time()
    fens = []
    for i in range(n):
        fen = board_array_to_fen(
            data["board_array"][i], data["turn"][i],
            data["castling"][i], data["ep_square"][i])
        fens.append((i, fen))
    print(f"  FEN conversion: {time.time()-t0:.1f}s")

    # Split into chunks for parallel workers
    chunk_size = max(1, len(fens) // workers)
    chunks = []
    for start in range(0, len(fens), chunk_size):
        chunks.append(fens[start:start + chunk_size])

    print(f"  Analyzing with SF depth={depth} pvs={pvs} ({workers} workers)...")
    t0 = time.time()
    all_results = []

    if workers <= 1:
        # Single-threaded: call directly in main process (avoids subprocess issues)
        chunk_results = analyze_chunk(fens, depth, pvs, SF_PATH)
        all_results.extend(chunk_results)
        elapsed = time.time() - t0
        rate = len(all_results) / elapsed if elapsed > 0 else 0
        print(f"    done ({len(all_results):,}/{n:,} positions, "
              f"{rate:.0f} pos/s)")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(analyze_chunk, chunk, depth, pvs, SF_PATH): i
                for i, chunk in enumerate(chunks)
            }
            done = 0
            for future in as_completed(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                done += 1
                elapsed = time.time() - t0
                rate = sum(1 for _ in all_results) / elapsed if elapsed > 0 else 0
                eta = (n - len(all_results)) / rate if rate > 0 else 0
                print(f"    chunk {done}/{len(chunks)} done "
                      f"({len(all_results):,}/{n:,} positions, "
                      f"{rate:.0f} pos/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Analysis complete: {len(all_results):,} positions in {elapsed:.1f}s "
          f"({len(all_results)/elapsed:.0f} pos/s)")

    # Build soft target tensors
    # K = max top moves across all positions
    K = pvs
    soft_indices = torch.full((n, K), -1, dtype=torch.int16)
    soft_cp = torch.zeros((n, K), dtype=torch.int16)

    valid = 0
    for idx, indices, cp_vals in all_results:
        if idx >= n:
            continue
        k = min(len(indices), K)
        if k > 0:
            soft_indices[idx, :k] = torch.tensor(indices[:k], dtype=torch.int16)
            soft_cp[idx, :k] = torch.tensor(
                [max(-32000, min(32000, c)) for c in cp_vals[:k]],
                dtype=torch.int16)
            valid += 1

    print(f"  Valid soft targets: {valid:,}/{n:,} ({100*valid/n:.1f}%)")

    torch.save({
        "soft_indices": soft_indices,
        "soft_cp": soft_cp,
        "K": K,
        "depth": depth,
        "pvs": pvs,
        "n_positions": n,
        "n_valid": valid,
    }, out_path)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved {out_path.name} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build soft policy targets")
    parser.add_argument("--shard", type=int, default=None,
                        help="Process single shard index (default: all)")
    parser.add_argument("--depth", type=int, default=6,
                        help="Stockfish depth (default: 6)")
    parser.add_argument("--pvs", type=int, default=5,
                        help="Number of PV lines (default: 5)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel Stockfish workers (default: 4)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Max positions per shard (for testing)")
    args = parser.parse_args()

    # Find shards
    shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    shards = [s for s in shards if "_soft" not in s.stem]
    print(f"Found {len(shards)} shards in {SHARD_DIR}")

    if args.shard is not None:
        if args.shard >= len(shards):
            print(f"Error: shard {args.shard} >= {len(shards)}")
            return
        shards = [shards[args.shard]]
        print(f"Processing shard {args.shard} only")

    t_total = time.time()
    for i, shard_path in enumerate(shards):
        print(f"\n[{i+1}/{len(shards)}] {shard_path.name}")
        process_shard(shard_path, args.depth, args.pvs, args.workers,
                      args.max_positions)

    print(f"\nAll done in {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
