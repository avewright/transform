#!/usr/bin/env python3
"""Quick analysis of downloaded Lichess parquet shard."""
import pyarrow.parquet as pq
import time
from collections import Counter
from pathlib import Path

# Find the parquet file
cache = Path("outputs/lichess_cache")
parquet_files = list(cache.rglob("*.parquet"))
if not parquet_files:
    print("No parquet files found!")
    exit(1)

path = str(parquet_files[0])
print(f"File: {path}")

t0 = time.time()
pf = pq.read_metadata(path)
print(f"Rows: {pf.num_rows:,}")

schema = pq.read_schema(path)
for i in range(len(schema)):
    print(f"  {schema.field(i)}")

# Read all columns
print("\nLoading data...")
table = pq.read_table(path, columns=["fen", "line", "depth", "cp", "mate"])
print(f"Loaded {len(table):,} rows in {time.time()-t0:.1f}s")

# Depth distribution
depths = [int(d) for d in table.column("depth").to_pylist() if d is not None]
depth_counts = Counter(depths)
print("\nDepth distribution (top 10):")
for d, c in sorted(depth_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  depth {d}: {c:,} ({c/len(depths)*100:.1f}%)")

high15 = sum(1 for d in depths if d >= 15)
high20 = sum(1 for d in depths if d >= 20)
print(f"\nDepth >= 15: {high15:,} ({high15/len(depths)*100:.1f}%)")
print(f"Depth >= 20: {high20:,} ({high20/len(depths)*100:.1f}%)")

# Eval distribution
cps = [c for c in table.column("cp").to_pylist() if c is not None]
mates = sum(1 for m in table.column("mate").to_pylist() if m is not None)
print(f"\nCentipawn evals: {len(cps):,}, Mate evals: {mates:,}")
if cps:
    abs_cps = sorted(abs(c) for c in cps)
    print(f"CP |abs| median: {abs_cps[len(abs_cps)//2]}, p95: {abs_cps[int(len(abs_cps)*0.95)]}")

# Sample data
print("\nSample FENs:")
for f in table.column("fen").to_pylist()[:3]:
    print(f"  {f}")
print("\nSample lines:")
for l in table.column("line").to_pylist()[:3]:
    print(f"  {str(l)[:80]}")

# Check how many have valid best moves in our vocab
import sys
sys.path.insert(0, ".")
from move_vocab import UCI_TO_IDX
import chess

print("\nChecking move validity (sampling 50K)...")
fens = table.column("fen").to_pylist()[:50000]
lines = table.column("line").to_pylist()[:50000]
valid = 0
invalid_move = 0
no_line = 0
not_in_vocab = 0
for fen, line in zip(fens, lines):
    if not line:
        no_line += 1
        continue
    best = line.split()[0]
    if best not in UCI_TO_IDX:
        not_in_vocab += 1
        continue
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(best)
        if move in board.legal_moves:
            valid += 1
        else:
            invalid_move += 1
    except Exception:
        invalid_move += 1

print(f"  Valid: {valid:,}/{50000} ({valid/500:.1f}%)")
print(f"  Not in vocab: {not_in_vocab:,}")
print(f"  Invalid/illegal: {invalid_move:,}")
print(f"  No line: {no_line:,}")
print(f"\nValid rate: {valid/50000*100:.1f}%")
print(f"Estimated usable from this shard: {int(pf.num_rows * valid/50000):,}")
