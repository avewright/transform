"""Merge multiple dataset directories into one for combined training.

Usage:
    python merge_datasets.py \
        --output-dir outputs/exp100_diverse_data \
        --sources "outputs/exp087_relabeled_d8/dataset" \
                  "outputs/exp095_endgame_harvest/dataset" \
                  "outputs/exp099_middlegame_harvest/dataset"
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000, help="Records per output shard")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all records from all sources
    total_by_source = {}
    all_records = []
    for src in args.sources:
        files = sorted(src.glob("positions_*.jsonl"))
        count = 0
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        all_records.append(rec)
                        count += 1
        total_by_source[str(src)] = count
        print(f"  {src}: {count} records")

    print(f"Total: {len(all_records)} records")

    # Analyze phase distribution
    phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
    for rec in all_records:
        ply = rec.get("ply", 0)
        if ply == 0 or ply is None:
            # Synthetic endgame (no ply) - classify by material
            phase_counts["endgame"] += 1
        elif ply < 20:
            phase_counts["opening"] += 1
        elif ply < 60:
            phase_counts["middlegame"] += 1
        else:
            phase_counts["endgame"] += 1
    
    total = len(all_records)
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} ({100*count/total:.1f}%)")

    # Write sharded output
    shard_idx = 0
    for i in range(0, len(all_records), args.shard_size):
        shard_idx += 1
        shard = all_records[i:i + args.shard_size]
        path = out_dir / f"positions_{shard_idx:06d}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for rec in shard:
                fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {shard_idx} shards to {out_dir}")


if __name__ == "__main__":
    main()
