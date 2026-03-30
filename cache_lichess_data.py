"""Cache a subset of the lichess-sf HF dataset locally for fast training.

Downloads positions from the streaming HF dataset and saves them as a local
JSONL file for much faster training throughput (100x faster than streaming).
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LARGE_REPO = "avewright/chess-positions-lichess-sf"
OUTPUT_FILE = Path("data/lichess_sf_cached_200k.jsonl")
TARGET_COUNT = 200_000


def _load_hf_token():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def main():
    from datasets import load_dataset

    token = _load_hf_token()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {TARGET_COUNT:,} positions from {LARGE_REPO}...")
    print(f"Saving to: {OUTPUT_FILE}")

    ds = load_dataset(LARGE_REPO, split="train", streaming=True, token=token)

    count = 0
    t_start = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            # Write row as JSON line
            record = {
                "fen": row["fen"],
                "best_move": row["best_move"],
                "eval_type": row.get("eval_type", "cp"),
                "eval_value": row.get("eval_value", 0),
                "wdl_win": row.get("wdl_win", 0.33),
                "wdl_draw": row.get("wdl_draw", 0.34),
                "wdl_loss": row.get("wdl_loss", 0.33),
                "phase": row.get("phase", "unknown"),
                "top_moves": row.get("top_moves", "[]"),
                "depth": row.get("depth", 0),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

            if count % 10000 == 0:
                elapsed = time.time() - t_start
                rate = count / elapsed
                eta = (TARGET_COUNT - count) / rate if rate > 0 else 0
                print(f"  {count:>8,}/{TARGET_COUNT:,} ({count*100/TARGET_COUNT:.1f}%) "
                      f"rate={rate:.0f}/s ETA={eta:.0f}s")

            if count >= TARGET_COUNT:
                break

    elapsed = time.time() - t_start
    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"\nDone! {count:,} positions in {elapsed:.0f}s ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
