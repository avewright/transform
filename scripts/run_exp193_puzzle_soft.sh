#!/bin/bash
# Fast Lichess puzzle → soft_cache (hard one-hot). Safe beside GPU train + SF harvests.
set -euo pipefail
cd /root/transform
OUT=outputs/exp193_puzzle_soft
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
mkdir -p "$OUT"
echo "=== exp193 puzzle soft harvest $(date -Is) ===" | tee -a "$OUT/run.log"
python -u experiments/exp193_puzzle_soft_harvest.py \
  --go \
  --target 200000 \
  --min-rating 1500 \
  --max-rating 2800 \
  --shard-size 10000 \
  --cache-every 25000 \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp193 done $(date -Is) ===" | tee -a "$OUT/run.log"
