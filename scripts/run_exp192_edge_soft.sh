#!/bin/bash
# Edge-case / max-Elo soft MultiPV harvest (Black STM, checks, puzzles, promos).
set -euo pipefail
cd /root/transform
OUT=outputs/exp192_edge_soft
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH=/root/transform/stockfish/stockfish-latest
mkdir -p "$OUT"
echo "=== exp192 edge soft $(date -Is) workers=28 hash=128 SF=$STOCKFISH_PATH ===" | tee -a "$OUT/run.log"
python -u experiments/exp192_edge_soft_harvest.py \
  --go \
  --workers 28 \
  --target 500000 \
  --multipv 8 \
  --tau 100 \
  --hash-mb 128 \
  --shard-size 5000 \
  --cache-every 20000 \
  --min-puzzle-rating 1400 \
  --max-puzzle-rating 2800 \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp192 done $(date -Is) ===" | tee -a "$OUT/run.log"
