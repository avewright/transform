#!/bin/bash
# Deep endgame soft harvest (synthetic + trade-down), Syzygy-aware via STOCKFISH.
set -euo pipefail
cd /root/transform
OUT=outputs/exp095_endgame_deep
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH=/root/transform/stockfish/stockfish-latest
mkdir -p "$OUT"
echo "=== exp095 deep endgame $(date -Is) workers=16 depth=14 ===" | tee -a "$OUT/run.log"
python -u experiments/exp095_endgame_harvest.py \
  --output-dir "$OUT" \
  --depth 14 \
  --workers 16 \
  --generators 4 \
  --sf-threads 1 \
  --sf-hash-mb 128 \
  --max-records 200000 \
  --shard-records 5000 \
  --synthetic-weight 0.45 \
  --tradedown-weight 0.35 \
  --random-weight 0.20 \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp095 done $(date -Is) ===" | tee -a "$OUT/run.log"
