#!/bin/bash
set -euo pipefail
cd /root/transform
OUT=outputs/exp190_phase_deep_continue
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH=/root/transform/stockfish/stockfish-latest
mkdir -p "$OUT"
echo "=== exp190 harvest restart $(date -Is) workers=24 hash=128 ===" | tee -a "$OUT/run.log"
python -u experiments/exp190_phase_deep_harvest.py \
  --go \
  --workers 24 \
  --target 1000000 \
  --multipv 8 \
  --tau 120 \
  --hash-mb 128 \
  --shard-size 5000 \
  --cache-every 25000 \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp190 done $(date -Is) ===" | tee -a "$OUT/run.log"
