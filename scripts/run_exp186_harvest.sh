#!/bin/bash
# CPU Stockfish MultiPV harvest (full strength). Safe alongside GPU train.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/exp186_sf_multipv
export PYTHONUNBUFFERED=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"
export MOVE_VOCAB_VERSION=compact
mkdir -p "$OUT"
echo "=== exp186 MP harvest start $(date -Is) ===" | tee -a "$OUT/run.log"
python -u experiments/exp186_sf_multipv_harvest_mp.py \
  --go \
  --workers 72 \
  --target 2000000 \
  --depth-min 2 \
  --depth-max 8 \
  --multipv 8 \
  --tau 120 \
  --hash-mb 32 \
  --shard-size 5000 \
  --cache-every 50000 \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp186 harvest end $(date -Is) ===" | tee -a "$OUT/run.log"
