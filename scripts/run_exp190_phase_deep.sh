#!/bin/bash
# Phase-balanced deep MultiPV harvest (CPU). Safe alongside GPU exp189.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/exp190_phase_deep
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish-latest}"

mkdir -p "$OUT"
echo "=== exp190 phase-deep harvest $(date -Is) SF=$STOCKFISH_PATH ===" | tee -a "$OUT/run.log"
# 64 workers × 96MB hash ≈ 6GB; leave cores for GPU host
python -u experiments/exp190_phase_deep_harvest.py \
  --go \
  --workers 40 \
  --target 1000000 \
  --multipv 8 \
  --tau 120 \
  --hash-mb 128 \
  --shard-size 5000 \
  --cache-every 25000 \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp190 done $(date -Is) ===" | tee -a "$OUT/run.log"
