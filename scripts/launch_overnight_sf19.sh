#!/usr/bin/env bash
set -euo pipefail
cd /root/transform
export MOVE_VOCAB_VERSION=compact PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export STOCKFISH_PATH=/root/.local/bin/stockfish-19
OUT=/root/transform/outputs/sf19_ft/overnight_20260908
# Let the current bounded 4k-step job finish; its final live checkpoint is the source.
while kill -0 39799 2>/dev/null; do sleep 5; done
exec python3 -u scripts/overnight_sf19.py \
  --out "$OUT" --source outputs/sf19_ft/run2/latest.pt --hours 12 \
  >> "$OUT/controller.log" 2>&1
