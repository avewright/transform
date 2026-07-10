#!/bin/bash
# A40 exp184: wide full-dim attn + soft MultiPV + NorMuon (~3h schedule)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/exp184_a40_wide_soft
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

echo "=== exp184 start $(date -Is) ===" | tee -a "$OUT/run.log"
python -u experiments/exp184_a40_wide_soft_normuon.py \
  --go \
  --batch-size 384 \
  --accum-steps 1 \
  --steps 6000 \
  --warmup 400 \
  --soft-frac 0.55 \
  --soft-alpha 0.7 \
  --value-weight 0.15 \
  --muon-lr 0.02 \
  --adam-lr 3e-4 \
  --min-depth 12 \
  --shuffle-buffer 4096 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 500 \
  --output-dir "$OUT" \
  --soft-cache "$OUT/soft_cache.pt" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== exp184 end $(date -Is) ===" | tee -a "$OUT/run.log"
