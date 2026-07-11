#!/bin/bash
# Resume exp189 mixing in exp190 deep soft targets (40% of soft steps).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/exp189_200m_maxelo_policy
CKPT="$OUT/best.pt"
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
DEEP=outputs/exp190_phase_deep/soft_cache.pt

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact

mkdir -p "$OUT"
echo "=== exp189 deep-mix resume $(date -Is) ===" | tee -a "$OUT/run.log"

python -u experiments/exp189_200m_maxelo_policy.py \
  --go \
  --resume "$CKPT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$DEEP" \
  --deep-mix-frac 0.40 \
  --output-dir "$OUT" \
  --batch-size 448 \
  --accum-steps 2 \
  --steps 8000 \
  --warmup 200 \
  --soft-frac 0.72 \
  --soft-alpha 0.40 \
  --lr 4e-6 \
  --hflip-p 0.5 \
  --value-weight 0.08 \
  --min-depth 15 \
  --shuffle-buffer 4096 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 500 \
  2>&1 | tee -a "$OUT/run.log"

echo "=== exp189 deep-mix done $(date -Is) ===" | tee -a "$OUT/run.log"
