#!/bin/bash
# exp189: continue HF 200M compact-soft on 2M MultiPV soft + deep hard.
# Goal: max Elo with pure next-move prediction (no MCTS).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/exp189_200m_maxelo_policy
CKPT=outputs/hf_checkpoint/best_model.pt
SOFT=outputs/exp186_sf_multipv/soft_cache.pt

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact

mkdir -p "$OUT"

echo "=== exp189 200M max-Elo policy $(date -Is) ===" | tee -a "$OUT/run.log"
python -u experiments/exp189_200m_maxelo_policy.py \
  --go \
  --checkpoint "$CKPT" \
  --soft-cache "$SOFT" \
  --output-dir "$OUT" \
  --batch-size 448 \
  --accum-steps 2 \
  --steps 16000 \
  --warmup 400 \
  --soft-frac 0.72 \
  --soft-alpha 0.35 \
  --lr 6e-6 \
  --hflip-p 0.5 \
  --value-weight 0.08 \
  --min-depth 15 \
  --shuffle-buffer 4096 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 500 \
  2>&1 | tee -a "$OUT/run.log"

echo "=== exp189 done $(date -Is) ===" | tee -a "$OUT/run.log"
