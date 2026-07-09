#!/bin/bash
# A40 pipeline: pretrain (resume) → RL (resume iterations). Run inside tmux:
#   tmux new-session -d -s pretrain "bash scripts/a40_pipeline.sh"
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_OUT=outputs/exp182_pretrain_a40_309m
RL_OUT=outputs/rl_selfplay_a40

echo "=== Stage 1: Pretrain resume from ${PRETRAIN_OUT}/latest.pt ==="
python experiments/exp182_pretrain_700m.py \
  --go --a100 --a100-309m --force --resume \
  --batch-size 192 --accum-steps 1 \
  --output-dir "$PRETRAIN_OUT" \
  --save-interval 1000 --log-interval 25 \
  2>&1 | tee -a "$PRETRAIN_OUT/run.log"

echo ""
echo "=== Stage 2: RL resume (continues iter counter if ${RL_OUT}/latest.pt exists) ==="
python experiments/exp183_selfplay.py \
  --go --preset a40 --mode sf \
  --checkpoint "$PRETRAIN_OUT/latest.pt" \
  --output-dir "$RL_OUT" \
  2>&1 | tee -a "$RL_OUT/run.log"

echo ""
echo "=== Pipeline complete ==="
