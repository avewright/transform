#!/bin/bash
# A40 exp185: deep-small 28L/256d pretrain (~3.5h) → full-strength SF RL
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_OUT=outputs/exp185_a40_deep_small
RL_OUT=outputs/exp185_rl_sf
# Prefer large local MultiPV harvest when available; else fall back to exp085 cache.
HARVEST_CACHE=outputs/exp186_sf_multipv/soft_cache_merged.pt
OLD_CACHE=outputs/exp184_a40_wide_soft/soft_cache.pt
if [[ -f "$HARVEST_CACHE" ]]; then
  SOFT_CACHE="$HARVEST_CACHE"
elif [[ -f outputs/exp186_sf_multipv/soft_cache.pt ]]; then
  SOFT_CACHE=outputs/exp186_sf_multipv/soft_cache.pt
else
  SOFT_CACHE="$OLD_CACHE"
fi
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

mkdir -p "$PRETRAIN_OUT" "$RL_OUT"

echo "=== Stage 1: deep-small soft+hard pretrain soft_cache=$SOFT_CACHE $(date -Is) ===" | tee -a "$PRETRAIN_OUT/run.log"
# 8000 × 1024 ≈ 8.2M positions ≈ ~3.5h @ ~650 pos/s (real HF-stream throughput)
python -u experiments/exp185_a40_deep_small.py \
  --go \
  --batch-size 1024 \
  --accum-steps 1 \
  --steps 8000 \
  --warmup 500 \
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
  --output-dir "$PRETRAIN_OUT" \
  --soft-cache "$SOFT_CACHE" \
  2>&1 | tee -a "$PRETRAIN_OUT/run.log"

CKPT="$PRETRAIN_OUT/best.pt"
if [[ ! -f "$CKPT" ]]; then CKPT="$PRETRAIN_OUT/latest.pt"; fi
echo "=== Stage 2: SF expert-iter RL from $CKPT $(date -Is) ===" | tee -a "$RL_OUT/run.log"

python -u experiments/exp183_selfplay.py \
  --go --preset a40 --mode sf \
  --checkpoint "$CKPT" \
  --output-dir "$RL_OUT" \
  --iterations 8 \
  --games 16 \
  --sims 128 \
  2>&1 | tee -a "$RL_OUT/run.log"

echo "=== Pipeline complete $(date -Is) ===" | tee -a "$RL_OUT/run.log"
