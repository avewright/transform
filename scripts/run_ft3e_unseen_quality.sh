#!/usr/bin/env bash
# FT3e: from FT3b best → unseen quality/diverse mix (no recycled FT3/FT3b/FT3c keys).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-outputs/exp191_soft_ft3e_unseen_quality}"
INIT="${INIT:-outputs/exp191_soft_ft3b_unseen/best.pt}"
POOL="${POOL:-outputs/lichess_evals_soft/soft_cache_20m.pt}"
MIX_DIR="${MIX_DIR:-outputs/unseen_quality_mix}"
DEEP="$MIX_DIR/soft_cache.pt"
SOFT="${SOFT:-outputs/exp186_sf_multipv/soft_cache.pt}"
HARD="${HARD:-outputs/data_cache/hard_ballast_d15_n2000000_s42.pt}"
LOG="$OUT/run.log"
mkdir -p "$OUT" "$MIX_DIR"

EXCLUDE=(
  outputs/lichess_evals_soft/soft_cache.pt
  outputs/lichess_evals_soft/soft_cache_unseen_ft3.pt
  outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt
  outputs/lichess_evals_soft/quality_deep_mix.pt
  outputs/lichess_evals_soft/soft_cache_ft3style_remainder.pt
)

echo "[$(date -Is)] wait for pool $POOL…" | tee -a "$LOG"
while [[ ! -f "$POOL" ]]; do sleep 30; done

if [[ ! -f "$DEEP" ]]; then
  echo "[$(date -Is)] building unseen quality mix…" | tee -a "$LOG"
  python3 -u scripts/build_unseen_quality_mix.py \
    --pool "$POOL" \
    --exclude "${EXCLUDE[@]}" \
    --output-dir "$MIX_DIR" \
    --target 4000000 \
    --deep-frac 0.72 \
    --puzzle-frac 0.12 \
    --harvest-frac 0.10 \
    --syzygy-frac 0.06 \
    2>&1 | tee -a "$MIX_DIR/build.log" | tee -a "$LOG"
fi

echo "[$(date -Is)] wait for GPU free…" | tee -a "$LOG"
while pgrep -f 'experiments/exp191_400m_meta_attention.py' >/dev/null 2>&1; do
  sleep 20
done
sleep 5

echo "[$(date -Is)] FT3e start init=$INIT deep=$DEEP → $OUT" | tee -a "$LOG"
exec python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$DEEP" \
  --deep-mix-frac 0.55 \
  --soft-frac 0.95 \
  --soft-alpha 0.55 \
  --batch-size 256 \
  --accum-steps 1 \
  --steps 10000 \
  --warmup 400 \
  --muon-lr 0.0025 \
  --adam-lr 8e-5 \
  --value-weight 0.08 \
  --grad-clip 0.5 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --hard-cache "$HARD" \
  --hard-n 1000000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 1000 \
  --eval-interval 2000 \
  --eval-n 1024 \
  --select-metric soft_loss \
  --compile \
  --polar \
  --cautious-wd \
  2>&1 | tee -a "$LOG"
