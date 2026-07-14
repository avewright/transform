#!/usr/bin/env bash
# FT3d: continue FT3b on leftover FT3-style Lichess deep soft (8M ∖ FT3b's 3.2M),
# with torch.compile + Polar Express.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-outputs/exp191_soft_ft3d_lichess}"
INIT="${INIT:-outputs/exp191_soft_ft3b_unseen/best.pt}"
DEEP="${DEEP:-outputs/lichess_evals_soft/soft_cache_ft3style_remainder.pt}"
SOFT="${SOFT:-outputs/exp186_sf_multipv/soft_cache.pt}"
HARD="${HARD:-outputs/data_cache/hard_ballast_d15_n2000000_s42.pt}"
LOG="$OUT/run.log"
mkdir -p "$OUT"

echo "[$(date -Is)] wait for deep cache $DEEP…" | tee -a "$LOG"
while [[ ! -f "$DEEP" ]]; do sleep 15; done

echo "[$(date -Is)] wait for GPU free…" | tee -a "$LOG"
while pgrep -f 'experiments/exp191_400m_meta_attention.py' >/dev/null 2>&1; do
  sleep 20
done
sleep 5

if [[ ! -f "$INIT" ]]; then
  echo "[$(date -Is)] ERROR missing init $INIT" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] FT3d start init=$INIT deep=$DEEP → $OUT (compile+polar, FT3 recipe)" | tee -a "$LOG"
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
