#!/bin/bash
# After FT4 (or now): continue FT3-style training on healthy_soft_mix.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=outputs/healthy_soft_mix/ft_chain.log
DEEP=outputs/healthy_soft_mix/soft_cache.pt
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
FT3=outputs/exp191_soft_ft3_lichess/best.pt
FT4=outputs/exp191_soft_ft4_quality
OUT=outputs/exp191_soft_ft5_healthy

mkdir -p "$OUT" outputs/healthy_soft_mix

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Wait for mix
log "waiting for $DEEP"
while [[ ! -f "$DEEP" ]]; do sleep 30; done
log "mix ready"

# Wait for GPU (FT4) to finish
log "waiting for FT4 / free GPU"
while pgrep -f 'output-dir outputs/exp191_soft_ft4_quality' >/dev/null \
   || pgrep -f 'exp191_soft_ft4_quality' >/dev/null; do
  # still training
  if pgrep -f 'exp191_400m_meta_attention.py' >/dev/null; then
    sleep 60
  else
    break
  fi
done
# Also wait quality_tail FT4 process specifically
while pgrep -f 'output-dir outputs/exp191_soft_ft4_quality' >/dev/null; do sleep 30; done

# Prefer FT3 (proven Elo) as init; if FT4 elo clearly better later we can switch
INIT="$FT3"
if [[ -f "$FT4/best.pt" ]]; then
  # use FT3 still as user asked to continue FT3 recipe; FT4 weights optional
  INIT="$FT3"
fi
log "FT5-healthy init=$INIT deep=$DEEP"

python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$DEEP" \
  --deep-mix-frac 0.60 \
  --soft-frac 0.95 \
  --soft-alpha 0.55 \
  --batch-size 256 \
  --accum-steps 1 \
  --steps 10000 \
  --warmup 400 \
  --muon-lr 0.002 \
  --adam-lr 7e-5 \
  --value-weight 0.08 \
  --grad-clip 0.5 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --hard-cache "$HARD" \
  --hard-n 1000000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 1000 \
  --eval-n 2048 \
  --select-metric soft_loss \
  2>&1 | tee -a "$OUT/run.log" | tee -a "$LOG"

log "FT5-healthy done — running elo"
python -u elo_eval_latest.py "$OUT/best.pt" exp191_soft_ft5_healthy \
  --movetime 0.05 --games-per-opening-per-color 1 --stop-after-bracket \
  --elos 1450 1600 1750 1900 2050 \
  2>&1 | tee -a "$OUT/elo.log" | tee -a "$LOG"

echo "$OUT/best.pt" > outputs/healthy_soft_mix/BEST_CKPT.txt
log "done best=$OUT/best.pt"
