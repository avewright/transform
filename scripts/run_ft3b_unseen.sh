#!/usr/bin/env bash
# FT3-style continue on novel Lichess deep soft (positions not in FT3/FT4 mixes).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1

INIT="${INIT:-outputs/exp191_soft_ft3_lichess/best.pt}"
OUT="${OUT:-outputs/exp191_soft_ft3b_unseen}"
# Prefer late-shard Lichess (shards 2–19, never used by FT3/8M). Fallback: 8M∖seen.
DEEP="${DEEP:-outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt}"
FALLBACK_DEEP="${FALLBACK_DEEP:-outputs/lichess_evals_soft/soft_cache_unseen_ft3.pt}"
LOG="$OUT/run.log"
mkdir -p "$OUT"

echo "[$(date -Is)] wait for novel deep cache ($DEEP or $FALLBACK_DEEP)…" | tee -a "$LOG"
while [[ ! -f "$DEEP" && ! -f "$FALLBACK_DEEP" ]]; do
  sleep 20
done
# Prefer late-shard 12M if present; else unseen-from-8M
if [[ -f "$DEEP" ]]; then
  :
elif [[ -f "$FALLBACK_DEEP" ]]; then
  DEEP="$FALLBACK_DEEP"
fi
echo "[$(date -Is)] using deep=$DEEP" | tee -a "$LOG"

echo "[$(date -Is)] wait for GPU free (no other exp191)…" | tee -a "$LOG"
while pgrep -f 'experiments/exp191_400m_meta_attention.py' >/dev/null 2>&1; do
  sleep 30
done
# If late-shard cache still building and fallback ready, start on fallback;
# if neither ready, wait a bit more for late shards (higher novelty).
if [[ ! -f "$DEEP" ]]; then
  echo "[$(date -Is)] deep not ready after GPU free — wait up to 45m for $DEEP" | tee -a "$LOG"
  for _ in $(seq 1 90); do
    [[ -f "$DEEP" || -f "$FALLBACK_DEEP" ]] && break
    sleep 30
  done
  if [[ -f outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt ]]; then
    DEEP=outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt
  elif [[ -f "$FALLBACK_DEEP" ]]; then
    DEEP="$FALLBACK_DEEP"
  fi
fi
sleep 10

if [[ ! -f "$DEEP" ]]; then
  echo "[$(date -Is)] ERROR missing deep cache $DEEP" | tee -a "$LOG"
  exit 1
fi
if [[ ! -f "$INIT" ]]; then
  echo "[$(date -Is)] ERROR missing init $INIT" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] FT3b start init=$INIT deep=$DEEP → $OUT" | tee -a "$LOG"
# Match FT3 recipe closely; slight bump deep_mix since labels are all new/deep.
exec python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache outputs/exp186_sf_multipv/soft_cache.pt \
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
  --hard-cache outputs/data_cache/hard_ballast_d15_n2000000_s42.pt \
  --hard-n 1000000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 1000 \
  --eval-n 2048 \
  --select-metric soft_loss \
  2>&1 | tee -a "$LOG"
