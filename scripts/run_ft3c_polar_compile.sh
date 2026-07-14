#!/usr/bin/env bash
# FT3c: continue from FT3b (or FT3) on late-shard 12M Lichess soft,
# with torch.compile + Polar Express NorMuon (modded-nanogpt style).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-outputs/exp191_soft_ft3c_polar}"
DEEP="${DEEP:-outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt}"
SOFT="${SOFT:-outputs/exp186_sf_multipv/soft_cache.pt}"
HARD="${HARD:-outputs/data_cache/hard_ballast_d15_n2000000_s42.pt}"
LOG="$OUT/run.log"
mkdir -p "$OUT"

pick_init() {
  local ft3b=outputs/exp191_soft_ft3b_unseen/best.pt
  local ft3=outputs/exp191_soft_ft3_lichess/best.pt
  if [[ -f "$ft3b" ]]; then
    echo "$ft3b"
  else
    echo "$ft3"
  fi
}

echo "[$(date -Is)] wait for GPU free (no other exp191)…" | tee -a "$LOG"
while pgrep -f 'experiments/exp191_400m_meta_attention.py' >/dev/null 2>&1; do
  sleep 30
done
sleep 10

INIT="${INIT:-$(pick_init)}"
if [[ ! -f "$INIT" ]]; then
  echo "[$(date -Is)] ERROR missing init $INIT" | tee -a "$LOG"
  exit 1
fi
if [[ ! -f "$DEEP" ]]; then
  echo "[$(date -Is)] ERROR missing deep $DEEP" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] FT3c start init=$INIT deep=$DEEP → $OUT (compile+polar)" | tee -a "$LOG"
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
