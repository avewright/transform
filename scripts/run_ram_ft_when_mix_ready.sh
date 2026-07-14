#!/usr/bin/env bash
# When quality mix is ready: stop stream → RAM FT; export HF; keep packing next set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source .env
set +a
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MIX=outputs/unseen_quality_mix/soft_cache.pt
STREAM_OUT=outputs/exp191_soft_ft3e_stream
CACHE_OUT=outputs/exp191_soft_ft3e_ram
INIT_FALLBACK=outputs/exp191_soft_ft3b_unseen/best.pt
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
HF_REPO="${HF_REPO:-avewright/chess-soft-multipv-lichess}"
LOG=outputs/overnight_maxelo/fast_ram_chain.log
mkdir -p outputs/overnight_maxelo "$CACHE_OUT"

echo "[$(date -Is)] wait for mix $MIX…" | tee -a "$LOG"
while [[ ! -f "$MIX" ]]; do sleep 10; done
while [[ $(stat -c%s "$MIX" 2>/dev/null || echo 0) -lt 50000000 ]]; do sleep 5; done
echo "[$(date -Is)] mix ready — stopping stream" | tee -a "$LOG"
touch "$STREAM_OUT/SWAP_TO_CACHE"
pkill -INT -f 'output-dir outputs/exp191_soft_ft3e_stream' 2>/dev/null || true
for _ in $(seq 1 24); do
  pgrep -f 'output-dir outputs/exp191_soft_ft3e_stream' >/dev/null 2>&1 || break
  sleep 5
done
pkill -9 -f 'output-dir outputs/exp191_soft_ft3e_stream' 2>/dev/null || true
sleep 5

if [[ -f "$STREAM_OUT/best.pt" ]]; then
  INIT="$STREAM_OUT/best.pt"
else
  INIT="$INIT_FALLBACK"
fi

echo "[$(date -Is)] RAM FT start init=$INIT deep=$MIX → $CACHE_OUT" | tee -a "$LOG"
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$CACHE_OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$MIX" \
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
  2>&1 | tee -a "$CACHE_OUT/run.log" | tee -a "$LOG"
