#!/usr/bin/env bash
# Phase A: stream Lichess soft (GPU busy now).
# Phase B: when virgin/quality RAM cache is ready, stop stream and continue
#          from stream best on the fast RAM mix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT="${INIT:-outputs/exp191_soft_ft3b_unseen/best.pt}"
STREAM_OUT="${STREAM_OUT:-outputs/exp191_soft_ft3e_stream}"
CACHE_OUT="${CACHE_OUT:-outputs/exp191_soft_ft3e_unseen_quality}"
POOL="${POOL:-outputs/lichess_evals_soft/soft_cache_virgin_6m.pt}"
MIX_DIR="${MIX_DIR:-outputs/unseen_quality_mix}"
DEEP_CACHE="$MIX_DIR/soft_cache.pt"
SOFT="${SOFT:-outputs/exp186_sf_multipv/soft_cache.pt}"
HARD="${HARD:-outputs/data_cache/hard_ballast_d15_n2000000_s42.pt}"
LOG="$STREAM_OUT/chain.log"
mkdir -p "$STREAM_OUT" "$CACHE_OUT" "$MIX_DIR"

EXCLUDE=(
  outputs/lichess_evals_soft/soft_cache.pt
  outputs/lichess_evals_soft/soft_cache_unseen_ft3.pt
  outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt
  outputs/lichess_evals_soft/quality_deep_mix.pt
  outputs/lichess_evals_soft/soft_cache_ft3style_remainder.pt
)

echo "[$(date -Is)] PHASE A: stream train from $INIT → $STREAM_OUT" | tee -a "$LOG"
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$STREAM_OUT" \
  --soft-cache "$SOFT" \
  --deep-stream-lichess \
  --deep-stream-shard-start 4 \
  --deep-stream-buffer-fens 8192 \
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
  >> "$STREAM_OUT/run.log" 2>&1 &
STREAM_PID=$!
echo "[$(date -Is)] stream_pid=$STREAM_PID" | tee -a "$LOG"
tail -f "$STREAM_OUT/run.log" >> "$LOG" &
TAIL_PID=$!

# Background: wait for pool → build mix → stop stream
(
  echo "[$(date -Is)] wait for RAM pool $POOL…" | tee -a "$LOG"
  while [[ ! -f "$POOL" ]]; do sleep 20; done
  while [[ $(stat -c%s "$POOL" 2>/dev/null || echo 0) -lt 100000000 ]]; do sleep 10; done
  if [[ ! -f "$DEEP_CACHE" ]]; then
    echo "[$(date -Is)] building unseen quality mix…" | tee -a "$LOG"
    python3 -u scripts/build_unseen_quality_mix.py \
      --pool "$POOL" \
      --exclude "${EXCLUDE[@]}" \
      --output-dir "$MIX_DIR" \
      --target 4000000 \
      --deep-frac 0.72 --puzzle-frac 0.12 --harvest-frac 0.10 --syzygy-frac 0.06 \
      2>&1 | tee -a "$MIX_DIR/build.log" | tee -a "$LOG"
  fi
  echo "[$(date -Is)] CACHE READY — stopping stream for RAM FT" | tee -a "$LOG"
  touch "$STREAM_OUT/SWAP_TO_CACHE"
  kill -INT "$STREAM_PID" 2>/dev/null || true
  for _ in $(seq 1 36); do
    kill -0 "$STREAM_PID" 2>/dev/null || break
    sleep 5
  done
  kill -9 "$STREAM_PID" 2>/dev/null || true
) &
WATCH_PID=$!

wait "$STREAM_PID" || true
kill "$TAIL_PID" 2>/dev/null || true
wait "$WATCH_PID" || true

if [[ ! -f "$DEEP_CACHE" ]]; then
  echo "[$(date -Is)] stream ended without cache — exit" | tee -a "$LOG"
  exit 0
fi

if [[ -f "$STREAM_OUT/best.pt" ]]; then
  CACHE_INIT="$STREAM_OUT/best.pt"
else
  CACHE_INIT="$INIT"
fi

echo "[$(date -Is)] PHASE B: RAM cache FT init=$CACHE_INIT deep=$DEEP_CACHE → $CACHE_OUT" | tee -a "$LOG"
exec python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$CACHE_INIT" \
  --output-dir "$CACHE_OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$DEEP_CACHE" \
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
