#!/usr/bin/env bash
# Wait for exp273 puzzle FT to release the GPU, then start exp274 Syzygy FT.
set -euo pipefail
cd /root/transform
OUT=outputs/exp274_syzygy_finetune
LOG="$OUT/watchdog.log"
mkdir -p "$OUT"
WAIT_PID="${1:-}"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] exp274 watchdog start wait_pid=${WAIT_PID:-auto}"

if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID="$(pgrep -f 'experiments/exp273_puzzle_finetune.py --go' | head -1 || true)"
fi
if [[ -n "${WAIT_PID:-}" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "[$(date -Is)] waiting for puzzle pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 20
  done
  echo "[$(date -Is)] pid $WAIT_PID exited"
  sleep 5
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "[$(date -Is)] GPU still busy — abort"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  exit 1
fi

export MOVE_VOCAB_VERSION=compact PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/transform/.hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[$(date -Is)] starting exp274 --go"
exec python3 -u experiments/exp274_syzygy_finetune.py --go \
  --max-steps 8000 --train-minutes 240 \
  2>&1 | tee -a "$OUT/run.log"
