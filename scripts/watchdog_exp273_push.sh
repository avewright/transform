#!/usr/bin/env bash
# Wait for the live exp273 train process, then push latest.pt to avewright/puzzle-model.
set -euo pipefail
cd /root/transform
OUT=outputs/exp273_puzzle_finetune
LOG="$OUT/hf_watchdog.log"
mkdir -p "$OUT"
TRAIN_PID="${1:-}"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] watchdog start train_pid=${TRAIN_PID:-auto}"

if [[ -z "$TRAIN_PID" ]]; then
  TRAIN_PID="$(pgrep -f 'experiments/exp273_puzzle_finetune.py --go' | head -1 || true)"
fi
if [[ -n "${TRAIN_PID:-}" ]]; then
  echo "[$(date -Is)] waiting for pid $TRAIN_PID"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 20
  done
  echo "[$(date -Is)] train pid $TRAIN_PID exited"
  sleep 3
else
  echo "[$(date -Is)] no live train pid; uploading current latest.pt if present"
fi

if [[ ! -f "$OUT/latest.pt" ]]; then
  echo "[$(date -Is)] missing $OUT/latest.pt — abort"
  exit 1
fi

export MOVE_VOCAB_VERSION=compact PYTHONUNBUFFERED=1
python3 -u scripts/upload_exp273_hf.py --ckpt "$OUT/latest.pt"
echo "[$(date -Is)] push done"
