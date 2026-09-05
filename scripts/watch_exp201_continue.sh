#!/bin/bash
# After the first 30k run exits: push to HF, then keep training in 30k-step segments.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
OUT=outputs/exp201_recurrent_64
LOG="$OUT/continue.log"
mkdir -p "$OUT"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

wait_for_first_30k() {
  log "waiting for first 30k job to finish"
  while pgrep -f "experiments/exp201_recurrent_64.py --go --skip-mix --max-steps 30000" >/dev/null; do
    sleep 30
  done
  log "first 30k process gone"
}

latest_steps() {
  python3 - <<'PY'
import torch
from pathlib import Path
p = Path("outputs/exp201_recurrent_64/latest.pt")
if not p.exists():
    print(0)
else:
    c = torch.load(p, map_location="cpu", weights_only=False)
    print(int(c.get("steps", 0)))
PY
}

upload() {
  log "uploading to HF"
  python3 -u scripts/upload_exp201_hf.py >>"$LOG" 2>&1 || log "HF upload failed (will retry next segment)"
}

wait_for_first_30k
upload

SEGMENT=30000
while true; do
  if [[ -f "$OUT/STOP_FOREVER" ]]; then
    log "STOP_FOREVER seen; exiting continue loop"
    exit 0
  fi
  cur=$(latest_steps)
  nxt=$((cur + SEGMENT))
  log "registering READY disjoint shards (will not rewrite live cache while training)"
  python3 -u scripts/queue_exp201_disjoint.py --register >>"$LOG" 2>&1 || log "shard register failed"
  log "continue train steps ${cur} -> ${nxt}"
  python3 -u experiments/exp201_recurrent_64.py --go --skip-mix \
    --resume "$OUT/latest.pt" \
    --max-steps "$nxt" \
    --train-minutes 10080 \
    --continue-min-lr \
    --deep-mix-frac 0.4 \
    >>"$OUT/tmux.log" 2>&1 || log "train segment exited rc=$?"
  upload
done
