#!/usr/bin/env bash
# More incumbent-vs-SF19 harvest. CPU analyze first (GPU stays with training).
# GPU scan of unseen SF19 starts only after the train arm releases the card.
set -euo pipefail
cd /root/transform
set -a
# shellcheck disable=SC1091
source .env
set +a
export PATH="/root/transform/.venv/bin:/root/.local/bin:$PATH"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="/root/.local/bin/stockfish-19"

ROOT=/root/transform
OUT="$ROOT/outputs/swa_harvest_more"
PY="$ROOT/.venv/bin/python"
CKPT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"
LOG="$OUT/controller.log"
mkdir -p "$OUT"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "=== harvest-more start pid=$$ ==="
"$PY" -u scripts/prepare_harvest_more.py 2>&1 | tee -a "$LOG"
[[ -f "$OUT/PLAN.json" ]] || { log "FATAL prepare failed"; exit 1; }

log "CPU analyze unresolved off-PV (250k SF19 nodes, 16 workers)"
"$PY" -u scripts/harvest_swa_mistakes.py --analyze \
  --out-dir "$OUT" \
  --ckpt "$CKPT" \
  --nodes 250000 \
  --workers 16 \
  --analyze-limit 110000 \
  --shard-size 4096 \
  2>&1 | tee -a "$OUT/analyze.log"

log "waiting for GPU (overnight-corr15 train must finish)"
while true; do
  if pgrep -f "exp201_recurrent_64.py" >/dev/null; then
    sleep 60
    continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "${used:-0}" -gt 500 ]]; then
    sleep 30
    continue
  fi
  break
done
log "GPU free; scanning unseen SF19 with incumbent SWA"

"$PY" -u scripts/harvest_swa_mistakes.py --go \
  --ckpt "$CKPT" \
  --out-dir "$OUT" \
  --cache "sf19_unseen=$OUT/unseen_sf19.pt" \
  --block-manifest "$ROOT/outputs/overnight_corr15/blocked_manifest.json" \
  --micro-batch 96 \
  --shard-size 4096 \
  --no-compile \
  2>&1 | tee -a "$OUT/scan.log"

log "=== harvest-more finished ==="
