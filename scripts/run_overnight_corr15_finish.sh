#!/usr/bin/env bash
# Finish corr15 from the live full checkpoint. Frozen mix files are not written.
# Play-harvest mistakes are ingested from a copied inbox onto bonus_live.pt.
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
export MKL_NUM_THREADS=1
export STOCKFISH_PATH="/root/.local/bin/stockfish-19"

ROOT=/root/transform
MIX="$ROOT/outputs/overnight_corr15"
TRAIN="$MIX/train"
PY="$ROOT/.venv/bin/python"
INCUMBENT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"
LOG="$MIX/finish.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

[[ -f "$MIX/FROZEN.json" ]] || { log "FATAL: frozen mix missing"; exit 1; }
[[ -f "$TRAIN/latest.pt" ]] || { log "FATAL: missing live latest.pt"; exit 1; }
[[ -f "$TRAIN/bonus_live.pt" ]] || { log "FATAL: missing bonus_live.pt copy"; exit 1; }
[[ -d "$TRAIN/play_inbox" ]] || { log "FATAL: missing play_inbox copy"; exit 1; }

# Do not overwrite frozen caches. bonus_live.pt is the writable sprinkle target.
log "=== corr15 finish start pid=$$ ==="
log "resume=$TRAIN/latest.pt (full optimizer). frozen bonus/soft/deep not written."
log "bonus_live=$TRAIN/bonus_live.pt inbox=$TRAIN/play_inbox"

"$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --resume "$TRAIN/latest.pt" \
  --soft-cache "$MIX/soft_cache.pt" \
  --deep-cache "$MIX/deep_cache.pt" \
  --bonus-cache "$TRAIN/bonus_live.pt" \
  --bonus-inbox "$TRAIN/play_inbox" \
  --deep-mix-frac 0.05 --bonus-mix-frac 0.15 \
  --optimizer polar_normuon --muon-lr 0.0007 --adam-lr 1e-5 --warmup 100 \
  --force-lr \
  --batch-size 64 --max-steps 8000 --train-minutes 180 \
  --val-every 500 --save-every 500 --elo-every 0 \
  --torch-compile --compile-polar \
  --output-dir "$TRAIN" \
  --block-manifest "$MIX/blocked_manifest.json" \
  --external-eval "sf19=$MIX/eval_sf19.pt" \
  --external-eval "syzygy=$MIX/eval_syzygy.pt" \
  >>"$TRAIN/train.log" 2>&1 || true

log "train process exited"
if [[ -f "$TRAIN/STOP" ]]; then
  log "STOP still present; not evaluating"
  exit 0
fi
log "starting 128-game screens (cached incumbent reference if protocol matches)"
"$PY" -u scripts/master_v1_eval_and_report.py \
  --base "$MIX" \
  --incumbent "$INCUMBENT" | tee -a "$LOG"
log "=== finish controller done; no publish ==="
