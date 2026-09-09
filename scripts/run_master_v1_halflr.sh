#!/usr/bin/env bash
# Isolated half-LR arm. Same authorized mix + incumbent init as master-v1 baseline.
# Only muon/adam LRs change (0.0007/1e-5 -> 0.00035/5e-6). Does not overwrite the incumbent.
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
export HF_HOME=/root/transform/.hf_cache
export HUGGINGFACE_HUB_CACHE=/root/transform/.hf_cache

ROOT=/root/transform
BASE="$ROOT/outputs/master_v1_halflr"
EXPORT="$ROOT/outputs/master_v1_baseline/export"
TRAIN="$BASE/train"
EVAL="$BASE/eval"
LOG="$BASE/run.log"
PY="$ROOT/.venv/bin/python"
INCUMBENT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"

mkdir -p "$BASE" "$TRAIN" "$EVAL"
if [[ ! -e "$BASE/export" ]]; then
  ln -s "$EXPORT" "$BASE/export"
fi
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "=== master-v1 half-LR controller start pid=$$ tmux=${TMUX:-none} ==="
log "question: same mix + incumbent init, half muon/adam LR; clip rate and Elo vs cached incumbent"
log "git=$(git -C "$ROOT" rev-parse HEAD) stockfish=$STOCKFISH_PATH"
log "init=$INCUMBENT muon_lr=0.00035 adam_lr=5e-6"

[[ -f "$EXPORT/AUTHORIZED.json" ]] || { log "FATAL: missing authorized export $EXPORT/AUTHORIZED.json"; exit 1; }
[[ -f "$INCUMBENT" ]] || { log "FATAL: missing incumbent $INCUMBENT"; exit 1; }

if [[ ! -f "$TRAIN/train_summary.json" && ! -f "$TRAIN/latest.pt" ]]; then
  log "starting bounded train 8000 steps / 120 min"
  "$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
    --resume "$INCUMBENT" \
    --soft-cache "$EXPORT/soft_cache.pt" \
    --deep-cache "$EXPORT/deep_cache.pt" \
    --deep-mix-frac 0.05 --bonus-mix-frac 0 \
    --optimizer polar_normuon --muon-lr 0.00035 --adam-lr 5e-6 --warmup 100 \
    --batch-size 64 --max-steps 8000 --train-minutes 120 \
    --val-every 500 --save-every 500 --elo-every 0 \
    --torch-compile --compile-polar \
    --output-dir "$TRAIN" \
    --external-eval "sf19=$EXPORT/sf19_eval.pt" \
    --external-eval "lichess=$EXPORT/lichess_eval.pt" \
    --external-eval "puzzles=$EXPORT/puzzles_eval.pt" \
    --external-eval "syzygy=$EXPORT/syzygy_eval.pt" \
    >>"$TRAIN/train.log" 2>&1 || true
  log "train process exited"
else
  log "train artifacts already present; not restarting"
fi

log "starting 128-game screens (cached incumbent reference if protocol matches)"
"$PY" -u scripts/master_v1_eval_and_report.py \
  --base "$BASE" \
  --incumbent "$INCUMBENT" | tee -a "$LOG"
log "=== controller finished ==="
