#!/usr/bin/env bash
# Isolated overnight-base + 15% verified-correction arm.
# Starts from incumbent SWA with a fresh optimizer. Does not overwrite the incumbent.
# Does not load training_resume.pt.
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
MIX="$ROOT/outputs/overnight_corr15"
TRAIN="$MIX/train"
EVAL="$MIX/eval"
LOG="$MIX/run.log"
PY="$ROOT/.venv/bin/python"
INCUMBENT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"
RESUME_LIVE="$ROOT/outputs/sf19_ft/overnight_20260908/training_resume.pt"

mkdir -p "$MIX" "$TRAIN" "$EVAL"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "=== overnight corr15 controller start pid=$$ tmux=${TMUX:-none} ==="
log "question: 15% verified SWA corrections replace overnight Lichess-replay bonus; same SWA init + original LRs"
log "git=$(git -C "$ROOT" rev-parse HEAD) stockfish=$STOCKFISH_PATH"
log "init=$INCUMBENT (SWA weights only). live resume=$RESUME_LIVE is not passed to --resume"

if [[ ! -f "$MIX/FROZEN.json" ]]; then
  log "building frozen mix"
  "$PY" -u scripts/build_overnight_corr15.py --out "$MIX" 2>&1 | tee -a "$LOG"
fi
[[ -f "$MIX/FROZEN.json" ]] || { log "FATAL: mix not frozen"; exit 1; }
[[ -f "$INCUMBENT" ]] || { log "FATAL: missing incumbent SWA"; exit 1; }
if [[ -f "$RESUME_LIVE" ]]; then
  log "training_resume.pt present and will not be used this arm"
fi

if [[ ! -f "$TRAIN/train_summary.json" && ! -f "$TRAIN/latest.pt" ]]; then
  log "starting 8000-step train muon_lr=0.0007 adam_lr=1e-5 bonus=0.15"
  "$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
    --resume "$INCUMBENT" \
    --soft-cache "$MIX/soft_cache.pt" \
    --deep-cache "$MIX/deep_cache.pt" \
    --bonus-cache "$MIX/bonus_cache.pt" \
    --deep-mix-frac 0.05 --bonus-mix-frac 0.15 \
    --optimizer polar_normuon --muon-lr 0.0007 --adam-lr 1e-5 --warmup 100 \
    --batch-size 64 --max-steps 8000 --train-minutes 120 \
    --val-every 500 --save-every 500 --elo-every 0 \
    --torch-compile --compile-polar \
    --output-dir "$TRAIN" \
    --block-manifest "$MIX/blocked_manifest.json" \
    --external-eval "sf19=$MIX/eval_sf19.pt" \
    --external-eval "syzygy=$MIX/eval_syzygy.pt" \
    >>"$TRAIN/train.log" 2>&1 || true
  log "train process exited"
else
  log "train artifacts already present; not restarting"
fi

log "starting 128-game screens (cached incumbent reference if protocol matches)"
"$PY" -u scripts/master_v1_eval_and_report.py \
  --base "$MIX" \
  --incumbent "$INCUMBENT" | tee -a "$LOG"
log "=== controller finished; no publish, no extra training ==="
