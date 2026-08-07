#!/usr/bin/env bash
# Wait for wider_shallower exp197, then launch meta_shaw_elo challenger.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish)}"

WIDER_PID="${WIDER_PID:-}"
WIDER_DIR="${WIDER_DIR:-outputs/exp197_wider_shallower}"
META_DIR="${META_DIR:-outputs/exp197_meta_shaw_elo}"
SOFT="${SOFT_CACHE:-outputs/hf_soft_mix/soft_cache.pt}"
DEEP="${DEEP_CACHE:-outputs/hf_soft_mix/deep_soft.pt}"
TRAIN_MINUTES="${TRAIN_MINUTES:-240}"
MAX_STEPS="${MAX_STEPS:-16000}"
LOG=logs/exp197_meta_after_wider.log
mkdir -p logs "$META_DIR"

echo "[$(date +%Y-%m-%dT%H:%M:%S)] waiting for wider train to finish…" | tee -a "$LOG"

if [[ -n "$WIDER_PID" ]]; then
  while kill -0 "$WIDER_PID" 2>/dev/null; do sleep 30; done
else
  # Prefer PID file / process match; else wait for summary.json
  while pgrep -f 'exp197_hf_soft_elo.py --go --trial wider_shallower' >/dev/null 2>&1; do
    sleep 60
  done
  # Also wait for summary if train just exited into elo eval
  for _ in $(seq 1 120); do
    [[ -f "$WIDER_DIR/summary.json" || -f "$WIDER_DIR/train_result.json" ]] && break
    pgrep -f 'exp197_hf_soft_elo.py' >/dev/null 2>&1 || break
    sleep 30
  done
fi

echo "[$(date +%Y-%m-%dT%H:%M:%S)] wider done; launching meta_shaw_elo → $META_DIR" | tee -a "$LOG"
if [[ -f "$META_DIR/summary.json" ]]; then
  echo "meta already has summary — skip" | tee -a "$LOG"
  exit 0
fi

python -u experiments/exp197_hf_soft_elo.py --go \
  --trial meta_shaw_elo \
  --soft-cache "$SOFT" \
  --deep-cache "$DEEP" \
  --max-steps "$MAX_STEPS" \
  --train-minutes "$TRAIN_MINUTES" \
  --output-dir "$META_DIR" \
  2>&1 | tee -a "$META_DIR/run.log" | tee -a "$LOG"

echo "[$(date +%Y-%m-%dT%H:%M:%S)] meta done" | tee -a "$LOG"
