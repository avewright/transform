#!/bin/bash
# 8GB Elo Autoresearch — architecture / data / optimizer search with Elo promotion.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-outputs/autoresearch_8gb}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ -n "${MAX_TRIALS:-}" ]]; then
  EXTRA+=(--max-trials "$MAX_TRIALS")
fi
if [[ -n "${SOFT_CACHE:-}" ]]; then
  EXTRA+=(--soft-cache "$SOFT_CACHE")
fi
if [[ -n "${DEEP_CACHE:-}" ]]; then
  EXTRA+=(--deep-cache "$DEEP_CACHE")
fi
if [[ -n "${TRAIN_MINUTES:-}" ]]; then
  EXTRA+=(--train-minutes "$TRAIN_MINUTES")
fi

echo "=== autoresearch_8gb start $(date -Is) ===" | tee -a "$OUT/run.log"
python -u experiments/exp194_autoresearch_8gb.py --go "${EXTRA[@]}" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"
echo "=== autoresearch_8gb end $(date -Is) ===" | tee -a "$OUT/run.log"
