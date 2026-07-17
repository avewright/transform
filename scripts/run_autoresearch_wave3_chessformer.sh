#!/bin/bash
# Wave 3: Chessformer CF-240M ideas (Monroe & Chalmers, arXiv:2409.12272).
# Soft-policy T=4 aux, SWA, Shaw recipe, heavier value.
# Run AFTER wave-1 finishes (or on a free GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-outputs/autoresearch_8gb}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOFT="${SOFT_CACHE:-$OUT/soft_cache.pt}"
STOCKFISH_PATH="${STOCKFISH_PATH:-stockfish/stockfish/stockfish-windows-x86-64-avx2.exe}"
export STOCKFISH_PATH

echo "=== wave3 chessformer start $(date -Is) ===" | tee -a "$OUT/wave3_chessformer.log"
python -u experiments/exp194_autoresearch_8gb.py --go \
  --soft-cache "$SOFT" \
  --train-minutes "${TRAIN_MINUTES:-180}" \
  --max-steps "${MAX_STEPS:-5000}" \
  --min-steps-done 4000 \
  --only cf_soft_temp cf_soft_temp_heavy cf_swa cf_shaw_recipe cf_value_heavy \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/wave3_chessformer.log"
echo "=== wave3 chessformer end $(date -Is) ===" | tee -a "$OUT/wave3_chessformer.log"
