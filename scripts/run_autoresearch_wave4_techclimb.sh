#!/bin/bash
# Wave 4: GAB + fusion stacks + train knobs (tech-tree climb).
# Run after wave1 (and ideally after 2/3) frees the GPU.
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

echo "=== wave4 techclimb start $(date -Is) ===" | tee -a "$OUT/wave4.log"
python -u experiments/exp194_autoresearch_8gb.py --go \
  --soft-cache "$SOFT" \
  --train-minutes "${TRAIN_MINUTES:-180}" \
  --max-steps "${MAX_STEPS:-5000}" \
  --min-steps-done 4000 \
  --only gab gab_no_relbias gab_qk_norm stack_ultimate meta_shaw_soft_swa muon_hot label_smooth dropout_zero warmup_long \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/wave4.log"
echo "=== wave4 techclimb end $(date -Is) ===" | tee -a "$OUT/wave4.log"
