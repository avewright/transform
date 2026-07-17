#!/bin/bash
# Wave 2: modded-nanogpt inspired ablations (QK-Norm, zero-init, Polar NorMuon, meta+QK).
# Run AFTER wave-1 architecture queue finishes (or in parallel on another GPU).
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

echo "=== wave2 start $(date -Is) ===" | tee -a "$OUT/wave2.log"
python -u experiments/exp194_autoresearch_8gb.py --go \
  --soft-cache "$SOFT" \
  --train-minutes "${TRAIN_MINUTES:-180}" \
  --max-steps "${MAX_STEPS:-5000}" \
  --min-steps-done 4000 \
  --only qk_norm zero_init_out qk_norm_zero_init meta_qk_norm polar_normuon \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/wave2.log"
echo "=== wave2 end $(date -Is) ===" | tee -a "$OUT/wave2.log"
