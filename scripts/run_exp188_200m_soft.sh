#!/bin/bash
# Finetune HF 200M → compact vocab on exp186 soft MultiPV harvest.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/exp188_200m_compact_soft
CKPT=outputs/hf_checkpoint/best_model.pt
SOFT=outputs/exp186_sf_multipv/soft_cache_merged.pt
HARVEST_SOFT=outputs/exp186_sf_multipv/soft_cache.pt
OLD_SOFT=outputs/exp184_a40_wide_soft/soft_cache.pt

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

mkdir -p "$OUT"

# Refresh merged soft cache from latest harvest + old MultiPV
if [[ -f "$HARVEST_SOFT" ]]; then
  echo "=== merge soft caches $(date -Is) ===" | tee -a "$OUT/run.log"
  python -u scripts/merge_soft_caches.py "$OLD_SOFT" "$HARVEST_SOFT" -o "$SOFT" 2>&1 | tee -a "$OUT/run.log"
fi

echo "=== exp188 200M compact soft FT $(date -Is) ===" | tee -a "$OUT/run.log"
# Gentle FT: compact vocab is already free (spatial head). Soft harvest is
# shallow (d2-8) and disagrees with 200M prior (~6% zero-shot top1) — keep
# soft as light mix, mostly hard CE on soft best-move + some deep HF hard.
python -u experiments/exp188_200m_compact_soft.py \
  --go \
  --checkpoint "$CKPT" \
  --soft-cache "$SOFT" \
  --output-dir "$OUT" \
  --batch-size 384 \
  --accum-steps 2 \
  --steps 4000 \
  --warmup 400 \
  --soft-frac 0.70 \
  --soft-alpha 0.30 \
  --lr 8e-6 \
  --value-weight 0.10 \
  --min-depth 12 \
  --shuffle-buffer 2048 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 500 \
  2>&1 | tee -a "$OUT/run.log"

echo "=== exp188 done $(date -Is) ===" | tee -a "$OUT/run.log"
