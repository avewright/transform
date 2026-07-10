#!/bin/bash
# After exp186 harvest has a large soft_cache, finetune exp185 best.pt on it.
# Safe to run once soft_cache has >= ~200k rows (ideally 500k+).
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_OUT=outputs/exp185_a40_deep_small
FT_OUT=outputs/exp186_finetune_soft
HARVEST_CACHE=outputs/exp186_sf_multipv/soft_cache.pt
OLD_CACHE=outputs/exp184_a40_wide_soft/soft_cache.pt
MERGED_CACHE=outputs/exp186_sf_multipv/soft_cache_merged.pt

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

mkdir -p "$FT_OUT"

if [[ ! -f "$HARVEST_CACHE" ]]; then
  echo "Missing $HARVEST_CACHE — rebuild from JSONL"
  python -u experiments/exp186_sf_multipv_harvest_mp.py --build-cache-only --output-dir outputs/exp186_sf_multipv
fi

# Merge old MultiPV + new harvest (later wins on dups)
python -u scripts/merge_soft_caches.py "$OLD_CACHE" "$HARVEST_CACHE" -o "$MERGED_CACHE"

CKPT="$PRETRAIN_OUT/best.pt"
if [[ ! -f "$CKPT" ]]; then CKPT="$PRETRAIN_OUT/latest.pt"; fi
if [[ ! -f "$CKPT" ]]; then
  echo "No exp185 checkpoint yet at $PRETRAIN_OUT"; exit 1
fi

echo "=== exp186 soft finetune from $CKPT cache=$MERGED_CACHE $(date -Is) ===" | tee -a "$FT_OUT/run.log"
# Shorter finetune: more soft, less over-cycle risk as harvest grows
python -u experiments/exp185_a40_deep_small.py \
  --go \
  --batch-size 1024 \
  --accum-steps 1 \
  --steps 3000 \
  --warmup 200 \
  --soft-frac 0.70 \
  --soft-alpha 0.75 \
  --value-weight 0.15 \
  --muon-lr 0.01 \
  --adam-lr 1.5e-4 \
  --min-depth 12 \
  --shuffle-buffer 4096 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 500 \
  --output-dir "$FT_OUT" \
  --soft-cache "$MERGED_CACHE" \
  --init-checkpoint "$CKPT" \
  2>&1 | tee -a "$FT_OUT/run.log"

echo "=== finetune done $(date -Is) ===" | tee -a "$FT_OUT/run.log"
