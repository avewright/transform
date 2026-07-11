#!/bin/bash
# A40 RunPod: 437M meta-factored attention + soft MultiPV + NorMuon
# Goal: maximize pos/s on ~48GB — large batch, no grad-ckpt unless OOM.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-outputs/exp191_400m_meta_attention}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact

SOFT="${SOFT_CACHE:-$OUT/soft_cache.pt}"
if [[ ! -f "$SOFT" && -f outputs/exp186_sf_multipv/soft_cache.pt ]]; then
  SOFT=outputs/exp186_sf_multipv/soft_cache.pt
fi
DEEP_ARGS=()
if [[ -n "${DEEP_SOFT_CACHE:-}" && -f "$DEEP_SOFT_CACHE" ]]; then
  DEEP_ARGS=(--deep-soft-cache "$DEEP_SOFT_CACHE" --deep-mix-frac "${DEEP_MIX_FRAC:-0.40}")
elif [[ -f outputs/exp190_phase_deep/soft_cache.pt ]]; then
  DEEP_ARGS=(--deep-soft-cache outputs/exp190_phase_deep/soft_cache.pt --deep-mix-frac 0.40)
fi

# Optional: GRAD_CHECKPOINT=1 if OOM at batch 256
CKPT_ARGS=()
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  CKPT_ARGS=(--grad-checkpoint)
fi

echo "=== exp191 A40 start $(date -Is) ===" | tee -a "$OUT/run.log"
echo "soft=$SOFT deep_args=${DEEP_ARGS[*]:-none} ckpt=${GRAD_CHECKPOINT:-0}" | tee -a "$OUT/run.log"

python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --batch-size "${BATCH_SIZE:-256}" \
  --accum-steps 1 \
  --steps "${STEPS:-8000}" \
  --warmup 400 \
  --soft-frac 0.72 \
  --soft-alpha 0.40 \
  --value-weight 0.08 \
  --muon-lr 0.02 \
  --adam-lr 3e-4 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --shuffle-buffer 4096 \
  --log-interval 25 \
  --save-interval 500 \
  --eval-interval 500 \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  "${DEEP_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  2>&1 | tee -a "$OUT/run.log"

echo "=== exp191 A40 end $(date -Is) ===" | tee -a "$OUT/run.log"
