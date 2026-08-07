#!/bin/bash
# A40 RunPod: 437M meta-factored attention — overnight ELO chase defaults.
# Soft-heavy MultiPV + local hard ballast + async prep. No mid-step HF stream.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-outputs/exp191_400m_meta_attention}"
mkdir -p "$OUT" outputs/data_cache
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOFT="${SOFT_CACHE:-$OUT/soft_cache.pt}"
if [[ ! -f "$SOFT" && -f outputs/exp186_sf_multipv/soft_cache.pt ]]; then
  SOFT=outputs/exp186_sf_multipv/soft_cache.pt
fi
DEEP_ARGS=()
if [[ -n "${DEEP_SOFT_CACHE:-}" && -f "$DEEP_SOFT_CACHE" ]]; then
  DEEP_ARGS=(--deep-soft-cache "$DEEP_SOFT_CACHE" --deep-mix-frac "${DEEP_MIX_FRAC:-0.20}")
elif [[ -f outputs/exp190_phase_deep/soft_cache.pt ]]; then
  DEEP_ARGS=(--deep-soft-cache outputs/exp190_phase_deep/soft_cache.pt --deep-mix-frac "${DEEP_MIX_FRAC:-0.20}")
fi

HARD_CACHE="${HARD_CACHE:-outputs/data_cache/hard_ballast_d15_n2000000_s42.pt}"
HARD_ARGS=(--hard-cache "$HARD_CACHE" --hard-n "${HARD_N:-1000000}" --prefetch-depth "${PREFETCH_DEPTH:-3}")
if [[ "${NO_HARD_CACHE:-0}" == "1" ]]; then
  HARD_ARGS=(--no-hard-cache --prefetch-depth "${PREFETCH_DEPTH:-3}")
fi

CKPT_ARGS=()
if [[ "${GRAD_CHECKPOINT:-0}" == "1" ]]; then
  CKPT_ARGS=(--grad-checkpoint)
fi

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
  if [[ "${RESUME_OPT:-1}" == "1" ]]; then
    RESUME_ARGS+=(--resume-opt)
  fi
fi

INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  INIT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
fi

# Path-to-2500 defaults: prefer hf_soft_mix_5m when present
if [[ -z "${SOFT_CACHE:-}" && -f outputs/hf_soft_mix/soft_cache.pt ]]; then
  SOFT=outputs/hf_soft_mix/soft_cache.pt
fi
if [[ -z "${DEEP_SOFT_CACHE:-}" && -f outputs/hf_soft_mix/deep_soft.pt ]]; then
  DEEP_ARGS=(--deep-soft-cache outputs/hf_soft_mix/deep_soft.pt --deep-mix-frac "${DEEP_MIX_FRAC:-0.25}")
fi

echo "=== exp191 A40 start $(date -Is) ===" | tee -a "$OUT/run.log"
echo "soft=$SOFT soft_frac=${SOFT_FRAC:-0.88} soft_alpha=${SOFT_ALPHA:-0.50} steps=${STEPS:-22000} deep_args=${DEEP_ARGS[*]:-none} hard=${HARD_ARGS[*]} resume=${RESUME:-none} init=${INIT_CHECKPOINT:-none}" | tee -a "$OUT/run.log"

python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --batch-size "${BATCH_SIZE:-256}" \
  --accum-steps 1 \
  --steps "${STEPS:-22000}" \
  --warmup "${WARMUP:-800}" \
  --soft-frac "${SOFT_FRAC:-0.88}" \
  --soft-alpha "${SOFT_ALPHA:-0.50}" \
  --value-weight "${VALUE_WEIGHT:-0.08}" \
  --muon-lr "${MUON_LR:-0.015}" \
  --adam-lr 3e-4 \
  --grad-clip "${GRAD_CLIP:-0.5}" \
  --hflip-p 0.5 \
  --min-depth "${MIN_DEPTH:-12}" \
  --shuffle-buffer 4096 \
  --log-interval 25 \
  --save-interval "${SAVE_INTERVAL:-500}" \
  --eval-interval "${EVAL_INTERVAL:-1500}" \
  --eval-n "${EVAL_N:-2048}" \
  --select-metric "${SELECT_METRIC:-top1}" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  "${HARD_ARGS[@]}" \
  "${DEEP_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  2>&1 | tee -a "$OUT/run.log"

echo "=== exp191 A40 end $(date -Is) ===" | tee -a "$OUT/run.log"
