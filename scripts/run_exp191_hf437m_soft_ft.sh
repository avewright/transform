#!/usr/bin/env bash
# Phase 2: soft FT of 437M via max-Elo harness (top1 select + pure-policy Elo gate).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT="${INIT_CHECKPOINT:-}"
if [[ -z "$INIT" ]]; then
  for c in outputs/champion/champion.pt outputs/hf_437m_ft3h_hub/best_model.pt outputs/hf_437m/best_model.pt; do
    if [[ -f "$c" ]]; then INIT="$c"; break; fi
  done
fi
SOFT="${SOFT_CACHE:-outputs/hf_soft_mix/soft_cache.pt}"
DEEP="${DEEP_SOFT_CACHE:-outputs/hf_soft_mix/deep_soft.pt}"
NAME="${NAME:-exp191_hf437m_soft_ft}"
OUT="${OUT:-outputs/runs/$NAME}"
STEPS="${STEPS:-12000}"
BATCH="${BATCH_SIZE:-64}"
mkdir -p "$OUT" logs

if [[ ! -f "$SOFT" ]]; then
  echo "missing soft cache $SOFT — run scripts/run_path_2500_p1_soft_mix.sh first"
  exit 1
fi

DEEP_ARGS=()
if [[ -f "$DEEP" ]]; then
  DEEP_ARGS=(--deep-soft-cache "$DEEP" --deep-mix-frac "${DEEP_MIX_FRAC:-0.42}")
fi

echo "=== harness.loop init=$INIT soft=$SOFT out=$OUT ==="
python -u -m harness.loop \
  --name "$NAME" \
  --init "$INIT" \
  --soft-cache "$SOFT" \
  "${DEEP_ARGS[@]}" \
  --soft-frac "${SOFT_FRAC:-0.85}" \
  --soft-alpha "${SOFT_ALPHA:-0.38}" \
  --steps "$STEPS" \
  --batch-size "$BATCH" \
  2>&1 | tee -a "$OUT/run.log"

# Compatibility aliases for older path_2500 consumers
mkdir -p outputs/exp191_hf437m_soft_ft
if [[ -f "$OUT/best.pt" ]]; then
  ln -sfn "$(cd "$OUT" && pwd)/best.pt" outputs/exp191_hf437m_soft_ft/best.pt
  ln -sfn "$(cd "$OUT" && pwd)/best.pt" outputs/exp191_hf437m_soft_ft/best_model.pt
fi

echo "=== Phase 2 soft FT done ==="
