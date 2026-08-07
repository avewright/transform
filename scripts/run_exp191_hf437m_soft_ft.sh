#!/usr/bin/env bash
# Phase 2: soft FT of hf_437m on HF soft mix (A40 preferred; works on CUDA).
# Elo-gates via periodic probe — never crown on soft_loss alone.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT="${INIT_CHECKPOINT:-outputs/hf_437m/best_model.pt}"
SOFT="${SOFT_CACHE:-outputs/hf_soft_mix/soft_cache.pt}"
DEEP="${DEEP_SOFT_CACHE:-outputs/hf_soft_mix/deep_soft.pt}"
OUT="${OUT:-outputs/exp191_hf437m_soft_ft}"
STEPS="${STEPS:-12000}"
BATCH="${BATCH_SIZE:-64}"   # 256 on A40; 64 safer on smaller GPUs
mkdir -p "$OUT" logs

if [[ ! -f "$SOFT" ]]; then
  echo "missing soft cache $SOFT — run scripts/run_path_2500_p1_soft_mix.sh first"
  exit 1
fi

DEEP_ARGS=()
if [[ -f "$DEEP" ]]; then
  DEEP_ARGS=(--deep-soft-cache "$DEEP" --deep-mix-frac "${DEEP_MIX_FRAC:-0.25}")
fi

echo "=== exp191 soft FT init=$INIT soft=$SOFT out=$OUT ==="
python -u experiments/exp191_400m_meta_attention.py --go \
  --init-checkpoint "$INIT" \
  --soft-cache "$SOFT" \
  "${DEEP_ARGS[@]}" \
  --soft-frac "${SOFT_FRAC:-0.85}" \
  --soft-alpha "${SOFT_ALPHA:-0.45}" \
  --steps "$STEPS" \
  --warmup "${WARMUP:-600}" \
  --batch-size "$BATCH" \
  --min-depth "${MIN_DEPTH:-12}" \
  --hflip-p 0.5 \
  --value-weight "${VALUE_WEIGHT:-0.08}" \
  --log-interval 25 \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --eval-interval "${EVAL_INTERVAL:-2000}" \
  --select-metric top1 \
  --no-hard-cache \
  --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/run.log"

# Elo-gate latest / best
for CK in "$OUT/best_model.pt" "$OUT/latest.pt"; do
  if [[ -f "$CK" ]]; then
    TAG="elo_$(basename "$(dirname "$CK")")_$(basename "$CK" .pt)"
    echo "=== Elo probe $CK ==="
    python -u elo_eval_latest.py "$CK" "path2500_${TAG}" \
      --elos 1600 1750 1900 2050 \
      --games-per-opening-per-color 1 \
      --stop-after-bracket \
      2>&1 | tee -a "$OUT/elo_probe.log" || true
  fi
done

echo "=== Phase 2 soft FT done ==="
