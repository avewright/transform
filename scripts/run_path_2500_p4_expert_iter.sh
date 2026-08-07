#!/usr/bin/env bash
# Phase 4: KL-anchored expert-iter — ONLY after policy Elo ≥~1900–2000.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Student = soft-FT champ; prior = frozen copy of same (or base hf_437m)
CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
  for c in \
    outputs/exp191_hf437m_soft_ft/best_model.pt \
    outputs/lora_soft_hfmix/merged_model.pt \
    outputs/hf_437m/best_model.pt; do
    if [[ -f "$c" ]]; then CKPT="$c"; break; fi
  done
fi
PRIOR="${PRIOR_CHECKPOINT:-$CKPT}"
OUT="${OUT:-outputs/rl_selfplay_kl_anchored}"
mkdir -p "$OUT" logs

echo "=== Phase 4 KL expert-iter student=$CKPT prior=$PRIOR ==="
echo "NOTE: skip if policy Elo still <1900 — RL will collapse."

if [[ "${SMOKE:-0}" == "1" ]]; then
  python -u experiments/exp183_selfplay.py --preset kl --go --smoke \
    --checkpoint "$CKPT" \
    --prior-checkpoint "$PRIOR" \
    --output-dir "$OUT" \
    2>&1 | tee "$OUT/run.log"
else
  python -u experiments/exp183_selfplay.py --preset kl --go \
    --checkpoint "$CKPT" \
    --prior-checkpoint "$PRIOR" \
    --mode "${MODE:-sf}" \
    --iterations "${ITERS:-5}" \
    --output-dir "$OUT" \
    2>&1 | tee "$OUT/run.log"
fi

if [[ -f "$OUT/latest.pt" ]]; then
  python -u elo_eval_latest.py "$OUT/latest.pt" path2500_kl_policy \
    --elos 1750 1900 2050 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    2>&1 | tee -a "$OUT/elo_probe.log" || true
fi

echo "=== Phase 4 done ==="
