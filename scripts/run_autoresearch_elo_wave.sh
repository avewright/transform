#!/bin/bash
# Elo-max wave: arch + data + train efficiency/speed. Skips completed trials.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-outputs/autoresearch_8gb}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish)}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# Prefer largest available soft cache
if [[ -z "${SOFT_CACHE:-}" ]]; then
  for c in "$OUT/soft_cache_200k.pt" "$OUT/soft_cache_50k.pt" "$OUT/soft_cache_40k.pt"; do
    if [[ -f "$c" ]]; then SOFT="$c"; break; fi
  done
else
  SOFT="$SOFT_CACHE"
fi
TRAIN_MINUTES="${TRAIN_MINUTES:-60}"
MAX_STEPS="${MAX_STEPS:-5000}"

ONLY=(
  # speed / efficiency first (fast Elo signal)
  speed_micro_hot eff_mps_fat micro_qk_swa_soft wider_shallower
  # data mixes
  data_soft_peak soft_heavy_mix elo_safe_mix cf_soft_temp soft_temp_t2
  # architecture
  meta_shaw_elo qk_norm_zero_init gab fused_encoder arch_deep_thin
  stack_ultimate sota_stack_v1 infer_speed_recipe
  # neural search (slower)
  meta_latent_search
  # optimizers
  polar_normuon adamw_only muon_hot cf_swa
)

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== elo_wave start $TS soft=$SOFT minutes=$TRAIN_MINUTES steps=$MAX_STEPS ===" | tee -a "$OUT/run_mac.log"
python -u experiments/exp194_autoresearch_8gb.py --go \
  --soft-cache "$SOFT" \
  --train-minutes "$TRAIN_MINUTES" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUT" \
  --shuffle \
  --only "${ONLY[@]}" \
  2>&1 | tee -a "$OUT/run_mac.log"
echo "=== elo_wave end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$OUT/run_mac.log"
