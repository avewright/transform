#!/usr/bin/env bash
# Canonical exp197 entry: reuse HF mix, train wider then meta (or whichever missing).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish)}"

MIX_OUT="${MIX_OUT:-outputs/hf_soft_mix}"
SOFT="${SOFT_CACHE:-$MIX_OUT/soft_cache.pt}"
DEEP="${DEEP_CACHE:-$MIX_OUT/deep_soft.pt}"
TRAIN_MINUTES="${TRAIN_MINUTES:-240}"
MAX_STEPS="${MAX_STEPS:-16000}"
mkdir -p logs

if [[ ! -f "$SOFT" ]]; then
  echo "=== building HF soft mix ==="
  python -u scripts/build_hf_soft_mix.py --go --target 1500000 \
    2>&1 | tee logs/build_hf_soft_mix.log
fi

echo "=== mix ready: $SOFT ==="
ls -lah "$SOFT" "$DEEP" 2>/dev/null || true

run_trial() {
  local trial="$1"
  local out="outputs/exp197_${trial}"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" ]]; then
    echo "skip $trial (summary exists)"
    return 0
  fi
  if pgrep -f "exp197_hf_soft_elo.py --go --trial ${trial}" >/dev/null 2>&1; then
    echo "already running $trial — wait"
    while pgrep -f "exp197_hf_soft_elo.py --go --trial ${trial}" >/dev/null 2>&1; do sleep 60; done
    return 0
  fi
  echo "=== train $trial → $out ==="
  python -u experiments/exp197_hf_soft_elo.py --go \
    --trial "$trial" \
    --soft-cache "$SOFT" \
    --deep-cache "$DEEP" \
    --max-steps "$MAX_STEPS" \
    --train-minutes "$TRAIN_MINUTES" \
    --output-dir "$out" \
    2>&1 | tee "$out/run.log"
}

run_trial wider_shallower
run_trial meta_shaw_elo

echo "=== exp197 complete ==="
for t in wider_shallower meta_shaw_elo; do
  s="outputs/exp197_${t}/summary.json"
  [[ -f "$s" ]] && python -c "import json; d=json.load(open('$s')); print(d.get('trial'), 'elo_est=', d.get('elo_estimate'), 'steps=', d.get('train',{}).get('steps'))"
done
