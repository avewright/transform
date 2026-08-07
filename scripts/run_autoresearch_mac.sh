#!/bin/bash
# Apple Silicon (MPS) Elo autoresearch — small-model architecture/data/train search.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-outputs/autoresearch_8gb}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish)}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

# Prefer local venv if present
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

SOFT="${SOFT_CACHE:-$OUT/soft_cache_200k.pt}"
if [[ ! -f "$SOFT" ]]; then
  echo "Missing soft cache: $SOFT"
  echo "Harvest first:"
  echo "  python scripts/harvest_local_multipv.py --go --target 40000 --workers 12"
  exit 1
fi

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--smoke)
fi
if [[ -n "${MAX_TRIALS:-}" ]]; then
  EXTRA+=(--max-trials "$MAX_TRIALS")
fi
if [[ -n "${ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(--only $ONLY)
fi

TRAIN_MINUTES="${TRAIN_MINUTES:-45}"
MAX_STEPS="${MAX_STEPS:-4000}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== autoresearch_mac start $TS device=mps soft=$SOFT ===" | tee -a "$OUT/run_mac.log"
python -u experiments/exp194_autoresearch_8gb.py --go "${EXTRA[@]}" \
  --soft-cache "$SOFT" \
  --train-minutes "$TRAIN_MINUTES" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUT" \
  --shuffle \
  2>&1 | tee -a "$OUT/run_mac.log"
echo "=== autoresearch_mac end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$OUT/run_mac.log"
