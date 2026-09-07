#!/bin/bash
# 30-minute Polar-NorMuon FT on harvested disagreements, then policy Elo gauntlet.
set -euo pipefail
cd "$(dirname "$0")/.."
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-/Users/avewright/.local/bin/stockfish}"

HARVEST="${HARVEST:-outputs/hf100m_lapse_ft/soft_cache.pt}"
INIT="${INIT:-outputs/hf100m_lapse_ft/init_from_hf.pt}"
OUT="${OUT:-outputs/hf100m_lapse_ft_30m}"
MINUTES="${MINUTES:-30}"

mkdir -p "$OUT"
if [[ ! -f "$HARVEST" ]]; then
  echo "missing harvest cache: $HARVEST" >&2
  exit 1
fi
if [[ ! -f "$INIT" ]]; then
  echo "missing warm-start: $INIT" >&2
  exit 1
fi

echo "Train ${MINUTES}m Polar-NorMuon on disagreements -> $OUT"
.venv/bin/python -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --resume "$INIT" \
  --soft-cache "$HARVEST" \
  --output-dir "$OUT" \
  --max-steps 1200 \
  --train-minutes "$MINUTES" \
  --deep-mix-frac 0 \
  --batch-size 48 \
  --optimizer polar_normuon \
  --muon-lr 0.002 \
  --adam-lr 3e-5 \
  --warmup 100 \
  --val-every 0 \
  --val-eval-n 256

CKPT="$OUT/latest.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "training finished without $CKPT" >&2
  exit 1
fi

echo "Elo gauntlet (policy, no book/syzygy) -> $CKPT"
.venv/bin/python -u -m harness.elo \
  --ckpt "$CKPT" \
  --out-prefix hf100m_lapse_ft_30m \
  --mode policy \
  --movetime 0.05 \
  --games-per-opening-per-color 2 \
  --elos 1450 1600 1750 1900 2050 2200 \
  --stop-after-bracket
