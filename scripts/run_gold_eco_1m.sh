#!/usr/bin/env bash
# 1M gold SF19 soft labels from Lichess ECO openings (3,704 card / 3,810 live).
# Starts on every unique book prefix + leaf, then epsilon/wild explores deep.
# Separate from expand50m. Does not touch the 99M incumbent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"

OUT="${GOLD_OUT:-$ROOT/outputs/sf19_soft/gold_eco_1m}"
TARGET="${GOLD_TARGET:-1000000}"
WORKERS="${GOLD_WORKERS:-10}"
NODES="${GOLD_NODES:-100000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') gold-eco start target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

exec "$PY" -u scripts/sf19_soft_dataset.py generate --go \
  --mode eco \
  --out-dir "$OUT" \
  --target "$TARGET" \
  --workers "$WORKERS" \
  --nodes "$NODES" \
  --play-nodes 4000 \
  --multipv 8 \
  --tau 120 \
  --ply-stride 2 \
  --ply-skip-open 0 \
  --ply-cap 180 \
  --book-noise 2 \
  --watchdog-s 20 \
  --hash-mb 64 \
  --shard-size 5000 \
  --holdout-frac 0.02 \
  --seed 27041 \
  >> "$OUT/harvest.stdout.log" 2>&1
