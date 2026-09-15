#!/usr/bin/env bash
# Fresh Stockfish 19 harvest: 1–8 MultiPV soft targets + WDL/cp values,
# with n_pieces tracked on every row and in stats.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${SF19_HARVEST_OUT:-$ROOT/outputs/sf19_soft/harvest}"
TARGET="${SF19_HARVEST_TARGET:-300000}"
WORKERS="${SF19_HARVEST_WORKERS:-8}"
NODES="${SF19_HARVEST_NODES:-100000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') sf19 harvest start target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

exec "$PY" -u scripts/sf19_soft_dataset.py generate --go \
  --mode piece_curve \
  --no-ingest \
  --out-dir "$OUT" \
  --target "$TARGET" \
  --workers "$WORKERS" \
  --nodes "$NODES" \
  --play-nodes 4000 \
  --multipv 8 \
  --tau 120 \
  --ply-stride 2 \
  --ply-skip-open 0 \
  --ply-cap 220 \
  --book-noise 2 \
  --watchdog-s 20 \
  --hash-mb 64 \
  --shard-size 5000 \
  --holdout-frac 0.05 \
  --seed 19019 \
  --no-seed-caches \
  >> "$OUT/harvest.stdout.log" 2>&1
