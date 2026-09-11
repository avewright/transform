#!/usr/bin/env bash
# Full-game SF19 harvest shaped to N(17, 6) over piece count.
# Starts from Lichess ECO, plays to the end, keeps a row only if that
# n_pieces is still under the bell. Mid/end streams fast-forward so the
# left side of the curve is fillable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${SF19_CURVE_OUT:-$ROOT/outputs/sf19_soft/piece_curve}"
TARGET="${SF19_CURVE_TARGET:-300000}"
WORKERS="${SF19_CURVE_WORKERS:-8}"
NODES="${SF19_CURVE_NODES:-100000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') piece_curve start target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

# Sample from local SF19 / mix / master first. Do not exclude those pools.
exec "$PY" -u scripts/sf19_soft_dataset.py generate --go \
  --mode piece_curve \
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
  --seed 19017 \
  --no-seed-caches \
  >> "$OUT/harvest.stdout.log" 2>&1
