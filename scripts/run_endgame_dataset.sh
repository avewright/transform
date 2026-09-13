#!/usr/bin/env bash
# Parallel Stockfish 19 self-play, label only 6–12 piece boards.
# SF is CPU. This box: 96 cores / 500GB RAM → 80 workers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${ENDGAME_OUT:-$ROOT/outputs/endgame_dataset}"
TARGET="${ENDGAME_TARGET:-300000}"
WORKERS="${ENDGAME_WORKERS:-64}"
NODES="${ENDGAME_NODES:-100000}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"

echo "$(date '+%Y-%m-%dT%H:%M:%S') endgame harvest target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

exec "$PY" -u scripts/sf19_soft_dataset.py generate --go \
  --mode endgame \
  --out-dir "$OUT" \
  --target "$TARGET" \
  --workers "$WORKERS" \
  --nodes "$NODES" \
  --play-nodes 4000 \
  --multipv 8 \
  --tau 120 \
  --ply-stride 1 \
  --ply-skip-open 0 \
  --ply-cap 220 \
  --book-noise 2 \
  --watchdog-s 20 \
  --hash-mb 64 \
  --shard-size 5000 \
  --holdout-frac 0.05 \
  --seed 27412 \
  --no-seed-caches \
  2>&1 | tee -a "$OUT/harvest.stdout.log"
