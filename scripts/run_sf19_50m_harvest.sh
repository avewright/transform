#!/usr/bin/env bash
# Long-running SF19 MultiPV harvest toward 50M hashed positions.
# Writes READY shards under outputs/sf19_soft/expand50m/inbox/.
# Does NOT attach them to the exp270 first-night mix (no live ingest).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"

OUT="${SF19_HARVEST_OUT:-$ROOT/outputs/sf19_soft/expand50m}"
TARGET="${SF19_HARVEST_TARGET:-50000000}"
WORKERS="${SF19_HARVEST_WORKERS:-6}"
NODES="${SF19_HARVEST_NODES:-100000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') harvest start target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

EXCLUDE=()
for p in \
  "$ROOT/outputs/hf_elo_mix/soft_cache.pt" \
  "$ROOT/outputs/hf_elo_mix/deep_cache.pt" \
  "$ROOT/outputs/exp270_mix_v1/soft_cache.pt" \
  "$ROOT/outputs/exp270_mix_v1/deep_cache.pt" \
  "$ROOT/outputs/organized_chess_v1/soft_cache.pt"
do
  [[ -f "$p" ]] && EXCLUDE+=("$p")
done

exec "$PY" -u scripts/sf19_soft_dataset.py generate --go \
  --mode mix \
  --out-dir "$OUT" \
  --target "$TARGET" \
  --workers "$WORKERS" \
  --nodes "$NODES" \
  --play-nodes 4000 \
  --multipv 8 \
  --tau 120 \
  --epsilon 0.25 \
  --ply-stride 3 \
  --ply-skip-open 4 \
  --ply-cap 140 \
  --book-noise 4 \
  --hash-mb 64 \
  --shard-size 5000 \
  --holdout-frac 0.01 \
  --seed 27050 \
  --seed-fens-n 4096 \
  ${EXCLUDE[@]+--exclude-caches "${EXCLUDE[@]}"} \
  >> "$OUT/harvest.stdout.log" 2>&1
