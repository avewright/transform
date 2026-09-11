#!/usr/bin/env bash
# Gold-standard SF19 MultiPV harvest from Lichess ECO openings toward 1M unique positions.
# Seeds: data/lichess_openings/{a-e}.tsv (leaves + book prefixes), then ε / wild play deep.
# Writes READY shards under outputs/sf19_soft/eco_1m/inbox/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${SF19_ECO_OUT:-$ROOT/outputs/sf19_soft/eco_1m}"
TARGET="${SF19_ECO_TARGET:-1000000}"
WORKERS="${SF19_ECO_WORKERS:-8}"
NODES="${SF19_ECO_NODES:-100000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') eco harvest start target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

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
  --holdout-frac 0.05 \
  --seed 19010 \
  --no-seed-caches \
  ${EXCLUDE[@]+--exclude-caches "${EXCLUDE[@]}"} \
  >> "$OUT/harvest.stdout.log" 2>&1
