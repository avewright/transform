#!/usr/bin/env bash
# Official Stockfish 19 UCI WDL harvest from Lichess ECO starts.
# Writes READY shards under outputs/sf19_wdl/eco/inbox/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${SF19_WDL_OUT:-$ROOT/outputs/sf19_wdl/eco}"
TARGET="${SF19_WDL_TARGET:-1000000}"
WORKERS="${SF19_WDL_WORKERS:-16}"
NODES="${SF19_WDL_NODES:-25000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') wdl harvest start target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"

EXCLUDE=()
for p in \
  "$ROOT/outputs/organized_chess_v1/sf19_train.pt" \
  "$ROOT/outputs/organized_chess_v1/soft_cache.pt" \
  "$ROOT/outputs/sf19_soft/eco_1m"
do
  if [[ -f "$p" ]]; then
    EXCLUDE+=("$p")
  elif [[ -d "$p" ]]; then
    while IFS= read -r cache; do
      EXCLUDE+=("$cache")
    done < <(find "$p" -name 'soft_cache.pt' -print)
  fi
done

exec "$PY" -u scripts/sf19_wdl_dataset.py generate --go \
  --mode eco \
  --out-dir "$OUT" \
  --target "$TARGET" \
  --workers "$WORKERS" \
  --nodes "$NODES" \
  --play-nodes 2000 \
  --ply-stride 2 \
  --ply-skip-open 0 \
  --ply-cap 180 \
  --watchdog-s 12 \
  --hash-mb 64 \
  --shard-size 5000 \
  --holdout-frac 0.05 \
  --seed 19019 \
  ${EXCLUDE[@]+--exclude-caches "${EXCLUDE[@]}"} \
  >> "$OUT/harvest.stdout.log" 2>&1
