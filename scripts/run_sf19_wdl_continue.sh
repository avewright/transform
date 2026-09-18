#!/usr/bin/env bash
# Continue official SF19 WDL harvest and append to avewright/local-wdl.
# Does not overwrite shards already on HF. CPU-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${SF19_WDL_OUT:-$ROOT/outputs/sf19_wdl/eco}"
TARGET="${SF19_WDL_TARGET:-2000000}"
# Stay off the trainer's CPU. 16 workers halved Value99 pos/s.
WORKERS="${SF19_WDL_WORKERS:-4}"
NODES="${SF19_WDL_NODES:-25000}"
EVERY="${SF19_WDL_PUSH_EVERY:-50000}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT" "$ROOT/logs"

if [[ ! -x "$STOCKFISH_PATH" ]]; then
  echo "Stockfish 19 missing at $STOCKFISH_PATH" >&2
  exit 1
fi

if [[ ! -f "$OUT/hf_upload.json" ]]; then
  echo "$(date '+%Y-%m-%dT%H:%M:%S') seed HF keys" | tee -a "$OUT/controller.log"
  "$PY" -u scripts/sf19_wdl_dataset.py seed-hf --out-dir "$OUT" --repo avewright/local-wdl
fi

echo "$(date '+%Y-%m-%dT%H:%M:%S') harvest target=$TARGET workers=$WORKERS nodes=$NODES out=$OUT" | tee -a "$OUT/controller.log"
nohup "$PY" -u scripts/sf19_wdl_dataset.py generate --go \
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
  >> "$OUT/harvest.stdout.log" 2>&1 &
echo $! > "$OUT/harvest.pid"

nohup "$PY" -u scripts/sf19_wdl_dataset.py watch-push \
  --out-dir "$OUT" \
  --repo avewright/local-wdl \
  --every "$EVERY" \
  --poll 120 \
  >> "$OUT/push_watch.stdout.log" 2>&1 &
echo $! > "$OUT/push.pid"

echo "harvest pid=$(cat "$OUT/harvest.pid") push pid=$(cat "$OUT/push.pid")"
