#!/usr/bin/env bash
# Push READY ECO shards to HF each time another 50k rows land.
# Stop with: touch outputs/sf19_soft/eco_1m/HALT_PUSH
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
OUT="${SF19_ECO_OUT:-$ROOT/outputs/sf19_soft/eco_1m}"
REPO="${SF19_ECO_REPO:-avewright/stockfish-19-soft-targets}"
EVERY="${SF19_ECO_PUSH_EVERY:-50000}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') push watch repo=$REPO every=$EVERY out=$OUT" | tee -a "$OUT/push_watch.log"
exec "$PY" -u scripts/sf19_soft_dataset.py watch-push \
  --out-dir "$OUT" \
  --repo "$REPO" \
  --every "$EVERY" \
  --poll 30 \
  >> "$OUT/push_watch.stdout.log" 2>&1
