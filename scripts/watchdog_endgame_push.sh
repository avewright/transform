#!/usr/bin/env bash
# Push READY endgame shards to HuggingFace as they accumulate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"
OUT="${ENDGAME_OUT:-$ROOT/outputs/endgame_dataset}"
REPO="${ENDGAME_HF_REPO:-avewright/endgame-dataset}"
exec python3 -u scripts/sf19_soft_dataset.py watch-push \
  --out-dir "$OUT" \
  --repo "$REPO" \
  --every "${ENDGAME_PUSH_EVERY:-25000}" \
  --poll 45
