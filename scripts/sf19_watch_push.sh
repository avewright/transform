#!/bin/bash
# Append new expand shards onto avewright/chess-soft-sf19 without touching shard_000000.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
OUT="${OUT:-outputs/sf19_soft/expand1}"
OFFSET="${OFFSET:-51}"
BASE_ROWS="${BASE_ROWS:-1010000}"
INTERVAL="${INTERVAL:-90}"

while true; do
  python -u scripts/sf19_soft_dataset.py push \
    --out-dir "$OUT" \
    --repo avewright/chess-soft-sf19 \
    --shard-offset "$OFFSET" \
    --base-rows "$BASE_ROWS" || true
  sleep "$INTERVAL"
done
