#!/bin/bash
# Stockfish 19 soft-target dataset: bench, generate, verify, push.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

CMD="${1:-generate}"
OUT="${OUT:-outputs/sf19_soft/prod}"
shift || true

case "$CMD" in
  bench)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py bench --out-dir "$OUT" "$@"
    ;;
  audit)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py audit --out-dir "$OUT" "$@"
    ;;
  freeze-eval)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py freeze-eval --out-dir "$OUT" "$@"
    ;;
  generate)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py generate --go --out-dir "$OUT" \
      --mode mix --nodes 100000 --multipv 8 "$@"
    ;;
  compare-ft)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py compare-ft --out-dir "$OUT" "$@"
    ;;
  pilot)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py generate --go --pilot \
      --out-dir "$OUT" --target 25000 --workers 14 "$@"
    ;;
  verify)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py verify --out-dir "$OUT" "$@"
    ;;
  push)
    exec .venv/bin/python -u scripts/sf19_soft_dataset.py push --out-dir "$OUT" \
      --repo avewright/chess-soft-sf19 "$@"
    ;;
  *)
    echo "usage: $0 [bench|audit|freeze-eval|generate|compare-ft|pilot|verify|push]" >&2
    exit 1
    ;;
esac
