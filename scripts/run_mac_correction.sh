#!/usr/bin/env bash
# Resume the Mac correction loop. Safe to re-run; finished phases are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-$ROOT/outputs/mac_correction_v1}"
CKPT="${CKPT:-$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt}"
PHASE="${1:-go}"
shift || true
exec "$PY" -u scripts/mac_correction_pipeline.py --"$PHASE" \
  --ckpt "$CKPT" --out-dir "$OUT" "$@"
