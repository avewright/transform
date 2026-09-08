#!/usr/bin/env bash
# 128-game SF2050/2200 searchless screen. Same openings/colors as the ~2150 reference.
# Does not promote. Overnight SWA stays incumbent until the report says otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

OUT="${OUT:-$ROOT/outputs/searchless_gauntlet}"
PY="${PY:-python3}"
NODES="${NODES:-8000}"
REPEATS="${REPEATS:-4}"
mkdir -p "$OUT"
LOG="$OUT/gauntlet.log"

INCUMBENT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"
CONTROL="$ROOT/outputs/astra_compare/control/eval_swa.pt"
ASTRA="$ROOT/outputs/astra_compare/astra/eval_swa.pt"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_elo() {
  local ckpt="$1" name="$2"
  local prefix="searchless_${name}_n${NODES}_r${REPEATS}"
  local dest="$ROOT/outputs/elo_eval_${prefix}.json"
  if [[ -f "$dest" ]]; then
    log "skip $name (exists $dest)"
    return
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" >&2
    exit 1
  fi
  log "begin $name ckpt=$ckpt games=128 nodes=$NODES elos=2050,2200"
  "$PY" -u -m harness.elo \
    --ckpt "$ckpt" \
    --out-prefix "$prefix" \
    --mode policy \
    --no-book --no-syzygy \
    --nodes "$NODES" \
    --games-per-opening-per-color "$REPEATS" \
    --no-stop-after-bracket \
    --elos 2050 2200
  log "done $name"
}

log "=== searchless gauntlet start ==="
run_elo "$INCUMBENT" incumbent
run_elo "$CONTROL" control
run_elo "$ASTRA" astra
log "=== report ==="
"$PY" -u scripts/report_searchless_gauntlet.py \
  --out-dir "$OUT" \
  --write "$OUT/report.json"
log "=== gauntlet finished ==="
cat "$OUT/report.json"
