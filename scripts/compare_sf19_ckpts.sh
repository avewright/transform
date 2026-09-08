#!/usr/bin/env bash
# Search-free, harvest-safe checkpoint comparison.
# Same 8 protocol openings, no book, no Syzygy, fixed-node Stockfish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a
source .env
set +a

export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"
export OMP_NUM_THREADS=1

PY="${PY:-$(command -v python3)}"
OUT="${OUT:-$ROOT/outputs/sf19_ft/compare1}"
NODES="${NODES:-8000}"
mkdir -p "$OUT"
LOG="$OUT/compare.log"

INIT="$ROOT/outputs/sf19_ft/init.pt"
S1500="$ROOT/outputs/sf19_ft/run1/step_001500.pt"
SWA="$ROOT/outputs/sf19_ft/run1/eval_swa.pt"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_h2h() {
  local a="$1" b="$2" name="$3"
  local dest="$OUT/h2h_${name}.json"
  if [[ -f "$dest" ]]; then
    log "skip h2h $name (exists)"
    return
  fi
  log "h2h $name"
  "$PY" -u scripts/match_two_ckpts.py --go \
    --a "$a" --b "$b" --out "$dest" --max-openings 8
}

run_sf() {
  local ckpt="$1" prefix="$2" games="$3"
  shift 3
  local dest="$ROOT/outputs/elo_eval_${prefix}.json"
  if [[ -f "$dest" ]]; then
    log "skip sf $prefix (exists)"
    return
  fi
  log "sf $prefix games=$games nodes=$NODES elos=$*"
  "$PY" -u -m harness.elo \
    --ckpt "$ckpt" \
    --out-prefix "$prefix" \
    --mode policy \
    --no-book --no-syzygy \
    --nodes "$NODES" \
    --games-per-opening-per-color "$games" \
    --no-stop-after-bracket \
    --elos "$@"
}

log "=== screen: pairwise policy (8 openings, both colors) ==="
run_h2h "$INIT" "$S1500" "init_vs_1500"
run_h2h "$INIT" "$SWA" "init_vs_swa"
run_h2h "$S1500" "$SWA" "1500_vs_swa"

log "=== screen: fixed-node SF, 1 game/opening/color, 1750+1900 ==="
run_sf "$INIT" "sf19_screen_init_n${NODES}" 1 1750 1900
run_sf "$S1500" "sf19_screen_1500_n${NODES}" 1 1750 1900
run_sf "$SWA" "sf19_screen_swa_n${NODES}" 1 1750 1900

log "=== rank screen ==="
"$PY" -u scripts/rank_sf19_compare.py --dir "$OUT" --write "$OUT/screen_rank.json"

# Larger comparison of the top two (always include init if it is not already one of them).
mapfile -t TOP < <("$PY" -u scripts/rank_sf19_compare.py --dir "$OUT" --top 2 --names-only)
log "top two: ${TOP[*]}"

larger_prefix() {
  case "$1" in
    init) echo "sf19_large_init_n${NODES}" ;;
    step1500) echo "sf19_large_1500_n${NODES}" ;;
    swa) echo "sf19_large_swa_n${NODES}" ;;
    *) echo "sf19_large_${1}_n${NODES}" ;;
  esac
}

ckpt_for() {
  case "$1" in
    init) echo "$INIT" ;;
    step1500) echo "$S1500" ;;
    swa) echo "$SWA" ;;
    *) echo "" ;;
  esac
}

NEED=("init")
for n in "${TOP[@]}"; do
  NEED+=("$n")
done
# unique
declare -A SEEN=()
UNIQUE=()
for n in "${NEED[@]}"; do
  if [[ -z "${SEEN[$n]:-}" ]]; then
    SEEN[$n]=1
    UNIQUE+=("$n")
  fi
done

log "=== larger: 2 games/opening/color, 1600-2050, nodes=$NODES ==="
for n in "${UNIQUE[@]}"; do
  ckpt="$(ckpt_for "$n")"
  pref="$(larger_prefix "$n")"
  run_sf "$ckpt" "$pref" 2 1600 1750 1900 2050
done

"$PY" -u scripts/rank_sf19_compare.py --dir "$OUT" --write "$OUT/final_rank.json" --include-large
log "=== compare done ==="
cat "$OUT/final_rank.json"
