#!/usr/bin/env bash
# One 8k-step arm: refreshed verified-correction mix from the gauntlet winner.
# Reuses the existing Astra 8k run when the incumbent won (same init/settings).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

OUT="${OUT:-$ROOT/outputs/searchless_train}"
MIX="${MIX:-$ROOT/outputs/astra_mix_v2}"
OLD_MIX="${OLD_MIX:-$ROOT/outputs/astra_mix}"
OVERNIGHT="${OVERNIGHT:-$ROOT/outputs/sf19_ft/overnight_20260908}"
GAUNTLET="${GAUNTLET:-$ROOT/outputs/searchless_gauntlet/report.json}"
STEPS="${STEPS:-8000}"
MINUTES="${MINUTES:-180}"
PY="${PY:-python3}"
mkdir -p "$OUT"
LOG="$OUT/train_ctrl.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ ! -f "$MIX/mix_report.json" ]]; then
  echo "refreshed mix not ready: $MIX/mix_report.json" >&2
  exit 1
fi
if [[ ! -f "$GAUNTLET" ]]; then
  echo "gauntlet report missing: $GAUNTLET" >&2
  exit 1
fi

WINNER="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$GAUNTLET").read_text())
print(d["decision"]["winner"])
PY
)"
REASON="$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$GAUNTLET").read_text())
print(d["decision"]["reason"])
PY
)"

case "$WINNER" in
  incumbent) CKPT="$OVERNIGHT/eval_swa.pt" ;;
  control) CKPT="$ROOT/outputs/astra_compare/control/eval_swa.pt" ;;
  astra) CKPT="$ROOT/outputs/astra_compare/astra/eval_swa.pt" ;;
  *) echo "unknown winner $WINNER" >&2; exit 1 ;;
esac

log "winner=$WINNER ckpt=$CKPT"
log "reason=$REASON"
echo "$WINNER" > "$OUT/start_from.txt"
printf '%s\n' "$REASON" > "$OUT/start_reason.txt"

SHARED_EVAL=(
  --external-eval "soft=$OVERNIGHT/eval_soft.pt"
  --external-eval "replay=$OVERNIGHT/eval_replay.pt"
  --external-eval "deep=$OVERNIGHT/eval_deep.pt"
  --block-manifest "$OVERNIGHT/val_manifest_soft.json"
  --block-manifest "$OVERNIGHT/val_manifest_deep.json"
)
if [[ -f "$OVERNIGHT/val_manifest_replay.json" ]]; then
  SHARED_EVAL+=(--block-manifest "$OVERNIGHT/val_manifest_replay.json")
fi

log "train refreshed mix 8k from $WINNER"
"$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --resume "$CKPT" --output-dir "$OUT/refreshed" \
  --soft-cache "$MIX/soft_cache.pt" \
  --deep-cache "$MIX/deep_cache.pt" \
  --bonus-cache "$MIX/bonus_cache.pt" \
  --deep-mix-frac 0.05 --bonus-mix-frac 0.15 \
  --optimizer polar_normuon --torch-compile --compile-polar --force-lr \
  --muon-lr 0.0007 --adam-lr 0.00001 --warmup 100 --batch-size 64 \
  --max-steps "$STEPS" --train-minutes "$MINUTES" \
  --val-every 500 --val-eval-n 2000 --save-every 500 --elo-every 0 \
  "${SHARED_EVAL[@]}"

NEW="$OUT/refreshed/eval_swa.pt"
if [[ ! -f "$NEW" ]]; then
  NEW="$OUT/refreshed/latest.pt"
fi
BASE="$CKPT"
if [[ "$WINNER" == "incumbent" && -f "$ROOT/outputs/astra_compare/astra/eval_swa.pt" ]]; then
  OLD="$ROOT/outputs/astra_compare/astra/eval_swa.pt"
  log "reuse existing astra 8k as old-mix arm (same init/settings)"
else
  OLD=""
fi

log "shared holdouts"
HOLD_CKPTS=(--ckpt "start=$BASE" --ckpt "refreshed=$NEW")
if [[ -n "$OLD" ]]; then
  HOLD_CKPTS+=(--ckpt "oldmix=$OLD")
fi
"$PY" -u scripts/eval_shared_holdouts.py \
  "${HOLD_CKPTS[@]}" \
  --eval "soft=$OVERNIGHT/eval_soft.pt" \
  --eval "replay=$OVERNIGHT/eval_replay.pt" \
  --eval "deep=$OVERNIGHT/eval_deep.pt" \
  --val-n 2000 --write "$OUT/holdout.json"

run_elo() {
  local ckpt="$1" name="$2"
  local dest="$ROOT/outputs/elo_eval_searchless_${name}_n8000_r4.json"
  if [[ -f "$dest" ]]; then
    log "skip elo $name"
    return
  fi
  log "elo $name"
  "$PY" -u -m harness.elo --ckpt "$ckpt" --out-prefix "searchless_${name}_n8000_r4" \
    --mode policy --no-book --no-syzygy --nodes 8000 \
    --games-per-opening-per-color 4 --no-stop-after-bracket \
    --elos 2050 2200
}

log "2050/2200 gauntlet on refreshed SWA"
run_elo "$NEW" refreshed
"$PY" -u scripts/report_searchless_gauntlet.py \
  --out-dir "$ROOT/outputs/searchless_gauntlet" \
  --write "$OUT/gauntlet_after.json" || true

log "=== train stage done ==="
