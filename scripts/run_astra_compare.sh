#!/usr/bin/env bash
# Equal-compute screen: overnight mix vs Astra mix, both from the same SWA.
# Shared overnight holdouts. Paired-opening Elo after both runs.
# Do not start this while the SWA harvest still owns the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1

OUT="${OUT:-$ROOT/outputs/astra_compare}"
MIX="${MIX:-$ROOT/outputs/astra_mix}"
OVERNIGHT="${OVERNIGHT:-$ROOT/outputs/sf19_ft/overnight_20260908}"
SWA="${SWA:-$OVERNIGHT/eval_swa.pt}"
STEPS="${STEPS:-8000}"
MINUTES="${MINUTES:-180}"
ELO_REPEATS="${ELO_REPEATS:-1}"
PY="${PY:-python3}"

EVAL_SOFT="$OVERNIGHT/eval_soft.pt"
EVAL_REPLAY="$OVERNIGHT/eval_replay.pt"
EVAL_DEEP="$OVERNIGHT/eval_deep.pt"
MAN_SOFT="$OVERNIGHT/val_manifest_soft.json"
MAN_DEEP="$OVERNIGHT/val_manifest_deep.json"
MAN_REPLAY="$OVERNIGHT/val_manifest_replay.json"

if [[ ! -f "$SWA" ]]; then
  echo "missing SWA checkpoint: $SWA" >&2
  exit 1
fi
if [[ ! -f "$MIX/mix_report.json" ]]; then
  echo "mix not assembled: $MIX/mix_report.json" >&2
  exit 1
fi
if [[ ! -f "$MIX/bonus_cache.pt" ]]; then
  echo "mistakes not ready: $MIX/bonus_cache.pt" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -q harvest_swa_mistakes; then
  echo "GPU still running harvest_swa_mistakes; not starting comparison" >&2
  exit 1
fi
for p in "$EVAL_SOFT" "$EVAL_REPLAY" "$EVAL_DEEP" "$MAN_SOFT" "$MAN_DEEP"; do
  if [[ ! -f "$p" ]]; then
    echo "missing shared holdout: $p" >&2
    exit 1
  fi
done

SHARED_EVAL=(
  --external-eval "soft=$EVAL_SOFT"
  --external-eval "replay=$EVAL_REPLAY"
  --external-eval "deep=$EVAL_DEEP"
  --block-manifest "$MAN_SOFT"
  --block-manifest "$MAN_DEEP"
)
if [[ -f "$MAN_REPLAY" ]]; then
  SHARED_EVAL+=(--block-manifest "$MAN_REPLAY")
fi

mkdir -p "$OUT/control" "$OUT/astra" "$OUT"
echo "control: overnight 75/20/5 from $SWA"
"$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --resume "$SWA" --output-dir "$OUT/control" \
  --soft-cache "$OVERNIGHT/soft_cache.pt" \
  --deep-cache "$OVERNIGHT/deep_cache.pt" \
  --bonus-cache "$OVERNIGHT/replay_cache.pt" \
  --deep-mix-frac 0.05 --bonus-mix-frac 0.20 \
  --optimizer polar_normuon --torch-compile --compile-polar --force-lr \
  --muon-lr 0.0007 --adam-lr 0.00001 --warmup 100 --batch-size 64 \
  --max-steps "$STEPS" --train-minutes "$MINUTES" \
  --val-every 500 --val-eval-n 2000 --save-every 500 --elo-every 0 \
  "${SHARED_EVAL[@]}"

echo "astra: 40/30/15/10/5 from $SWA"
"$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --resume "$SWA" --output-dir "$OUT/astra" \
  --soft-cache "$MIX/soft_cache.pt" \
  --deep-cache "$MIX/deep_cache.pt" \
  --bonus-cache "$MIX/bonus_cache.pt" \
  --deep-mix-frac 0.05 --bonus-mix-frac 0.15 \
  --optimizer polar_normuon --torch-compile --compile-polar --force-lr \
  --muon-lr 0.0007 --adam-lr 0.00001 --warmup 100 --batch-size 64 \
  --max-steps "$STEPS" --train-minutes "$MINUTES" \
  --val-every 500 --val-eval-n 2000 --save-every 500 --elo-every 0 \
  "${SHARED_EVAL[@]}"

pick_ckpt() {
  local d="$1"
  if [[ -f "$d/eval_swa.pt" ]]; then
    echo "$d/eval_swa.pt"
  else
    echo "$d/latest.pt"
  fi
}

CTRL=$(pick_ckpt "$OUT/control")
ASTRA=$(pick_ckpt "$OUT/astra")

echo "shared holdout eval"
"$PY" -u scripts/eval_shared_holdouts.py \
  --ckpt "control=$CTRL" --ckpt "astra=$ASTRA" \
  --eval "soft=$EVAL_SOFT" --eval "replay=$EVAL_REPLAY" --eval "deep=$EVAL_DEEP" \
  --val-n 2000 --write "$OUT/holdout.json"

run_elo() {
  local ckpt="$1" name="$2"
  local prefix="astra_compare_${name}_n8000_r${ELO_REPEATS}"
  "$PY" -u -m harness.elo --ckpt "$ckpt" --out-prefix "$prefix" \
    --mode policy --no-book --no-syzygy --nodes 8000 \
    --games-per-opening-per-color "$ELO_REPEATS" --no-stop-after-bracket \
    --elos 1750 1900
}

echo "paired-opening Elo control"
run_elo "$CTRL" control
echo "paired-opening Elo astra"
run_elo "$ASTRA" astra

"$PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT")
hold = json.loads((out / "holdout.json").read_text()) if (out / "holdout.json").exists() else {}
report = {"holdout": hold, "ckpts": {"control": "$CTRL", "astra": "$ASTRA"}, "elo": {}}
root = Path("$ROOT") / "outputs"
for name in ("control", "astra"):
    p = root / f"elo_eval_astra_compare_{name}_n8000_r${ELO_REPEATS}.json"
    if p.exists():
        d = json.loads(p.read_text())
        games = d.get("games") or []
        report["elo"][name] = {
            "path": str(p),
            "n_games": len(games),
            "score": sum(g.get("score", 0) for g in games) / max(1, len(games)),
            "estimate": d.get("estimate"),
        }
(out / "compare_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
