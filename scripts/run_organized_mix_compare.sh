#!/usr/bin/env bash
# Equal-compute screen: frozen organized_chess_v1 vs the 15% correction swap.
# Same overnight SWA, same architecture, same hparams. Only soft_cache differs.
# Corrections are inside the variant soft_cache; bonus_mix_frac stays 0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

BASE="${BASE:-$ROOT/outputs/organized_chess_v1}"
VAR="${VAR:-$ROOT/outputs/organized_chess_v1_corr15}"
SWA="${SWA:-$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt}"
OUT="${OUT:-$ROOT/outputs/organized_mix_compare}"
STEPS="${STEPS:-8000}"
MINUTES="${MINUTES:-180}"
PY="${PY:-$ROOT/.venv/bin/python}"

if [[ ! -f "$BASE/FROZEN.json" ]]; then
  echo "baseline is not frozen: $BASE/FROZEN.json" >&2
  exit 1
fi
if [[ ! -f "$VAR/manifest.json" ]]; then
  echo "variant mix missing: $VAR/manifest.json" >&2
  exit 1
fi
if [[ ! -f "$SWA" ]]; then
  echo "missing incumbent SWA: $SWA" >&2
  exit 1
fi

DEVICE="${DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
  DEVICE="$("$PY" - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
fi

COMPILE=()
BATCH="${BATCH:-64}"
if [[ "$DEVICE" == "cuda" ]]; then
  COMPILE=(--torch-compile --compile-polar)
elif [[ "$DEVICE" == "mps" ]]; then
  BATCH="${BATCH:-32}"
  COMPILE=(--no-torch-compile --no-compile-polar)
else
  echo "no CUDA/MPS; 100M compare on CPU is not useful" >&2
  exit 1
fi

SHARED_EVAL=(
  --external-eval "sf19=$BASE/sf19_eval.pt"
  --external-eval "lichess=$BASE/lichess_eval.pt"
  --external-eval "puzzles=$BASE/puzzles_eval.pt"
  --external-eval "syzygy=$BASE/syzygy_eval.pt"
  --block-manifest "$BASE/blocked_manifest.json"
)

mkdir -p "$OUT/control" "$OUT/corr15" "$OUT"
echo "device=$DEVICE batch=$BATCH steps=$STEPS from $SWA"

echo "control: frozen v1 45/35/15/5"
"$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --device "$DEVICE" \
  --resume "$SWA" --output-dir "$OUT/control" \
  --soft-cache "$BASE/soft_cache.pt" \
  --deep-cache "$BASE/deep_cache.pt" \
  --deep-mix-frac 0.05 --bonus-mix-frac 0 \
  --optimizer polar_normuon --force-lr \
  --muon-lr 0.0007 --adam-lr 0.00001 --warmup 100 --batch-size "$BATCH" \
  --max-steps "$STEPS" --train-minutes "$MINUTES" \
  --val-every 500 --val-eval-n 1000 --save-every 500 --elo-every 0 \
  "${COMPILE[@]}" \
  "${SHARED_EVAL[@]}"

echo "corr15: 30/15/35/15/5 (SF19-family still 45%)"
"$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --device "$DEVICE" \
  --resume "$SWA" --output-dir "$OUT/corr15" \
  --soft-cache "$VAR/soft_cache.pt" \
  --deep-cache "$VAR/deep_cache.pt" \
  --deep-mix-frac 0.05 --bonus-mix-frac 0 \
  --optimizer polar_normuon --force-lr \
  --muon-lr 0.0007 --adam-lr 0.00001 --warmup 100 --batch-size "$BATCH" \
  --max-steps "$STEPS" --train-minutes "$MINUTES" \
  --val-every 500 --val-eval-n 1000 --save-every 500 --elo-every 0 \
  "${COMPILE[@]}" \
  "${SHARED_EVAL[@]}"

CTRL="$OUT/control/eval_swa.pt"
CORR="$OUT/corr15/eval_swa.pt"
[[ -f "$CTRL" ]] || CTRL="$OUT/control/latest.pt"
[[ -f "$CORR" ]] || CORR="$OUT/corr15/latest.pt"

"$PY" -u scripts/eval_shared_holdouts.py \
  --ckpt "incumbent=$SWA" \
  --ckpt "control=$CTRL" \
  --ckpt "corr15=$CORR" \
  --eval "sf19=$BASE/sf19_eval.pt" \
  --eval "lichess=$BASE/lichess_eval.pt" \
  --eval "puzzles=$BASE/puzzles_eval.pt" \
  --eval "syzygy=$BASE/syzygy_eval.pt" \
  --val-n 1000 \
  --write "$OUT/holdout_compare.json"

echo "wrote $OUT/holdout_compare.json"
