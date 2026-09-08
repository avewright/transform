#!/usr/bin/env bash
# Low-LR NorMuon continuation from step 1500 on original SF19 + expand harvest.
# Does not mutate harvest READY flags (attaches cache files, not shard dirs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source .env
set +a

export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PY="${PY:-$(command -v python3)}"
OUT="${OUT:-$ROOT/outputs/sf19_ft/run2}"
INBOX="${INBOX:-$ROOT/outputs/sf19_soft/expand1/inbox}"
BASE="${BASE:-$ROOT/outputs/sf19_ft/soft_cache.pt}"
SRC="${SRC:-$ROOT/outputs/sf19_ft/run1/step_001500.pt}"
INIT="$OUT/init_from_1500.pt"
LATEST="$OUT/latest.pt"
MANIFEST="$OUT/dataset_manifest.json"
LATE="$OUT/late_inbox"
HF_MIX="${HF_MIX:-$ROOT/outputs/hf_elo_mix/soft_cache.pt}"
HF_BONUS="$OUT/hf_replay.pt"
DEEP="${DEEP:-$ROOT/outputs/hf_elo_mix/deep_cache.pt}"
mkdir -p "$OUT" "$LATE"

if [[ -f "$LATEST" ]]; then
  RESUME="$LATEST"
  echo "resume full $LATEST"
else
  RESUME="$INIT"
  if [[ ! -f "$INIT" ]]; then
    echo "stripping $SRC -> $INIT"
    "$PY" - <<PY
import torch
from pathlib import Path
src = Path("$SRC")
dst = Path("$INIT")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
torch.save({
    "model_state_dict": state,
    "config": ckpt.get("config"),
    "steps": 0,
    "global_step": 0,
    "source": str(src),
    "note": "weights-only from step 1500; fresh NorMuon",
}, dst)
print(f"wrote {dst} keys={len(state)}")
PY
  fi
fi

if [[ -f "$HF_MIX" ]]; then
  cp -f "$HF_MIX" "$HF_BONUS"
fi
if [[ ! -f "$HF_BONUS" ]]; then
  echo "missing Lichess mix: $HF_MIX" >&2
  exit 1
fi
if [[ ! -f "$DEEP" ]]; then
  echo "missing syzygy deep cache: $DEEP" >&2
  exit 1
fi
echo "mix SF19=75% lichess=20% syzygy=5%"
echo "  bonus $HF_BONUS"
echo "  deep  $DEEP"

echo "manifest of READY expand caches"
"$PY" - <<PY
import json
from pathlib import Path
inbox = Path("$INBOX")
shards = []
for sh in sorted(inbox.glob("shard_*")):
    cache = sh / "soft_cache.pt"
    if cache.exists() and (sh / "READY").exists():
        shards.append(str(cache.resolve()))
man = {
    "live": str(Path("$BASE").resolve()),
    "shards": shards,
    "notes": "SF19 expand READY caches attached as files so harvest READY stays intact",
}
Path("$MANIFEST").write_text(json.dumps(man, indent=2) + "\n")
print(f"attach {len(shards)} caches")
PY

exec "$PY" -u experiments/exp201_recurrent_64.py --go \
  --skip-mix \
  --soft-cache "$BASE" \
  --deep-cache "$DEEP" \
  --deep-mix-frac 0.05 \
  --resume "$RESUME" \
  --output-dir "$OUT" \
  --attach-from-manifest "$MANIFEST" \
  --bonus-cache "$HF_BONUS" \
  --bonus-mix-frac 0.20 \
  --bonus-exclude "$BASE" \
  --optimizer polar_normuon \
  --torch-compile --compile-polar \
  --force-lr \
  --muon-lr 0.002 \
  --adam-lr 3e-5 \
  --warmup 100 \
  --batch-size 64 \
  --max-steps 4000 \
  --val-every 250 \
  --val-eval-n 2000 \
  --save-every 500 \
  --elo-every 0 \
  --train-minutes 1440
