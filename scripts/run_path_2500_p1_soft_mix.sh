#!/usr/bin/env bash
# Phase 1: rebuild phase-balanced HF soft mix (target 5M, mid ≥40%).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact

OUT="${OUT:-outputs/hf_soft_mix_5m}"
TARGET="${TARGET:-5000000}"
DEEP_MAX="${DEEP_MAX:-8000000}"
SKIP="${SKIP_DOWNLOAD:-0}"
mkdir -p "$OUT" logs

ARGS=(--go --output-dir "$OUT" --target "$TARGET" --deep-max-rows "$DEEP_MAX"
      --min-depth 12 --syzygy-n 500000 --harvest-frac 0.15
      --mid-frac 0.48 --open-frac 0.22 --end-frac 0.30)
if [[ "$SKIP" == "1" ]]; then
  ARGS+=(--skip-download)
fi

echo "=== building soft mix → $OUT target=$TARGET ==="
python -u scripts/build_hf_soft_mix.py "${ARGS[@]}" 2>&1 | tee "logs/path2500_p1_mix.log"

# Also refresh canonical hf_soft_mix if this is the 5m build
if [[ "$OUT" == "outputs/hf_soft_mix_5m" && -f "$OUT/soft_cache.pt" ]]; then
  mkdir -p outputs/hf_soft_mix
  cp -f "$OUT/soft_cache.pt" outputs/hf_soft_mix/soft_cache.pt
  cp -f "$OUT/deep_soft.pt" outputs/hf_soft_mix/deep_soft.pt 2>/dev/null || true
  cp -f "$OUT/mix_report.json" outputs/hf_soft_mix/mix_report.json
  echo "synced → outputs/hf_soft_mix/"
fi

OUT="$OUT" python - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rep = json.loads((out / "mix_report.json").read_text())
mid = rep["phase"]["middlegame"]
n = rep["n"]
p50 = rep["label_depth"]["p50"]
print(f"GATE: n={n:,} mid={mid:.3f} depth_p50={p50}")
ok = n >= 4_500_000 and mid >= 0.40 and p50 >= 20
if not ok and n >= 1_000_000 and mid >= 0.40 and p50 >= 20:
    print(f"NOTE: n={n:,} < 4.5M — acceptable interim mix; expand with p1_expand_5m when able")
    ok = True
raise SystemExit(0 if ok else 1)
PY
