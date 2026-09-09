#!/usr/bin/env bash
# Fresh 128-game held-out screen: incumbent SWA vs corr15 SWA.
# Cached 8-opening results are not reused (different openings).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export STOCKFISH_PATH="/root/.local/bin/stockfish-19"
OUT="$ROOT/outputs/heldout_v1"
PROTO="$ROOT/harness/protocol_heldout_v1.json"
INC="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"
CAND="$ROOT/outputs/overnight_corr15/train/eval_swa.pt"
PY="$ROOT/.venv/bin/python"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/gauntlet.log") 2>&1

echo "$(date -Is) === heldout_v1 start ==="
"$PY" -u scripts/verify_overnight_resume_and_provenance.py "$OUT"

run_one() {
  local ckpt="$1" prefix="$2" dest="$3"
  if [[ -f "$dest" ]]; then
    echo "reuse $dest"
    return 0
  fi
  echo "$(date -Is) elo $prefix"
  "$PY" -u -m harness.elo \
    --ckpt "$ckpt" \
    --protocol "$PROTO" \
    --out-prefix "$prefix" \
    --mode policy \
    --no-book --no-syzygy \
    --nodes 8000 \
    --games-per-opening-per-color 2 \
    --no-stop-after-bracket \
    --elos 2050 2200
  local produced="$ROOT/outputs/elo_eval_${prefix}.json"
  cp -f "$produced" "$dest"
}

run_one "$INC" "heldout_v1_incumbent_n8000_r2" "$OUT/elo_incumbent.json"
run_one "$CAND" "heldout_v1_corr15_n8000_r2" "$OUT/elo_corr15.json"

"$PY" -u - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "/root/transform")
from scripts.report_searchless_gauntlet import _summarize, paired_opening_bootstrap

out = Path("/root/transform/outputs/heldout_v1")
inc = json.loads((out / "elo_incumbent.json").read_text())
cand = json.loads((out / "elo_corr15.json").read_text())
inc_s = _summarize(inc)
cand_s = _summarize(cand)
diff = paired_opening_bootstrap(cand.get("games") or [], inc.get("games") or [])
replace = bool(diff and diff.get("a_better"))
report = {
    "screen": "heldout_v1",
    "protocol": "/root/transform/harness/protocol_heldout_v1.json",
    "games_each": 128,
    "openings": 16,
    "repeats_per_color": 2,
    "elos": [2050, 2200],
    "cached_incumbent_reused": False,
    "incumbent": inc_s,
    "candidate": cand_s,
    "paired_opening_candidate_minus_incumbent": diff,
    "recommendation": "replace_incumbent" if replace else "retain_incumbent",
    "do_not_overwrite_incumbent": True,
    "do_not_publish": True,
    "note": "corr15 is the latest 0.500 challenger. half-LR also 0.500 on the original 8 openings and was not replayed here.",
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print("WROTE", out / "report.json")
PY

echo "$(date -Is) === heldout_v1 done; no publish ==="
