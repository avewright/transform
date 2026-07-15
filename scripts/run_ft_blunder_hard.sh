#!/usr/bin/env bash
# Gentle hard-CE corrective FT from FT3h on blunder-hard pack, then Elo gate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
[[ -f .env ]] && source .env
set +a
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export STOCKFISH_PATH="${STOCKFISH_PATH:-$ROOT/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

MIX=outputs/policy_blunder_hard/soft_cache.pt
OUT=outputs/exp191_soft_ft3h_blunder_hard_v2
INIT=outputs/exp191_soft_ft3h_edge_end/best.pt
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
LOG=outputs/overnight_maxelo/ft_blunder_hard.log
mkdir -p outputs/overnight_maxelo "$OUT"

if [[ ! -f "$MIX" ]]; then
  echo "missing $MIX — run filter_blunder_hard_pack.py first" >&2
  exit 1
fi

# ~1 deep epoch at deep_mix=0.2 soft_frac=0.9 batch=256 ≈ deep_n/(256*0.18) steps,
# then a little shallow/hard polish — never reshuffle deep (--deep-max-epochs 1).
echo "[$(date -Is)] gentle hard FT (deep≤1 epoch) init=$INIT deep=$MIX → $OUT" | tee "$LOG"
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$MIX" \
  --deep-mix-frac 0.20 \
  --deep-max-epochs 1 \
  --soft-frac 0.90 \
  --soft-alpha 0.0 \
  --batch-size 256 \
  --accum-steps 1 \
  --steps 200 \
  --warmup 40 \
  --muon-lr 0.0007 \
  --adam-lr 3e-5 \
  --value-weight 0.10 \
  --grad-clip 0.5 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --hard-cache "$HARD" \
  --hard-n 1000000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 100 \
  --eval-interval 100 \
  --eval-n 1024 \
  --select-metric soft_loss \
  --compile \
  --polar \
  --cautious-wd \
  2>&1 | tee "$OUT/run.log" | tee -a "$LOG"

CKPT="$OUT/best.pt"
[[ -f "$CKPT" ]] || CKPT="$OUT/last.pt"
echo "[$(date -Is)] Elo gauntlet ckpt=$CKPT" | tee -a "$LOG"
python -u elo_eval_latest.py "$CKPT" exp191_soft_ft3h_blunder_hard_v2 \
  --movetime 0.05 \
  --games-per-opening-per-color 1 \
  --stop-after-bracket \
  --elos 1750 1900 2050 \
  2>&1 | tee "$OUT/elo_gauntlet.log" | tee -a "$LOG"

python -u - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

new_p = Path("outputs/elo_eval_exp191_soft_ft3h_blunder_hard_v2.json")
base_p = Path("outputs/elo_eval_exp191_soft_ft3h_edge_end.json")
out = Path("outputs/exp191_soft_ft3h_blunder_hard_v2/elo_gate.json")

def scores(path: Path) -> dict:
    d = json.loads(path.read_text())
    by = {s["sf_elo"]: s["score"] for s in d.get("summaries", [])}
    return {"estimate": d.get("estimate"), "by_elo": by}

new = scores(new_p)
base = scores(base_p)
b1900 = base["by_elo"].get(1900)
n1900 = new["by_elo"].get(1900)
b1750 = base["by_elo"].get(1750)
n1750 = new["by_elo"].get(1750)
keep = (
    n1900 is not None
    and b1900 is not None
    and n1900 >= 0.55
    and n1900 >= (b1900 - 0.05)
    and (n1750 is None or b1750 is None or n1750 >= b1750 - 0.10)
)
gate = {
    "keep": keep,
    "baseline": base,
    "candidate": new,
    "rules": {
        "sf1900_min": 0.55,
        "sf1900_vs_baseline_slack": 0.05,
        "sf1750_vs_baseline_slack": 0.10,
    },
    "play_checkpoint": (
        "outputs/exp191_soft_ft3h_blunder_hard_v2/best.pt"
        if keep
        else "outputs/exp191_soft_ft3h_edge_end/best.pt"
    ),
}
out.write_text(json.dumps(gate, indent=2))
print(json.dumps(gate, indent=2))
print("GATE_KEEP=" + str(keep))
print("PLAY=" + gate["play_checkpoint"])
PY

echo "[$(date -Is)] done" | tee -a "$LOG"
