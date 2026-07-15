#!/usr/bin/env bash
# Tactics hard FT from FT3h → Elo gate. Deep seen ≤1 epoch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
[[ -f .env ]] && source .env
set +a
export MOVE_VOCAB_VERSION=compact PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export STOCKFISH_PATH="${STOCKFISH_PATH:-$ROOT/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

MIX=outputs/policy_tactics_hard/soft_cache.pt
OUT=outputs/exp191_soft_ft3h_tactics_hard
INIT=outputs/exp191_soft_ft3h_edge_end/best.pt
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
LOG=outputs/overnight_maxelo/ft_tactics_hard.log
mkdir -p outputs/overnight_maxelo "$OUT"

echo "[$(date -Is)] build tactics pack…" | tee "$LOG"
python -u scripts/build_tactics_hard_pack.py \
  --min-rating 2200 --max-n 20000 \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] FT deep≤1 epoch → $OUT" | tee -a "$LOG"
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$MIX" \
  --deep-mix-frac 0.18 \
  --deep-max-epochs 1 \
  --soft-frac 0.90 \
  --soft-alpha 0.0 \
  --batch-size 256 \
  --accum-steps 1 \
  --steps 250 \
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
  --save-interval 125 \
  --eval-interval 125 \
  --eval-n 1024 \
  --select-metric soft_loss \
  --compile --polar --cautious-wd \
  2>&1 | tee "$OUT/run.log" | tee -a "$LOG"

CKPT="$OUT/best.pt"
[[ -f "$CKPT" ]] || CKPT="$OUT/last.pt"
echo "[$(date -Is)] Elo $CKPT" | tee -a "$LOG"
python -u elo_eval_latest.py "$CKPT" exp191_soft_ft3h_tactics_hard \
  --movetime 0.05 \
  --games-per-opening-per-color 1 \
  --stop-after-bracket \
  --elos 1750 1900 2050 \
  2>&1 | tee "$OUT/elo_gauntlet.log" | tee -a "$LOG"

python -u - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
new_p = Path("outputs/elo_eval_exp191_soft_ft3h_tactics_hard.json")
base_p = Path("outputs/elo_eval_exp191_soft_ft3h_edge_end.json")
out = Path("outputs/exp191_soft_ft3h_tactics_hard/elo_gate.json")

def scores(path):
    d = json.loads(path.read_text())
    return {"estimate": d.get("estimate"), "by_elo": {s["sf_elo"]: s["score"] for s in d.get("summaries", [])}}

new, base = scores(new_p), scores(base_p)
n1900, b1900 = new["by_elo"].get(1900), base["by_elo"].get(1900)
n1750, b1750 = new["by_elo"].get(1750), base["by_elo"].get(1750)
keep = (
    n1900 is not None and b1900 is not None
    and n1900 >= 0.55 and n1900 >= b1900 - 0.05
    and (n1750 is None or b1750 is None or n1750 >= b1750 - 0.10)
)
gate = {
    "keep": keep,
    "baseline": base,
    "candidate": new,
    "play_checkpoint": (
        "outputs/exp191_soft_ft3h_tactics_hard/best.pt" if keep
        else "outputs/exp191_soft_ft3h_edge_end/best.pt"
    ),
}
out.write_text(json.dumps(gate, indent=2))
print(json.dumps(gate, indent=2))
print("GATE_KEEP=" + str(keep))
print("PLAY=" + gate["play_checkpoint"])
PY
echo "[$(date -Is)] done" | tee -a "$LOG"
