#!/usr/bin/env bash
# Fine-tune FT3h on policy-loss soft cache (blunders + lost-game SF MultiPV).
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

MIX=outputs/policy_loss_soft/soft_cache.pt
OUT=outputs/exp191_soft_ft3h_loss_fix
INIT=outputs/exp191_soft_ft3h_edge_end/best.pt
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
LOG=outputs/overnight_maxelo/ft_policy_loss.log
mkdir -p outputs/overnight_maxelo "$OUT"

echo "[$(date -Is)] wait for $MIX…" | tee -a "$LOG"
while [[ ! -f outputs/policy_loss_soft/DONE ]]; do sleep 20; done
while [[ $(stat -c%s "$MIX" 2>/dev/null || echo 0) -lt 1000000 ]]; do sleep 5; done

# Don't fight harvest if still holding GPU — DONE means harvest finished
echo "[$(date -Is)] FT start init=$INIT deep=$MIX → $OUT" | tee -a "$LOG"
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$MIX" \
  --deep-mix-frac 0.80 \
  --soft-frac 0.95 \
  --soft-alpha 0.60 \
  --batch-size 256 \
  --accum-steps 1 \
  --steps 6000 \
  --warmup 300 \
  --muon-lr 0.0012 \
  --adam-lr 5e-5 \
  --value-weight 0.10 \
  --grad-clip 0.5 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --hard-cache "$HARD" \
  --hard-n 1000000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 1000 \
  --eval-interval 1000 \
  --eval-n 1024 \
  --select-metric soft_loss \
  --compile \
  --polar \
  --cautious-wd \
  2>&1 | tee -a "$OUT/run.log" | tee -a "$LOG"

echo "[$(date -Is)] FT done; Elo gauntlet" | tee -a "$LOG"
CKPT="$OUT/best.pt"
[[ -f "$CKPT" ]] || CKPT="$OUT/last.pt"
python -u elo_eval_latest.py "$CKPT" exp191_soft_ft3h_loss_fix \
  --movetime 0.05 \
  --games-per-opening-per-color 1 \
  --stop-after-bracket \
  --elos 1750 1900 2050 \
  2>&1 | tee -a "$OUT/elo_gauntlet.log" | tee -a "$LOG"
