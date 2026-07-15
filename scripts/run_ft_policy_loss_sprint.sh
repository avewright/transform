#!/usr/bin/env bash
# AFK sprint: short FT on loss soft cache → Elo. Fits ~1h wall.
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

echo "[$(date -Is)] AFK sprint FT init=$INIT deep=$MIX → $OUT" | tee -a "$LOG"
# ~1.4s/step → 2000 steps ≈ 47min; leave room for Elo
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$MIX" \
  --deep-mix-frac 0.85 \
  --soft-frac 0.95 \
  --soft-alpha 0.65 \
  --batch-size 288 \
  --accum-steps 1 \
  --steps 2000 \
  --warmup 150 \
  --muon-lr 0.0015 \
  --adam-lr 6e-5 \
  --value-weight 0.08 \
  --grad-clip 0.5 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --hard-cache "$HARD" \
  --hard-n 800000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 400 \
  --eval-interval 400 \
  --eval-n 1024 \
  --select-metric soft_loss \
  --compile \
  --polar \
  --cautious-wd \
  2>&1 | tee "$OUT/run.log" | tee -a "$LOG"

echo "[$(date -Is)] FT done; Elo gauntlet" | tee -a "$LOG"
CKPT="$OUT/best.pt"
[[ -f "$CKPT" ]] || CKPT="$OUT/last.pt"
python -u elo_eval_latest.py "$CKPT" exp191_soft_ft3h_loss_fix \
  --movetime 0.05 \
  --games-per-opening-per-color 1 \
  --stop-after-bracket \
  --elos 1750 1900 2050 \
  2>&1 | tee "$OUT/elo_gauntlet.log" | tee -a "$LOG"

echo "[$(date -Is)] sprint complete ckpt=$CKPT" | tee -a "$LOG"
