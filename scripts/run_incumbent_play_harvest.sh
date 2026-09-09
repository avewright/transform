#!/usr/bin/env bash
# Play games with the incumbent vs Stockfish 19 max. CPU only.
# Half the games start from random real boards; half are book + noise games.
# Unique positions only; keep inaccuracy / blunder / conversion / major.
set -euo pipefail
cd /root/transform
export PATH="/root/transform/.venv/bin:/root/.local/bin:$PATH"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/.local/bin/stockfish-19}"

ROOT=/root/transform
CKPT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"
OUT="$ROOT/outputs/swa_play_harvest"
SEED="$ROOT/outputs/swa_harvest_more/unseen_sf19.pt"
[[ -f "$SEED" ]] || SEED="$ROOT/outputs/overnight_corr15/soft_cache.pt"
HASHES="$ROOT/outputs/swa_sf19_disagree_v1/seen_all_tagged_hashes.npy"

mkdir -p "$OUT"
echo "$(date -Is) play-harvest start ckpt=$CKPT seed=$SEED" | tee -a "$OUT/controller.log"

exec "$ROOT/.venv/bin/python" -u scripts/harvest_exp201_lapses.py --go --loop \
  --ckpt "$CKPT" \
  --out-dir "$OUT" \
  --inbox "$OUT/inbox" \
  --games 48 --workers 24 \
  --screen-nodes 8000 --teacher-nodes 80000 --play-nodes 4000 \
  --confirm-drop 50 \
  --unlimited-frac 1.0 \
  --require-sf-name "Stockfish 19" \
  --book-noise-plies 4 \
  --seed-cache "$SEED" --seed-frac 0.5 \
  --tau 120 --sf-movetime 0.0 --ply-cap 120 \
  --no-keep-all-from-losses \
  --holdout-game-frac 0.05 --flush-every 8 \
  --explore-epsilon 0.25 --explore-temperature 0.85 \
  --explore-top-k 4 --explore-plies 48 --seed 1909 \
  --exclude-hashes "$HASHES" \
  --exclude-cache \
    "$ROOT/outputs/swa_sf19_disagree_v1/all_disagreements.pt" \
    "$ROOT/outputs/swa_sf19_disagree_v1/verified_substantial.pt" \
    "$ROOT/outputs/overnight_corr15/bonus_cache.pt" \
  --exclude-dir \
    "$ROOT/outputs/swa_harvest_more/analyzed" \
    "$OUT/inbox"
