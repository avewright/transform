#!/usr/bin/env bash
# Phase 0: pure-policy Elo baseline (+ optional MCTS report) via max-Elo harness.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK=1

CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
  for c in \
    outputs/champion/champion.pt \
    outputs/hf_437m_ft3h_hub/best_model.pt \
    outputs/hf_437m_ft3h/best_model.pt \
    outputs/hf_437m/best_model.pt; do
    if [[ -f "$c" ]]; then CKPT="$c"; break; fi
  done
fi
MCTS_SIMS="${MCTS_SIMS:-64}"
mkdir -p outputs logs

echo "=== P0 pure-policy Elo (no book / no syzygy) ckpt=$CKPT ==="
python -u -m harness.elo --ckpt "$CKPT" --mode policy \
  --out-prefix hf437m_p0_pure \
  --no-book --no-syzygy \
  --elos 1450 1600 1750 1900 2050 \
  --games-per-opening-per-color "${GAMES:-2}" \
  --stop-after-bracket \
  2>&1 | tee logs/path2500_p0_policy.log

echo "=== P0 MCTS report @${MCTS_SIMS} sims (not for promotion) ==="
python -u -m harness.elo --ckpt "$CKPT" --mode mcts \
  --out-prefix "hf437m_p0_mcts${MCTS_SIMS}" \
  --sims "$MCTS_SIMS" --search-mode auto --batch-size 16 \
  --elos 1750 1900 \
  --games-per-opening-per-color "${MCTS_GAMES:-1}" \
  --stop-after-bracket \
  2>&1 | tee "logs/path2500_p0_mcts${MCTS_SIMS}.log"

if [[ "${RUN_MCTS800:-0}" == "1" ]]; then
  echo "=== P0 MCTS @800 sims ==="
  python -u -m harness.elo --ckpt "$CKPT" --mode mcts \
    --out-prefix hf437m_p0_mcts800 \
    --sims 800 --search-mode puct --batch-size 16 \
    --elos 1900 2050 2200 \
    --games-per-opening-per-color "${MCTS_GAMES:-1}" \
    --stop-after-bracket \
    2>&1 | tee logs/path2500_p0_mcts800.log
fi

echo "=== P0 done ==="
