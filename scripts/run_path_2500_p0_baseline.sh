#!/usr/bin/env bash
# Phase 0: policy + MCTS Elo baselines on hf_437m.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK=1

CKPT="${CKPT:-outputs/hf_437m/best_model.pt}"
MCTS_SIMS="${MCTS_SIMS:-64}"
mkdir -p outputs logs

echo "=== P0 policy Elo ==="
python -u elo_eval_latest.py "$CKPT" hf437m_p0 \
  --elos 1450 1600 1750 1900 2050 \
  --games-per-opening-per-color "${GAMES:-2}" \
  --stop-after-bracket \
  2>&1 | tee logs/path2500_p0_policy.log

echo "=== P0 MCTS Elo @${MCTS_SIMS} sims (auto) ==="
python -u scripts/elo_eval_mcts.py "$CKPT" "hf437m_p0_mcts${MCTS_SIMS}" \
  --sims "$MCTS_SIMS" --search-mode auto --batch-size 16 \
  --elos 1750 1900 \
  --games-per-opening-per-color "${MCTS_GAMES:-1}" \
  --stop-after-bracket \
  2>&1 | tee "logs/path2500_p0_mcts${MCTS_SIMS}.log"

if [[ "${RUN_MCTS800:-0}" == "1" ]]; then
  echo "=== P0 MCTS Elo @800 sims (puct tournament) ==="
  python -u scripts/elo_eval_mcts.py "$CKPT" hf437m_p0_mcts800 \
    --sims 800 --search-mode puct --batch-size 16 \
    --elos 1900 2050 2200 \
    --games-per-opening-per-color "${MCTS_GAMES:-1}" \
    --stop-after-bracket \
    2>&1 | tee logs/path2500_p0_mcts800.log
fi

echo "=== P0 done ==="
