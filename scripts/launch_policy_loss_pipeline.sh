#!/usr/bin/env bash
# Launch parallel policy-loss harvest then FT waiter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export STOCKFISH_PATH="${STOCKFISH_PATH:-$ROOT/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"

mkdir -p outputs/policy_loss_soft outputs/overnight_maxelo
rm -f outputs/policy_loss_soft/DONE \
      outputs/policy_loss_soft/harvest.log \
      outputs/policy_loss_soft/nohup.log

# Reap orphan SF from stopped harvests
pkill -9 -f 'stockfish-ubuntu' 2>/dev/null || true
sleep 1

nohup python -u scripts/harvest_policy_loss_soft.py --go \
  --workers 6 \
  --games 480 \
  --black-frac 0.78 \
  --sf-elos 1750 1900 2050 \
  --teacher-nodes 1200000 \
  --train-after \
  > outputs/policy_loss_soft/nohup.log 2>&1 &
echo $! > outputs/policy_loss_soft/harvest.pid

nohup bash scripts/run_ft_policy_loss.sh \
  > outputs/overnight_maxelo/ft_policy_loss_nohup.log 2>&1 &
echo $! > outputs/overnight_maxelo/ft_policy_loss.pid

echo "harvest_pid=$(cat outputs/policy_loss_soft/harvest.pid)"
echo "ft_pid=$(cat outputs/overnight_maxelo/ft_policy_loss.pid)"
