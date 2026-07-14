#!/usr/bin/env bash
# After FT3b hits 10k: Elo gauntlet on best.pt, then hand off to FT3c (compile+polar).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FT3B="${FT3B:-outputs/exp191_soft_ft3b_unseen}"
CKPT="${CKPT:-$FT3B/best.pt}"
PREFIX="${PREFIX:-exp191_soft_ft3b_unseen}"
LOG="$FT3B/elo_gauntlet.log"
ELO_TIMEOUT_SEC="${ELO_TIMEOUT_SEC:-7200}"
# Full gauntlet levels (same ladder as FT1/FT2 pipeline)
ELOS=(${ELOS:-1450 1600 1750 1900 2050})

mkdir -p "$FT3B"
echo "[$(date -Is)] wait for FT3b / any exp191 to finish…" | tee -a "$LOG"
while pgrep -f 'experiments/exp191_400m_meta_attention.py' >/dev/null 2>&1; do
  sleep 20
done
# Also wait until Done appears or best.pt is fresh after 10k
for _ in $(seq 1 30); do
  [[ -f "$CKPT" ]] && break
  sleep 10
done
if [[ ! -f "$CKPT" ]]; then
  echo "[$(date -Is)] ERROR missing $CKPT" | tee -a "$LOG"
  exit 1
fi
sleep 5

echo "[$(date -Is)] Elo gauntlet ckpt=$CKPT elos=${ELOS[*]}" | tee -a "$LOG"
set +e
timeout --signal=INT --kill-after=60 "$ELO_TIMEOUT_SEC" \
  python -u elo_eval_latest.py "$CKPT" "$PREFIX" \
    --movetime 0.05 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    --elos "${ELOS[@]}" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[$(date -Is)] Elo finished rc=$rc" | tee -a "$LOG"

# Hand off to FT3c (compile+polar on 12M late shards)
echo "[$(date -Is)] starting FT3c polar/compile…" | tee -a "$LOG"
exec bash scripts/run_ft3c_polar_compile.sh
