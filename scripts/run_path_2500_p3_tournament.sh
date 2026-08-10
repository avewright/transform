#!/usr/bin/env bash
# Phase 3: MCTS Elo report (not champion metric) — 800/1600 sims vs SF 2200–2500.
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
    outputs/runs/exp191_hf437m_soft_ft/best.pt \
    outputs/exp191_hf437m_soft_ft/best.pt \
    outputs/exp191_hf437m_soft_ft/best_model.pt \
    outputs/lora_soft_hfmix/merged_model.pt \
    outputs/hf_437m_ft3h_hub/best_model.pt \
    outputs/hf_437m/best_model.pt; do
    if [[ -f "$c" ]]; then CKPT="$c"; break; fi
  done
fi
PREFIX="${PREFIX:-path2500_tour}"
GAMES="${GAMES:-2}"
mkdir -p outputs logs

echo "=== tournament MCTS report ckpt=$CKPT ==="
SMOKE_SIMS="${SMOKE_SIMS:-64}"

python -u -m harness.elo --ckpt "$CKPT" --mode mcts \
  --out-prefix "${PREFIX}_gumbel${SMOKE_SIMS}" \
  --sims "$SMOKE_SIMS" --search-mode gumbel --batch-size 16 \
  --elos 1750 1900 \
  --games-per-opening-per-color 1 \
  --stop-after-bracket \
  2>&1 | tee "logs/${PREFIX}_gumbel${SMOKE_SIMS}.log"

if [[ "${RUN_FULL:-0}" == "1" ]]; then
  python -u -m harness.elo --ckpt "$CKPT" --mode mcts \
    --out-prefix "${PREFIX}_puct800" \
    --sims 800 --search-mode puct --batch-size 16 \
    --elos 2050 2200 2350 2500 \
    --games-per-opening-per-color "$GAMES" \
    --stop-after-bracket \
    2>&1 | tee "logs/${PREFIX}_puct800.log"

  python -u -m harness.elo --ckpt "$CKPT" --mode mcts \
    --out-prefix "${PREFIX}_puct1600" \
    --sims 1600 --search-mode puct --batch-size 16 \
    --elos 2200 2350 2500 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    2>&1 | tee "logs/${PREFIX}_puct1600.log"
else
  echo "SKIP heavy 800/1600 gauntlet (set RUN_FULL=1 to enable)"
  python -u -m harness.elo --ckpt "$CKPT" --mode mcts \
    --out-prefix "${PREFIX}_puct${SMOKE_SIMS}" \
    --sims "$SMOKE_SIMS" --search-mode puct --batch-size 16 \
    --elos 1900 2050 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    2>&1 | tee "logs/${PREFIX}_puct${SMOKE_SIMS}.log"
fi

echo "=== UCI tournament command (CuteChess / Arena) ==="
cat <<EOF
python uci_engine.py \\
  --checkpoint $CKPT \\
  --syzygy syzygy \\
  --default-sims 800 \\
  --batch-size 16 \\
  --search-mode auto \\
  --dynamic-cpuct
EOF

echo "=== Phase 3 done ==="
