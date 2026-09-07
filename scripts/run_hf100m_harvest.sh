#!/bin/bash
# Harvest inaccuracies/blunders + soft targets for the 100M squares64 model.
# Bounded CPU harvest for a 24GB Mac. Run explicitly when ready.
set -euo pipefail
cd "$(dirname "$0")/.."
export MOVE_VOCAB_VERSION=compact
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

MODE="${1:-pilot}"
case "$MODE" in
  smoke) GAMES=2;   WORKERS=1; NODES=100000 ;;
  pilot) GAMES=60;  WORKERS=4; NODES=600000 ;;
  full)  GAMES=240; WORKERS=4; NODES=600000 ;;
  sf19)
    # New games vs unlimited Stockfish 19. Do not rescan the existing MultiPV stream.
    CKPT="${CKPT:-outputs/hf100m_lapse_ft_30m/latest.pt}"
    if [[ ! -f "$CKPT" ]]; then
      CKPT=$(.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "avewright/chess-transformer-100m-squares64"
revision = HfApi().model_info(repo).sha
print(hf_hub_download(repo, "latest.pt", revision=revision))
PY
      )
    fi
    OUT="${OUT:-outputs/hf100m_sf19_games/20260906_154325}"
    export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"
    echo "Checkpoint: $CKPT"
    echo "Output: $OUT (SF19 max-elo, reuse engine, screen 8k / deep 80k / play 8k)"
    exec .venv/bin/python -u scripts/harvest_exp201_lapses.py --go --loop \
      --ckpt "$CKPT" \
      --out-dir "$OUT" \
      --inbox "$OUT/inbox" \
      --games 48 --workers 10 \
      --screen-nodes 8000 --teacher-nodes 80000 --play-nodes 8000 \
      --unlimited-frac 1.0 \
      --require-sf-name "Stockfish 19" \
      --book-noise-plies 3 \
      --tau 120 --sf-movetime 0.0 --ply-cap 120 \
      --no-keep-all-from-losses \
      --holdout-game-frac 0.08 --flush-every 8 \
      --explore-epsilon 0.25 --explore-temperature 0.85 \
      --explore-top-k 4 --explore-plies 48 --seed 1901 \
      --exclude-dir \
        outputs/hf100m_bulk/20260905_175038_bulk \
        outputs/hf100m_lapse_ft \
        outputs/hf100m_sf_games
    ;;
  sf-games)
    # Efficient SF soft targets: play games, MultiPV=1 screen, deep-label disagreements only.
    CKPT="${CKPT:-outputs/hf100m_lapse_ft_30m/latest.pt}"
    if [[ ! -f "$CKPT" ]]; then
      CKPT=$(.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "avewright/chess-transformer-100m-squares64"
revision = HfApi().model_info(repo).sha
print(hf_hub_download(repo, "latest.pt", revision=revision))
PY
      )
    fi
    OUT="${OUT:-outputs/hf100m_sf_games/$(date +%Y%m%d_%H%M%S)}"
    SEED="${SEED_CACHE:-outputs/hf100m_lapse_ft/soft_cache.pt}"
    echo "Checkpoint: $CKPT"
    echo "Output: $OUT (screen 25k / deep 250k, seed=$SEED)"
    exec .venv/bin/python -u scripts/harvest_exp201_lapses.py --go --loop \
      --ckpt "$CKPT" \
      --out-dir "$OUT" \
      --inbox "$OUT/inbox" \
      --games 36 --workers 6 \
      --screen-nodes 25000 --teacher-nodes 250000 \
      --sf-elos 1450 1600 1750 1900 2050 --unlimited-frac 0.15 \
      --tau 120 --sf-movetime 0.05 --ply-cap 140 \
      --no-keep-all-from-losses \
      --holdout-game-frac 0.1 --flush-every 6 \
      --explore-epsilon 0.2 --explore-temperature 0.85 \
      --explore-top-k 4 --explore-plies 48 --seed 201 \
      --seed-cache "$SEED" --seed-frac 0.5
    ;;
  bulk)
    # 10M path: reuse HF MultiPV labels, filter with the 100M. No Stockfish.
    CKPT="${CKPT:-outputs/hf100m_lapse_ft_30m/latest.pt}"
    if [[ ! -f "$CKPT" ]]; then
      CKPT=$(.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "avewright/chess-transformer-100m-squares64"
revision = HfApi().model_info(repo).sha
print(hf_hub_download(repo, "latest.pt", revision=revision))
PY
      )
    fi
    OUT="${OUT:-outputs/hf100m_bulk/20260905_175038_bulk}"
    SKIP="${SKIP_SEEN:-0}"
    if [[ "$SKIP" -eq 0 && -f "$OUT/cursor.json" ]]; then
      SKIP=$(.venv/bin/python -c "import json; print(int(json.load(open('$OUT/cursor.json'))['seen']))")
    fi
    echo "Checkpoint: $CKPT"
    echo "Output: $OUT (bulk filter, resume skip_seen=$SKIP, target 10M)"
    exec .venv/bin/python -u scripts/harvest_hf100m_bulk.py --go \
      --ckpt "$CKPT" \
      --out-dir "$OUT" \
      --target 10000000 \
      --skip-seen "$SKIP" \
      --batch-rows 4096 \
      --micro-batch 128 \
      --shard-size 100000
    ;;
  *) echo "usage: $0 [smoke|pilot|full|bulk|sf-games|sf19]"; exit 1 ;;
esac

# Pin the current HF revision for this entire run; do not silently use stale weights.
CKPT=$(.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "avewright/chess-transformer-100m-squares64"
revision = HfApi().model_info(repo).sha
print(hf_hub_download(repo, "latest.pt", revision=revision))
PY
)
OUT="outputs/hf100m_lapses/$(date +%Y%m%d_%H%M%S)_${MODE}_$$"
echo "Checkpoint: $CKPT"
echo "Output: $OUT; workers=$WORKERS; games=$GAMES (bounded run)"

.venv/bin/python -u scripts/harvest_exp201_lapses.py --go \
  --ckpt "$CKPT" \
  --out-dir "$OUT" \
  --games "$GAMES" --workers "$WORKERS" --teacher-nodes "$NODES" \
  --sf-elos 1600 1750 1900 2200 --unlimited-frac 0.2 \
  --tau 120 --sf-movetime 0.06 --ply-cap 160 \
  --no-keep-all-from-losses \
  --inbox "$OUT/inbox" --holdout-game-frac 0.2 --flush-every 8 \
  --explore-epsilon 0.15 --explore-temperature 0.8 \
  --explore-top-k 4 --explore-plies 40 --seed 100
