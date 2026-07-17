#!/usr/bin/env bash
# A40 (~45GB): piece-square dual on high-Elo puzzles + Syzygy.
#
# On the pod (after git pull):
#   1) Sync the mix once from your laptop (preferred):
#        scp outputs/autoresearch_8gb/highelo_puzzle_syzygy_mix.pt \
#            user@pod:~/transform/outputs/autoresearch_8gb/
#      Or set MIX=/path/to/highelo_puzzle_syzygy_mix.pt
#   2) Ensure Stockfish is on PATH or STOCKFISH_PATH=...
#   3) tmux new -s dual 'bash scripts/run_dual_highelo_a40.sh'
#
# Env knobs:
#   STEPS=12000  TRAIN_MINUTES=480  BATCH_SIZE=768  MIX=...  OUT=...
#   SKIP_ELO=1   FORCE=1  ELO_EVERY=500
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION="${MOVE_VOCAB_VERSION:-compact}"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-outputs/autoresearch_8gb_a40}"
MIX="${MIX:-outputs/autoresearch_8gb/highelo_puzzle_syzygy_mix.pt}"
PUZZLE="${PUZZLE:-outputs/exp193_puzzle_highelo/soft_cache.pt}"
SYZ="${SYZ:-outputs/syzygy_hf/soft_cache.pt}"
STEPS="${STEPS:-12000}"
TRAIN_MINUTES="${TRAIN_MINUTES:-480}"
TRIAL="${TRIAL:-dual_highelo_a40}"
mkdir -p "$OUT" "$(dirname "$MIX")" outputs/syzygy_hf

if [[ -z "${STOCKFISH_PATH:-}" ]]; then
  if command -v stockfish >/dev/null 2>&1; then
    export STOCKFISH_PATH="$(command -v stockfish)"
  elif [[ -x stockfish/stockfish/stockfish-ubuntu-x86-64-avx2 ]]; then
    export STOCKFISH_PATH="$(pwd)/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
  fi
fi

echo "=== dual_highelo A40 $(date -Is) ==="
echo "gpu:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "trial=$TRIAL steps=$STEPS minutes=$TRAIN_MINUTES mix=$MIX out=$OUT"
echo "stockfish=${STOCKFISH_PATH:-MISSING}"

ensure_mix() {
  if [[ -f "$MIX" ]]; then
    python -u -c "import torch; d=torch.load('$MIX', map_location='cpu', weights_only=False); print(f'mix ok n={d[\"board_array\"].shape[0]:,}')"
    return
  fi
  echo "mix missing at $MIX — building"
  if [[ ! -f "$SYZ" ]]; then
    echo "downloading avewright/chess-soft-syzygy -> $SYZ"
    python -u scripts/autoresearch_8gb/hf_soft_to_cache.py \
      --repo avewright/chess-soft-syzygy --out "$SYZ"
  fi
  if [[ ! -f "$PUZZLE" ]]; then
    cat <<EOF
ERROR: high-Elo puzzle cache missing: $PUZZLE
Sync from laptop, e.g.:
  scp outputs/exp193_puzzle_highelo/soft_cache.pt POD:$(pwd)/outputs/exp193_puzzle_highelo/
  scp outputs/autoresearch_8gb/highelo_puzzle_syzygy_mix.pt POD:$(pwd)/outputs/autoresearch_8gb/
Or set MIX= to a prebuilt mix.
EOF
    exit 2
  fi
  python -u scripts/autoresearch_8gb/merge_soft_caches.py "$PUZZLE" "$SYZ" --out "$MIX"
  python -u -c "import torch; d=torch.load('$MIX', map_location='cpu', weights_only=False); print(f'mix built n={d[\"board_array\"].shape[0]:,}')"
}

ensure_mix

# Optional: override batch in search_space via a one-shot patched run is hard;
# probe inside train_trial starts at trial batch_size=768 and shrinks to max_vram_gb=40.
FORCE_ARGS=(--force)
if [[ "${FORCE:-1}" != "1" ]]; then
  FORCE_ARGS=()
fi
SKIP_ARGS=()
if [[ "${SKIP_ELO:-0}" == "1" ]]; then
  SKIP_ARGS=(--skip-elo)
fi

LOG="$OUT/dual_highelo_a40.log"
echo "logging -> $LOG"
python -u experiments/exp194_autoresearch_8gb.py --go \
  "${FORCE_ARGS[@]}" \
  "${SKIP_ARGS[@]}" \
  --soft-cache "$MIX" \
  --train-minutes "$TRAIN_MINUTES" \
  --max-steps "$STEPS" \
  --min-steps-done 0 \
  --only "$TRIAL" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$LOG"

echo "=== dual_highelo A40 done $(date -Is) ==="
echo "ckpt: $OUT/trials/$TRIAL/latest.pt"
echo "elo history: $OUT/trials/$TRIAL/elo_gauntlet.jsonl"
