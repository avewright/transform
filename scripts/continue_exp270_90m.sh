#!/usr/bin/env bash
# Wait for the live exp270 process to exit on its own, then resume in tmux
# exp270 with every packed Lichess shard and a ~90M-draw budget.
set -euo pipefail
cd /root/transform
TRAIN_PID="${1:-}"
LOG=outputs/exp270_squares64_pretrain/continue_90m.log
mkdir -p outputs/exp270_squares64_pretrain
exec >>"$LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] watchdog start train_pid=${TRAIN_PID:-none}"

if [[ -n "$TRAIN_PID" ]]; then
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 20
  done
  echo "[$(date -u +%H:%M:%S)] train pid $TRAIN_PID exited"
  sleep 3
fi

for _ in $(seq 1 180); do
  packing=0
  pgrep -f 'pack_exp270_lichess_rest.py' >/dev/null && packing=1
  pgrep -f 'pack_exp270_bonus.py' >/dev/null && packing=1
  if [[ $packing -eq 0 ]]; then
    break
  fi
  echo "[$(date -u +%H:%M:%S)] waiting for packer"
  sleep 20
done

mapfile -t SHARDS < <(ls -1 outputs/exp270_mix_v1/lichess_rest_*.pt)
if [[ ${#SHARDS[@]} -eq 0 ]]; then
  echo "no extra shards; abort"
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] attaching ${#SHARDS[@]} shards → tmux exp270"

EXTRA=()
for p in "${SHARDS[@]}"; do
  EXTRA+=(--extra-soft-cache "$p")
done

# One line so tmux send-keys does not break on continuations.
CMD="cd /root/transform && export MOVE_VOCAB_VERSION=compact PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -u experiments/exp270_squares64_pretrain.py --go --output-dir outputs/exp270_squares64_pretrain --soft-cache outputs/exp270_mix_v1/soft_cache.pt --deep-cache outputs/exp270_mix_v1/deep_cache.pt --bonus-cache outputs/exp270_mix_v1/bonus_cache.pt --quality-cache outputs/exp270_mix_v1/sf19_eco_train.pt --puzzle-cache outputs/exp270_mix_v1/puzzle_cache.pt --resume outputs/exp270_squares64_pretrain/latest.pt ${EXTRA[*]} --train-minutes 22000 --max-steps 1200000 --batch-size 80 --no-fill-vram --optimizer polar_normuon --compile-polar --force-lr --muon-lr 0.007407 --adam-lr 0.000111 --warmup 0 --torch-compile --compile-mode default --deep-mix-frac 0.20 --bonus-mix-frac 0.08 --quality-mix-frac 0.05 --puzzle-mix-frac 0.05 --deep-in-each-batch --soft-alpha 0.55 --external-eval sf19=outputs/exp270_mix_v1/sf19_eval.pt --external-eval sf19eco=outputs/exp270_mix_v1/sf19_eco_eval.pt 2>&1 | tee -a outputs/exp270_squares64_pretrain/train.log"

tmux send-keys -t exp270 "$CMD" C-m
echo "[$(date -u +%H:%M:%S)] resume command sent"
