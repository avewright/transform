#!/usr/bin/env bash
# RunPod pack for exp270 (~270M squares64, fresh init). Does not start training
# unless you pass the train subcommand on the pod.
#
#   bash scripts/runpod_exp270.sh setup
#   bash scripts/runpod_exp270.sh mix
#   bash scripts/runpod_exp270.sh bench
#   bash scripts/runpod_exp270.sh train          # only after bench
#   bash scripts/runpod_exp270.sh eval-backup
#
# 99M incumbent is the benchmark. Do not overwrite it. Do not --resume from it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${EXP270_OUT:-$ROOT/outputs/exp270_squares64_pretrain}"
MIX="${EXP270_MIX:-$ROOT/outputs/exp270_mix_v1}"
HOURS="${EXP270_HOURS:-12}"
TRAIN_HOURS="${EXP270_TRAIN_HOURS:-10.5}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT" "$MIX"

log() { echo "$(date -Is) $*" | tee -a "$OUT/controller.log"; }

cmd_setup() {
  bash scripts/runpod_setup.sh
  "$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  MOVE_VOCAB_VERSION=compact "$PY" -u experiments/exp270_squares64_pretrain.py --smoke --device cpu
  log "setup ok"
}

cmd_mix() {
  "$PY" -u scripts/build_exp270_mix.py --go --output "$MIX"
  log "mix frozen at $MIX"
}

cmd_bench() {
  log "benchmark 270M on this GPU (throughput / peak mem / latency / LR stability)"
  "$PY" -u scripts/bench_exp270.py --out "$OUT/bench.json" --mix "$MIX"
  log "wrote $OUT/bench.json — read it before train"
}

cmd_train() {
  if [[ ! -f "$OUT/bench.json" && "${EXP270_SKIP_BENCH:-0}" != "1" ]]; then
    echo "missing $OUT/bench.json — run bench first, or EXP270_SKIP_BENCH=1" >&2
    exit 1
  fi
  if [[ ! -f "$MIX/soft_cache.pt" ]]; then
    echo "missing mix $MIX/soft_cache.pt — run mix first" >&2
    exit 1
  fi
  EXTRA=()
  if [[ -f "$OUT/bench.json" ]]; then
    # shellcheck disable=SC2207
    EXTRA=($("$PY" - <<PY
import json
b=json.load(open("$OUT/bench.json"))
rec=b.get("recommend") or {}
args=[]
if rec.get("grad_checkpoint"): args.append("--grad-checkpoint")
if rec.get("batch_size"): args += ["--batch-size", str(int(rec["batch_size"]))]
if rec.get("accum_steps"): args += ["--accum-steps", str(int(rec["accum_steps"]))]
if rec.get("muon_lr"): args += ["--muon-lr", str(rec["muon_lr"])]
if rec.get("adam_lr"): args += ["--adam-lr", str(rec["adam_lr"])]
print(" ".join(args))
PY
))
  fi
  MINUTES="$(python3 - <<PY
print(int(float("$TRAIN_HOURS") * 60))
PY
)"
  log "train deadline ${TRAIN_HOURS}h of ${HOURS}h pod. extra=${EXTRA[*]:-none}"
  mkdir -p "$OUT"
  exec "$PY" -u experiments/exp270_squares64_pretrain.py --go \
    --output-dir "$OUT" \
    --soft-cache "$MIX/soft_cache.pt" \
    --deep-cache "$MIX/deep_cache.pt" \
    --train-minutes "$MINUTES" \
    --block-manifest "$MIX/blocked_manifest.json" \
    --external-eval "sf19=$MIX/sf19_eval.pt" \
    --external-eval "lichess=$MIX/lichess_eval.pt" \
    --external-eval "syzygy=$MIX/syzygy_eval.pt" \
    ${EXTRA[@]+"${EXTRA[@]}"}
}

cmd_eval_backup() {
  log "eval + snapshot (incumbent stays untouched)"
  if [[ -f "$OUT/eval_swa.pt" ]]; then
    "$PY" -u scripts/eval_exp201_searchfree.py --ckpt "$OUT/eval_swa.pt" --out "$OUT/eval_swa.json" || true
  fi
  if [[ -f "$OUT/latest.pt" ]]; then
    "$PY" -u scripts/eval_exp201_searchfree.py --ckpt "$OUT/latest.pt" --out "$OUT/eval_live.json" || true
  fi
  mkdir -p "$OUT/backup"
  cp -n "$OUT/latest.pt" "$OUT/backup/latest.pt" 2>/dev/null || true
  cp -n "$OUT/eval_swa.pt" "$OUT/backup/eval_swa.pt" 2>/dev/null || true
  cp -n "$OUT/config.json" "$OUT/backup/config.json" 2>/dev/null || true
  cp -n "$OUT/dataset_manifest.json" "$OUT/backup/" 2>/dev/null || true
  log "backup in $OUT/backup — upload as avewright/chess-transformer-270m-squares64-exp270"
}

usage() {
  sed -n '2,14p' "$0"
}

case "${1:-}" in
  setup) cmd_setup ;;
  mix) cmd_mix ;;
  bench) cmd_bench ;;
  train) cmd_train ;;
  eval-backup) cmd_eval_backup ;;
  *) usage; exit 1 ;;
esac
