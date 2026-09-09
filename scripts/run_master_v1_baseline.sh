#!/usr/bin/env bash
# Isolated master-v1 baseline arm. Does not overwrite the incumbent.
set -euo pipefail
cd /root/transform
set -a
# shellcheck disable=SC1091
source .env
set +a
export PATH="/root/transform/.venv/bin:/root/.local/bin:$PATH"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export STOCKFISH_PATH="/root/.local/bin/stockfish-19"
export HF_HOME=/root/transform/.hf_cache
export HUGGINGFACE_HUB_CACHE=/root/transform/.hf_cache

ROOT=/root/transform
BASE="$ROOT/outputs/master_v1_baseline"
EXPORT="$BASE/export"
PRE="$BASE/preflight"
TRAIN="$BASE/train"
EVAL="$BASE/eval"
LOG="$BASE/run.log"
PY="$ROOT/.venv/bin/python"
INCUMBENT="$ROOT/outputs/sf19_ft/overnight_20260908/eval_swa.pt"

mkdir -p "$BASE" "$EXPORT" "$PRE" "$TRAIN" "$EVAL"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "=== master-v1 baseline controller start pid=$$ tmux=${TMUX:-none} ==="
log "git=$(git -C "$ROOT" rev-parse HEAD) stockfish=$STOCKFISH_PATH"

# 1. Wait for in-flight export if any
if [[ ! -f "$EXPORT/manifest.json" ]]; then
  log "waiting for export manifest at $EXPORT/manifest.json"
  while ! [[ -f "$EXPORT/manifest.json" ]]; do
    if pgrep -f "build_chess_master.py.*export" >/dev/null; then
      sleep 15
      continue
    fi
    log "export process gone; launching export in this session"
    "$PY" -u scripts/build_chess_master.py --output outputs/chess_master_v1 export \
      --recipe pilot_45_35_15_5 \
      --dest "$EXPORT" | tee -a "$LOG"
    break
  done
fi
[[ -f "$EXPORT/manifest.json" ]] || { log "FATAL: export did not complete"; exit 1; }
log "export manifest present"

# 2. Validate
if [[ ! -f "$EXPORT/AUTHORIZED.json" ]]; then
  log "validating export"
  "$PY" -u scripts/validate_master_export.py --export "$EXPORT" --recipe pilot_45_35_15_5 | tee -a "$LOG"
fi
[[ -f "$EXPORT/AUTHORIZED.json" ]] || { log "FATAL: export not authorized"; exit 1; }
log "export authorized"

# 3. Preflight
if [[ ! -f "$PRE/PREFLIGHT_OK.json" ]]; then
  log "preflight train step (compile on)"
  set +e
  "$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
    --resume "$INCUMBENT" \
    --soft-cache "$EXPORT/soft_cache.pt" \
    --deep-cache "$EXPORT/deep_cache.pt" \
    --deep-mix-frac 0.05 --bonus-mix-frac 0 \
    --optimizer polar_normuon --muon-lr 0.0007 --adam-lr 1e-5 --warmup 100 \
    --batch-size 64 --max-steps 1 --train-minutes 15 \
    --val-every 1 --save-every 1 --elo-every 0 \
    --torch-compile --compile-polar \
    --output-dir "$PRE" \
    --external-eval "sf19=$EXPORT/sf19_eval.pt" \
    --external-eval "lichess=$EXPORT/lichess_eval.pt" \
    --external-eval "puzzles=$EXPORT/puzzles_eval.pt" \
    --external-eval "syzygy=$EXPORT/syzygy_eval.pt" \
    >>"$PRE/train.log" 2>&1
  rc=$?
  set -e
  compile=on
  if grep -q "torch.compile skipped" "$PRE/train.log"; then
    compile=off
    log "torch.compile failed; cause follows"
    grep -n "torch.compile skipped" "$PRE/train.log" | tee -a "$LOG"
    log "re-running preflight with compile explicitly disabled"
    rm -f "$PRE/latest.pt" "$PRE/eval_swa.pt"
    "$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
      --resume "$INCUMBENT" \
      --soft-cache "$EXPORT/soft_cache.pt" \
      --deep-cache "$EXPORT/deep_cache.pt" \
      --deep-mix-frac 0.05 --bonus-mix-frac 0 \
      --optimizer polar_normuon --muon-lr 0.0007 --adam-lr 1e-5 --warmup 100 \
      --batch-size 64 --max-steps 1 --train-minutes 15 \
      --val-every 1 --save-every 1 --elo-every 0 \
      --no-torch-compile --no-compile-polar \
      --output-dir "$PRE" \
      --external-eval "sf19=$EXPORT/sf19_eval.pt" \
      --external-eval "lichess=$EXPORT/lichess_eval.pt" \
      --external-eval "puzzles=$EXPORT/puzzles_eval.pt" \
      --external-eval "syzygy=$EXPORT/syzygy_eval.pt" \
      >>"$PRE/train.log" 2>&1
  fi
  [[ $rc -eq 0 || -f "$PRE/latest.pt" ]] || { log "FATAL: preflight train failed"; tail -50 "$PRE/train.log" | tee -a "$LOG"; exit 1; }
  COMPILE_FLAG=$compile "$PY" -u scripts/master_v1_preflight_checks.py \
    --preflight-dir "$PRE" \
    --export-dir "$EXPORT" \
    --incumbent "$INCUMBENT" | tee -a "$LOG"
  [[ -f "$PRE/PREFLIGHT_OK.json" ]] || { log "FATAL: preflight checks failed"; exit 1; }
  log "preflight ok compile=$compile"
else
  compile=$(python3 -c "import json; print(json.load(open('$PRE/PREFLIGHT_OK.json')).get('torch_compile','on'))")
  log "preflight already ok compile=$compile"
fi

COMPILE_ARGS=(--torch-compile --compile-polar)
if [[ "${compile}" == "off" ]]; then
  COMPILE_ARGS=(--no-torch-compile --no-compile-polar)
fi

# 4. Bounded train
if [[ ! -f "$TRAIN/train_summary.json" && ! -f "$TRAIN/latest.pt" ]]; then
  log "starting bounded train 8000 steps / 120 min"
  "$PY" -u experiments/exp201_recurrent_64.py --go --skip-mix \
    --resume "$INCUMBENT" \
    --soft-cache "$EXPORT/soft_cache.pt" \
    --deep-cache "$EXPORT/deep_cache.pt" \
    --deep-mix-frac 0.05 --bonus-mix-frac 0 \
    --optimizer polar_normuon --muon-lr 0.0007 --adam-lr 1e-5 --warmup 100 \
    --batch-size 64 --max-steps 8000 --train-minutes 120 \
    --val-every 500 --save-every 500 --elo-every 0 \
    "${COMPILE_ARGS[@]}" \
    --output-dir "$TRAIN" \
    --external-eval "sf19=$EXPORT/sf19_eval.pt" \
    --external-eval "lichess=$EXPORT/lichess_eval.pt" \
    --external-eval "puzzles=$EXPORT/puzzles_eval.pt" \
    --external-eval "syzygy=$EXPORT/syzygy_eval.pt" \
    >>"$TRAIN/train.log" 2>&1 || true
  log "train process exited"
else
  log "train artifacts already present; not restarting"
fi

# 5. Eval
log "starting 128-game screens (1h reserve)"
"$PY" -u scripts/master_v1_eval_and_report.py \
  --base "$BASE" \
  --incumbent "$INCUMBENT" | tee -a "$LOG"
log "=== controller finished ==="
