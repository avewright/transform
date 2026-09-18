#!/bin/bash
# Re-pull avewright/local-wdl and append new split=0 shards into the mix pack.
# Waits out any in-flight packer so we do not race train_*.npz.
set -u
ROOT=/root/transform
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
OUT=outputs/value99_data_sfwdl_train
LOG=logs/value99_sfwdl_train.log
EVERY="${EVERY:-1800}"
mkdir -p logs
echo "$(date -Is) sfwdl HF refresh watch start every=${EVERY}s" >> "$LOG"
while true; do
  if pgrep -f 'scripts/prepare_value99_sfwdl_train.py' >/dev/null; then
    sleep 30
    continue
  fi
  python3 -u scripts/prepare_value99_sfwdl_train.py \
    --out "$OUT" --watch --refresh --every "$EVERY" >> "$LOG" 2>&1 || true
  echo "$(date -Is) sfwdl pack watch exited; restart" >> "$LOG"
  sleep 30
done
