#!/bin/bash
# Keep the 50M ChessFENS harvest alive. Never starts training.
set -u
ROOT=/root/transform
cd "$ROOT"
DATA=outputs/value99_data_50m
LOG=logs/value99_harvest.log
mkdir -p logs
echo "$(date -Is) harvest watchdog start" >> "$LOG"
while [[ ! -f "$DATA/manifest.json" ]]; do
  if ! pgrep -f 'scripts/prepare_value99_data.py --rows 50000000' >/dev/null; then
    echo "$(date -Is) resume 50M prepare" >> "$LOG"
    python3 -u scripts/prepare_value99_data.py --rows 50000000 --valmix-rows 8192 --out "$DATA" --seed 294 >> "$LOG" 2>&1
  fi
  sleep 30
done
echo "$(date -Is) harvest complete accepted=$(python3 -c "import json;print(json.load(open('$DATA/manifest.json'))['accepted'])")" >> "$LOG"
