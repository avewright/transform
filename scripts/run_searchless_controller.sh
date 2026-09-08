#!/usr/bin/env bash
# Wait for the 2050/2200 gauntlet and mix v2, then start the 8k train without
# leaving the GPU idle. Does not start while harness.elo still owns the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
LOG="$ROOT/outputs/searchless_gauntlet/controller.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

GAUNTLET="$ROOT/outputs/searchless_gauntlet/report.json"
MIX="$ROOT/outputs/astra_mix_v2/mix_report.json"

log "waiting for gauntlet + mix v2"
while [[ ! -f "$GAUNTLET" || ! -f "$MIX" ]]; do
  sleep 20
done
log "both ready"

# Do not overlap the still-running gauntlet python if report was written early.
while pgrep -f 'python3 -u -m harness.elo' >/dev/null; do
  log "elo still running; wait"
  sleep 15
done
while pgrep -f 'exp201_recurrent_64.py' >/dev/null; do
  log "trainer already running; wait"
  sleep 15
done

if ! pgrep -f 'harvest_swa_mistakes.py --analyze' >/dev/null; then
  log "start 8 CPU analyzers on remaining off-PV inbox"
  tmux new-session -d -s swa-analyze -c "$ROOT" \
    "MOVE_VOCAB_VERSION=compact python3 -u scripts/harvest_swa_mistakes.py --analyze --watch --out-dir outputs/swa_mistakes --nodes 250000 --workers 8 --analyze-limit 210000 2>&1 | tee -a outputs/swa_mistakes/analyze.log"
fi

chmod +x "$ROOT/scripts/run_searchless_train.sh"
log "start train"
exec bash "$ROOT/scripts/run_searchless_train.sh"
