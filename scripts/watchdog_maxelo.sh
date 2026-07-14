#!/bin/bash
# Watchdog: if maxelo tmux session dies before stages/all.done, restart pipeline.
set -uo pipefail
cd /root/transform
MERGED=outputs/exp191_soft_merged
LOG="$MERGED/watchdog.log"
mkdir -p "$MERGED/stages"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

while true; do
  if [[ -f "$MERGED/stages/all.done" ]]; then
    log "all.done present — watchdog exiting"
    exit 0
  fi
  if ! tmux has-session -t maxelo 2>/dev/null; then
    log "maxelo session missing — restarting pipeline"
    tmux new-session -d -s maxelo \
      "bash /root/transform/scripts/run_exp191_maxelo_pipeline.sh; ec=\$?; echo EXIT=\$ec; date -Is >> $MERGED/pipeline.log; sleep 7200"
    log "restarted maxelo"
  else
    # heartbeat
    if [[ -f "$MERGED/STATUS.txt" ]]; then
      log "ok: $(cat "$MERGED/STATUS.txt")"
    else
      log "ok: maxelo alive (no STATUS yet)"
    fi
  fi
  sleep 120
done
