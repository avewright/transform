#!/bin/bash
# ─────────────────────────────────────────────────────────
# Watchdog for exp076: Continue V2 model training
# ─────────────────────────────────────────────────────────
# Run in tmux: tmux new -s train
# Then:        bash watchdog_exp076.sh
#
# This script:
#   1. Launches exp076 training
#   2. Monitors for crashes every 60s
#   3. Auto-restarts with checkpoint resume
#   4. Logs everything to outputs/exp076_continue_v2/watchdog.log
#   5. Prints status summaries periodically
# ─────────────────────────────────────────────────────────

set -u

cd /root/transform

EXP_DIR="outputs/exp076_continue_v2"
LOG_FILE="$EXP_DIR/train.log"
WATCHDOG_LOG="$EXP_DIR/watchdog.log"
TRAIN_SCRIPT="experiments/exp076_continue_v2.py"
MAX_RESTARTS=50
RESTART_DELAY=30      # seconds to wait before restart
CHECK_INTERVAL=60     # seconds between alive checks
RESTART_COUNT=0

mkdir -p "$EXP_DIR"

log() {
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $*" | tee -a "$WATCHDOG_LOG"
}

check_gpu_alive() {
    # Check if our training process is using the GPU
    local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "$gpu_procs"
}

check_process_alive() {
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "1"
    else
        echo "0"
    fi
}

get_latest_step() {
    if [ -f "$EXP_DIR/training_log.json" ]; then
        python3 -c "
import json
with open('$EXP_DIR/training_log.json') as f:
    log = json.load(f)
if log:
    last = log[-1]
    step = last.get('step', '?')
    acc = last.get('accuracy', last.get('policy_loss', '?'))
    typ = last.get('type', '?')
    pos = last.get('positions_seen', '?')
    print(f'step={step} type={typ} acc/loss={acc} pos={pos}')
else:
    print('no entries')
" 2>/dev/null || echo "parse error"
    else
        echo "no log yet"
    fi
}

get_gpu_stats() {
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "N/A"
}

launch_training() {
    log "Launching training (attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)..."
    log "  Script: $TRAIN_SCRIPT"
    log "  Log: $LOG_FILE"

    # Launch in background, append to log
    nohup python3 -u "$TRAIN_SCRIPT" >> "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    log "  PID: $TRAIN_PID"

    # Wait for the process to start up
    sleep 10

    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        log "  Training process started OK"
    else
        log "  WARNING: Process died immediately! Check $LOG_FILE"
    fi
}

# ── Main watchdog loop ──

log "=========================================="
log " WATCHDOG STARTED for exp076_continue_v2"
log "=========================================="
log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
log "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)"
log "  Script: $TRAIN_SCRIPT"
log "  Max restarts: $MAX_RESTARTS"

launch_training

while true; do
    sleep "$CHECK_INTERVAL"

    proc_alive=$(check_process_alive)
    gpu_alive=$(check_gpu_alive)
    gpu_stats=$(get_gpu_stats)
    latest=$(get_latest_step)

    ts=$(date '+%H:%M:%S')

    if [ "$proc_alive" -eq 1 ]; then
        # Process is alive
        log "OK | PID=$TRAIN_PID | GPU procs=$gpu_alive | $gpu_stats | $latest"
    else
        # Process died
        log "DEAD | Training process $TRAIN_PID is not running"
        log "  Last status: $latest"

        # Check exit code if available
        wait "$TRAIN_PID" 2>/dev/null
        EXIT_CODE=$?
        log "  Exit code: $EXIT_CODE"

        # Check for NaN crash
        if [ -f "$EXP_DIR/checkpoints/nan_crash.pt" ]; then
            log "  NaN crash detected. Will restart with checkpoint resume."
        fi

        RESTART_COUNT=$((RESTART_COUNT + 1))

        if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
            log "FATAL: Max restarts ($MAX_RESTARTS) reached. Giving up."
            log "  Last checkpoint: $(ls -la $EXP_DIR/checkpoints/latest.pt 2>/dev/null)"
            exit 1
        fi

        log "  Waiting ${RESTART_DELAY}s before restart..."
        sleep "$RESTART_DELAY"

        # Clear GPU memory
        log "  Clearing GPU memory..."
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

        launch_training
    fi
done
