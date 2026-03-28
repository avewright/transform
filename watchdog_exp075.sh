#!/bin/bash
# Watchdog + monitor for exp075 training
# Run this in a tmux session: tmux new -s train
# It monitors the 4 workers. If the orchestrator dies, it restarts training
# with checkpoint resume.

cd /root/transform

LOG_DIR="outputs/exp075_ddp_4gpu"

check_alive() {
    # Check if any worker processes are running
    local alive=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "$alive"
}

monitor() {
    while true; do
        alive=$(check_alive)
        ts=$(date '+%H:%M:%S')
        
        if [ "$alive" -eq 0 ]; then
            echo "[$ts] NO GPU PROCESSES FOUND -- restarting training..."
            nohup python3 experiments/exp075_ddp_4gpu.py >> "$LOG_DIR/launch.log" 2>&1 &
            echo "[$ts] Relaunched (PID=$!). Waiting 120s for startup..."
            sleep 120
            continue
        fi
        
        echo "[$ts] $alive GPU processes active"
        for i in 0 1 2 3; do
            log="$LOG_DIR/worker$i/train.log"
            if [ -f "$log" ]; then
                last=$(grep -E "step.*pos/s" "$log" | tail -1 | sed 's/.*\(step.*\)/\1/' | head -c 80)
                echo "  W$i: $last"
            fi
        done
        
        # Show GPU memory
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
        echo ""
        
        sleep 60
    done
}

echo "=== EXP075 WATCHDOG STARTED ==="
echo "Press Ctrl+C to stop monitoring (training continues in background)"
echo ""
monitor
