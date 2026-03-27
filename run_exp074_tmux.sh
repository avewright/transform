#!/usr/bin/env bash
# Launch exp074 (resume 200M training) in a detached tmux session.
# Logs go to outputs/exp074_resume_200m/train.log
# Attach with: tmux attach -t exp074

set -e
cd "$(dirname "$0")/.."

mkdir -p outputs/exp074_resume_200m

SESSION="exp074"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
  "cd /root/transform && python experiments/exp074_resume_200m.py 2>&1 | tee outputs/exp074_resume_200m/train.log; echo 'DONE'; read"

echo "Started tmux session: $SESSION"
echo "  Attach:  tmux attach -t $SESSION"
echo "  Logs:    tail -f outputs/exp074_resume_200m/train.log"
