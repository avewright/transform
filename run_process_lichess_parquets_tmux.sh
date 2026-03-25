#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed on this machine."
  exit 1
fi

SESSION_NAME="${SESSION_NAME:-lichess_parquet_pipeline}"
WORK_ROOT="${WORK_ROOT:-/workspace/chess_hf_pipeline}"
LOG_DIR="${LOG_DIR:-$WORK_ROOT/logs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$WORK_ROOT"

CMD=(
  python process_lichess_parquets.py
  --work-root "$WORK_ROOT"
  "$@"
)

printf -v CMD_STR '%q ' "${CMD[@]}"

tmux new-session -d -s "$SESSION_NAME" \
  "cd $(printf '%q' "$ROOT_DIR") && \
   echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $CMD_STR\" | tee -a $(printf '%q' "$RUN_LOG") && \
   $CMD_STR 2>&1 | tee -a $(printf '%q' "$RUN_LOG")"

echo "Started tmux session: $SESSION_NAME"
echo "Run log: $RUN_LOG"
echo "Work root: $WORK_ROOT"
echo "State: $WORK_ROOT/orchestrator_state.json"
echo "Events: $WORK_ROOT/orchestrator_events.jsonl"
echo "Attach with: tmux attach -t $SESSION_NAME"
