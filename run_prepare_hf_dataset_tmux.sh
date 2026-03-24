#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed on this machine."
  echo "Install it first, then rerun this launcher."
  exit 1
fi

SESSION_NAME="${SESSION_NAME:-hf_prepare_lichess}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/outputs/tmux_logs}"
SHARD_DIR="${SHARD_DIR:-$ROOT_DIR/outputs/hf_dataset_shards/lichess_sf_formatted}"

mkdir -p "$LOG_DIR" "$SHARD_DIR"

RUN_LOG="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"
CMD=(
  python prepare_hf_dataset.py
  --shard-dir "$SHARD_DIR"
  "$@"
)

printf -v CMD_STR '%q ' "${CMD[@]}"

tmux new-session -d -s "$SESSION_NAME" \
  "cd $(printf '%q' "$ROOT_DIR") && \
   echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $CMD_STR\" | tee -a $(printf '%q' "$RUN_LOG") && \
   $CMD_STR 2>&1 | tee -a $(printf '%q' "$RUN_LOG")"

echo "Started tmux session: $SESSION_NAME"
echo "Run log: $RUN_LOG"
echo "Shard dir: $SHARD_DIR"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Inspect progress with:"
echo "  tail -f $RUN_LOG"
echo "  tail -f $SHARD_DIR/events.jsonl"
echo "  cat $SHARD_DIR/progress.json"
