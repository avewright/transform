#!/bin/bash
# Wait for exp184 training to finish, then run holdout + optional Elo probe.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/exp184_a40_wide_soft
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2}"
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1

echo "=== watchdog waiting for exp184 tmux session to end $(date -Is) ===" | tee -a "$OUT/watchdog.log"
while tmux has-session -t exp184 2>/dev/null; do
  sleep 60
done
echo "=== exp184 session ended $(date -Is) ===" | tee -a "$OUT/watchdog.log"

CKPT=""
for c in "$OUT/best.pt" "$OUT/latest.pt"; do
  if [[ -f "$c" ]]; then CKPT="$c"; break; fi
done
if [[ -z "$CKPT" ]]; then
  echo "No checkpoint found" | tee -a "$OUT/watchdog.log"
  exit 1
fi
echo "Evaluating $CKPT" | tee -a "$OUT/watchdog.log"

python - <<PY 2>&1 | tee -a "$OUT/watchdog.log"
import json, os, sys
from pathlib import Path
os.environ.setdefault('MOVE_VOCAB_VERSION','compact')
sys.path.insert(0,'.')
import torch
from chess_transformer_factory import build_model, ChessTransformerConfig
from experiments.exp184_a40_wide_soft_normuon import eval_soft_top1, cache_soft_dataset

ckpt_path = Path('$CKPT')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = ChessTransformerConfig(**ckpt['config'])
model = build_model(cfg).cuda()
model.load_state_dict(ckpt['model_state_dict'], strict=False)
soft = cache_soft_dataset(Path('$OUT/soft_cache.pt'))
m = eval_soft_top1(model, soft, torch.device('cuda'), n=5000)
payload = {'checkpoint': str(ckpt_path), 'step': ckpt.get('step'), **m}
print(json.dumps(payload, indent=2))
Path('$OUT/final_holdout.json').write_text(json.dumps(payload, indent=2))
PY

# Optional gameplay probe (may take a while; non-fatal)
python -u elo_eval_latest.py "$CKPT" "$OUT/elo_probe" \
  --movetime 0.05 \
  2>&1 | tee -a "$OUT/watchdog.log" || true

echo "=== watchdog done $(date -Is) ===" | tee -a "$OUT/watchdog.log"
