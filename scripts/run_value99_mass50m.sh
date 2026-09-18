#!/bin/bash
set -euo pipefail
ROOT=/root/transform
cd "$ROOT"
export PYTHONUNBUFFERED=1
DATA=outputs/value99_data_50m
LOG=logs/value99_mass50m_pipeline.log
mkdir -p outputs logs "$DATA"

if [[ ! -f "$DATA/manifest.json" ]]; then
  echo "$(date -Is) preparing 50M" | tee -a "$LOG"
  python3 -u scripts/prepare_value99_data.py --rows 50000000 --valmix-rows 8192 --out "$DATA" --seed 294
fi

python3 - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/value99_data_50m/manifest.json').read_text())
accepted=int(m['accepted']); train=int(m['counts']['train'])
if accepted<50000000:
    raise SystemExit(f'Accepted {accepted} < 50M')
if train<49000000:
    raise SystemExit(f'Train {train} too small for a 50M run')
print('data_ok', accepted, m['counts'], m.get('valmix',{}).get('filled'))
PY

if [[ ! -f outputs/value99_smoke50m/events.jsonl ]]; then
  echo "$(date -Is) smoke" | tee -a "$LOG"
  rm -rf outputs/value99_smoke50m
  python3 -u experiments/value99_pretrain.py --config configs/value99_smoke50m.json --out outputs/value99_smoke50m --device cuda
  python3 - <<'PY'
import json
from pathlib import Path
events=[json.loads(l) for l in Path('outputs/value99_smoke50m/events.jsonl').read_text().splitlines() if l.strip()]
train=[e for e in events if e.get('stage')=='train']
if len(train)<2: raise SystemExit('smoke: not enough train steps')
if not all(float(e['loss'])==float(e['loss']) for e in train): raise SystemExit('smoke: nonfinite loss')
if not all(float(e['grad_norm'])>0 for e in train): raise SystemExit('smoke: zero gradients')
if abs(train[-1]['loss']-train[0]['loss'])<1e-8: raise SystemExit('smoke: loss did not change')
print('smoke_train_ok', train[-1])
PY
  echo "$(date -Is) smoke resume" | tee -a "$LOG"
  python3 -u experiments/value99_pretrain.py --config configs/value99_smoke50m.json --out outputs/value99_smoke50m --device cuda --resume
fi

if [[ ! -f outputs/value99_mass50m/manifest.json ]]; then
  echo "$(date -Is) launching mass run" | tee -a "$LOG"
  python3 -u experiments/value99_pretrain.py --config configs/value99_mass50m.json --out outputs/value99_mass50m --device cuda
else
  echo "$(date -Is) resuming mass run" | tee -a "$LOG"
  python3 -u experiments/value99_pretrain.py --config configs/value99_mass50m.json --out outputs/value99_mass50m --device cuda --resume
fi
