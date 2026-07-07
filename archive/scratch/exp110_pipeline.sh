#!/bin/bash
# exp110_pipeline.sh: Automated pipeline that waits for exp110 training,
# runs ELO eval, then starts exp110b syzygy training.

set -o pipefail
cd /root/transform

LOG="outputs/pipeline.log"
log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "Pipeline started. Waiting for exp110 training to finish..."

# Wait for training to finish
while pgrep -f "exp110_diverse_training" > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date +%H:%M:%S)] exp110 training finished!"

# Check if best model exists
if [[ ! -f outputs/exp110_diverse_training/best_model.pt ]]; then
    echo "ERROR: No best_model.pt found!"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Running quick ELO eval on exp110 best checkpoint..."
python elo_eval_latest.py \
    outputs/exp110_diverse_training/best_model.pt \
    outputs/exp110_diverse_training/elo_eval \
    --elos 1600 1750 1900 2050 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    2>&1 | tee outputs/exp110_diverse_training/elo_eval.log

echo "[$(date +%H:%M:%S)] ELO eval complete!"

echo "[$(date +%H:%M:%S)] Starting exp110b syzygy training..."
python -u experiments/exp110b_syzygy_training.py 2>&1 | tee outputs/exp110b_syzygy_training/exp110b.log

echo "[$(date +%H:%M:%S)] exp110b training complete!"

echo "[$(date +%H:%M:%S)] Running ELO eval on exp110b best checkpoint..."
python elo_eval_latest.py \
    outputs/exp110b_syzygy_training/best_model.pt \
    outputs/exp110b_syzygy_training/elo_eval \
    --elos 1320 1600 1750 1900 2050 \
    --games-per-opening-per-color 2 \
    --stop-after-bracket \
    2>&1 | tee outputs/exp110b_syzygy_training/elo_eval.log

echo "[$(date +%H:%M:%S)] FULL PIPELINE COMPLETE!"
echo "Check results in:"
echo "  outputs/exp110_diverse_training/elo_eval.log"
echo "  outputs/exp110b_syzygy_training/elo_eval.log"

# Phase 5: Weakness harvest using exp110b best model
echo "[$(date +%H:%M:%S)] Starting weakness harvest on exp110b model..."
BEST_CKPT="outputs/exp110b_syzygy_training/best_model.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
    BEST_CKPT="outputs/exp110_diverse_training/best_model.pt"
fi

python -u experiments/exp110_weakness_harvest.py \
    --checkpoint "$BEST_CKPT" \
    --games 500 \
    --depth 8 \
    --multipv 5 \
    --sf-elos 1600 1750 1900 2050 \
    --seed 42 \
    2>&1 | tee outputs/exp110_weakness_harvest/weakness.log

echo "[$(date +%H:%M:%S)] Weakness harvest complete!"
echo "  $(wc -l outputs/exp110_weakness_harvest/dataset/*.jsonl 2>/dev/null | tail -1)"

# Phase 6: Train exp110c with weakness data
echo "[$(date +%H:%M:%S)] Starting exp110c weakness-targeted training..."
python -u experiments/exp110c_weakness_training.py 2>&1 | tee outputs/exp110c_weakness_training/exp110c.log

echo "[$(date +%H:%M:%S)] exp110c training complete!"

# Phase 7: Full ELO eval on exp110c
echo "[$(date +%H:%M:%S)] Running full ELO eval on exp110c..."
python elo_eval_latest.py \
    outputs/exp110c_weakness_training/best_model.pt \
    outputs/exp110c_weakness_training/elo_eval \
    --elos 1320 1600 1750 1900 2050 \
    --games-per-opening-per-color 2 \
    --stop-after-bracket \
    2>&1 | tee outputs/exp110c_weakness_training/elo_eval.log

echo "[$(date +%H:%M:%S)] ALL PHASES COMPLETE!"
echo "Results:"
echo "  exp110:  $(tail -1 outputs/exp110_diverse_training/elo_eval.log 2>/dev/null)"
echo "  exp110b: $(tail -1 outputs/exp110b_syzygy_training/elo_eval.log 2>/dev/null)"
echo "  exp110c: $(tail -1 outputs/exp110c_weakness_training/elo_eval.log 2>/dev/null)"
