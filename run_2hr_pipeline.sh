#!/bin/bash
# ============================================================
# AUTONOMOUS 2-HOUR TRAINING PIPELINE
# Deploy in tmux, runs 4 experiments back-to-back
# Each saves best_model.pt; later experiments init from winners
# ============================================================

set -e
cd /root/transform

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="outputs/tmux_pipeline_${TIMESTAMP}.log"
mkdir -p outputs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

log "=========================================="
log "AUTONOMOUS PIPELINE START"
log "=========================================="

# ── PHASE 1: exp105 chess bias (from best ckpt, ~30 min) ──
# Hypothesis: chess-relative attention bias improves representation
# Init: exp101_v2 best (0.1726 on 10K), LR=8e-6, 2000 steps
log ""
log "PHASE 1: exp105 chess-relative attention bias"
log "  Init: outputs/exp101_long_v2/best_model.pt (0.1726)"
log "  Config: LR=8e-6, steps=2000, batch=256x2, warmup=200, value_w=0.25"

python3 experiments/exp105_chess_bias.py \
    --init-checkpoint outputs/exp101_long_v2/best_model.pt \
    --n-files 50 \
    --label-smoothing 0.0 \
    --max-steps 2000 \
    --eval-interval 200 \
    --save-interval 500 \
    --warmup-steps 200 \
    --lr 8e-6 \
    --value-weight 0.25 \
    --n-eval 10000 \
    --output-dir outputs/exp105_chess_bias \
    2>&1 | tee -a outputs/exp105_full_console.log

EXP105_BEST=$(python3 -c "import json; print(json.load(open('outputs/exp105_chess_bias/status.json'))['best_accuracy'])" 2>/dev/null || echo "0")
log "PHASE 1 RESULT: exp105 best_accuracy = $EXP105_BEST"

# ── PHASE 2: exp104b label smoothing (from best ckpt, ~30 min) ──
# Hypothesis: label smoothing improves generalization on noisy labels
# Init: exp101_v2 best, but with label_smoothing=0.1
log ""
log "PHASE 2: exp104b cross-file shuffle + label smoothing"
log "  Init: outputs/exp101_long_v2/best_model.pt (0.1726)"
log "  Config: LR=8e-6, steps=2000, LS=0.1, value_w=0.25"

python3 experiments/exp104_shuffled_training.py \
    --init-checkpoint outputs/exp101_long_v2/best_model.pt \
    --n-files 50 \
    --label-smoothing 0.1 \
    --max-steps 2000 \
    --eval-interval 200 \
    --save-interval 500 \
    --warmup-steps 200 \
    --lr 8e-6 \
    --value-weight 0.25 \
    --n-eval 10000 \
    --output-dir outputs/exp104b_shuffle_LS \
    2>&1 | tee -a outputs/exp104b_console.log

EXP104B_BEST=$(python3 -c "import json; print(json.load(open('outputs/exp104b_shuffle_LS/status.json'))['best_accuracy'])" 2>/dev/null || echo "0")
log "PHASE 2 RESULT: exp104b best_accuracy = $EXP104B_BEST"

# ── PHASE 3: Pick winner, continue longer (from best of 1+2, ~30 min) ──
log ""
log "PHASE 3: Continue best winner for 3000 more steps"

WINNER_PATH=""
WINNER_NAME=""
python3 << 'PYEOF' > /tmp/winner.txt
import json, os
results = {}
for name, path in [("exp105", "outputs/exp105_chess_bias/status.json"),
                    ("exp104b", "outputs/exp104b_shuffle_LS/status.json"),
                    ("exp101v2", None)]:
    if path and os.path.exists(path):
        results[name] = json.load(open(path))["best_accuracy"]
    elif name == "exp101v2":
        results[name] = 0.1726  # known baseline
best_name = max(results, key=results.get)
best_acc = results[best_name]
ckpt_map = {
    "exp105": "outputs/exp105_chess_bias/best_model.pt",
    "exp104b": "outputs/exp104b_shuffle_LS/best_model.pt",
    "exp101v2": "outputs/exp101_long_v2/best_model.pt",
}
print(f"{best_name} {best_acc} {ckpt_map[best_name]}")
PYEOF
read WINNER_NAME WINNER_ACC WINNER_PATH < /tmp/winner.txt
log "Winner: $WINNER_NAME (acc=$WINNER_ACC) from $WINNER_PATH"

# Determine which script to use for continuation
if [ "$WINNER_NAME" = "exp105" ]; then
    CONT_SCRIPT="experiments/exp105_chess_bias.py"
else
    CONT_SCRIPT="experiments/exp104_shuffled_training.py"
fi

log "Continuing with $CONT_SCRIPT from $WINNER_PATH"
log "  Config: LR=3e-6 (lower for continuation), steps=3000, no LS, value_w=0.25"

python3 $CONT_SCRIPT \
    --init-checkpoint "$WINNER_PATH" \
    --n-files 100 \
    --label-smoothing 0.0 \
    --max-steps 3000 \
    --eval-interval 200 \
    --save-interval 500 \
    --warmup-steps 100 \
    --lr 3e-6 \
    --value-weight 0.25 \
    --n-eval 10000 \
    --output-dir outputs/exp106_continuation \
    2>&1 | tee -a outputs/exp106_console.log

EXP106_BEST=$(python3 -c "import json; print(json.load(open('outputs/exp106_continuation/status.json'))['best_accuracy'])" 2>/dev/null || echo "0")
log "PHASE 3 RESULT: exp106 best_accuracy = $EXP106_BEST"

# ── PHASE 4: Final continuation from absolute best (~30 min) ──
log ""
log "PHASE 4: Ultra-low LR polish from overall best"

python3 << 'PYEOF' > /tmp/final_winner.txt
import json, os
results = {}
for name, path in [("exp105", "outputs/exp105_chess_bias/status.json"),
                    ("exp104b", "outputs/exp104b_shuffle_LS/status.json"),
                    ("exp106", "outputs/exp106_continuation/status.json"),
                    ("exp101v2", None)]:
    if path and os.path.exists(path):
        results[name] = json.load(open(path))["best_accuracy"]
    elif name == "exp101v2":
        results[name] = 0.1726
best_name = max(results, key=results.get)
best_acc = results[best_name]
ckpt_map = {
    "exp105": "outputs/exp105_chess_bias/best_model.pt",
    "exp104b": "outputs/exp104b_shuffle_LS/best_model.pt",
    "exp106": "outputs/exp106_continuation/best_model.pt",
    "exp101v2": "outputs/exp101_long_v2/best_model.pt",
}
script_map = {
    "exp105": "experiments/exp105_chess_bias.py",
    "exp104b": "experiments/exp104_shuffled_training.py",
    "exp106": "experiments/exp104_shuffled_training.py",
    "exp101v2": "experiments/exp104_shuffled_training.py",
}
print(f"{best_name} {best_acc} {ckpt_map[best_name]} {script_map[best_name]}")
PYEOF
read FINAL_NAME FINAL_ACC FINAL_PATH FINAL_SCRIPT < /tmp/final_winner.txt
log "Overall best: $FINAL_NAME (acc=$FINAL_ACC)"

# Use different seed for diversity
python3 $FINAL_SCRIPT \
    --init-checkpoint "$FINAL_PATH" \
    --n-files 100 \
    --label-smoothing 0.0 \
    --max-steps 3000 \
    --eval-interval 200 \
    --save-interval 500 \
    --warmup-steps 50 \
    --lr 1e-6 \
    --value-weight 0.15 \
    --n-eval 10000 \
    --seed 123 \
    --output-dir outputs/exp107_polish \
    2>&1 | tee -a outputs/exp107_console.log

EXP107_BEST=$(python3 -c "import json; print(json.load(open('outputs/exp107_polish/status.json'))['best_accuracy'])" 2>/dev/null || echo "0")
log "PHASE 4 RESULT: exp107 best_accuracy = $EXP107_BEST"

# ── SUMMARY ──
log ""
log "=========================================="
log "PIPELINE COMPLETE"
log "=========================================="
log ""
log "Results summary:"
log "  Baseline (exp101_v2):  0.1726"
log "  exp105 (chess bias):   $EXP105_BEST"
log "  exp104b (label smooth): $EXP104B_BEST"
log "  exp106 (continuation): $EXP106_BEST"
log "  exp107 (polish):       $EXP107_BEST"
log ""

# Final comprehensive eval of all best models
python3 << 'PYEOF'
import sys, os, json
sys.path.insert(0, '/root/transform')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

checkpoints = [
    ("exp101_v2 (baseline)", "outputs/exp101_long_v2/best_model.pt", "exp101"),
    ("exp105 (chess bias)", "outputs/exp105_chess_bias/best_model.pt", "exp105"),
    ("exp104b (label smooth)", "outputs/exp104b_shuffle_LS/best_model.pt", "exp104"),
    ("exp106 (continuation)", "outputs/exp106_continuation/best_model.pt", "exp106"),
    ("exp107 (polish)", "outputs/exp107_polish/best_model.pt", "exp107"),
]

# Only import once
from data_loader import build_eval_from_hf
print("Loading 10K eval set for final comparison...")
eval_data, eval_tensors = build_eval_from_hf("avewright/chess-positions-lichess-sf", n_eval=10000, encoder_type="fused")

results = {}
for name, path, exp_id in checkpoints:
    if not os.path.exists(path):
        print(f"  {name}: MISSING")
        continue
    try:
        if exp_id == "exp105":
            from experiments.exp105_chess_bias import load_model, evaluate
        else:
            from experiments.exp104_shuffled_training import load_model, evaluate
        model = load_model(path, 'cuda')
        m = evaluate(model, eval_data, eval_tensors, 'cuda')
        results[name] = m
        print(f"  {name}: acc={m['accuracy']:.4f}  top3={m['top3_accuracy']:.4f}  val={m['value_accuracy']:.4f}")
        del model
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

# Save results
with open("outputs/pipeline_final_results.json", "w") as f:
    json.dump({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in results.items()}, f, indent=2)
print("\nResults saved to outputs/pipeline_final_results.json")
PYEOF

log "Final eval complete. Check outputs/pipeline_final_results.json"
log "=========================================="
