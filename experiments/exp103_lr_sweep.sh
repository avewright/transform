#!/bin/bash
# exp103: LR sweep from exp101 v2 best (0.1676 acc)
# 3 runs of 500 steps each, eval every 100 steps
# Hypothesis: A lower LR prevents the degradation seen in longer runs

INIT_CKPT="/root/transform/outputs/exp101_long_v2/best_model.pt"
SCRIPT="experiments/exp101_hf_scale_training.py"
COMMON="--init-checkpoint $INIT_CKPT --batch-size 256 --accum-steps 2 --value-weight 0.50 --max-steps 500 --eval-interval 100 --save-interval 250 --warmup-steps 25 --max-files 10 --seed 42"

cd /root/transform

echo "=== LR Sweep from exp101 v2 best (0.1676 acc) ==="
echo ""

# Run 1: LR=3e-6 (very conservative)
echo ">>> Run 1/3: LR=3e-6"
python3 $SCRIPT $COMMON --lr 3e-6 --output-dir outputs/exp103_lr3e6 2>&1 | tee outputs/exp103_lr3e6_console.log
echo ""

# Run 2: LR=8e-6 (moderate)
echo ">>> Run 2/3: LR=8e-6"
python3 $SCRIPT $COMMON --lr 8e-6 --output-dir outputs/exp103_lr8e6 2>&1 | tee outputs/exp103_lr8e6_console.log
echo ""

# Run 3: LR=2e-5 (same as exp101 v2)
echo ">>> Run 3/3: LR=2e-5"
python3 $SCRIPT $COMMON --lr 2e-5 --output-dir outputs/exp103_lr2e5 2>&1 | tee outputs/exp103_lr2e5_console.log
echo ""

echo "=== LR Sweep Complete ==="
echo "Compare best_acc from each run's status.json:"
for d in outputs/exp103_lr3e6 outputs/exp103_lr8e6 outputs/exp103_lr2e5; do
    if [ -f "$d/status.json" ]; then
        echo "  $d: $(cat $d/status.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"best_acc={d.get(\"best_acc\",\"?\")}")')"
    fi
done
