"""Comprehensive value head analysis.

Runs on CPU to avoid interrupting GPU training.
Provides: confusion matrix, per-class metrics, calibration, per-phase breakdown.

Usage:
  python experiments/_value_analysis.py [--checkpoint PATH] [--eval-set PATH]
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_transformer_factory import build_model
from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl

SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"


def classify_phase(board_array):
    """Classify position phase by piece count."""
    n_pieces = (board_array > 0).sum().item()
    if n_pieces >= 28:
        return "opening"
    elif n_pieces >= 14:
        return "middlegame"
    else:
        return "endgame"


def classify_material_balance(board_array, turn):
    """Classify material balance from side-to-move perspective."""
    PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0,
                    7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0}
    white_mat = sum(PIECE_VALUES.get(int(p), 0) for p in board_array if 1 <= int(p) <= 6)
    black_mat = sum(PIECE_VALUES.get(int(p), 0) for p in board_array if 7 <= int(p) <= 12)
    if int(turn) == 0:  # white to move
        diff = white_mat - black_mat
    else:
        diff = black_mat - white_mat
    if diff > 2:
        return "ahead"
    elif diff < -2:
        return "behind"
    else:
        return "equal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(ROOT / "outputs" / "exp149_scratch_204m" / "best_model.pt"))
    parser.add_argument("--eval-set", type=str,
                        default=str(SHARD_DIR / "eval.pt"))
    parser.add_argument("--max-positions", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_model()
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    print(f"Loading eval set: {args.eval_set}")
    data = torch.load(args.eval_set, map_location=device, weights_only=False)
    n = min(args.max_positions, data["board_array"].shape[0])

    # Compute WDL from cp/mate
    wdl = compute_wdl(data["cp"][:n].float(), data["mate"][:n].float())
    print(f"Evaluating {n:,} positions on CPU...")

    # Run inference in batches
    BS = 64
    all_value_logits = []
    all_true_wdl = []
    all_phases = []
    all_balances = []

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, BS):
            end = min(start + BS, n)
            batch = {
                "fused_ids": board_array_to_fused(data["board_array"][start:end]).to(device),
                "turn": data["turn"][start:end].long().to(device),
                "castling": data["castling"][start:end].long().to(device),
                "ep_file": ep_square_to_file(data["ep_square"][start:end].long()).to(device),
            }
            out = model(batch)
            all_value_logits.append(out["value_logits"].cpu())
            all_true_wdl.append(wdl[start:end])

            for i in range(start, end):
                all_phases.append(classify_phase(data["board_array"][i]))
                all_balances.append(classify_material_balance(
                    data["board_array"][i], data["turn"][i]))

            if (start // BS) % 20 == 0:
                elapsed = time.time() - t0
                rate = end / elapsed if elapsed > 0 else 0
                print(f"  {end:,}/{n:,} ({rate:.0f} pos/s)")

    elapsed = time.time() - t0
    print(f"Inference: {n:,} positions in {elapsed:.1f}s ({n/elapsed:.0f} pos/s)")

    value_logits = torch.cat(all_value_logits, dim=0)  # [N, 3]
    true_wdl = torch.cat(all_true_wdl, dim=0)  # [N, 3]
    value_probs = F.softmax(value_logits, dim=-1)  # [N, 3]

    pred_class = value_logits.argmax(dim=-1)  # 0=W, 1=D, 2=L
    true_class = true_wdl.argmax(dim=-1)

    CLASS_NAMES = ["Win", "Draw", "Loss"]

    # ========== 1. Overall Accuracy ==========
    correct = (pred_class == true_class).float().mean().item()
    print(f"\n{'='*60}")
    print(f"OVERALL VALUE ACCURACY: {100*correct:.2f}%")
    print(f"{'='*60}")

    # ========== 2. Confusion Matrix ==========
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    conf = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(true_class, pred_class):
        conf[t, p] += 1

    print(f"{'':>10} {'Pred_W':>8} {'Pred_D':>8} {'Pred_L':>8} {'Total':>8}")
    for i in range(3):
        row_total = conf[i].sum().item()
        pcts = [f"{100*conf[i,j].item()/max(1,row_total):.1f}%" for j in range(3)]
        print(f"  True_{CLASS_NAMES[i]:<4} {conf[i,0]:>5} ({pcts[0]:>5}) "
              f"{conf[i,1]:>5} ({pcts[1]:>5}) "
              f"{conf[i,2]:>5} ({pcts[2]:>5}) = {row_total:>5}")

    # ========== 3. Per-Class Precision / Recall / F1 ==========
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for c in range(3):
        tp = conf[c, c].item()
        fn = conf[c].sum().item() - tp
        fp = conf[:, c].sum().item() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        support = conf[c].sum().item()
        print(f"  {CLASS_NAMES[c]:>8} {100*precision:>9.1f}% {100*recall:>9.1f}% "
              f"{100*f1:>9.1f}% {support:>9,}")

    # ========== 4. Value Loss ==========
    # Cross-entropy: true_wdl is soft targets [N, 3], value_logits is raw [N, 3]
    log_probs = F.log_softmax(value_logits, dim=-1)
    ce_loss = -(true_wdl * log_probs).sum(dim=-1).mean().item()
    # Also compute with hard targets for comparison
    hard_ce = F.cross_entropy(value_logits, true_class).item()
    print(f"\nValue Loss (soft CE): {ce_loss:.4f}")
    print(f"Value Loss (hard CE): {hard_ce:.4f}")
    print(f"Random baseline CE: {1.0986:.4f} (ln 3)")

    # ========== 5. Prediction Distribution ==========
    print(f"\nPredicted Class Distribution:")
    for c in range(3):
        count = (pred_class == c).sum().item()
        print(f"  {CLASS_NAMES[c]}: {count:>5,} ({100*count/n:.1f}%)")
    print(f"\nTrue Class Distribution:")
    for c in range(3):
        count = (true_class == c).sum().item()
        print(f"  {CLASS_NAMES[c]}: {count:>5,} ({100*count/n:.1f}%)")

    # ========== 6. Confidence Distribution ==========
    max_probs = value_probs.max(dim=-1)[0]
    print(f"\nConfidence Distribution (max predicted probability):")
    bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7),
            (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        mask = (max_probs >= lo) & (max_probs < hi)
        count = mask.sum().item()
        if count > 0:
            acc = (pred_class[mask] == true_class[mask]).float().mean().item()
            print(f"  [{lo:.1f}, {hi:.1f}): {count:>5,} positions, "
                  f"accuracy={100*acc:.1f}%")

    # ========== 7. Calibration (by predicted win probability) ==========
    print(f"\nCalibration: Predicted Win Prob vs Actual Win Rate:")
    cal_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    win_probs = value_probs[:, 0]  # P(win)
    actual_win = (true_class == 0).float()
    for i in range(len(cal_bins) - 1):
        lo, hi = cal_bins[i], cal_bins[i + 1]
        mask = (win_probs >= lo) & (win_probs < hi)
        count = mask.sum().item()
        if count > 0:
            actual = actual_win[mask].mean().item()
            pred_mean = win_probs[mask].mean().item()
            gap = abs(actual - pred_mean)
            print(f"  P(win) [{lo:.1f}, {hi:.1f}): n={count:>5,}, "
                  f"predicted={100*pred_mean:.1f}%, actual={100*actual:.1f}%, "
                  f"gap={100*gap:.1f}pp")

    # ========== 8. Per-Phase Analysis ==========
    print(f"\n{'='*60}")
    print(f"PER-PHASE VALUE ANALYSIS")
    print(f"{'='*60}")
    for phase in ["opening", "middlegame", "endgame"]:
        mask = torch.tensor([p == phase for p in all_phases])
        if mask.sum() == 0:
            continue
        pc = pred_class[mask]
        tc = true_class[mask]
        acc = (pc == tc).float().mean().item()
        count = mask.sum().item()

        # Phase confusion
        pconf = torch.zeros(3, 3, dtype=torch.long)
        for t, p in zip(tc, pc):
            pconf[t, p] += 1

        print(f"\n  {phase.upper()}: {count:,} positions, accuracy={100*acc:.1f}%")
        print(f"    {'':>8} {'Pred_W':>8} {'Pred_D':>8} {'Pred_L':>8}")
        for i in range(3):
            row_total = pconf[i].sum().item()
            pcts = [f"{100*pconf[i,j].item()/max(1,row_total):.1f}%" for j in range(3)]
            print(f"    True_{CLASS_NAMES[i]:<4} {pconf[i,0]:>5} ({pcts[0]:>5}) "
                  f"{pconf[i,1]:>5} ({pcts[1]:>5}) "
                  f"{pconf[i,2]:>5} ({pcts[2]:>5})")

        # Phase value loss
        phase_logits = value_logits[mask]
        phase_true = true_class[mask]
        phase_ce = F.cross_entropy(phase_logits, phase_true).item()
        print(f"    Value CE loss: {phase_ce:.4f}")

    # ========== 9. Per-Balance Analysis ==========
    print(f"\n{'='*60}")
    print(f"PER-MATERIAL-BALANCE VALUE ANALYSIS")
    print(f"{'='*60}")
    for balance in ["ahead", "equal", "behind"]:
        mask = torch.tensor([b == balance for b in all_balances])
        if mask.sum() == 0:
            continue
        pc = pred_class[mask]
        tc = true_class[mask]
        acc = (pc == tc).float().mean().item()
        count = mask.sum().item()

        print(f"\n  {balance.upper()}: {count:,} positions, accuracy={100*acc:.1f}%")

        # Per-class recall
        for c in range(3):
            cmask = tc == c
            if cmask.sum() > 0:
                cacc = (pc[cmask] == c).float().mean().item()
                print(f"    {CLASS_NAMES[c]} recall: {100*cacc:.1f}% ({cmask.sum().item()} samples)")

    # ========== 10. Draw Analysis ==========
    print(f"\n{'='*60}")
    print(f"DRAW ANALYSIS (common failure mode)")
    print(f"{'='*60}")
    draw_mask = true_class == 1
    draw_count = draw_mask.sum().item()
    if draw_count > 0:
        draw_pred = pred_class[draw_mask]
        for c in range(3):
            count = (draw_pred == c).sum().item()
            print(f"  True draws predicted as {CLASS_NAMES[c]}: "
                  f"{count:,} ({100*count/draw_count:.1f}%)")

        # When model predicts draw
        pred_draw_mask = pred_class == 1
        pred_draw_count = pred_draw_mask.sum().item()
        if pred_draw_count > 0:
            print(f"\n  When model predicts Draw ({pred_draw_count:,} positions):")
            for c in range(3):
                count = (true_class[pred_draw_mask] == c).sum().item()
                print(f"    Actually {CLASS_NAMES[c]}: {count:,} ({100*count/pred_draw_count:.1f}%)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
