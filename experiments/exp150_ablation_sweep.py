"""exp150: Short ablation sweep from exp149 epoch-1 checkpoint.

Hypothesis: exp149's label_smoothing=0.1, weight_decay=0.1, or value_weight=0.5
may over-regularize exact best-move prediction. Top-3 and value improve strongly
but top-1 lags — suggesting the model learns move families but not exact moves.

Protocol:
  - Resume from exp149 epoch_1.pt (or latest.pt if epoch_1 not reached)
  - Change ONE hyperparameter per ablation
  - Train 5K steps (~1.5h each) on same data (epoch 2 territory)
  - Eval on 20K positions for reliable comparison
  - Compare against a 5K-step control run with unchanged settings

Ablations:
  A: label_smoothing 0.1 → 0.0   (no smoothing = sharper exact-move signal)
  B: label_smoothing 0.1 → 0.02  (mild smoothing)
  C: weight_decay 0.1 → 0.01     (less L2 regularization)
  D: weight_decay 0.1 → 0.03     (moderate reduction)
  E: value_weight 0.5 → 0.25     (more gradient budget for policy)

Usage:
  python experiments/exp150_ablation_sweep.py --ablation A
  python experiments/exp150_ablation_sweep.py --ablation control
  python experiments/exp150_ablation_sweep.py --ablation all  # runs all sequentially

Requires: exp149 epoch_1.pt or latest.pt checkpoint.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model, DEFAULT_200M_CONFIG
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
EXP149_DIR = ROOT / "outputs" / "exp149_scratch_204m"
OUTPUT_DIR = ROOT / "outputs" / "exp150_ablations"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
EVAL_PATH = SHARD_DIR / "eval_20k.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None

MODEL_CONFIG = DEFAULT_200M_CONFIG

# Baseline settings from exp149
BASELINE = {
    "label_smoothing": 0.1,
    "weight_decay": 0.1,
    "value_weight": 0.5,
    "lr": 2e-4,          # exp149 LR at step 105K (epoch 1) ≈ 1.5e-4
    "min_lr_frac": 0.01,
}

ABLATIONS = {
    "control": {},  # no changes
    "A": {"label_smoothing": 0.0},
    "B": {"label_smoothing": 0.02},
    "C": {"weight_decay": 0.01},
    "D": {"weight_decay": 0.03},
    "E": {"value_weight": 0.25},
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
    PIECE_CHARS = ".PNBRQKpnbrqk"
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file_idx in range(8):
            sq = rank * 8 + file_idx
            p = int(ba_row[sq])
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += PIECE_CHARS[p]
        if empty > 0:
            row += str(empty)
        fen_rows.append(row)
    board_str = "/".join(fen_rows)
    turn_str = "w" if int(turn_val) == 0 else "b"
    castle_str = ""
    cv = int(castling_val)
    if cv & 8: castle_str += "K"
    if cv & 4: castle_str += "Q"
    if cv & 2: castle_str += "k"
    if cv & 1: castle_str += "q"
    if not castle_str: castle_str = "-"
    ev = int(ep_val)
    if 0 <= ev < 64:
        ep_str = chr(ord('a') + ev % 8) + str(ev // 8 + 1)
    else:
        ep_str = "-"
    return f"{board_str} {turn_str} {castle_str} {ep_str} 0 1"


def load_eval_data(eval_path):
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    wdl = compute_wdl(raw["cp"], raw["mate"])
    eval_data = []
    surviving = []
    for i in range(raw["board_array"].shape[0]):
        try:
            fen = _board_array_to_fen(
                raw["board_array"][i], raw["turn"][i],
                raw["castling"][i], raw["ep_square"][i])
            board = chess.Board(fen)
            uci = IDX_TO_UCI[raw["move_idx"][i].item()]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            eval_data.append({
                "board": board, "move": move,
                "wdl": (wdl[i, 0].item(), wdl[i, 1].item(), wdl[i, 2].item()),
            })
            surviving.append(i)
        except Exception:
            continue
    idx = torch.tensor(surviving, dtype=torch.long)
    eval_tensors = {
        "turn": raw["turn"][idx].long(),
        "castling": raw["castling"][idx].long(),
        "ep_file": ep_square_to_file(raw["ep_square"][idx].long()),
        "fused_ids": board_array_to_fused(raw["board_array"][idx]),
    }
    return eval_data, eval_tensors


def run_eval(model, eval_data, eval_tensors, batch_size=32):
    model.eval()
    correct = top3 = top5 = val_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            n = len(chunk)
            idx_slice = slice(i, i + n)
            batch_input = {
                "fused_ids": eval_tensors["fused_ids"][idx_slice].to(DEVICE),
                "turn": eval_tensors["turn"][idx_slice].to(DEVICE),
                "castling": eval_tensors["castling"][idx_slice].to(DEVICE),
                "ep_file": eval_tensors["ep_file"][idx_slice].to(DEVICE),
            }
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()
            value_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                board, true_move = d["board"], d["move"]
                l = logits[j].clone()
                mask = legal_move_mask(board).to(DEVICE)
                l[~mask] = float("-inf")
                true_idx = move_to_index(true_move)
                if l.argmax().item() == true_idx:
                    correct += 1
                topk3 = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk3:
                    top3 += 1
                topk5 = l.topk(min(5, l.shape[0])).indices.tolist()
                if true_idx in topk5:
                    top5 += 1
                vp = F.softmax(value_logits[j], dim=-1)
                pred_class = vp.argmax().item()
                true_class = max(range(3), key=lambda k: d["wdl"][k])
                if pred_class == true_class:
                    val_correct += 1
                total += 1
    return {
        "top1": correct / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "val": val_correct / max(total, 1),
        "total": total,
    }


def find_checkpoint():
    """Find the best starting checkpoint from exp149."""
    epoch1 = EXP149_DIR / "epoch_1.pt"
    if epoch1.exists():
        return epoch1, "epoch_1"
    latest = EXP149_DIR / "latest.pt"
    if latest.exists():
        return latest, "latest"
    raise FileNotFoundError(
        f"No exp149 checkpoint found. Need epoch_1.pt or latest.pt in {EXP149_DIR}")


def run_ablation(name, overrides, steps=5000, batch_size=24, accum_steps=4,
                 eval_data=None, eval_tensors=None):
    """Run a single ablation for `steps` optimizer steps."""
    global LOG_PATH

    abl_dir = OUTPUT_DIR / name
    abl_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = abl_dir / "training.log"

    settings = {**BASELINE, **overrides}
    label_smoothing = settings["label_smoothing"]
    weight_decay = settings["weight_decay"]
    value_weight = settings["value_weight"]

    log("=" * 60)
    log(f"ABLATION {name}: {overrides if overrides else 'control (no changes)'}")
    log(f"  label_smoothing={label_smoothing}, weight_decay={weight_decay}, "
        f"value_weight={value_weight}")

    # Load checkpoint
    ckpt_path, ckpt_label = find_checkpoint()
    log(f"  Loading checkpoint: {ckpt_path} ({ckpt_label})")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model(MODEL_CONFIG)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.train()

    base_step = ckpt.get("step", 0)
    log(f"  Base step: {base_step}")

    # Optimizer — fresh optimizer from checkpoint weights
    # (we don't restore optimizer state because weight_decay changes
    #  would make the old state incompatible)
    optimizer = AdamW(model.parameters(), lr=settings["lr"],
                      weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    # For control: restore optimizer state since no weight_decay change
    if name == "control" and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            log("  Restored optimizer state for control")
        except Exception:
            log("  Could not restore optimizer state, starting fresh")

    # Data loader (start from epoch 1)
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=batch_size,
        encoder_type="fused", device=DEVICE, seed=42)
    loader.set_epoch(1)  # Start from epoch 2 data ordering

    # LR schedule — continue cosine from base_step
    total_steps_exp149 = 315_933
    warmup_steps = 2000

    def get_lr(step):
        if step < warmup_steps:
            return settings["lr"] * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps_exp149 - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return settings["lr"] * (settings["min_lr_frac"] + (1.0 - settings["min_lr_frac"]) * cosine)

    # Set initial LR
    lr = get_lr(base_step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    log(f"  Starting LR: {lr:.2e}")

    # Initial eval
    if eval_data:
        torch.cuda.empty_cache()
        r0 = run_eval(model, eval_data, eval_tensors)
        log(f"  INITIAL: top1={r0['top1']:.2%} top3={r0['top3']:.2%} "
            f"top5={r0['top5']:.2%} val={r0['val']:.2%}")

    # Training loop
    eff_bs = batch_size * accum_steps
    accum_p = accum_v = 0.0
    accum_n = 0
    step = base_step
    local_step = 0
    t0 = time.time()
    best_acc = r0['top1'] if eval_data else 0.0
    eval_interval = 500

    for batch_input, move_targets, wdl_targets in loader:
        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            p_loss = F.cross_entropy(
                result["policy_logits"], move_targets,
                label_smoothing=label_smoothing)
            v_loss = F.cross_entropy(result["value_logits"], wdl_targets)
            loss = (p_loss + value_weight * v_loss) / accum_steps

        scaler.scale(loss).backward()
        accum_p += p_loss.item()
        accum_v += v_loss.item()
        accum_n += 1

        if accum_n >= accum_steps:
            scaler.unscale_(optimizer)
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1
            local_step += 1

            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if local_step % 100 == 0:
                avg_p = accum_p / accum_n
                avg_v = accum_v / accum_n
                elapsed = time.time() - t0
                pos_s = local_step * eff_bs / max(elapsed, 1)
                log(f"  [{name}] step {local_step}/{steps} (global {step}) | "
                    f"p={avg_p:.4f} v={avg_v:.4f} | lr={lr:.2e} gn={gn:.2f} | "
                    f"{pos_s:.0f} pos/s")

            if local_step % eval_interval == 0 and eval_data:
                torch.cuda.empty_cache()
                r = run_eval(model, eval_data, eval_tensors)
                tag = "**NEW BEST**" if r['top1'] > best_acc else ""
                log(f"  [{name}] EVAL step {local_step}: top1={r['top1']:.2%} "
                    f"top3={r['top3']:.2%} top5={r['top5']:.2%} val={r['val']:.2%} {tag}")
                if r['top1'] > best_acc:
                    best_acc = r['top1']
                model.train()

            accum_p = accum_v = 0.0
            accum_n = 0

            if local_step >= steps:
                break

    # Final eval
    final = None
    if eval_data:
        torch.cuda.empty_cache()
        final = run_eval(model, eval_data, eval_tensors)
        log(f"  [{name}] FINAL: top1={final['top1']:.2%} top3={final['top3']:.2%} "
            f"top5={final['top5']:.2%} val={final['val']:.2%}")

    elapsed = time.time() - t0
    log(f"  [{name}] Done in {timedelta(seconds=int(elapsed))}. "
        f"Best top1={best_acc:.2%}")

    # Save final checkpoint and summary
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": MODEL_CONFIG.to_dict(),
        "step": step,
        "local_steps": local_step,
        "best_acc": best_acc,
        "settings": settings,
        "overrides": overrides,
    }, abl_dir / "final.pt")

    summary = {
        "name": name,
        "overrides": overrides,
        "settings": settings,
        "base_step": base_step,
        "local_steps": local_step,
        "initial": {k: round(v, 4) for k, v in r0.items()} if eval_data else None,
        "final": {k: round(v, 4) for k, v in final.items()} if final else None,
        "best_top1": round(best_acc, 4),
        "elapsed_s": round(elapsed, 1),
    }
    with open(abl_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Clean up GPU
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True,
                    help="Ablation name (A-E, control, or 'all')")
    ap.add_argument("--steps", type=int, default=5000,
                    help="Optimizer steps per ablation (default: 5000)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval
    eval_data, eval_tensors = None, None
    if EVAL_PATH.exists():
        print(f"Loading eval from {EVAL_PATH}...")
        eval_data, eval_tensors = load_eval_data(EVAL_PATH)
        print(f"Eval positions: {len(eval_data)}")
    else:
        print(f"WARNING: No 20K eval found at {EVAL_PATH}, using 5K fallback")
        fallback = SHARD_DIR / "eval.pt"
        if fallback.exists():
            eval_data, eval_tensors = load_eval_data(fallback)
            print(f"Eval positions: {len(eval_data)}")

    if args.ablation == "all":
        # Run all ablations sequentially
        results = {}
        for name in ["control", "A", "B", "C", "D", "E"]:
            r = run_ablation(name, ABLATIONS[name], steps=args.steps,
                           eval_data=eval_data, eval_tensors=eval_tensors)
            results[name] = r

        # Print comparison table
        print("\n" + "=" * 80)
        print("ABLATION SWEEP RESULTS")
        print("=" * 80)
        print(f"{'Name':<10} {'Override':<30} {'Initial':>8} {'Final':>8} {'Delta':>8}")
        print("-" * 80)
        for name in ["control", "A", "B", "C", "D", "E"]:
            r = results[name]
            init_t1 = r["initial"]["top1"] if r["initial"] else 0
            final_t1 = r["final"]["top1"] if r["final"] else 0
            delta = final_t1 - init_t1
            override_str = str(ABLATIONS[name]) if ABLATIONS[name] else "baseline"
            print(f"{name:<10} {override_str:<30} {init_t1:>7.2%} {final_t1:>7.2%} "
                  f"{delta:>+7.2%}")
        print("=" * 80)

        # Save combined results
        with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        if args.ablation not in ABLATIONS:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ABLATIONS.keys())}")
            sys.exit(1)
        run_ablation(args.ablation, ABLATIONS[args.ablation], steps=args.steps,
                    eval_data=eval_data, eval_tensors=eval_tensors)


if __name__ == "__main__":
    main()
