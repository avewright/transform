"""exp158: Phase-Weighted Policy Loss — Downsample Openings in Gradient Space.

MOTIVATION (from Ruoss et al. 2024 + data analysis):
  Our training data is 58% openings (≥28 pieces), 23% middlegame (14-27),
  19% endgame (<14). Ruoss et al. found uniform sampling >> natural frequency.
  
  In openings, many moves are roughly equivalent (14.6% quiet accuracy),
  making hard one-hot policy targets especially noisy. Middlegame positions
  have more decisive move distinctions and are harder — the model should
  spend more gradient there.

  This experiment reweights the loss per-position based on game phase
  (derived from piece count). No data loader changes needed.

Approach:
  - Compute piece_count from board_array per batch
  - Assign weights: opening (28+) → 0.5, middlegame (14-27) → 1.5, endgame (<14) → 1.2
  - Normalize weights so batch mean ≈ 1 (preserves effective LR)
  - Apply weights to policy + value loss per sample
  
  Effective training distribution becomes:
  opening 58% × 0.5 = 29 → 38% (was 58%)
  middlegame 23% × 1.5 = 34.5 → 44% (was 23%) 
  endgame 19% × 1.2 = 22.8 → 29% (was 19%)
  
  Much more balanced, especially for middlegame!

Architecture: Same 204M backbone, continues from epoch_1.pt
Expected gain: +30-60 Elo (from Ruoss's uniform sampling insight)

Usage:
  python experiments/exp158_phase_weighted.py
  python experiments/exp158_phase_weighted.py --max-steps 5000  # quick ablation
"""

import argparse
import gc
import json
import math
import os
import signal
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

from chess_transformer_factory import (
    build_model, ChessTransformerConfig, count_parameters,
)
from move_vocab import VOCAB_SIZE
from data_loader import ShardedChessLoader

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp158_phase_weighted"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
SOURCE_CKPT = ROOT / "outputs" / "exp149_scratch_204m" / "epoch_1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = ChessTransformerConfig()

# Phase weight constants
PHASE_WEIGHT_OPENING = 0.5    # ≥28 pieces — downsample heavily
PHASE_WEIGHT_MIDDLEGAME = 1.5 # 14-27 pieces — upsample
PHASE_WEIGHT_ENDGAME = 1.2    # <14 pieces — slight upsample


def _signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    log("SHUTDOWN requested. Saving checkpoint...")

signal.signal(signal.SIGINT, _signal_handler)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def compute_phase_weights(fused_ids):
    """Compute per-sample loss weights based on game phase (piece count).
    
    Args:
        fused_ids: (B, 64) board representation. Values > 0 are pieces.
    
    Returns:
        (B,) weight tensor, normalized so mean ≈ 1.
    """
    piece_count = (fused_ids > 0).sum(dim=1).float()  # (B,)
    
    weights = torch.ones_like(piece_count)
    weights[piece_count >= 28] = PHASE_WEIGHT_OPENING
    weights[(piece_count >= 14) & (piece_count < 28)] = PHASE_WEIGHT_MIDDLEGAME
    weights[piece_count < 14] = PHASE_WEIGHT_ENDGAME
    
    # Normalize so mean ≈ 1 (preserves effective learning rate)
    weights = weights / weights.mean()
    
    return weights


def save_checkpoint(model, optimizer, scaler, step, epoch, best_acc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": MODEL_CONFIG.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_acc": best_acc,
    }, tmp)
    os.replace(str(tmp), str(path))


def evaluate(model, eval_path, device):
    """Quick eval: top-1, top-3, value acc on eval set."""
    from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl
    
    data = torch.load(eval_path, map_location="cpu", weights_only=True)
    n = data["board_array"].shape[0]
    
    fused = board_array_to_fused(data["board_array"])
    turn = data["turn"].long()
    castling = data["castling"].long()
    ep_file = ep_square_to_file(data["ep_square"].long())
    move_idx = data["move_idx"].long()
    wdl = compute_wdl(data["cp"], data["mate"]).float()
    
    model.eval()
    bs = 128
    top1_correct = 0
    top3_correct = 0
    val_correct = 0
    
    # Phase-wise tracking
    phase_correct = {p: 0 for p in ["opening", "middlegame", "endgame"]}
    phase_total = {p: 0 for p in ["opening", "middlegame", "endgame"]}
    
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            inp = {
                "fused_ids": fused[i:j].to(device),
                "turn": turn[i:j].to(device),
                "castling": castling[i:j].to(device),
                "ep_file": ep_file[i:j].to(device),
            }
            with autocast("cuda", dtype=torch.float16):
                policy_logits, value_logits = model(inp)
            
            pred_move = policy_logits.argmax(dim=-1).cpu()
            top3_pred = policy_logits.topk(3, dim=-1).indices.cpu()
            pred_wdl = value_logits.argmax(dim=-1).cpu()
            true_wdl = wdl[i:j].argmax(dim=-1)
            targets = move_idx[i:j]
            
            top1_correct += (pred_move == targets).sum().item()
            top3_correct += (top3_pred == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            val_correct += (pred_wdl == true_wdl).sum().item()
            
            # Phase tracking
            piece_count = (fused[i:j] > 0).sum(dim=1)
            for k in range(j - i):
                pc = piece_count[k].item()
                phase = "opening" if pc >= 28 else "middlegame" if pc >= 14 else "endgame"
                phase_total[phase] += 1
                if pred_move[k] == targets[k]:
                    phase_correct[phase] += 1
    
    model.train()
    
    phase_acc = {}
    for p in phase_total:
        if phase_total[p] > 0:
            phase_acc[p] = phase_correct[p] / phase_total[p]
    
    return {
        "top1": top1_correct / n,
        "top3": top3_correct / n,
        "value": val_correct / n,
        "n": n,
        "phase_acc": phase_acc,
    }


def main():
    global LOG_PATH
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Max steps for quick ablation (e.g. 5000)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"
    
    log("=" * 60)
    log("exp158: Phase-Weighted Policy/Value Training")
    log(f"Phase weights: opening={PHASE_WEIGHT_OPENING}, "
        f"middlegame={PHASE_WEIGHT_MIDDLEGAME}, endgame={PHASE_WEIGHT_ENDGAME}")
    log("=" * 60)
    
    # Load source checkpoint
    ckpt_path = args.checkpoint or str(SOURCE_CKPT)
    log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    model = build_model(MODEL_CONFIG)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    log(f"Model: {count_parameters(model) / 1e6:.1f}M params on {DEVICE}")
    
    # Data loader with hflip (stacks with phase weighting)
    loader = ShardedChessLoader(
        SHARD_DIR, args.batch_size, encoder_type="fused",
        device=DEVICE, seed=42, hflip=True,
    )
    total_steps_per_epoch = len(loader) // args.accum_steps
    total_steps = total_steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    log(f"Training: {total_steps:,} steps ({args.epochs} epochs)")
    
    # Cosine LR schedule
    min_lr = args.lr * args.min_lr_frac
    
    def get_lr(step):
        if step < 1000:  # warmup
            return args.lr * step / 1000
        progress = (step - 1000) / max(1, total_steps - 1000)
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")
    
    # Resume if requested
    start_step = 0
    start_epoch = 0
    best_acc = 0
    if args.resume:
        resume_path = OUTPUT_DIR / "latest.pt"
        if resume_path.exists():
            rckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(rckpt["model_state_dict"])
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            scaler.load_state_dict(rckpt["scaler_state_dict"])
            start_step = rckpt.get("step", 0)
            start_epoch = rckpt.get("epoch", 0)
            best_acc = rckpt.get("best_acc", 0)
            log(f"Resumed from step {start_step}")
            del rckpt
    
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.eval_only:
        eval_path = SHARD_DIR / "eval.pt"
        result = evaluate(model, eval_path, DEVICE)
        log(f"Eval: top1={result['top1']:.4f} top3={result['top3']:.4f} "
            f"val={result['value']:.4f}")
        for phase, acc in result.get("phase_acc", {}).items():
            log(f"  {phase}: {acc:.4f}")
        return
    
    # Setup loss
    policy_criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing, reduction='none')
    value_criterion = nn.CrossEntropyLoss(reduction='none')
    
    step = start_step
    optimizer.zero_grad()
    eval_path = SHARD_DIR / "eval.pt"
    t0 = time.time()
    positions_done = 0
    
    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch)
        
        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN or (args.max_steps and step >= args.max_steps):
                save_checkpoint(model, optimizer, scaler, step, epoch,
                               best_acc, OUTPUT_DIR / "latest.pt")
                log(f"Stopped at step {step}")
                return
            
            with autocast("cuda", dtype=torch.float16):
                policy_logits, value_logits = model(batch_input)
                
                # Per-sample losses
                p_loss_per = policy_criterion(policy_logits, move_targets)
                v_loss_per = value_criterion(value_logits, wdl_targets)
                
                # Phase weights from piece count
                phase_w = compute_phase_weights(batch_input["fused_ids"])
                
                # Weighted losses
                p_loss = (p_loss_per * phase_w).mean()
                v_loss = (v_loss_per * phase_w).mean()
                
                loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
            
            scaler.scale(loss).backward()
            positions_done += move_targets.shape[0]
            
            if (step + 1) % args.accum_steps == 0 or step == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update LR
                real_step = step // args.accum_steps
                lr = get_lr(real_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            step += 1
            real_step = step // args.accum_steps
            
            # Log
            if real_step % 25 == 0 and step % args.accum_steps == 0:
                elapsed = time.time() - t0
                pos_per_sec = positions_done / elapsed if elapsed > 0 else 0
                eta = timedelta(seconds=(total_steps * args.accum_steps - step) / max(1, step / elapsed))
                log(f"  step {real_step:,}/{total_steps:,} | "
                    f"p={p_loss.item() * args.accum_steps:.4f} "
                    f"v={v_loss.item() * args.accum_steps:.4f} | "
                    f"lr={lr:.2e} | {pos_per_sec:.0f} pos/s | ETA {eta}")
            
            # Save
            if real_step % 1000 == 0 and step % args.accum_steps == 0 and real_step > 0:
                save_checkpoint(model, optimizer, scaler, step, epoch,
                               best_acc, OUTPUT_DIR / "latest.pt")
            
            # Eval
            if real_step % 1000 == 0 and step % args.accum_steps == 0:
                result = evaluate(model, eval_path, DEVICE)
                log(f"  EVAL step {real_step}: top1={result['top1']:.4f} "
                    f"top3={result['top3']:.4f} val={result['value']:.4f}")
                for phase, acc in result.get("phase_acc", {}).items():
                    log(f"    {phase}: {acc:.4f}")
                
                if result["top1"] > best_acc:
                    best_acc = result["top1"]
                    save_checkpoint(model, optimizer, scaler, step, epoch,
                                   best_acc, OUTPUT_DIR / "best_model.pt")
                    log(f"  ★ New best: {best_acc:.4f}")
        
        # End of epoch
        save_checkpoint(model, optimizer, scaler, step, epoch,
                       best_acc, OUTPUT_DIR / f"epoch_{epoch + 1}.pt")
        log(f"Epoch {epoch + 1} complete, step {real_step}")
    
    log("Training complete!")


if __name__ == "__main__":
    main()
