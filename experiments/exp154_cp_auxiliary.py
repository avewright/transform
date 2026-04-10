"""exp154: Continue from exp149 epoch_1 with centipawn auxiliary loss.

Hypothesis:
  The value head is the primary bottleneck for MCTS ELO (verified: 1506 ELO vs
  1845 baseline at 100 sims, despite higher policy accuracy). The centipawn data
  in training shards is unused — adding a cp regression head provides dense,
  continuous gradient signal about positional quality, helping both the value head
  AND shared representations learn better positional features.

  The cp auxiliary loss encourages the encoder to represent fine-grained
  evaluation differences (e.g., +0.5 vs +1.0 pawns), which the coarse WDL
  labels cannot provide.

Design:
  - Continue from exp149 epoch_1 checkpoint (same as exp153)
  - Add a cp_head: Linear(1024, 512) + ReLU + Linear(512, 1)
  - cp_loss: Huber loss on tanh-scaled centipawns (cp/1000, clipped to [-3, 3])
  - Total: p_loss + 0.5*v_loss + 0.1*cp_loss
  - hflip=True for data augmentation
  - cp_head initialized randomly, other weights from checkpoint

Comparison:
  exp149 epoch_2-3 (no aux, no hflip) vs exp153 (hflip only) vs exp154 (hflip + cp aux)

Usage:
  python experiments/exp154_cp_auxiliary.py
  python experiments/exp154_cp_auxiliary.py --resume
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

from chess_transformer_factory import build_model, DEFAULT_200M_CONFIG
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp154_cp_aux"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
SOURCE_CKPT = ROOT / "outputs" / "exp149_scratch_204m" / "epoch_1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG


class CpHead(nn.Module):
    """Centipawn regression head from CLS hidden state."""
    def __init__(self, hidden_dim=1024, inner_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, 1)
    
    def forward(self, cls_hidden):
        return self.fc2(F.relu(self.fc1(cls_hidden))).squeeze(-1)


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


def save_checkpoint(model, cp_head, optimizer, scaler, step, epoch, best_acc, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "cp_head_state_dict": cp_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_acc": best_acc,
    }, path)
    log(f"  Saved checkpoint: {path.name} (step {step}, acc={best_acc:.2f}%)")


def evaluate(model, eval_data, device, max_positions=5000):
    model.eval()
    correct = 0
    top3 = 0
    value_correct = 0
    total = 0
    n = min(len(eval_data["move_idx"]), max_positions)
    bs = 64

    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_input = {
                "fused_ids": eval_data["fused_ids"][start:end].to(device),
                "turn": eval_data["turn"][start:end].to(device),
                "castling": eval_data["castling"][start:end].to(device),
                "ep_file": eval_data["ep_file"][start:end].to(device),
            }
            batch_moves = eval_data["move_idx"][start:end].to(device)
            batch_wdl = eval_data["wdl"][start:end].to(device)
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"]
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_moves).sum().item()
            _, top3_idx = logits.topk(3, dim=-1)
            top3 += (top3_idx == batch_moves.unsqueeze(-1)).any(dim=-1).sum().item()
            v_pred = result["value_logits"].argmax(dim=-1)
            value_correct += (v_pred == batch_wdl).sum().item()
            total += end - start

    model.train()
    return correct / total, top3 / total, value_correct / total


def prepare_eval_data(shard_dir, n_eval=5000, seed=42):
    eval_path = shard_dir / "eval.pt"
    data = torch.load(eval_path, map_location="cpu", weights_only=True)
    n = min(n_eval, data["board_array"].shape[0])
    ba = data["board_array"][:n]
    fused = board_array_to_fused(ba)
    ep_file = ep_square_to_file(data["ep_square"][:n].long())
    wdl = compute_wdl(data["cp"][:n], data["mate"][:n])
    wdl_class = wdl.argmax(dim=-1)
    return {
        "fused_ids": fused,
        "turn": data["turn"][:n].long(),
        "castling": data["castling"][:n].long(),
        "ep_file": ep_file,
        "move_idx": data["move_idx"][:n].long(),
        "wdl": wdl_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2,
                    help="Number of continuation epochs (epochs 2-3)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--cp-weight", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--source-ckpt", type=str, default=str(SOURCE_CKPT),
                    help="exp149 epoch_1 checkpoint to continue from")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = OUTPUT_DIR / "training.log"

    log("=" * 60)
    log("exp154: cp auxiliary loss + hflip, continue from exp149 epoch_1")
    log(f"  device: {DEVICE}")
    log(f"  config: {MODEL_CONFIG}")

    model = build_model(MODEL_CONFIG)
    cp_head = CpHead(hidden_dim=MODEL_CONFIG.hidden_dim,
                     inner_dim=MODEL_CONFIG.value_hidden)
    params = sum(p.numel() for p in model.parameters())
    cp_params = sum(p.numel() for p in cp_head.parameters())
    log(f"  model params: {params/1e6:.1f}M, cp_head params: {cp_params/1e6:.3f}M")

    if args.eval_only:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model = model.to(DEVICE).eval()
        eval_data = prepare_eval_data(SHARD_DIR)
        log(f"  Loaded checkpoint for eval: {ckpt_path}")
        acc, t3, val = evaluate(model, eval_data, DEVICE, max_positions=20000)
        log(f"  EVAL: acc={acc*100:.2f}% top3={t3*100:.2f}% val={val*100:.2f}%")
        return

    # Resume or load from source
    resume_path = OUTPUT_DIR / "latest.pt"
    start_step = 0
    start_epoch = 0
    best_acc = 0.0

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        if "cp_head_state_dict" in ckpt:
            cp_head.load_state_dict(ckpt["cp_head_state_dict"])
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Resumed: step={start_step}, epoch={start_epoch}, best_acc={best_acc:.2f}%")
    else:
        # Load from exp149 epoch_1
        src = args.source_ckpt
        if not Path(src).exists():
            log(f"  ERROR: Source checkpoint not found: {src}")
            log(f"  Run exp149 to completion (epoch 1) first!")
            return
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        # cp_head stays random (new head)
        log(f"  Loaded source: {src}")
        log(f"  cp_head initialized randomly (new auxiliary head)")

    model = model.to(DEVICE)
    cp_head = cp_head.to(DEVICE)

    # Combine parameters for optimizer
    all_params = list(model.parameters()) + list(cp_head.parameters())
    optimizer = AdamW(all_params, lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))

    if args.resume and resume_path.exists() and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = GradScaler('cuda')
    if args.resume and resume_path.exists() and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=args.batch_size,
        encoder_type="fused", device=DEVICE, seed=42,
        hflip=True, include_cp=True,
    )
    steps_per_epoch = len(loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    eff_bs = args.batch_size * args.accum_steps
    total_pos = len(loader) * args.batch_size

    # Cosine LR schedule continuing from exp149 epoch_1 position
    # exp149 had 3 epochs total. epoch_1 = 1/3 done. We continue for 2/3.
    # So our schedule should decay from the epoch_1 LR position.
    total_steps_full = steps_per_epoch * 3  # full 3-epoch schedule
    epoch1_steps = steps_per_epoch  # steps already done in epoch 0

    log(f"  {total_pos:,} positions, bs={args.batch_size}, accum={args.accum_steps}, eff_bs={eff_bs}")
    log(f"  {steps_per_epoch:,} steps/epoch, {total_steps:,} total (continuing epochs 2-3)")

    eval_data = prepare_eval_data(SHARD_DIR)
    log(f"  Eval: {len(eval_data['move_idx'])} positions")

    def get_lr(step):
        # Continue cosine schedule from epoch_1 position
        effective_step = epoch1_steps + step
        warmup_steps = 2000  # already past warmup
        if effective_step < warmup_steps:
            return args.lr * (effective_step + 1) / max(warmup_steps, 1)
        progress = (effective_step - warmup_steps) / max(total_steps_full - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    config_dict = {
        "experiment": "exp154_cp_auxiliary",
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "eff_bs": eff_bs, "lr": args.lr, "epochs": args.epochs,
        "value_weight": args.value_weight, "cp_weight": args.cp_weight,
        "hflip": True, "source": str(SOURCE_CKPT),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epochs (continuing from exp149 epoch_1)")
    log(f"  value_weight={args.value_weight}, cp_weight={args.cp_weight}")
    log(f"  hflip=True, label_smoothing={args.label_smoothing}")
    log(f"  weight_decay={args.weight_decay}, grad_clip={args.grad_clip}")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_cp_loss = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0

    model.train()
    cp_head.train()

    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch + 1)  # +1 because these are "epoch 2" and "epoch 3"

        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN:
                save_checkpoint(model, cp_head, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                log(f"Shutdown at step {step}")
                return

            # Get cp targets from the batch
            cp_targets = batch_input.pop("cp", None)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing)
                v_loss = F.cross_entropy(result["value_logits"], wdl_targets)
                
                loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
                
                # Centipawn auxiliary loss
                if cp_targets is not None:
                    cls_hidden = result["cls_hidden"]
                    cp_pred = cp_head(cls_hidden)
                    # Scale cp to [-3, 3] range
                    cp_scaled = cp_targets / 1000.0
                    cp_scaled = cp_scaled.clamp(-3.0, 3.0)
                    cp_l = F.huber_loss(cp_pred, cp_scaled, delta=1.0)
                    loss = loss + (args.cp_weight * cp_l) / args.accum_steps
                    accum_cp_loss += cp_l.item()

            scaler.scale(loss).backward()

            if torch.isnan(p_loss) or torch.isnan(v_loss):
                log(f"NaN detected at step {step}! Saving and aborting.")
                save_checkpoint(model, cp_head, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest_nan.pt")
                return

            accum_p_loss += p_loss.item()
            accum_v_loss += v_loss.item()
            accum_count += 1
            positions_seen += move_targets.shape[0]

            if accum_count >= args.accum_steps:
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                grad_norm_accum += gn.item()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # Update LR
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                if step % args.log_interval == 0:
                    avg_p = accum_p_loss / accum_count
                    avg_v = accum_v_loss / accum_count
                    avg_cp = accum_cp_loss / max(accum_count, 1)
                    avg_gn = grad_norm_accum / (step % args.log_interval or args.log_interval)
                    elapsed = time.time() - t0
                    rate = positions_seen / elapsed if elapsed > 0 else 0
                    remaining = (total_steps - step) * eff_bs / rate if rate > 0 else 0
                    eta = timedelta(seconds=int(remaining))
                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} cp={avg_cp:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | "
                        f"{rate:.0f} pos/s | pos={positions_seen:,} | ETA {eta}")
                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_cp_loss = 0.0
                    accum_count = 0
                    grad_norm_accum = 0.0

                if step % args.save_interval == 0:
                    save_checkpoint(model, cp_head, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")

                if step % args.eval_interval == 0:
                    acc, t3, val = evaluate(model, eval_data, DEVICE)
                    log(f"  EVAL step {step}: acc={acc*100:.2f}% "
                        f"top3={t3*100:.2f}% val={val*100:.2f}%")
                    if acc > best_acc / 100:
                        best_acc = acc * 100
                        log(f"  ** New best! acc={best_acc:.2f}%")
                        save_checkpoint(model, cp_head, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / "best_model.pt")

        # End of epoch
        epoch_path = OUTPUT_DIR / f"epoch_{epoch+2}.pt"  # epoch 2, 3
        save_checkpoint(model, cp_head, optimizer, scaler, step, epoch + 1, best_acc, epoch_path)
        log(f"Epoch {epoch+2} complete, saved {epoch_path.name}")

    log(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
