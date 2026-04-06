"""exp153: Continue training from exp149 epoch_1 with horizontal flip augmentation.

Hypothesis:
  Horizontal flip augmentation doubles effective training data variety at zero
  compute cost. By introducing it AFTER epoch 1 (clean training), we avoid
  confusing early learning while getting the diversity benefit for epochs 2-3.

Expected gain: +1-3% top-1 from data variety → ~+50-100 ELO over non-augmented.

Baseline: exp149 epoch_1.pt checkpoint (clean, no augmentation)
Control:  exp149 continuing without augmentation (epochs 2-3)
Variable: hflip=True in ShardedChessLoader

Architecture: unchanged 204M (16L/1024d/16H)
Data: same 10.1M positions, but 50% random horizontal flips per position
LR: resume cosine schedule from exp149 epoch 1 position

Usage:
  python experiments/exp153_hflip_continue.py
  python experiments/exp153_hflip_continue.py --resume
  python experiments/exp153_hflip_continue.py --eval-only --checkpoint outputs/exp153_hflip/best_model.pt
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
OUTPUT_DIR = ROOT / "outputs" / "exp153_hflip"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
SOURCE_CKPT = ROOT / "outputs" / "exp149_scratch_204m" / "epoch_1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG


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

    eval_data = []
    surviving = []
    wdl = compute_wdl(raw["cp"], raw["mate"])

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
                "board": board,
                "move": move,
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
            idx = slice(i, i + n)

            batch_input = {
                "fused_ids": eval_tensors["fused_ids"][idx].to(DEVICE),
                "turn": eval_tensors["turn"][idx].to(DEVICE),
                "castling": eval_tensors["castling"][idx].to(DEVICE),
                "ep_file": eval_tensors["ep_file"][idx].to(DEVICE),
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
                topk = l.topk(min(5, l.shape[0])).indices.tolist()
                if true_idx in topk[:3]:
                    top3 += 1
                if true_idx in topk:
                    top5 += 1

                vp = F.softmax(value_logits[j], dim=-1)
                pred_class = vp.argmax().item()
                wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: wdl[k])
                if pred_class == true_class:
                    val_correct += 1

                total += 1

    return correct / max(total, 1), top3 / max(total, 1), top5 / max(total, 1), val_correct / max(total, 1)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2,
                    help="Number of additional epochs (default: 2, so epoch 2 and 3)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=0.5)
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
                    help="exp149 epoch_1 checkpoint to start from")
    args = ap.parse_args()

    global LOG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"

    log("=" * 60)
    log("exp153: Continue from exp149 epoch_1 with hflip augmentation")
    log(f"  device: {DEVICE}")
    log(f"  config: {MODEL_CONFIG}")

    # Build model
    model = build_model(MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params/1e6:.1f}M")

    resume_path = OUTPUT_DIR / "latest.pt"

    if args.eval_only:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.to(DEVICE)
        log(f"  Loaded checkpoint for eval: {ckpt_path}")

        eval_path = SHARD_DIR / "eval.pt"
        eval_data, eval_tensors = load_eval_data(eval_path)
        acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
        log(f"  EVAL: acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
        return

    # Load source checkpoint or resume
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Resumed: step={start_step}, epoch={start_epoch}, best_acc={best_acc:.2%}")
    else:
        source = Path(args.source_ckpt)
        if not source.exists():
            log(f"ERROR: Source checkpoint not found: {source}")
            log("  exp149 must complete epoch 1 first.")
            return
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        # Start from where exp149 epoch 1 left off
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 1)  # epoch 1 complete → start at epoch 1 (0-indexed for epoch 2)
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Loaded exp149 epoch_1: step={start_step}, best_acc={best_acc:.2%}")
        log(f"  Continuing with hflip augmentation for {args.epochs} more epochs")

    model.to(DEVICE)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
    elif not args.resume:
        # Load optimizer state from source checkpoint to continue LR schedule
        source = Path(args.source_ckpt)
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Data loader WITH hflip augmentation
    log(f"Loading shards from {SHARD_DIR} (hflip=True)...")
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=args.batch_size,
        encoder_type="fused", device=DEVICE, seed=42,
        hflip=True)
    total_pos = loader.total_positions
    steps_per_epoch = len(loader) // args.accum_steps
    # Total steps = exp149's 3 full epochs (use same schedule)
    total_steps_full = steps_per_epoch * 3
    eff_bs = args.batch_size * args.accum_steps

    log(f"  {total_pos:,} positions, bs={args.batch_size}, accum={args.accum_steps}, eff_bs={eff_bs}")
    log(f"  {steps_per_epoch:,} steps/epoch")
    log(f"  hflip=True — 50% random horizontal flips per position")

    # LR schedule — continue exp149's cosine from step position
    warmup_steps = 2000

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps_full - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    # Save config
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump({"model": MODEL_CONFIG.to_dict(), "training": {
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "eff_bs": eff_bs, "lr": args.lr, "epochs": args.epochs,
            "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            "warmup_steps": warmup_steps, "total_steps_full": total_steps_full,
            "init": "exp149_epoch_1",
            "augmentation": "hflip_50pct",
            "source_ckpt": str(args.source_ckpt),
        }}, f, indent=2)

    # Eval data
    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path)
        log(f"  Eval: {len(eval_data)} positions")

    # Initial eval on loaded checkpoint
    if eval_data:
        torch.cuda.empty_cache()
        acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
        log(f"  BASELINE (epoch_1): acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
        if not (args.resume and resume_path.exists()):
            best_acc = acc

    # Training
    log(f"\n{'='*60}")
    log(f"Training: epochs {start_epoch+1}-{start_epoch+args.epochs}, LR={args.lr}")
    log(f"  weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")
    log(f"  grad_clip={args.grad_clip}, betas=(0.9, 0.95)")
    log(f"  HFLIP AUGMENTATION ENABLED")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0

    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        loader.set_epoch(epoch)
        log(f"  Starting epoch {epoch+1} (hflip augmentation active)")

        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN:
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                log(f"Shutdown at step {step}")
                return

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing)
                v_loss = F.cross_entropy(result["value_logits"], wdl_targets)
                loss = (p_loss + args.value_weight * v_loss) / args.accum_steps

            scaler.scale(loss).backward()

            if torch.isnan(p_loss) or torch.isnan(v_loss):
                log(f"NaN detected at step {step}! Saving and aborting.")
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest_nan.pt")
                return

            accum_p_loss += p_loss.item()
            accum_v_loss += v_loss.item()
            accum_count += 1
            positions_seen += move_targets.shape[0]

            if accum_count >= args.accum_steps:
                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_norm_accum += gn.item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                step += 1

                # Update LR (continuing exp149's cosine schedule)
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Log
                if step % args.log_interval == 0:
                    avg_p = accum_p_loss / accum_count
                    avg_v = accum_v_loss / accum_count
                    avg_gn = grad_norm_accum / args.log_interval
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_delta = positions_seen - start_step * eff_bs
                    pos_s = pos_delta / max(elapsed, 1)
                    remaining = total_pos * end_epoch - positions_seen
                    eta = remaining / max(pos_s, 1)

                    log(f"  step {step:,}/{total_steps_full:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"pos={positions_seen:,} | "
                        f"ETA {timedelta(seconds=int(eta))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0

                # Save
                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")

                # Eval
                if step % args.eval_interval == 0 and eval_data:
                    torch.cuda.empty_cache()
                    acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
                    log(f"  EVAL step {step}: acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
                    if acc > best_acc:
                        best_acc = acc
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / "best_model.pt")
                        log(f"  ** New best! acc={best_acc:.2%}")
                    model.train()

                accum_count = 0
                accum_p_loss = 0.0
                accum_v_loss = 0.0

        # End of epoch
        log(f"\nEpoch {epoch+1} complete. positions_seen={positions_seen:,}")
        save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                       OUTPUT_DIR / f"epoch_{epoch+1}.pt")

        if eval_data:
            torch.cuda.empty_cache()
            acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
            log(f"  EPOCH {epoch+1} EVAL: acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                              OUTPUT_DIR / "best_model.pt")
                log(f"  ** New best! acc={best_acc:.2%}")
            model.train()

    log(f"\nTraining complete. Final best_acc={best_acc:.2%}")
    log(f"Best checkpoint: {OUTPUT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
