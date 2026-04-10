"""exp156: CP-aware position weighting — upweight balanced/quiet positions.

Hypothesis:
  The training data is 58% opening and 80% quiet moves, yet quiet moves at
  14.6% accuracy are the primary bottleneck. Positions with |cp| near 0 have
  many roughly-equal candidate moves, so the hard one-hot target is especially
  noisy. By upweighting these balanced positions (where the model must learn
  finer distinctions), we can improve quiet-move accuracy without changing the
  architecture.

  Weight formula: w(cp) = 1 + alpha * exp(-|cp| / tau)
  - |cp| = 0 → weight ≈ 1 + alpha (maximum upweight for balanced positions)
  - |cp| = 500 → weight drops toward 1 (winning positions are easier, less useful)
  - alpha=1.0, tau=200 gives 2.0× weight at cp=0, 1.08× at cp=500

Expected gain: +0.5-2% quiet-move accuracy → better MCTS performance on typical positions.

Baseline: exp149 continuing from epoch_1 (uniform weighting, label_smoothing=0.1)
Control:  exp153 (hflip, uniform, same data)
Variable: Per-sample weight w(cp) applied to policy loss

Quick test: 5K steps from epoch_1 checkpoint (~35 min on RTX 4060).
If improved: extend to full epoch.

Architecture: unchanged 204M (16L/1024d/16H)
Data: same 10.1M positions, hflip=True, with cp-based sample weights
LR: resume cosine schedule from exp149 epoch 1 position

Usage:
  python experiments/exp156_balanced_weight.py
  python experiments/exp156_balanced_weight.py --alpha 0.5 --tau 150
  python experiments/exp156_balanced_weight.py --resume
  python experiments/exp156_balanced_weight.py --eval-only --checkpoint outputs/exp156_balanced_weight/best_model.pt
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
OUTPUT_DIR = ROOT / "outputs" / "exp156_balanced_weight"
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


PIECE_CHARS = ".PNBRQKpnbrqk"


def _board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
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


def compute_cp_weights(cp_tensor, mate_tensor, alpha, tau):
    """Compute per-sample weights: upweight balanced positions, downweight decisive.

    w(cp) = 1 + alpha * exp(-|cp| / tau)
    - |cp|=0: weight = 1+alpha (balanced position, maximum upweight)
    - |cp|→∞: weight → 1 (decisive position, uniform weight)

    Mate positions get weight = 1.0 (tactical, already well-learned at 41.8%).
    """
    cp = cp_tensor.float()
    has_mate = mate_tensor != 0

    weights = 1.0 + alpha * torch.exp(-cp.abs() / tau)

    # Mate positions: no upweight (already easy/tactical)
    weights[has_mate] = 1.0

    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Max CP weight boost for balanced positions (default: 1.0 = 2x weight at cp=0)")
    ap.add_argument("--tau", type=float, default=200.0,
                    help="CP decay scale in centipawns (default: 200)")
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Max steps for quick ablation (default: full training)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--source-ckpt", type=str, default=str(SOURCE_CKPT))
    args = ap.parse_args()

    global LOG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"

    log("=" * 60)
    log(f"exp156: CP-weighted policy loss (alpha={args.alpha}, tau={args.tau})")
    log(f"  device: {DEVICE}")
    log(f"  config: {MODEL_CONFIG}")

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
        eval_data, eval_tensors = load_eval_data(SHARD_DIR / "eval.pt")
        acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
        log(f"  EVAL: acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
        return

    # Load source or resume
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
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 1)
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Loaded exp149 epoch_1: step={start_step}, best_acc={best_acc:.2%}")

    model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Data loader WITH hflip + include_cp for weighting
    log(f"Loading shards from {SHARD_DIR} (hflip=True, include_cp=True)...")
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=args.batch_size,
        encoder_type="fused", device=DEVICE, seed=42,
        hflip=True, include_cp=True)
    total_pos = loader.total_positions
    steps_per_epoch = len(loader) // args.accum_steps
    total_steps_full = steps_per_epoch * 3
    eff_bs = args.batch_size * args.accum_steps

    log(f"  {total_pos:,} positions, bs={args.batch_size}, accum={args.accum_steps}, eff_bs={eff_bs}")
    log(f"  {steps_per_epoch:,} steps/epoch")
    log(f"  CP weighting: alpha={args.alpha}, tau={args.tau}")
    log(f"  Weight at |cp|=0: {1+args.alpha:.1f}x, |cp|=100: {1+args.alpha*math.exp(-100/args.tau):.2f}x, |cp|=500: {1+args.alpha*math.exp(-500/args.tau):.2f}x")

    warmup_steps = 2000

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps_full - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    # Save config
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump({"model": MODEL_CONFIG.to_dict(), "training": {
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "eff_bs": eff_bs, "lr": args.lr, "value_weight": args.value_weight,
            "epochs": args.epochs, "alpha": args.alpha, "tau": args.tau,
            "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            "warmup_steps": warmup_steps, "source_ckpt": str(args.source_ckpt),
            "augmentation": "hflip_50pct",
            "variable": "cp_weighted_policy_loss",
        }}, f, indent=2)

    # Eval
    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path)
        log(f"  Eval: {len(eval_data)} positions")

    if eval_data:
        torch.cuda.empty_cache()
        acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
        log(f"  BASELINE: acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
        if not (args.resume and resume_path.exists()):
            best_acc = acc

    log(f"\n{'='*60}")
    log(f"Training: epochs {start_epoch+1}-{start_epoch+args.epochs}")
    log(f"  LR={args.lr}, value_weight={args.value_weight}")
    log(f"  CP weight: alpha={args.alpha}, tau={args.tau}")
    log(f"  HFLIP=True")
    if args.max_steps:
        log(f"  MAX STEPS: {args.max_steps} (quick ablation)")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_avg_weight = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0

    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        loader.set_epoch(epoch)

        for batch_data in loader:
            if SHUTDOWN:
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                log(f"Shutdown at step {step}")
                return

            if args.max_steps and step >= start_step + args.max_steps:
                log(f"Reached max_steps={args.max_steps}. Stopping.")
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                # Final eval
                if eval_data:
                    acc, top3, top5, val_acc = run_eval(model, eval_data, eval_tensors)
                    log(f"  FINAL EVAL: acc={acc:.2%} top3={top3:.2%} top5={top5:.2%} val={val_acc:.2%}")
                return

            # Unpack — loader yields (batch_input, move_targets, wdl_targets)
            # when include_cp=True, cp is in batch_input["cp"]
            batch_input, move_targets, wdl_targets = batch_data
            cp_batch = batch_input.pop("cp", None)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)

                # Standard per-sample cross entropy (no reduction)
                p_loss_per_sample = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing, reduction='none')

                # Apply CP-based weights to policy loss
                if cp_batch is not None:
                    mate_batch = torch.zeros_like(cp_batch)  # mate info not in loader
                    weights = compute_cp_weights(
                        cp_batch, mate_batch, args.alpha, args.tau).to(DEVICE)
                    # Normalize weights so mean ≈ 1 (preserve effective LR)
                    weights = weights / weights.mean()
                    p_loss = (p_loss_per_sample * weights).mean()
                    avg_w = weights.mean().item()
                else:
                    p_loss = p_loss_per_sample.mean()
                    avg_w = 1.0

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
            accum_avg_weight += avg_w
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

                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if step % args.log_interval == 0:
                    avg_p = accum_p_loss / accum_count
                    avg_v = accum_v_loss / accum_count
                    avg_gn = grad_norm_accum / args.log_interval
                    avg_wt = accum_avg_weight / accum_count
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_delta = positions_seen - start_step * eff_bs
                    pos_s = pos_delta / max(elapsed, 1)
                    remaining = total_pos * end_epoch - positions_seen
                    eta = remaining / max(pos_s, 1)

                    log(f"  step {step:,}/{total_steps_full:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} w={avg_wt:.2f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"pos={positions_seen:,} | "
                        f"ETA {timedelta(seconds=int(eta))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_avg_weight = 0.0
                    accum_count = 0

                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")

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

        # End of epoch
        save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                       OUTPUT_DIR / f"epoch_{epoch + 1}.pt")
        log(f"  Epoch {epoch + 1} complete. best_acc={best_acc:.2%}")

    log("Training complete.")


if __name__ == "__main__":
    if "--resume" not in sys.argv:
        sys.argv.append("--resume")
    main()
