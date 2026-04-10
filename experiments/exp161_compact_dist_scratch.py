"""exp161: 204M FROM SCRATCH with Compact Vocab (1968) + Distributional Value (128-bin HL-Gauss).

Combines the two highest-impact improvements identified in the research log:
  1. Compact move vocabulary: 1968 geometrically reachable moves (vs legacy 5504)
     - Policy head 3× smaller, no wasted logits on impossible moves
     - Label smoothing no longer leaks probability to impossible moves
  2. Distributional value head: 128-bin HL-Gauss (vs 3-class WDL)
     - Farebrother 2024: 70% improvement in chess puzzle accuracy
     - Ruoss 2024: used at 2895 Elo
     - Better gradients, noise robustness, representation quality

Based on exp149 (from scratch 204M) training framework.

Hardware: RTX 4060 8GB
  - bs=24, accum=4 (eff_bs=96) at ~90-100 pos/s expected
  - 10M × 3 epochs ÷ 95 pos/s ≈ 88 hours (~3.7 days)

Usage:
  python experiments/exp161_compact_dist_scratch.py --epochs 3
  python experiments/exp161_compact_dist_scratch.py --resume
  python experiments/exp161_compact_dist_scratch.py --eval-only
  python experiments/exp161_compact_dist_scratch.py --max-steps 5000  # quick ablation
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
# CRITICAL: Set compact vocab BEFORE any imports that touch move_vocab
os.environ['MOVE_VOCAB_VERSION'] = 'compact'

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
from move_vocab import (
    VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask,
    LEGACY_UCI_TO_IDX, COMPACT_UCI_TO_IDX, legacy_to_compact_map,
)
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp161_compact_dist"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG

# ── Distributional value constants ─────────────────────────────────────
N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS  # ~0.006, per Farebrother et al. 2024


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


# ── Move index remapping (legacy shards → compact vocab) ──────────────

def build_remap_tensor():
    """Build a tensor mapping legacy move indices to compact indices.
    
    Returns tensor of shape (LEGACY_VOCAB_SIZE,) where t[legacy_idx] = compact_idx
    or -1 if the move doesn't exist in compact vocab.
    """
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t


# ── HL-Gauss distributional value ─────────────────────────────────────

def cp_to_win_percent(cp, mate):
    """Convert cp/mate to win percentage [0, 1]."""
    N = cp.shape[0]
    win_pct = torch.zeros(N, dtype=torch.float32, device=cp.device)
    mate_pos = mate > 0
    mate_neg = mate < 0
    win_pct[mate_pos] = 1.0
    win_pct[mate_neg] = 0.0
    no_mate = mate == 0
    if no_mate.any():
        k = 1.0 / 111.7
        win_pct[no_mate] = torch.sigmoid(k * cp[no_mate].float())
    return win_pct


def hl_gauss_loss(logits, win_pct, n_bins=N_VALUE_BINS, sigma=SIGMA_HL_GAUSS):
    """HL-Gauss: cross-entropy with Gaussian-smoothed categorical targets.
    
    Args:
        logits: (B, K) raw logits from distributional value head
        win_pct: (B,) win percentages in [0, 1]
    Returns:
        scalar mean loss
    """
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)  # (B, K)
    log_probs_target = -0.5 * (diff / sigma) ** 2
    targets = F.softmax(log_probs_target, dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def value_logits_to_win_pct(logits, n_bins=N_VALUE_BINS):
    """Convert distributional value logits to expected win percentage."""
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    probs = F.softmax(logits.float(), dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ── Model construction ─────────────────────────────────────────────────

def build_compact_dist_model(config):
    """Build model with compact vocab policy head + 128-bin distributional value head."""
    model = build_model(config)
    # Replace 3-class WDL value head → 128-bin distributional
    old_head = model.value_head
    assert isinstance(old_head, nn.Sequential) and len(old_head) == 3
    hidden_dim = old_head[0].out_features  # 512
    model.value_head = nn.Sequential(
        nn.Linear(config.hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, N_VALUE_BINS),
    )
    # Xavier init for the new head
    for layer in model.value_head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


# ── Eval ───────────────────────────────────────────────────────────────

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


def load_eval_data(eval_path, remap_tensor):
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)

    eval_data = []
    surviving = []

    for i in range(raw["board_array"].shape[0]):
        try:
            fen = _board_array_to_fen(
                raw["board_array"][i], raw["turn"][i],
                raw["castling"][i], raw["ep_square"][i])
            board = chess.Board(fen)
            # Remap legacy move index to compact
            legacy_idx = raw["move_idx"][i].item()
            compact_idx = remap_tensor[legacy_idx].item()
            if compact_idx < 0:
                continue  # shouldn't happen for legal moves
            uci = IDX_TO_UCI[compact_idx]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            # Store cp/mate for value eval
            cp_val = raw["cp"][i].item() if "cp" in raw else 0
            mate_val = raw["mate"][i].item() if "mate" in raw else 0
            eval_data.append({
                "board": board,
                "move": move,
                "compact_idx": compact_idx,
                "cp": cp_val,
                "mate": mate_val,
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
    correct = top3 = total = 0
    total_value_mae = 0.0
    # Phase-stratified tracking
    phase_correct = [0, 0, 0]  # opening, middlegame, endgame
    phase_total = [0, 0, 0]

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

            # Determine phase from piece count
            ba = batch_input["fused_ids"]
            non_king = ((ba >= 1) & (ba <= 5)) | ((ba >= 7) & (ba <= 11))
            piece_counts = non_king.sum(dim=1)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)

            logits = result["policy_logits"].float()
            value_logits = result["value_logits"].float()
            pred_win_pct = value_logits_to_win_pct(value_logits)

            for j, d in enumerate(chunk):
                board, true_move = d["board"], d["move"]
                l = logits[j].clone()
                mask = legal_move_mask(board).to(DEVICE)
                l[~mask] = float("-inf")

                true_idx = d["compact_idx"]
                hit = l.argmax().item() == true_idx
                if hit:
                    correct += 1
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk:
                    top3 += 1

                # Phase classification
                pc = piece_counts[j].item()
                phase = 0 if pc >= 14 else (2 if pc < 6 else 1)
                phase_total[phase] += 1
                if hit:
                    phase_correct[phase] += 1

                # Value MAE: predicted win% vs true win%
                cp_t = torch.tensor([d["cp"]], dtype=torch.float32)
                mate_t = torch.tensor([d["mate"]], dtype=torch.long)
                true_wp = cp_to_win_percent(cp_t, mate_t).item()
                pred_wp = pred_win_pct[j].item()
                total_value_mae += abs(pred_wp - true_wp)

                total += 1

    top1_acc = correct / max(total, 1)
    top3_acc = top3 / max(total, 1)
    value_mae = total_value_mae / max(total, 1)

    # Log per-phase accuracy
    phase_names = ["open", "mid", "end"]
    phase_strs = []
    for p in range(3):
        if phase_total[p] > 0:
            pa = phase_correct[p] / phase_total[p]
            phase_strs.append(f"{phase_names[p]}={pa:.1%}({phase_total[p]})")
    if phase_strs:
        log(f"    phase: {' '.join(phase_strs)}")

    return top1_acc, top3_acc, value_mae


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
        "vocab_version": "compact",
        "n_value_bins": N_VALUE_BINS,
    }, tmp)
    os.replace(str(tmp), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=1.0)  # higher for distributional
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)  # less smoothing with compact vocab
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0=full)")
    ap.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile for ~25% speedup")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    global LOG_PATH
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"

    log("=" * 60)
    log("exp161: 204M COMPACT VOCAB + DISTRIBUTIONAL VALUE from scratch")
    log(f"  device: {DEVICE}")
    log(f"  vocab: {VOCAB_SIZE} moves (compact)")
    log(f"  value: {N_VALUE_BINS}-bin HL-Gauss (σ={SIGMA_HL_GAUSS:.4f})")
    log(f"  config: {MODEL_CONFIG}")

    # Build move index remap tensor (legacy → compact)
    remap_tensor = build_remap_tensor()
    log(f"  remap: {(remap_tensor >= 0).sum().item()} legacy→compact mappings")

    # Build model from random init
    model = build_compact_dist_model(MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params/1e6:.1f}M")

    start_step = 0
    start_epoch = 0
    best_acc = 0.0
    resume_path = OUTPUT_DIR / "latest.pt"

    if args.eval_only:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.to(DEVICE)
        eval_path = SHARD_DIR / "eval.pt"
        eval_data, eval_tensors = load_eval_data(eval_path, remap_tensor)
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  EVAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        return

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
        log("  Training from RANDOM INITIALIZATION")

    model.to(DEVICE)
    if args.compile:
        log("  Compiling model with torch.compile (inductor backend)...")
        model = torch.compile(model)
        log("  Compilation triggered (first forward will be slow)")
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

    # Data loader (with cp/mate passthrough for HL-Gauss targets)
    log(f"Loading shards from {SHARD_DIR}...")
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=args.batch_size,
        encoder_type="fused", device=DEVICE, seed=42,
        include_cp=True, include_mate=True)
    total_pos = loader.total_positions
    steps_per_epoch = len(loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    eff_bs = args.batch_size * args.accum_steps

    log(f"  {total_pos:,} positions, bs={args.batch_size}, accum={args.accum_steps}, eff_bs={eff_bs}")
    log(f"  {steps_per_epoch:,} steps/epoch, {total_steps:,} total")

    # LR schedule
    warmup_steps = min(2000, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    # Save config
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump({"model": MODEL_CONFIG.to_dict(), "training": {
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "eff_bs": eff_bs, "lr": args.lr, "epochs": args.epochs,
            "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "init": "random", "vocab": "compact", "vocab_size": VOCAB_SIZE,
            "n_value_bins": N_VALUE_BINS, "sigma_hl_gauss": SIGMA_HL_GAUSS,
            "value_weight": args.value_weight,
        }}, f, indent=2)

    # Eval data
    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path, remap_tensor)
        log(f"  Eval: {len(eval_data)} positions")

    # Move remap to device
    remap_device = remap_tensor.to(DEVICE)

    # Initial eval (random init baseline)
    if eval_data and start_step == 0:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  RANDOM INIT: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        best_acc = acc

    # Training
    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epochs, LR={args.lr}, warmup={warmup_steps}")
    log(f"  value_weight={args.value_weight}, label_smoothing={args.label_smoothing}")
    log(f"  {N_VALUE_BINS}-bin HL-Gauss value, {VOCAB_SIZE}-move compact policy")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0
    skipped_moves = 0

    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch)

        for batch_input, move_targets_legacy, wdl_targets in loader:
            if SHUTDOWN:
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                log(f"Shutdown at step {step}")
                return

            # Remap move targets from legacy to compact
            move_targets = remap_device[move_targets_legacy]
            valid = move_targets >= 0
            if not valid.all():
                skipped = (~valid).sum().item()
                skipped_moves += skipped
                move_targets = move_targets.clamp(min=0)

            # Extract cp/mate for HL-Gauss value targets
            cp_vals = batch_input.pop("cp").to(DEVICE)
            mate_vals = batch_input.pop("mate").to(DEVICE)
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing,
                    ignore_index=-1)
                v_loss = hl_gauss_loss(result["value_logits"], win_pct)
                loss = (p_loss + args.value_weight * v_loss) / args.accum_steps

            scaler.scale(loss).backward()

            # NaN guard
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

                # Update LR
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
                    pos_s = positions_seen / max(elapsed, 1) if start_step == 0 else \
                             (positions_seen - start_step * eff_bs) / max(elapsed, 1)
                    remaining_steps = total_steps - step
                    remaining_pos = remaining_steps * eff_bs
                    eta = remaining_pos / max(pos_s, 1)

                    lr_str = f"{lr:.2e}" if lr >= 1e-5 else f"{lr:.2e}"
                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr_str} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"ETA {timedelta(seconds=int(eta))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0

                # Save
                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")
                    if step % 10000 == 0 and step > 0:
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / f"step_{step}.pt")

                # Eval
                if step % args.eval_interval == 0 and eval_data:
                    torch.cuda.empty_cache()
                    acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
                    log(f"  EVAL step {step}: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / "best_model.pt")
                        log(f"  ** New best! top1={best_acc:.2%}")
                    model.train()

                # Max steps check
                if args.max_steps > 0 and step >= args.max_steps:
                    log(f"Reached max_steps={args.max_steps}")
                    break

                accum_count = 0
                accum_p_loss = 0.0
                accum_v_loss = 0.0

        if args.max_steps > 0 and step >= args.max_steps:
            break

        # End of epoch
        log(f"\nEpoch {epoch+1}/{args.epochs} complete. positions_seen={positions_seen:,}")
        if skipped_moves > 0:
            log(f"  Skipped {skipped_moves} moves with no compact mapping")
            skipped_moves = 0

        save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                       OUTPUT_DIR / f"epoch_{epoch+1}.pt")

        if eval_data:
            torch.cuda.empty_cache()
            acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
            log(f"  EPOCH EVAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                              OUTPUT_DIR / "best_model.pt")
                log(f"  ** New best! top1={best_acc:.2%}")

    # Final save
    save_checkpoint(model, optimizer, scaler, step, args.epochs, best_acc,
                   OUTPUT_DIR / "best_model.pt")

    elapsed = time.time() - t0
    log(f"\nTraining complete: {step:,} steps, {positions_seen:,} positions")
    log(f"  Time: {timedelta(seconds=int(elapsed))}")
    log(f"  Speed: {positions_seen/max(elapsed,1):.0f} pos/s")
    log(f"  Best top1: {best_acc:.2%}")


if __name__ == "__main__":
    import traceback
    MAX_RETRIES = 5
    for _attempt in range(MAX_RETRIES):
        try:
            main()
            break
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                log(f"CUDA error (attempt {_attempt+1}/{MAX_RETRIES}): {e}")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(10)
            else:
                traceback.print_exc()
                break
    else:
        log("All retry attempts exhausted.")
