"""exp157: Distributional Value Head — 128-bin HL-Gauss classification.

MOTIVATION (from Ruoss et al. 2024 + Monroe et al. 2024):
  Our 3-class WDL value head has PLATEAUED at ~71% since step 30K.
  Both papers show that fine-grained value discretization (128 bins) with
  HL-Gauss loss dramatically outperforms coarse WDL classification.
  
  Why 3-class WDL fails:
  - Position with 99% and 55% win probability BOTH map to "Win" class
  - Model can't learn to distinguish close games from won ones
  - Value refinement (which MCTS needs) is impossible with 3 classes
  - At 100 sims MCTS, value quality is THE dominant Elo factor

  Ruoss et al. found 128 bins optimal (better than 16, 32, 64, or 256).
  HL-Gauss smoothing preserves ordinal structure: near-miss bins get partial
  credit, unlike hard cross-entropy.

Architecture:
  - Same 204M backbone (16L/1024d/16H)
  - Replace Linear(..., 3) → Linear(..., 128) in value head
  - Add HL-Gauss loss function for value targets
  - Load epoch_1 weights with strict=False (value head shape mismatch → reinit)

Win% mapping: cp → win% via sigmoid (symmetric, same as Lichess formula)
  - Mate-in-N: 100% (win) or 0% (loss)
  - No mate: win% = 100 / (1 + exp(-cp / 111.7))  [same scale as compute_wdl]
  - Binned into 128 uniform bins: bin_i covers [100*i/128, 100*(i+1)/128)

Expected gain: +5-15% value accuracy → +200-500 ELO at 100 MCTS sims.

Usage:
  python experiments/exp157_distributional_value.py
  python experiments/exp157_distributional_value.py --resume
  python experiments/exp157_distributional_value.py --eval-only --checkpoint outputs/exp157_dist_value/best_model.pt
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
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp157_dist_value"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
SOURCE_CKPT = ROOT / "outputs" / "exp149_scratch_204m" / "epoch_1.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

# ── Distributional Value Constants ──
N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS  # ~0.006, recommended by Farebrother et al.

# Use cls value head (same as exp149), but we'll replace the final layer
MODEL_CONFIG = ChessTransformerConfig(value_head_type="cls")


# ── Helper functions ──

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


# ── Win% and binning ──

def cp_to_win_percent(cp, mate):
    """Convert cp/mate to win percentage [0, 1].
    
    Uses same sigmoid as compute_wdl: win = 1/(1+exp(-cp/111.7))
    Mate: win% = 1.0 (mate > 0) or 0.0 (mate < 0).
    """
    N = cp.shape[0]
    win_pct = torch.zeros(N, dtype=torch.float32)
    
    mate_pos = mate > 0
    mate_neg = mate < 0
    win_pct[mate_pos] = 1.0
    win_pct[mate_neg] = 0.0
    
    no_mate = mate == 0
    if no_mate.any():
        k = 1.0 / 111.7
        win_pct[no_mate] = torch.sigmoid(k * cp[no_mate].float())
    
    return win_pct


def win_pct_to_bin(win_pct, n_bins=N_VALUE_BINS):
    """Map win percentage [0,1] to bin index [0, n_bins-1]."""
    return torch.clamp((win_pct * n_bins).long(), 0, n_bins - 1)


def hl_gauss_target(win_pct, n_bins=N_VALUE_BINS, sigma=SIGMA_HL_GAUSS):
    """Compute HL-Gauss soft target distribution.
    
    For each position, creates a soft categorical target where the mass is
    spread around the true bin with a Gaussian kernel. This preserves 
    ordinal structure: nearby bins get partial credit.
    
    Args:
        win_pct: (B,) tensor of win percentages in [0, 1]
        n_bins: number of value bins
        sigma: std dev of Gaussian smoothing (in [0,1] space)
    
    Returns:
        (B, n_bins) soft target distribution (sums to 1 per row)
    """
    # Bin centers in [0, 1] space
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=win_pct.device)  # (K,)
    
    # Gaussian around each true value
    # win_pct: (B,) → (B, 1)
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)  # (B, K)
    log_probs = -0.5 * (diff / sigma) ** 2  # (B, K)
    
    # Normalize to sum to 1
    targets = F.softmax(log_probs, dim=-1)
    
    return targets


def hl_gauss_loss(logits, win_pct):
    """HL-Gauss loss: cross-entropy with Gaussian-smoothed targets.
    
    Args:
        logits: (B, K) raw logits from value head
        win_pct: (B,) win percentages in [0, 1]
    
    Returns:
        scalar loss
    """
    targets = hl_gauss_target(win_pct)  # (B, K)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, K)
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss


def value_logits_to_expected_win_pct(logits, n_bins=N_VALUE_BINS):
    """Convert distributional value logits to expected win percentage."""
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    probs = F.softmax(logits, dim=-1)  # (B, K)
    return (probs * bin_centers).sum(dim=-1)  # (B,)


# ── FEN conversion for eval ──

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


# ── Eval ──

def load_eval_data(eval_path):
    """Load eval data. Returns (eval_data, eval_tensors, eval_win_pct)."""
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)

    eval_data = []
    surviving = []
    win_pct = cp_to_win_percent(raw["cp"], raw["mate"])

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
                "win_pct": win_pct[i].item(),
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
    eval_win_pct = win_pct[idx]

    return eval_data, eval_tensors, eval_win_pct


def run_eval(model, eval_data, eval_tensors, eval_win_pct, batch_size=32):
    """Evaluate policy + distributional value."""
    model.eval()
    correct = top3 = total = 0
    
    # Value metrics
    value_mae_sum = 0.0  # mean absolute error on win%
    value_correct_wdl = 0  # coarse WDL accuracy for comparison
    value_bin_correct = 0  # exact bin accuracy

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
            value_logits = result["value_logits"].float()  # (n, 128)

            # Policy metrics
            for j, d in enumerate(chunk):
                board, true_move = d["board"], d["move"]
                l = logits[j].clone()
                mask = legal_move_mask(board).to(DEVICE)
                l[~mask] = float("-inf")

                true_idx = move_to_index(true_move)
                if l.argmax().item() == true_idx:
                    correct += 1
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk:
                    top3 += 1
                total += 1

            # Value metrics (batch)
            batch_wp = eval_win_pct[idx].to(DEVICE)
            pred_wp = value_logits_to_expected_win_pct(value_logits)
            value_mae_sum += (pred_wp - batch_wp).abs().sum().item()
            
            # Coarse WDL comparison
            pred_wdl = torch.zeros(n, dtype=torch.long, device=DEVICE)
            pred_wdl[pred_wp > 0.55] = 0  # Win
            pred_wdl[(pred_wp >= 0.45) & (pred_wp <= 0.55)] = 1  # Draw
            pred_wdl[pred_wp < 0.45] = 2  # Loss
            
            true_wdl = torch.zeros(n, dtype=torch.long, device=DEVICE)
            true_wdl[batch_wp > 0.55] = 0
            true_wdl[(batch_wp >= 0.45) & (batch_wp <= 0.55)] = 1
            true_wdl[batch_wp < 0.45] = 2
            
            value_correct_wdl += (pred_wdl == true_wdl).sum().item()
            
            # Bin accuracy
            pred_bin = value_logits.argmax(dim=-1)
            true_bin = win_pct_to_bin(batch_wp)
            value_bin_correct += (pred_bin == true_bin).sum().item()

    return {
        "top1": correct / max(total, 1),
        "top3": top3 / max(total, 1),
        "val_wdl": value_correct_wdl / max(total, 1),
        "val_mae": value_mae_sum / max(total, 1),
        "val_bin_acc": value_bin_correct / max(total, 1),
    }


# ── Checkpoint ──

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
        "n_value_bins": N_VALUE_BINS,
    }, tmp)
    os.replace(str(tmp), str(path))


# ── Model surgery: replace 3-class value head with 128-bin ──

def replace_value_head(model, hidden_dim=512, n_bins=N_VALUE_BINS):
    """Replace the 3-output value head with N_VALUE_BINS-output head.
    
    The existing value head is Sequential(Linear(1024,512), ReLU, Linear(512,3)).
    We replace the final Linear(512,3) with Linear(512,128).
    The first two layers keep their trained weights (feature extraction).
    """
    old_head = model.value_head
    assert isinstance(old_head, nn.Sequential) and len(old_head) == 3
    
    # Keep the first two layers (Linear(1024,512) + ReLU)
    model.value_head = nn.Sequential(
        old_head[0],  # Linear(1024, 512) — KEEP WEIGHTS
        old_head[1],  # ReLU
        nn.Linear(hidden_dim, n_bins),  # NEW: random init
    )
    
    # Initialize the new final layer
    nn.init.xavier_uniform_(model.value_head[2].weight)
    nn.init.zeros_(model.value_head[2].bias)
    
    n_new = sum(p.numel() for p in model.value_head[2].parameters())
    log(f"  Replaced value head: Linear(512,3) → Linear(512,{n_bins})")
    log(f"  New parameters: {n_new:,} (rest of model frozen pretrained)")
    
    return model


# ── Main ──

def main():
    global LOG_PATH
    parser = argparse.ArgumentParser(description="exp157: Distributional Value Head")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs (default: 2 = epochs 2-3)")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Policy label smoothing")
    parser.add_argument("--value-weight", type=float, default=1.0,
                        help="Value loss weight (higher because this is the focus)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps (for quick ablation)")
    parser.add_argument("--value-lr-mult", type=float, default=5.0,
                        help="LR multiplier for value head (new params learn faster)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = str(OUTPUT_DIR / "training.log")

    eff_bs = args.batch_size * args.accum_steps
    log(f"exp157: Distributional Value Head ({N_VALUE_BINS} bins, HL-Gauss)")
    log(f"  Effective batch size: {eff_bs}")
    log(f"  Value weight: {args.value_weight}, Value LR mult: {args.value_lr_mult}")

    # ── Build model ──
    if args.resume:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "latest.pt")
        log(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ChessTransformerConfig(**ckpt["config"])
        model = build_model(config)
        # Model already has 128-bin value head if saved from this experiment
        if model.value_head[-1].out_features != N_VALUE_BINS:
            model = replace_value_head(model)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Resumed: step={start_step}, epoch={start_epoch}")
    elif args.eval_only:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ChessTransformerConfig(**ckpt["config"])
        model = build_model(config)
        if model.value_head[-1].out_features != N_VALUE_BINS:
            model = replace_value_head(model)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        log(f"Loaded model from {ckpt_path}")
    else:
        # Load from epoch_1 checkpoint, replace value head
        log(f"Loading epoch_1 weights from {SOURCE_CKPT}")
        if not SOURCE_CKPT.exists():
            log(f"ERROR: {SOURCE_CKPT} not found. Train exp149 first.")
            return
        
        ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
        config = ChessTransformerConfig(**ckpt["config"])
        model = build_model(config)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        
        # Surgery: replace value head
        model = replace_value_head(model)
        model.to(DEVICE)
        
        start_step = 0
        start_epoch = 0
        best_acc = 0.0

    total_params = count_parameters(model)
    log(f"  Model: {total_params/1e6:.1f}M parameters")

    # ── Eval-only mode ──
    if args.eval_only:
        eval_path = SHARD_DIR / "eval.pt"
        eval_data, eval_tensors, eval_win_pct = load_eval_data(eval_path)
        log(f"Eval set: {len(eval_data)} positions")
        metrics = run_eval(model, eval_data, eval_tensors, eval_win_pct)
        log(f"  Policy: top1={100*metrics['top1']:.2f}% top3={100*metrics['top3']:.2f}%")
        log(f"  Value:  wdl_acc={100*metrics['val_wdl']:.2f}% "
            f"mae={metrics['val_mae']:.4f} "
            f"bin_acc={100*metrics['val_bin_acc']:.2f}%")
        return

    # ── Data loader (include_cp + include_mate for win% computation) ──
    loader = ShardedChessLoader(
        shard_dir=SHARD_DIR,
        batch_size=args.batch_size,
        encoder_type="fused",
        device=DEVICE,
        hflip=True,
        include_cp=True,
        include_mate=True,
    )

    # ── Eval data ──
    eval_path = SHARD_DIR / "eval.pt"
    eval_data, eval_tensors, eval_win_pct = load_eval_data(eval_path)
    log(f"Eval: {len(eval_data)} positions")

    # ── Optimizer with separate LR for value head ──
    value_head_params = list(model.value_head.parameters())
    value_head_ids = {id(p) for p in value_head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in value_head_ids]

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": value_head_params, "lr": args.lr * args.value_lr_mult},
    ], betas=(0.9, 0.95), weight_decay=args.weight_decay)

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        log("  Restored optimizer state")

    scaler = GradScaler()
    if args.resume and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # ── LR schedule (cosine, continuing from epoch 1 position) ──
    total_positions = loader._total * args.epochs
    max_steps = total_positions // eff_bs
    if args.max_steps:
        max_steps = min(max_steps, args.max_steps)
    warmup_steps = min(1000, max_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if args.resume:
        for _ in range(start_step):
            scheduler.step()

    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epochs, max_steps={max_steps}")
    log(f"  LR={args.lr} (backbone), {args.lr * args.value_lr_mult} (value head)")
    log(f"  value_weight={args.value_weight}")
    log(f"  Value: {N_VALUE_BINS}-bin distributional (HL-Gauss σ={SIGMA_HL_GAUSS:.4f})")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0

    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch)

        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN:
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                log(f"Shutdown at step {step}")
                return

            if args.max_steps and step >= args.max_steps:
                log(f"Reached max_steps={args.max_steps}")
                break

            # Compute win% from cp + mate (available in batch_input)
            # Note: wdl_targets from loader is (B,3) soft WDL — we DON'T use it
            # Instead we compute fine-grained distributional target from cp/mate
            cp_vals = batch_input.pop("cp")  # Remove from model input
            mate_vals = batch_input.pop("mate")  # Remove from model input
            
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing)
                v_loss = hl_gauss_loss(
                    result["value_logits"].float(), win_pct)
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
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                if step % 25 == 0:
                    avg_p = accum_p_loss / accum_count
                    avg_v = accum_v_loss / accum_count
                    avg_gn = grad_norm_accum / max(1, step % 25)
                    lr_val = scheduler.get_last_lr()[0]
                    lr_val_v = scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else lr_val
                    elapsed = time.time() - t0
                    rate = positions_seen / elapsed if elapsed > 0 else 0
                    eta_s = (max_steps - step) * elapsed / max(step - start_step, 1)
                    eta = timedelta(seconds=int(eta_s))

                    log(f"  step {step:,}/{max_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr_val:.2e}/{lr_val_v:.2e} gn={avg_gn:.2f} | "
                        f"{rate:.0f} pos/s | pos={positions_seen:,} | ETA {eta}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0
                    grad_norm_accum = 0.0
                    t0 = time.time()

                if step % 1000 == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")

                    metrics = run_eval(model, eval_data, eval_tensors, eval_win_pct)
                    log(f"  EVAL step {step}: "
                        f"top1={100*metrics['top1']:.2f}% "
                        f"top3={100*metrics['top3']:.2f}% "
                        f"wdl_acc={100*metrics['val_wdl']:.2f}% "
                        f"mae={metrics['val_mae']:.4f} "
                        f"bin_acc={100*metrics['val_bin_acc']:.2f}%")

                    if metrics['top1'] > best_acc:
                        best_acc = metrics['top1']
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / "best_model.pt")
                        log(f"  ** New best: {100*best_acc:.2f}%")

                    model.train()

        log(f"Epoch {epoch + 1} complete at step {step}")
        save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                      OUTPUT_DIR / f"epoch_{epoch+1}.pt")

    log(f"\nTraining complete! Final step: {step}")
    log(f"Best top-1 accuracy: {100*best_acc:.2f}%")


if __name__ == "__main__":
    main()
