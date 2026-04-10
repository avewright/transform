"""exp159: Continue exp149 training with 128-bin HL-Gauss distributional value head.

MOTIVATION:
  exp149 at step 49,600 has:
  - Policy improving steadily: 16.98% top-1, 41.70% top-3
  - Value PLATEAUED at ~71% since step 30K (3-class WDL ceiling)
  
  The 3-class WDL value head is the #1 bottleneck for Elo:
  - MCTS value quality dominates Elo at 100-200 sims
  - 99% win and 55% win both map to "Win" class — no granularity
  - Ruoss 2024 + Farebrother 2024 show 128-bin HL-Gauss is the gold standard
  
  This experiment does VALUE HEAD SURGERY on the live exp149 checkpoint:
  - Replace Linear(512, 3) → Linear(512, 128) in value head
  - Use HL-Gauss loss (Gaussian-smoothed cross-entropy over bins)
  - 5× LR for value head new layer, continue trunk cosine schedule
  - Everything else identical to exp149

APPROACH:
  1. Load exp149/latest.pt (step 49,600)
  2. Replace value head final layer (3→128 outputs)  
  3. Continue exp149's cosine LR schedule seamlessly
  4. Value head final layer gets 5× LR multiplier (random init learns faster)
  5. HL-Gauss σ=0.75/128≈0.006 per Farebrother et al.
  6. value_weight=1.0 (doubled from 0.5) — value is the focus

WIN% MAPPING:
  cp → win% = sigmoid(cp / 111.7)  [same as compute_wdl]
  mate > 0 → 100%, mate < 0 → 0%
  Binned into 128 uniform bins in [0, 1]

EXPECTED GAIN: +100-300 Elo at 100 MCTS sims (from value quality alone)

Usage:
  python experiments/exp159_dist_value_continue.py                    # full training
  python experiments/exp159_dist_value_continue.py --max-steps 5000   # quick ablation
  python experiments/exp159_dist_value_continue.py --resume           # resume exp159
  python experiments/exp159_dist_value_continue.py --eval-only        # eval best
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
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp159_dist_value"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
SOURCE_CKPT = ROOT / "outputs" / "exp149_scratch_204m" / "latest.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

# Model config matches exp149
MODEL_CONFIG = ChessTransformerConfig()

# Distributional value constants
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


# ── Win% and HL-Gauss ──

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
    """HL-Gauss loss: cross-entropy with Gaussian-smoothed categorical targets.

    Args:
        logits: (B, K) raw logits from value head
        win_pct: (B,) win percentages in [0, 1]
    Returns:
        (B,) per-sample losses
    """
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)  # (B, K)
    log_probs_target = -0.5 * (diff / sigma) ** 2
    targets = F.softmax(log_probs_target, dim=-1)  # (B, K)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(targets * log_probs).sum(dim=-1)  # (B,)


def value_logits_to_win_pct(logits, n_bins=N_VALUE_BINS):
    """Convert distributional value logits to expected win percentage."""
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    probs = F.softmax(logits.float(), dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ── Value head surgery ──

def replace_value_head(model, n_bins=N_VALUE_BINS):
    """Replace 3-class WDL value head with N-bin distributional head.

    Keeps Linear(1024,512) + ReLU, replaces only Linear(512,3) → Linear(512,N).
    """
    old_head = model.value_head
    assert isinstance(old_head, nn.Sequential) and len(old_head) == 3
    hidden_dim = old_head[0].out_features  # 512

    model.value_head = nn.Sequential(
        old_head[0],  # Linear(1024, 512) — KEEP trained weights
        old_head[1],  # ReLU
        nn.Linear(hidden_dim, n_bins),  # NEW: random init
    )
    nn.init.xavier_uniform_(model.value_head[2].weight)
    nn.init.zeros_(model.value_head[2].bias)

    n_new = sum(p.numel() for p in model.value_head[2].parameters())
    log(f"  Value head surgery: Linear(512,3) → Linear(512,{n_bins})")
    log(f"  New params: {n_new:,} | Kept: Linear(1024,512)+ReLU weights")
    return model


# ── Eval ──

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
    """Load eval data with win% targets for distributional value eval."""
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)

    eval_data = []
    surviving = []
    win_pct = cp_to_win_percent(raw["cp"].float(), raw["mate"].long())

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
            eval_data.append({"board": board, "move": move})
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


def run_eval(model, eval_data, eval_tensors, eval_win_pct, batch_size=64):
    """Evaluate policy + distributional value quality."""
    model.eval()
    correct = top3 = total = 0
    value_mae_sum = 0.0
    value_wdl_correct = 0

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

            # Value metrics
            batch_wp = eval_win_pct[idx].to(DEVICE)
            pred_wp = value_logits_to_win_pct(value_logits)
            value_mae_sum += (pred_wp - batch_wp).abs().sum().item()

            # Coarse WDL for comparison with exp149
            pred_wdl = torch.zeros(n, dtype=torch.long, device=DEVICE)
            pred_wdl[pred_wp > 0.55] = 0   # Win
            pred_wdl[pred_wp < 0.45] = 2   # Loss
            pred_wdl[(pred_wp >= 0.45) & (pred_wp <= 0.55)] = 1  # Draw

            true_wdl = torch.zeros(n, dtype=torch.long, device=DEVICE)
            true_wdl[batch_wp > 0.55] = 0
            true_wdl[batch_wp < 0.45] = 2
            true_wdl[(batch_wp >= 0.45) & (batch_wp <= 0.55)] = 1

            value_wdl_correct += (pred_wdl == true_wdl).sum().item()

    return {
        "top1": correct / max(total, 1),
        "top3": top3 / max(total, 1),
        "val_mae": value_mae_sum / max(total, 1),
        "val_wdl": value_wdl_correct / max(total, 1),
    }


# ── Checkpoint ──

def save_checkpoint(model, optimizer, scaler, step, epoch, best_acc, path,
                    source_step=0):
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
        "source_step": source_step,
        "experiment": "exp159_dist_value_continue",
    }, tmp)
    os.replace(str(tmp), str(path))


# ── Main ──

def main():
    global LOG_PATH

    ap = argparse.ArgumentParser(description="exp159: Distributional value continue from exp149")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Total epochs (continues exp149's epoch count)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="Base LR (same as exp149)")
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=1.0,
                    help="Value loss weight (doubled from exp149's 0.5)")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--value-lr-mult", type=float, default=5.0,
                    help="LR multiplier for new value head layer")
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Max optimizer steps (for quick ablation)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"
    eff_bs = args.batch_size * args.accum_steps

    log("=" * 60)
    log("exp159: Distributional Value Head — Continue from exp149")
    log(f"  {N_VALUE_BINS}-bin HL-Gauss (σ={SIGMA_HL_GAUSS:.4f})")
    log(f"  value_weight={args.value_weight}, value_lr_mult={args.value_lr_mult}")
    log(f"  eff_bs={eff_bs}")
    log("=" * 60)

    # ── Load model ──
    if args.resume:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "latest.pt")
        log(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(ChessTransformerConfig(**ckpt["config"]))
        # Model was saved with 128-bin value head
        if model.value_head[-1].out_features != N_VALUE_BINS:
            model = replace_value_head(model)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0.0)
        source_step = ckpt.get("source_step", 0)
        log(f"  Resumed: step={start_step}, epoch={start_epoch}")
    elif args.eval_only:
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(ChessTransformerConfig(**ckpt["config"]))
        if model.value_head[-1].out_features != N_VALUE_BINS:
            model = replace_value_head(model)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        log(f"Loaded: {ckpt_path}")
    else:
        # Load exp149 checkpoint, do value head surgery
        ckpt_path = args.checkpoint or str(SOURCE_CKPT)
        log(f"Loading exp149 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = build_model(ChessTransformerConfig(**ckpt["config"]))
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

        # Surgery: replace value head final layer
        model = replace_value_head(model)
        model.to(DEVICE)

        source_step = ckpt.get("step", 0)
        start_step = 0
        start_epoch = 0
        best_acc = 0.0
        log(f"  Source checkpoint was at exp149 step {source_step}")

    log(f"  Model: {count_parameters(model)/1e6:.1f}M params on {DEVICE}")
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    # ── Eval only ──
    if args.eval_only:
        eval_path = SHARD_DIR / "eval.pt"
        eval_data, eval_tensors, eval_win_pct = load_eval_data(eval_path)
        log(f"Eval: {len(eval_data)} positions")
        metrics = run_eval(model, eval_data, eval_tensors, eval_win_pct)
        log(f"  Policy: top1={100*metrics['top1']:.2f}% top3={100*metrics['top3']:.2f}%")
        log(f"  Value:  wdl_acc={100*metrics['val_wdl']:.2f}% mae={metrics['val_mae']:.4f}")
        return

    # ── Data loader ──
    log(f"Loading shards from {SHARD_DIR}...")
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=args.batch_size,
        encoder_type="fused", device=DEVICE, seed=42,
        include_cp=True, include_mate=True,
    )
    total_pos = loader.total_positions
    steps_per_epoch = len(loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    log(f"  {total_pos:,} positions, {steps_per_epoch:,} steps/epoch, {total_steps:,} total")

    # ── LR schedule: continue exp149's cosine schedule ──
    # exp149 used: warmup=2000, total=318,900, lr=2e-4, min_lr=2e-6
    # We continue the SAME schedule but offset by source_step
    EXP149_TOTAL_STEPS = steps_per_epoch * args.epochs  # same data, same epochs
    EXP149_WARMUP = min(2000, EXP149_TOTAL_STEPS // 10)

    def get_lr(exp159_step):
        """Returns LR as if we're at source_step + exp159_step in exp149's schedule."""
        effective_step = source_step + exp159_step
        if effective_step < EXP149_WARMUP:
            return args.lr * (effective_step + 1) / max(EXP149_WARMUP, 1)
        progress = (effective_step - EXP149_WARMUP) / max(EXP149_TOTAL_STEPS - EXP149_WARMUP, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    # ── Optimizer with separate LR for value head ──
    value_final_layer_params = list(model.value_head[2].parameters())
    value_final_ids = {id(p) for p in value_final_layer_params}
    other_params = [p for p in model.parameters() if id(p) not in value_final_ids]

    optimizer = AdamW([
        {"params": other_params, "lr": args.lr, "name": "backbone"},
        {"params": value_final_layer_params, "lr": args.lr * args.value_lr_mult,
         "name": "value_new"},
    ], betas=(0.9, 0.95), weight_decay=args.weight_decay)

    scaler = GradScaler('cuda')

    if args.resume:
        resume_ckpt = torch.load(args.checkpoint or str(OUTPUT_DIR / "latest.pt"),
                                 map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        del resume_ckpt

    # ── Eval data ──
    eval_path = SHARD_DIR / "eval.pt"
    eval_data, eval_tensors, eval_win_pct = load_eval_data(eval_path)
    log(f"  Eval: {len(eval_data)} positions")

    # ── Config save ──
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": MODEL_CONFIG.to_dict(),
            "experiment": "exp159_dist_value_continue",
            "source": str(SOURCE_CKPT),
            "source_step": source_step,
            "n_value_bins": N_VALUE_BINS,
            "sigma_hl_gauss": SIGMA_HL_GAUSS,
            "training": {
                "batch_size": args.batch_size, "accum_steps": args.accum_steps,
                "eff_bs": eff_bs, "lr": args.lr, "epochs": args.epochs,
                "value_weight": args.value_weight, "value_lr_mult": args.value_lr_mult,
                "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            }
        }, f, indent=2)

    # ── Initial eval ──
    if start_step == 0:
        log("Running initial eval (value head is random)...")
        metrics = run_eval(model, eval_data, eval_tensors, eval_win_pct)
        log(f"  INIT: top1={100*metrics['top1']:.2f}% top3={100*metrics['top3']:.2f}% "
            f"wdl_acc={100*metrics['val_wdl']:.2f}% mae={metrics['val_mae']:.4f}")
        best_acc = metrics['top1']

    # ── Training ──
    log(f"\n{'='*60}")
    log(f"Training: {total_steps:,} steps ({args.epochs} epochs)")
    log(f"  Continuing exp149 LR schedule from effective step {source_step}")
    log(f"  LR at start: backbone={get_lr(0):.2e}, value={get_lr(0)*args.value_lr_mult:.2e}")
    log(f"{'='*60}")

    model.train()
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
                              OUTPUT_DIR / "latest.pt", source_step)
                log(f"Shutdown at step {step}")
                return

            if args.max_steps and step >= args.max_steps:
                log(f"Reached max_steps={args.max_steps}")
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt", source_step)
                break

            # Extract cp/mate for win% computation
            cp_vals = batch_input.pop("cp")
            mate_vals = batch_input.pop("mate")
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)

                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing)

                # Per-sample HL-Gauss value loss
                v_loss_per = hl_gauss_loss(result["value_logits"], win_pct)
                v_loss = v_loss_per.mean()

                loss = (p_loss + args.value_weight * v_loss) / args.accum_steps

            scaler.scale(loss).backward()

            # NaN guard
            if torch.isnan(p_loss) or torch.isnan(v_loss):
                log(f"NaN at step {step}! Saving and aborting.")
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest_nan.pt", source_step)
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
                step += 1

                # Update LR (continuing exp149's schedule)
                lr_backbone = get_lr(step)
                lr_value = lr_backbone * args.value_lr_mult
                optimizer.param_groups[0]["lr"] = lr_backbone
                optimizer.param_groups[1]["lr"] = lr_value

                # Log
                if step % args.log_interval == 0:
                    avg_p = accum_p_loss / accum_count
                    avg_v = accum_v_loss / accum_count
                    avg_gn = grad_norm_accum / args.log_interval
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_s = positions_seen / max(elapsed, 1) if start_step == 0 else \
                             (positions_seen - start_step * eff_bs) / max(elapsed, 1)
                    remaining = (total_steps - step) * eff_bs
                    eta = remaining / max(pos_s, 1)

                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr_backbone:.2e}/{lr_value:.2e} gn={avg_gn:.2f} | "
                        f"{pos_s:.0f} pos/s | ETA {timedelta(seconds=int(eta))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0

                # Save
                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt", source_step)
                    # SWA snapshots
                    if step % 10000 == 0:
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / f"step_{step}.pt", source_step)

                # Eval
                if step % args.eval_interval == 0:
                    torch.cuda.empty_cache()
                    metrics = run_eval(model, eval_data, eval_tensors, eval_win_pct)
                    log(f"  EVAL step {step}: "
                        f"top1={100*metrics['top1']:.2f}% "
                        f"top3={100*metrics['top3']:.2f}% "
                        f"wdl={100*metrics['val_wdl']:.2f}% "
                        f"mae={metrics['val_mae']:.4f}")

                    if metrics['top1'] > best_acc:
                        best_acc = metrics['top1']
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / "best_model.pt", source_step)
                        log(f"  ★ New best: {100*best_acc:.2f}%")

                    model.train()

                accum_count = 0
                accum_p_loss = 0.0
                accum_v_loss = 0.0

        else:
            # Only continue to next epoch if inner loop wasn't broken
            log(f"Epoch {epoch + 1} complete at step {step}")
            save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                          OUTPUT_DIR / f"epoch_{epoch+1}.pt", source_step)
            continue
        break  # max_steps was hit

    log(f"\nTraining complete! step={step}, best top-1={100*best_acc:.2f}%")


if __name__ == "__main__":
    main()
