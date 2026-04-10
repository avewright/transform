"""exp166: Phase-Weighted Fine-Tune — Rebalance Gradient Distribution by Game Phase.

MOTIVATION:
  Balanced eval (exp161 step 10K) shows middlegame is weakest:
    opening: 12.8% top-1, 0.14 MAE
    middlegame: 11.1% top-1, 0.23 MAE  ← worst by far
    endgame: 12.1% top-1, 0.18 MAE
  
  Training data is 79% opening, 6-8% middlegame, 12-14% endgame.
  Ruoss et al. (2024, 2895 Elo) found uniform phase sampling >> natural frequency.
  
  This experiment reweights the per-sample loss by game phase:
    opening (≥14 non-king pieces): weight=0.5  (downweight ~58%→38%)
    middlegame (6-13): weight=1.5              (upweight ~23%→44%)
    endgame (<6): weight=1.2                   (slight upweight ~19%→29%)
  
  Weights are normalized per-batch so mean≈1 (preserves effective LR).

Approach: Fine-tune from exp161 best checkpoint with lower LR.
Expected: +30-60 Elo from rebalanced gradient distribution.

Usage:
  python experiments/exp166_phase_weighted.py                     # 5K step ablation
  python experiments/exp166_phase_weighted.py --max-steps 20000   # longer run
  python experiments/exp166_phase_weighted.py --resume             # resume
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
OUTPUT_DIR = ROOT / "outputs" / "exp166_phase_weighted"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG

N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS

# ── Phase weight constants ─────────────────────────────────────────────
# Piece count thresholds (non-king pieces on board)
# These use fused_ids: pieces are values 1-5 (white) and 7-11 (black), excluding kings (6, 12)
PHASE_WEIGHT_OPENING = 0.5      # ≥14 non-king pieces
PHASE_WEIGHT_MIDDLEGAME = 1.5   # 6-13 non-king pieces
PHASE_WEIGHT_ENDGAME = 1.2      # <6 non-king pieces


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


# ── Move remapping ─────────────────────────────────────────────────────

def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t


# ── HL-Gauss distributional value ─────────────────────────────────────

def cp_to_win_percent(cp, mate):
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
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)
    log_probs_target = -0.5 * (diff / sigma) ** 2
    targets = F.softmax(log_probs_target, dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(targets * log_probs).sum(dim=-1)  # (B,) per-sample loss


def hl_gauss_loss_mean(logits, win_pct, n_bins=N_VALUE_BINS, sigma=SIGMA_HL_GAUSS):
    return hl_gauss_loss(logits, win_pct, n_bins, sigma).mean()


def value_logits_to_win_pct(logits, n_bins=N_VALUE_BINS):
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    probs = F.softmax(logits.float(), dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ── Phase weight computation ──────────────────────────────────────────

def compute_phase_weights(fused_ids):
    """Compute per-sample phase weights from fused board encoding.
    
    Args:
        fused_ids: (B, 64) — piece IDs. Non-king pieces: 1-5 (white), 7-11 (black).
    Returns:
        (B,) weight tensor, normalized so mean ≈ 1.
    """
    # Count non-king pieces (exclude 0=empty, 6=white king, 12=black king)
    non_king = ((fused_ids >= 1) & (fused_ids <= 5)) | ((fused_ids >= 7) & (fused_ids <= 11))
    piece_count = non_king.sum(dim=1).float()  # (B,)
    
    weights = torch.ones_like(piece_count)
    weights[piece_count >= 14] = PHASE_WEIGHT_OPENING
    weights[(piece_count >= 6) & (piece_count < 14)] = PHASE_WEIGHT_MIDDLEGAME
    weights[piece_count < 6] = PHASE_WEIGHT_ENDGAME
    
    # Normalize to mean=1 (preserves effective LR)
    weights = weights / (weights.mean() + 1e-8)
    return weights


# ── Model ──────────────────────────────────────────────────────────────

def build_compact_dist_model(config):
    model = build_model(config)
    old_head = model.value_head
    assert isinstance(old_head, nn.Sequential) and len(old_head) == 3
    hidden_dim = old_head[0].out_features
    model.value_head = nn.Sequential(
        nn.Linear(config.hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, N_VALUE_BINS),
    )
    for layer in model.value_head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


# ── Eval (same as exp161) ─────────────────────────────────────────────

def _board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
    PIECE_CHARS = ".PNBRQKpnbrqk"
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""; empty = 0
        for file_idx in range(8):
            sq = rank * 8 + file_idx
            p = int(ba_row[sq])
            if p == 0: empty += 1
            else:
                if empty > 0: row += str(empty); empty = 0
                row += PIECE_CHARS[p]
        if empty > 0: row += str(empty)
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
            legacy_idx = raw["move_idx"][i].item()
            compact_idx = remap_tensor[legacy_idx].item()
            if compact_idx < 0: continue
            uci = IDX_TO_UCI[compact_idx]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves: continue
            cp_val = raw["cp"][i].item() if "cp" in raw else 0
            mate_val = raw["mate"][i].item() if "mate" in raw else 0
            eval_data.append({"board": board, "move": move, "compact_idx": compact_idx,
                            "cp": cp_val, "mate": mate_val})
            surviving.append(i)
        except Exception: continue
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
    phase_correct = [0, 0, 0]
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
            ba = batch_input["fused_ids"]
            non_king = ((ba >= 1) & (ba <= 5)) | ((ba >= 7) & (ba <= 11))
            piece_counts = non_king.sum(dim=1)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()
            value_logits = result["value_logits"].float()
            pred_win_pct = value_logits_to_win_pct(value_logits)

            for j, d in enumerate(chunk):
                board = d["board"]
                l = logits[j].clone()
                mask = legal_move_mask(board).to(DEVICE)
                l[~mask] = float("-inf")
                true_idx = d["compact_idx"]
                hit = l.argmax().item() == true_idx
                if hit: correct += 1
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk: top3 += 1
                pc = piece_counts[j].item()
                phase = 0 if pc >= 14 else (2 if pc < 6 else 1)
                phase_total[phase] += 1
                if hit: phase_correct[phase] += 1
                cp_t = torch.tensor([d["cp"]], dtype=torch.float32)
                mate_t = torch.tensor([d["mate"]], dtype=torch.long)
                true_wp = cp_to_win_percent(cp_t, mate_t).item()
                pred_wp = pred_win_pct[j].item()
                total_value_mae += abs(pred_wp - true_wp)
                total += 1

    top1_acc = correct / max(total, 1)
    top3_acc = top3 / max(total, 1)
    value_mae = total_value_mae / max(total, 1)
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
        "step": step, "epoch": epoch, "best_acc": best_acc,
        "vocab_version": "compact", "n_value_bins": N_VALUE_BINS,
        "experiment": "exp166_phase_weighted",
    }, tmp)
    os.replace(str(tmp), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-checkpoint", type=str, default=None,
                    help="Checkpoint to fine-tune from (default: exp161_full/best_model.pt)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5, help="Lower LR for fine-tuning")
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=5000, help="Quick ablation (0=full)")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    # Control experiment: disable phase weighting
    ap.add_argument("--no-phase-weight", action="store_true",
                    help="Control: uniform weights (same as exp161 continuation)")
    args = ap.parse_args()

    global LOG_PATH, OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"

    source_ckpt = args.source_checkpoint or str(
        ROOT / "outputs" / "exp161_full" / "best_model.pt")

    log("=" * 60)
    log("exp166: Phase-Weighted Fine-Tune (compact + distributional)")
    log(f"  source: {source_ckpt}")
    log(f"  device: {DEVICE}")
    log(f"  phase weights: open={PHASE_WEIGHT_OPENING} mid={PHASE_WEIGHT_MIDDLEGAME} "
        f"end={PHASE_WEIGHT_ENDGAME}")
    if args.no_phase_weight:
        log("  ** CONTROL: phase weighting DISABLED **")
    log(f"  lr={args.lr}, max_steps={args.max_steps}")

    # Build remap tensor
    remap_tensor = build_remap_tensor()

    # Build model
    model = build_compact_dist_model(MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params/1e6:.1f}M")

    # Load source checkpoint
    start_step = 0
    start_epoch = 0
    best_acc = 0.0
    resume_path = OUTPUT_DIR / "latest.pt"

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(sd)
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Resumed from step {start_step}")
    else:
        ckpt = torch.load(source_ckpt, map_location="cpu", weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(sd)
        source_step = ckpt.get("step", "?")
        source_acc = ckpt.get("best_acc", "?")
        log(f"  Loaded source: step={source_step}, acc={source_acc}")

    model.to(DEVICE).train()

    # Optimizer (fresh — fine-tuning)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Data loader
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

    log(f"  {total_pos:,} positions, eff_bs={eff_bs}")
    log(f"  {total_steps:,} steps ({args.max_steps} max)")

    # LR schedule: short warmup + cosine
    warmup_steps = min(200, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    # Save config
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump({"model": MODEL_CONFIG.to_dict(), "training": {
            "source_checkpoint": source_ckpt,
            "phase_weighted": not args.no_phase_weight,
            "phase_weights": {"opening": PHASE_WEIGHT_OPENING,
                            "middlegame": PHASE_WEIGHT_MIDDLEGAME,
                            "endgame": PHASE_WEIGHT_ENDGAME},
            "lr": args.lr, "max_steps": args.max_steps,
            "warmup_steps": warmup_steps,
            "vocab": "compact", "vocab_size": VOCAB_SIZE,
            "n_value_bins": N_VALUE_BINS,
        }}, f, indent=2)

    # Eval data
    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path, remap_tensor)
        log(f"  Eval: {len(eval_data)} positions")

    remap_device = remap_tensor.to(DEVICE)

    # Initial eval (before fine-tuning)
    if eval_data:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  BASELINE: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        best_acc = acc

    # Training loop
    log(f"\n{'='*60}")
    log(f"Fine-tuning with phase-weighted loss")
    log(f"  LR={args.lr}, warmup={warmup_steps}, label_smoothing={args.label_smoothing}")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0
    phase_counts_accum = [0, 0, 0]  # track phase distribution

    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch)

        for batch_input, move_targets_legacy, wdl_targets in loader:
            if SHUTDOWN:
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest.pt")
                log(f"Shutdown at step {step}")
                return

            # Remap legacy → compact
            move_targets = remap_device[move_targets_legacy]
            valid = move_targets >= 0
            if not valid.all():
                move_targets = move_targets.clamp(min=0)

            # Value targets
            cp_vals = batch_input.pop("cp").to(DEVICE)
            mate_vals = batch_input.pop("mate").to(DEVICE)
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            # Phase weights
            if args.no_phase_weight:
                phase_w = torch.ones(move_targets.shape[0], device=DEVICE)
            else:
                phase_w = compute_phase_weights(batch_input["fused_ids"])

            # Track phase distribution
            fids = batch_input["fused_ids"]
            nk = ((fids >= 1) & (fids <= 5)) | ((fids >= 7) & (fids <= 11))
            pc = nk.sum(dim=1)
            phase_counts_accum[0] += (pc >= 14).sum().item()
            phase_counts_accum[1] += ((pc >= 6) & (pc < 14)).sum().item()
            phase_counts_accum[2] += (pc < 6).sum().item()

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)

                # Per-sample policy loss
                p_loss_per = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing,
                    ignore_index=-1, reduction='none')  # (B,)

                # Per-sample value loss
                v_loss_per = hl_gauss_loss(result["value_logits"], win_pct)  # (B,)

                # Weight and average
                weighted_p = (p_loss_per * phase_w).mean()
                weighted_v = (v_loss_per * phase_w).mean()
                loss = (weighted_p + args.value_weight * weighted_v) / args.accum_steps

            scaler.scale(loss).backward()

            if torch.isnan(loss):
                log(f"NaN at step {step}! Saving and aborting.")
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest_nan.pt")
                return

            accum_p_loss += weighted_p.item()
            accum_v_loss += weighted_v.item()
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
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_s = (positions_seen - start_step * eff_bs) / max(elapsed, 1)
                    remaining = (total_steps - step) * eff_bs / max(pos_s, 1)

                    # Phase distribution
                    total_phase = sum(phase_counts_accum) or 1
                    pct = [100 * c / total_phase for c in phase_counts_accum]

                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"phase={pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}% | "
                        f"ETA {timedelta(seconds=int(remaining))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0

                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")

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

                if args.max_steps > 0 and step >= args.max_steps:
                    log(f"Reached max_steps={args.max_steps}")
                    break

                accum_count = 0
                accum_p_loss = 0.0
                accum_v_loss = 0.0

        if args.max_steps > 0 and step >= args.max_steps:
            break

        log(f"\nEpoch {epoch+1} complete. {positions_seen:,} positions")
        save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                       OUTPUT_DIR / f"epoch_{epoch+1}.pt")

    # Final eval + save
    if eval_data:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  FINAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        if acc > best_acc:
            best_acc = acc

    save_checkpoint(model, optimizer, scaler, step, args.epochs, best_acc,
                   OUTPUT_DIR / "best_model.pt")

    elapsed = time.time() - t0
    total_phase = sum(phase_counts_accum) or 1
    log(f"\nTraining complete: {step:,} steps, {positions_seen:,} positions")
    log(f"  Time: {timedelta(seconds=int(elapsed))}")
    log(f"  Best top1: {best_acc:.2%}")
    log(f"  Data phase dist: open={phase_counts_accum[0]/total_phase:.1%} "
        f"mid={phase_counts_accum[1]/total_phase:.1%} "
        f"end={phase_counts_accum[2]/total_phase:.1%}")


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
                time.sleep(5)
            else:
                traceback.print_exc()
                break
    else:
        log("Max retries reached.")
