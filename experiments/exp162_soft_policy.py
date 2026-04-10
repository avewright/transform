"""exp162: Soft Policy Fine-Tuning with Multi-PV Targets from exp085.

Hypothesis: Fine-tuning from exp161 checkpoint (compact vocab + HL-Gauss value) 
with soft policy targets from Stockfish multi-PV analysis will improve the policy 
prior quality beyond what hard single-move supervision achieves.

Key insight (Ruoss et al. 2024):
  - Training on the full action distribution (all moves and their values) is
    ~30x more informative per position than behavioral cloning (best move only)
  - Action-value (AV) prediction >> state-value (SV) >> behavioral cloning (BC)
  - Our exp085 dataset has soft probability distributions over top-8 Stockfish moves

Data: avewright/exp085-parallel-multipv-harvest (224K positions)
  - 8 PVs per position with pre-computed probabilities (tau=120)
  - teacher_entropy metadata for curriculum/weighting
  - Phase distribution: ~86% opening, ~14% middlegame, ~0% endgame
  - Cached locally in outputs/exp162_soft_data/

Loss function:
  Soft CE: -sum_k(target_prob_k * log P(move_k | position))
  Combined: (1-alpha) * hard_CE + alpha * soft_CE + value_weight * HL-Gauss
  
  Ablation matrix (5K steps each, ~10 min with 224K data):
    control  : alpha=0.0 (hard targets only, exp161 baseline)
    soft_A   : alpha=0.5 (50% hard + 50% soft)
    soft_B   : alpha=1.0 (fully soft, no hard CE)
    soft_C   : alpha=0.3 (30% soft, preserve hard signal)
    soft_D   : alpha=0.7 (70% soft, more weight on distribution)

Protocol:
  1. Fine-tune from exp161 best checkpoint (compact vocab + 128-bin HL-Gauss)
  2. Train on cached soft target shards (224K positions)
  3. Eval every 500 steps on standard eval set
  4. Compare ablations on top-1, top-3, and phase-stratified accuracy

Usage:
  python experiments/exp162_soft_policy.py --checkpoint outputs/exp161_full/best_model.pt
  python experiments/exp162_soft_policy.py --checkpoint outputs/exp161_full/best_model.pt --ablation soft_A
  python experiments/exp162_soft_policy.py --checkpoint outputs/exp161_full/best_model.pt --ablation all
  python experiments/exp162_soft_policy.py --checkpoint outputs/exp161_full/best_model.pt --from-scratch
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
    board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp162_soft_policy"
SOFT_DATA_DIR = ROOT / "outputs" / "exp162_soft_data"
HARD_SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG
N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS


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


# ── Ablation configs ──

ABLATIONS = {
    "control": {
        "alpha": 0.0,
        "desc": "hard targets only (exp161 baseline)",
    },
    "soft_A": {
        "alpha": 0.5,
        "desc": "50% hard + 50% soft",
    },
    "soft_B": {
        "alpha": 1.0,
        "desc": "fully soft (no hard CE)",
    },
    "soft_C": {
        "alpha": 0.3,
        "desc": "30% soft, preserve hard signal",
    },
    "soft_D": {
        "alpha": 0.7,
        "desc": "70% soft, distribution-heavy",
    },
}


# ── HL-Gauss (same as exp161) ──

def cp_to_win_percent(cp, mate):
    N = cp.shape[0]
    win_pct = torch.zeros(N, dtype=torch.float32, device=cp.device)
    win_pct[mate > 0] = 1.0
    win_pct[mate < 0] = 0.0
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
    return -(targets * log_probs).sum(dim=-1).mean()


def value_logits_to_win_pct(logits, n_bins=N_VALUE_BINS):
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                 device=logits.device)
    probs = F.softmax(logits.float(), dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ── Soft policy loss ──

def soft_policy_loss(logits, soft_indices, soft_probs):
    """Sparse cross-entropy with soft targets.
    
    Args:
        logits:       (B, V) raw policy logits
        soft_indices: (B, K) compact move indices (-1 = padding)
        soft_probs:   (B, K) teacher probabilities (0 = padding)
    Returns:
        scalar loss = -sum_k(prob_k * log_softmax(logits)[idx_k])
    """
    B, K = soft_indices.shape
    
    # Log-softmax over full vocabulary
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, V)
    
    # Mask out padding
    valid = (soft_indices >= 0) & (soft_probs > 0)       # (B, K)
    
    # Gather log probs at target indices (clamp -1 to 0 for gather)
    safe_indices = soft_indices.clamp(min=0).long()       # (B, K)
    gathered = log_probs.gather(1, safe_indices)           # (B, K)
    
    # Weighted sum: -sum_k(prob_k * log_p_k)
    gathered = gathered * valid.float()
    weighted = soft_probs.float() * gathered               # (B, K)
    
    # Mean over batch (sum over K targets per position)
    return -weighted.sum(dim=-1).mean()


# ── Model construction (same as exp161) ──

def build_compact_dist_model(config):
    model = build_model(config)
    old_head = model.value_head
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


# ── Soft data loader ──

class SoftTargetLoader:
    """Loads cached soft target shards with shuffling."""
    
    def __init__(self, data_dir, batch_size, device, seed=42):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        
        self.shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {data_dir}")
        
        # Load all shards into memory (224K positions fits easily)
        all_data = {}
        total = 0
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu", weights_only=False)
            n = shard['board_array'].shape[0]
            for key in shard:
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(shard[key])
            total += n
        
        self.data = {k: torch.cat(v, dim=0) for k, v in all_data.items()}
        self.total_positions = total
        self._shuffle_idx = None
        self._pos = 0
    
    def set_epoch(self, epoch):
        rng = torch.Generator()
        rng.manual_seed(self.seed + epoch)
        self._shuffle_idx = torch.randperm(self.total_positions, generator=rng)
        self._pos = 0
    
    def __len__(self):
        return self.total_positions // self.batch_size
    
    def __iter__(self):
        if self._shuffle_idx is None:
            self.set_epoch(0)
        self._pos = 0
        return self
    
    def __next__(self):
        if self._pos >= self.total_positions - self.batch_size:
            raise StopIteration
        
        idx = self._shuffle_idx[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        
        batch_input = {
            "fused_ids": board_array_to_fused(self.data['board_array'][idx]).to(self.device),
            "turn": self.data['turn'][idx].long().to(self.device),
            "castling": self.data['castling'][idx].long().to(self.device),
            "ep_file": ep_square_to_file(self.data['ep_square'][idx].long()).to(self.device),
        }
        
        move_targets = self.data['move_idx'][idx].long().to(self.device)
        cp = self.data['cp'][idx].float().to(self.device)
        mate = self.data['mate'][idx].long().to(self.device)
        soft_indices = self.data['soft_indices'][idx].to(self.device)
        soft_probs = self.data['soft_probs'][idx].float().to(self.device)
        
        return batch_input, move_targets, cp, mate, soft_indices, soft_probs


# ── Eval (same as exp161) ──

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


def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t


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
            if compact_idx < 0:
                continue
            uci = IDX_TO_UCI[compact_idx]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            cp_val = raw["cp"][i].item() if "cp" in raw else 0
            mate_val = raw["mate"][i].item() if "mate" in raw else 0
            eval_data.append({
                "board": board, "move": move, "compact_idx": compact_idx,
                "cp": cp_val, "mate": mate_val,
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
                pc = piece_counts[j].item()
                phase = 0 if pc >= 14 else (2 if pc < 6 else 1)
                phase_total[phase] += 1
                if hit:
                    phase_correct[phase] += 1
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
        "step": step,
        "epoch": epoch,
        "best_acc": best_acc,
        "vocab_version": "compact",
        "n_value_bins": N_VALUE_BINS,
    }, tmp)
    os.replace(str(tmp), str(path))


# ── Main training ──

def run_ablation(args, ablation_name, ablation_config, checkpoint_path):
    """Run a single ablation configuration."""
    alpha = ablation_config["alpha"]
    desc = ablation_config["desc"]
    
    abl_dir = OUTPUT_DIR / ablation_name
    abl_dir.mkdir(parents=True, exist_ok=True)
    
    global LOG_PATH
    LOG_PATH = abl_dir / "training.log"
    
    log("=" * 60)
    log(f"exp162: Soft Policy Fine-Tuning — {ablation_name}")
    log(f"  description: {desc}")
    log(f"  alpha: {alpha} (hard weight={1-alpha:.1f}, soft weight={alpha:.1f})")
    log(f"  checkpoint: {checkpoint_path}")
    log(f"  device: {DEVICE}")
    log(f"  vocab: {VOCAB_SIZE} moves (compact)")
    log(f"  value: {N_VALUE_BINS}-bin HL-Gauss")
    
    # Build model
    model = build_compact_dist_model(MODEL_CONFIG)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    log(f"  Loaded checkpoint (step={ckpt.get('step', '?')}, acc={ckpt.get('best_acc', '?')})")
    
    model.to(DEVICE)
    model.train()
    
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params/1e6:.1f}M")
    
    # Optimizer — lower LR for fine-tuning
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')
    
    # Soft data loader
    log(f"Loading soft target data from {SOFT_DATA_DIR}...")
    soft_loader = SoftTargetLoader(SOFT_DATA_DIR, args.batch_size, DEVICE, seed=42)
    log(f"  {soft_loader.total_positions:,} positions with soft targets")
    
    steps_per_epoch = len(soft_loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    eff_bs = args.batch_size * args.accum_steps
    
    log(f"  bs={args.batch_size}, accum={args.accum_steps}, eff_bs={eff_bs}")
    log(f"  {steps_per_epoch} steps/epoch, {total_steps} total steps")
    
    # LR schedule — cosine decay with short warmup
    warmup_steps = min(200, total_steps // 10)
    
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (0.01 + 0.99 * cosine)
    
    # Eval data (legacy shards)
    remap_tensor = build_remap_tensor()
    eval_path = HARD_SHARD_DIR / "eval.pt"
    eval_data, eval_tensors = None, None
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path, remap_tensor)
        log(f"  Eval: {len(eval_data)} positions")
    
    # Save config
    config_path = abl_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": MODEL_CONFIG.to_dict(),
            "training": {
                "ablation": ablation_name,
                "alpha": alpha,
                "lr": args.lr, "epochs": args.epochs,
                "batch_size": args.batch_size, "accum_steps": args.accum_steps,
                "eff_bs": eff_bs, "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "value_weight": args.value_weight,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "checkpoint": str(checkpoint_path),
                "n_soft_positions": soft_loader.total_positions,
            }
        }, f, indent=2)
    
    # Initial eval
    if eval_data:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  INIT: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        best_acc = acc
    else:
        best_acc = 0.0
    
    # Training loop
    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epochs, alpha={alpha}, LR={args.lr}")
    log(f"{'='*60}")
    
    step = 0
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_soft_loss = 0.0
    accum_count = 0
    positions_seen = 0
    t0 = time.time()
    grad_norm_accum = 0.0
    
    for epoch in range(args.epochs):
        soft_loader.set_epoch(epoch)
        
        for batch_input, move_targets, cp_vals, mate_vals, soft_indices, soft_probs in soft_loader:
            if SHUTDOWN:
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              abl_dir / "latest.pt")
                log(f"Shutdown at step {step}")
                return best_acc
            
            win_pct = cp_to_win_percent(cp_vals, mate_vals)
            
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                policy_logits = result["policy_logits"]
                
                # Hard CE loss
                if alpha < 1.0:
                    p_loss = F.cross_entropy(
                        policy_logits, move_targets,
                        label_smoothing=args.label_smoothing)
                else:
                    p_loss = torch.tensor(0.0, device=DEVICE)
                
                # Soft CE loss
                if alpha > 0.0:
                    s_loss = soft_policy_loss(policy_logits, soft_indices, soft_probs)
                else:
                    s_loss = torch.tensor(0.0, device=DEVICE)
                
                # Combined policy loss
                combined_p = (1 - alpha) * p_loss + alpha * s_loss
                
                # Value loss (HL-Gauss)
                v_loss = hl_gauss_loss(result["value_logits"], win_pct)
                
                loss = (combined_p + args.value_weight * v_loss) / args.accum_steps
            
            scaler.scale(loss).backward()
            
            accum_p_loss += p_loss.item() if alpha < 1.0 else 0.0
            accum_v_loss += v_loss.item()
            accum_soft_loss += s_loss.item() if alpha > 0.0 else 0.0
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
                
                # Log
                if step % args.log_interval == 0:
                    avg_p = accum_p_loss / max(accum_count, 1)
                    avg_v = accum_v_loss / max(accum_count, 1)
                    avg_s = accum_soft_loss / max(accum_count, 1)
                    avg_gn = grad_norm_accum / max(args.log_interval, 1)
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_s = positions_seen / max(elapsed, 1)
                    remaining = (total_steps - step) * eff_bs / max(pos_s, 1)
                    
                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} s={avg_s:.4f} v={avg_v:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"ETA {timedelta(seconds=int(remaining))}")
                    
                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_soft_loss = 0.0
                    accum_count = 0
                
                # Save
                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  abl_dir / "latest.pt")
                
                # Eval
                if step % args.eval_interval == 0 and eval_data:
                    torch.cuda.empty_cache()
                    acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
                    log(f"  EVAL step {step}: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      abl_dir / "best_model.pt")
                        log(f"  ** New best! top1={best_acc:.2%}")
                    model.train()
                
                if args.max_steps > 0 and step >= args.max_steps:
                    break
                
                accum_count = 0
                accum_p_loss = 0.0
                accum_v_loss = 0.0
                accum_soft_loss = 0.0
        
        if args.max_steps > 0 and step >= args.max_steps:
            break
        
        log(f"\nEpoch {epoch+1}/{args.epochs} complete.")
    
    # Final eval
    if eval_data:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  FINAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                          abl_dir / "best_model.pt")
    
    save_checkpoint(model, optimizer, scaler, step, args.epochs, best_acc,
                   abl_dir / "final.pt")
    
    elapsed = time.time() - t0
    log(f"\n{ablation_name} done: {step} steps, {positions_seen:,} pos, "
        f"{timedelta(seconds=int(elapsed))}, best={best_acc:.2%}")
    
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to exp161 checkpoint")
    ap.add_argument("--ablation", type=str, default="soft_A",
                    help="Ablation name or 'all' for full sweep")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="Fine-tuning LR (lower than pre-training)")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Label smoothing for hard CE (0 with soft targets)")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if args.ablation == "all":
        results = {}
        for name, config in ABLATIONS.items():
            log(f"\n{'#'*60}")
            log(f"# ABLATION: {name}")
            log(f"{'#'*60}\n")
            best = run_ablation(args, name, config, checkpoint_path)
            results[name] = best
            gc.collect()
            torch.cuda.empty_cache()
        
        log("\n" + "=" * 60)
        log("ABLATION SUMMARY")
        log("=" * 60)
        for name, acc in sorted(results.items(), key=lambda x: -x[1]):
            log(f"  {name:12s}: top1={acc:.2%}  ({ABLATIONS[name]['desc']})")
        
        # Save summary
        with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
            json.dump({k: {"top1": v, "desc": ABLATIONS[k]["desc"]}
                      for k, v in results.items()}, f, indent=2)
    else:
        if args.ablation not in ABLATIONS:
            print(f"Unknown ablation: {args.ablation}. Choose from: {list(ABLATIONS.keys())}")
            sys.exit(1)
        run_ablation(args, args.ablation, ABLATIONS[args.ablation], checkpoint_path)


if __name__ == "__main__":
    main()
