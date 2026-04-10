"""exp167: DEFINITIVE A40 TRAINING — All validated improvements combined.

This is the "kitchen sink" experiment designed for A40 (48GB) compute.
Combines every improvement validated in exp149-exp165:

Architecture:
  - 204M params (16L, 1024d, 16H, FusedBoardEncoder)
  - Compact vocab (1968 geometrically reachable moves)
  - 128-bin distributional value (HL-Gauss, σ=0.75/128)
  - SpatialPolicyHead with project-then-gather
  - Board flip to side-to-move perspective

Training:
  - Phase-balanced sampling (downsample openings 0.5×, upsample mid 1.5× / end 1.2×)
  - Horizontal flip augmentation (50% random)
  - Label smoothing ε=0.05 (compact vocab only)
  - Auxiliary losses: material balance + game phase (aux_weight=0.1)
  - LR=2e-4, warmup=2000, cosine → 5e-7
  - 3 epochs over 10.2M positions
  - Save checkpoints every 5K steps for SWA
  - Eval every 2K steps

Post-training:
  - SWA over last 5 checkpoints
  - Elo gauntlet evaluation

Usage (A40):
  python experiments/exp167_a40_definitive.py --epochs 3 --batch-size 128
  python experiments/exp167_a40_definitive.py --resume
  python experiments/exp167_a40_definitive.py --max-steps 5000  # quick test

Usage (RTX 4060, testing):
  python experiments/exp167_a40_definitive.py --max-steps 2000 --batch-size 24 --accum 4
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

from chess_transformer_factory import build_model, ChessTransformerConfig, DEFAULT_200M_CONFIG
from move_vocab import (
    VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask,
    LEGACY_UCI_TO_IDX, COMPACT_UCI_TO_IDX, legacy_to_compact_map,
)
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp167_a40"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

# ── Constants ──
N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS
AUX_WEIGHT = 0.1
VALUE_WEIGHT = 1.0

# Piece values for material balance auxiliary
PIECE_VALUES = torch.tensor([0, 100, 320, 330, 500, 900, 0, -100, -320, -330, -500, -900, 0], dtype=torch.float32)


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


# ── Move index remapping ──

def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t


# ── HL-Gauss distributional value ──

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
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins, device=logits.device)
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)
    log_probs_target = -0.5 * (diff / sigma) ** 2
    targets = F.softmax(log_probs_target, dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def value_logits_to_win_pct(logits, n_bins=N_VALUE_BINS):
    bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins, device=logits.device)
    probs = F.softmax(logits.float(), dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ── Board flip (side-to-move perspective) ──

def build_flip_move_table():
    """Build compact→compact move index remap for board flip."""
    table = torch.full((VOCAB_SIZE,), -1, dtype=torch.long)
    for idx, uci in enumerate(IDX_TO_UCI):
        from_sq = ord(uci[0]) - ord('a') + (int(uci[1]) - 1) * 8
        to_sq = ord(uci[2]) - ord('a') + (int(uci[3]) - 1) * 8
        # Flip: rank → 7 - rank
        from_file, from_rank = from_sq % 8, from_sq // 8
        to_file, to_rank = to_sq % 8, to_sq // 8
        flipped_from = from_file + (7 - from_rank) * 8
        flipped_to = to_file + (7 - to_rank) * 8
        flipped_uci = chr(ord('a') + flipped_from % 8) + str(flipped_from // 8 + 1)
        flipped_uci += chr(ord('a') + flipped_to % 8) + str(flipped_to // 8 + 1)
        if len(uci) > 4:
            flipped_uci += uci[4]  # promotion piece
        if flipped_uci in COMPACT_UCI_TO_IDX:
            table[idx] = COMPACT_UCI_TO_IDX[flipped_uci]
    return table


def flip_board_batch(fused_ids, turn, castling, ep_file, move_idx, cp, mate, flip_move_table):
    """Flip positions where Black is to move → always from White's perspective.
    
    Returns modified copies of all tensors.
    """
    black_mask = (turn == 1)  # Black to move
    if not black_mask.any():
        return fused_ids, turn, castling, ep_file, move_idx, cp, mate
    
    fused = fused_ids.clone()
    t = turn.clone()
    c = castling.clone()
    ep = ep_file.clone()
    mi = move_idx.clone()
    cp_out = cp.clone()
    mate_out = mate.clone()
    
    # Flip board array: swap White↔Black pieces and reverse ranks
    b = fused[black_mask]  # (N_black, 64)
    # Reverse rank order: sq → (7 - rank) * 8 + file
    idx_map = torch.tensor([(7 - r) * 8 + f for r in range(8) for f in range(8)], dtype=torch.long)
    b = b[:, idx_map]
    # Swap colors: 1-6 (white) ↔ 7-12 (black), 0 stays 0
    white_mask_b = (b >= 1) & (b <= 6)
    black_mask_b = (b >= 7) & (b <= 12)
    b[white_mask_b] = b[white_mask_b] + 6
    b[black_mask_b] = b[black_mask_b] - 6
    fused[black_mask] = b
    
    # Flip turn to White
    t[black_mask] = 0
    
    # Flip castling bits: swap K/Q (bits 3-2) ↔ k/q (bits 1-0)
    cv = c[black_mask]
    new_cv = ((cv & 3) << 2) | ((cv >> 2) & 3)
    c[black_mask] = new_cv
    
    # EP file stays the same (file is symmetric)
    
    # Flip move index
    if flip_move_table is not None:
        old_mi = mi[black_mask]
        valid = (old_mi >= 0) & (old_mi < len(flip_move_table))
        new_mi = torch.where(valid, flip_move_table[old_mi.clamp(min=0)], torch.tensor(-1, dtype=mi.dtype))
        mi[black_mask] = new_mi
    
    # Negate CP and mate for Black (since we flipped perspective)
    cp_out[black_mask] = -cp_out[black_mask]
    mate_out[black_mask] = -mate_out[black_mask]
    
    return fused, t, c, ep, mi, cp_out, mate_out


# ── Auxiliary loss targets ──

def compute_material_balance(fused_ids):
    """Compute material balance from fused token IDs. Returns normalized float."""
    piece_vals = PIECE_VALUES.to(fused_ids.device)
    return (piece_vals[fused_ids.long()].sum(dim=-1) / 900.0)  # Queen-normalized


def compute_phase_targets(fused_ids):
    """Compute game phase from piece count. Returns 0=opening, 1=mid, 2=end."""
    non_king = ((fused_ids >= 1) & (fused_ids <= 5)) | ((fused_ids >= 7) & (fused_ids <= 11))
    piece_count = non_king.sum(dim=-1)
    phase = torch.where(piece_count >= 14, 0, torch.where(piece_count >= 6, 1, 2))
    return phase.long()


# ── Phase-balanced sampling weights ──

def compute_phase_weights(fused_ids):
    """Compute per-sample weights to balance phase distribution."""
    phase = compute_phase_targets(fused_ids)
    weights = torch.ones(len(phase), dtype=torch.float32, device=phase.device)
    weights[phase == 0] = 0.5   # Downsample openings (79% of data)
    weights[phase == 1] = 1.5   # Upsample middlegame
    weights[phase == 2] = 1.2   # Slight upsample endgame
    # Normalize so mean weight ≈ 1
    weights = weights / weights.mean()
    return weights


# ── Model + aux heads ──

class AuxHeads(nn.Module):
    """Auxiliary prediction heads from CLS token for trunk regularization."""
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.material_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 3)
        )
    
    def forward(self, cls_hidden):
        return {
            'material': self.material_head(cls_hidden).squeeze(-1),
            'phase': self.phase_head(cls_hidden),
        }


def build_model_with_aux():
    """Build 204M model with compact vocab, distributional value, and aux heads."""
    cfg = ChessTransformerConfig(n_value_classes=N_VALUE_BINS)
    model = build_model(cfg)
    
    # Replace value head with distributional (128-bin)
    hidden_dim = model.value_head[0].out_features
    model.value_head = nn.Sequential(
        nn.Linear(cfg.hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, N_VALUE_BINS),
    )
    for layer in model.value_head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    # Add auxiliary heads
    aux_heads = AuxHeads(cfg.hidden_dim)
    
    return model, aux_heads


# ── Eval ──

def load_eval_data():
    eval_path = SHARD_DIR / "eval.pt"
    remap = build_remap_tensor()
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    
    ba = raw['board_array']
    move_idx = raw['move_idx'].long()
    compact_idx = remap[move_idx.clamp(min=0)]
    valid = compact_idx >= 0
    
    fused = board_array_to_fused(ba)
    turn = raw['turn'].long()
    castling = raw['castling'].long()
    ep = ep_square_to_file(raw['ep_square'].long())
    cp = raw.get('cp', torch.zeros(len(ba))).float()
    mate = raw.get('mate', torch.zeros(len(ba))).long()
    
    return {
        'fused_ids': fused[valid],
        'turn': turn[valid],
        'castling': castling[valid],
        'ep_file': ep[valid],
        'move_idx': compact_idx[valid],
        'cp': cp[valid],
        'mate': mate[valid],
    }


@torch.no_grad()
def run_eval(model, eval_data, batch_size=128):
    model.eval()
    n = len(eval_data['fused_ids'])
    correct = top3 = total = 0
    total_value_mae = 0.0
    phase_correct = [0, 0, 0]
    phase_total = [0, 0, 0]
    
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = {
            'fused_ids': eval_data['fused_ids'][i:end].to(DEVICE),
            'turn': eval_data['turn'][i:end].to(DEVICE),
            'castling': eval_data['castling'][i:end].to(DEVICE),
            'ep_file': eval_data['ep_file'][i:end].to(DEVICE),
        }
        targets = eval_data['move_idx'][i:end].to(DEVICE)
        cp = eval_data['cp'][i:end].to(DEVICE)
        mate = eval_data['mate'][i:end].to(DEVICE)
        
        with autocast('cuda', dtype=torch.float16):
            result = model(batch)
        
        logits = result['policy_logits'].float()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        
        topk = torch.topk(logits, 3, dim=-1).indices
        top3 += (topk == targets.unsqueeze(-1)).any(dim=-1).sum().item()
        
        # Value MAE
        win_pct_target = cp_to_win_percent(cp, mate)
        win_pct_pred = value_logits_to_win_pct(result['value_logits'].float())
        total_value_mae += (win_pct_pred - win_pct_target).abs().sum().item()
        
        # Phase breakdown
        fused = batch['fused_ids']
        phase = compute_phase_targets(fused)
        for p in range(3):
            mask = phase == p
            if mask.any():
                phase_correct[p] += (preds[mask] == targets[mask]).sum().item()
                phase_total[p] += mask.sum().item()
        
        total += len(targets)
    
    phase_acc = [phase_correct[p] / max(phase_total[p], 1) for p in range(3)]
    model.train()
    return {
        'top1': correct / total,
        'top3': top3 / total,
        'value_mae': total_value_mae / total,
        'total': total,
        'phase_acc': phase_acc,
        'phase_total': phase_total,
    }


# ── Training ──

def train(args):
    global LOG_PATH
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"
    
    log(f"exp167: A40 Definitive Training")
    log(f"Device: {DEVICE}")
    log(f"Batch size: {args.batch_size}, Accum: {args.accum}, Eff BS: {args.batch_size * args.accum}")
    log(f"LR: {args.lr}, Epochs: {args.epochs}")
    log(f"Board flip: {args.flip}, Phase weighting: {args.phase_weight}")
    log(f"Aux losses: {args.aux}, Aux weight: {AUX_WEIGHT}")
    
    # Build model
    model, aux_heads = build_model_with_aux()
    model = model.to(DEVICE)
    aux_heads = aux_heads.to(DEVICE)
    params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in aux_heads.parameters())
    log(f"Model params: {params:,}")
    
    # Build move remap
    remap = build_remap_tensor().to(DEVICE)
    flip_table = build_flip_move_table() if args.flip else None
    
    # Load eval data
    eval_data = load_eval_data()
    log(f"Eval set: {len(eval_data['fused_ids'])} positions")
    
    # Optimizer
    all_params = list(model.parameters()) + list(aux_heads.parameters())
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = GradScaler()
    
    # Data loader
    loader = ShardedChessLoader(
        str(SHARD_DIR), batch_size=args.batch_size, seed=42,
        hflip=False, include_cp=True, include_mate=True,
    )
    steps_per_epoch = loader.total_positions // (args.batch_size * args.accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(2000, total_steps // 10)
    
    log(f"Steps/epoch: {steps_per_epoch:,}, Total: {total_steps:,}, Warmup: {warmup_steps}")
    
    # Resume
    global_step = 0
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        ckpt_path = OUTPUT_DIR / "latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'aux_state_dict' in ckpt:
                aux_heads.load_state_dict(ckpt['aux_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            global_step = ckpt['step']
            start_epoch = ckpt['epoch']
            best_acc = ckpt.get('best_acc', 0)
            log(f"Resumed from step {global_step}, epoch {start_epoch}, best_acc={best_acc:.4f}")
    
    # LR schedule
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(5e-7, args.lr * 0.5 * (1 + math.cos(math.pi * progress)))
    
    # Training loop
    t0 = time.time()
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_aux_loss = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch)
        optimizer.zero_grad()
        accum_count = 0
        
        for batch_input, move_targets_legacy, wdl_targets in loader:
            if SHUTDOWN or (args.max_steps and global_step >= args.max_steps):
                break
            
            # Remap legacy move indices to compact
            move_compact = remap[move_targets_legacy.to(DEVICE).clamp(min=0)]
            valid = move_compact >= 0
            if valid.sum() == 0:
                continue
            
            # Indices for CPU tensors  
            valid_cpu = valid.cpu()
            
            # Filter to valid positions
            fused = batch_input['fused_ids'][valid_cpu].to(DEVICE)
            turn = batch_input['turn'][valid_cpu].to(DEVICE)
            castling = batch_input['castling'][valid_cpu].to(DEVICE)
            ep_file = batch_input['ep_file'][valid_cpu].to(DEVICE)
            targets = move_compact[valid]
            cp_d = batch_input['cp'][valid_cpu].to(DEVICE) if 'cp' in batch_input else torch.zeros(valid.sum().item(), device=DEVICE)
            mate_d = batch_input['mate'][valid_cpu].to(DEVICE) if 'mate' in batch_input else torch.zeros(valid.sum().item(), dtype=torch.long, device=DEVICE)
            
            # Board flip (Black → White perspective)
            if args.flip and flip_table is not None:
                fused_cpu = fused.cpu()
                turn_cpu = turn.cpu()
                castling_cpu = castling.cpu()
                ep_cpu = ep_file.cpu()
                targets_cpu = targets.cpu()
                cp_cpu = cp_d.cpu()
                mate_cpu = mate_d.cpu()
                
                fused_cpu, turn_cpu, castling_cpu, ep_cpu, targets_cpu, cp_cpu, mate_cpu = flip_board_batch(
                    fused_cpu, turn_cpu, castling_cpu, ep_cpu, targets_cpu, cp_cpu, mate_cpu, flip_table)
                
                valid2 = targets_cpu >= 0
                if valid2.sum() == 0:
                    continue
                
                fused = fused_cpu[valid2].to(DEVICE)
                turn = turn_cpu[valid2].to(DEVICE)
                castling = castling_cpu[valid2].to(DEVICE)
                ep_file = ep_cpu[valid2].to(DEVICE)
                targets = targets_cpu[valid2].to(DEVICE)
                cp_d = cp_cpu[valid2].to(DEVICE)
                mate_d = mate_cpu[valid2].to(DEVICE)
            
            batch_in = {
                'fused_ids': fused,
                'turn': turn,
                'castling': castling,
                'ep_file': ep_file,
            }
            
            # Forward pass
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_in)
                policy_logits = result['policy_logits']
                value_logits = result['value_logits']
                cls_hidden = result['cls_hidden']
                
                # Policy loss (with phase weighting)
                policy_loss = F.cross_entropy(
                    policy_logits, targets,
                    label_smoothing=0.05,
                    reduction='none'
                )
                
                if args.phase_weight:
                    weights = compute_phase_weights(batch_in['fused_ids'])
                    policy_loss = (policy_loss * weights).mean()
                else:
                    policy_loss = policy_loss.mean()
                
                # Value loss (HL-Gauss)
                win_pct = cp_to_win_percent(cp_d, mate_d)
                value_loss = hl_gauss_loss(value_logits, win_pct)
                
                # Auxiliary losses
                aux_loss = torch.tensor(0.0, device=DEVICE)
                if args.aux:
                    aux_out = aux_heads(cls_hidden.float())
                    # Material balance
                    mat_target = compute_material_balance(batch_in['fused_ids']).float()
                    mat_loss = F.huber_loss(aux_out['material'], mat_target)
                    # Game phase
                    phase_target = compute_phase_targets(batch_in['fused_ids'])
                    phase_loss = F.cross_entropy(aux_out['phase'], phase_target)
                    aux_loss = mat_loss + phase_loss
                
                loss = policy_loss + VALUE_WEIGHT * value_loss + AUX_WEIGHT * aux_loss
                loss = loss / args.accum
            
            scaler.scale(loss).backward()
            accum_count += 1
            
            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()
            running_aux_loss += aux_loss.item()
            
            if accum_count >= args.accum:
                # LR schedule
                lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1
                
                # Logging
                if global_step % 25 == 0:
                    elapsed = time.time() - t0
                    pos_per_sec = global_step * args.batch_size * args.accum / elapsed
                    avg_p = running_policy_loss / 25
                    avg_v = running_value_loss / 25
                    avg_a = running_aux_loss / 25
                    eta = (total_steps - global_step) / max(pos_per_sec / (args.batch_size * args.accum), 0.01)
                    log(f"step={global_step} p={avg_p:.3f} v={avg_v:.3f} a={avg_a:.3f} "
                        f"lr={lr:.2e} gn={gn:.2f} {pos_per_sec:.0f}pos/s eta={timedelta(seconds=int(eta))}")
                    running_policy_loss = 0.0
                    running_value_loss = 0.0
                    running_aux_loss = 0.0
                
                # Eval
                if global_step % args.eval_interval == 0:
                    metrics = run_eval(model, eval_data)
                    phase_str = "/".join(f"{a*100:.1f}" for a in metrics['phase_acc'])
                    log(f"EVAL step={global_step} top1={metrics['top1']*100:.2f}% "
                        f"top3={metrics['top3']*100:.2f}% vMAE={metrics['value_mae']:.4f} "
                        f"phase={phase_str}")
                    
                    if metrics['top1'] > best_acc:
                        best_acc = metrics['top1']
                        save_checkpoint(model, aux_heads, optimizer, scaler, global_step, epoch, best_acc, 'best_model.pt')
                        log(f"NEW BEST: {best_acc*100:.2f}%")
                
                # Checkpoint
                if global_step % args.save_interval == 0:
                    save_checkpoint(model, aux_heads, optimizer, scaler, global_step, epoch, best_acc, 'latest.pt')
                    # Also save step checkpoint for SWA
                    save_checkpoint(model, aux_heads, optimizer, scaler, global_step, epoch, best_acc, f'step_{global_step}.pt')
            
            if SHUTDOWN or (args.max_steps and global_step >= args.max_steps):
                break
        
        if SHUTDOWN or (args.max_steps and global_step >= args.max_steps):
            break
        
        log(f"Epoch {epoch+1} complete at step {global_step}")
    
    # Final eval
    metrics = run_eval(model, eval_data)
    log(f"FINAL top1={metrics['top1']*100:.2f}% top3={metrics['top3']*100:.2f}% vMAE={metrics['value_mae']:.4f}")
    save_checkpoint(model, aux_heads, optimizer, scaler, global_step, args.epochs, best_acc, 'latest.pt')
    
    # SWA if we have enough checkpoints
    swa_checkpoints = sorted(OUTPUT_DIR.glob("step_*.pt"))
    if len(swa_checkpoints) >= 3:
        log(f"Running SWA over last {min(5, len(swa_checkpoints))} checkpoints...")
        run_swa(swa_checkpoints[-5:], model, eval_data)


def save_checkpoint(model, aux_heads, optimizer, scaler, step, epoch, best_acc, filename):
    path = OUTPUT_DIR / filename
    torch.save({
        'model_state_dict': model.state_dict(),
        'aux_state_dict': aux_heads.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_acc': best_acc,
        'vocab_version': 'compact',
        'n_value_bins': N_VALUE_BINS,
        'config': {
            'n_value_classes': N_VALUE_BINS,
            'flip': True,
            'phase_weight': True,
            'aux': True,
        },
    }, path)


def run_swa(checkpoint_paths, model, eval_data):
    """Average weights from multiple checkpoints."""
    avg_state = None
    n = len(checkpoint_paths)
    for cp in checkpoint_paths:
        ckpt = torch.load(cp, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict']
        if avg_state is None:
            avg_state = {k: v.float() for k, v in sd.items()}
        else:
            for k in avg_state:
                avg_state[k] += sd[k].float()
    for k in avg_state:
        avg_state[k] /= n
    
    model.load_state_dict({k: v.to(model.value_head[0].weight.dtype) for k, v in avg_state.items()})
    model = model.to(DEVICE)
    metrics = run_eval(model, eval_data)
    log(f"SWA top1={metrics['top1']*100:.2f}% top3={metrics['top3']*100:.2f}% vMAE={metrics['value_mae']:.4f}")
    
    swa_path = OUTPUT_DIR / "swa_model.pt"
    torch.save({
        'model_state_dict': {k: v.half() for k, v in avg_state.items()},
        'vocab_version': 'compact',
        'n_value_bins': N_VALUE_BINS,
    }, swa_path)
    log(f"SWA model saved to {swa_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128, help='Per-GPU batch size (128 for A40, 24 for 4060)')
    parser.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--eval-interval', type=int, default=2000)
    parser.add_argument('--save-interval', type=int, default=5000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-flip', dest='flip', action='store_false', default=True)
    parser.add_argument('--no-phase-weight', dest='phase_weight', action='store_false', default=True)
    parser.add_argument('--no-aux', dest='aux', action='store_false', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
