"""exp139: Large-scale training on 10M+ positions from lichess-sf.

Root cause analysis: The 204M model was trained on only 224K positions.
Ruoss et al. (2024) achieved 2895 ELO (without search!) with a 270M transformer 
on 10M positions. Our 204M model has ~17% top-1 policy accuracy vs their ~80%.
The gap is DATA, not architecture.

Strategy:
  Phase 1: Download + pretokenize 10M positions from avewright/chess-positions-lichess-sf
  Phase 2: Train 204M model (continuation from best checkpoint) with proper LR schedule
  Phase 3: Evaluate ELO

Hardware: RTX 4060 8GB
  - bs=16 with gradient checkpointing → ~50-70 pos/s
  - 10M positions × 1 epoch ≈ 40-55 hours
  - Use gradient accumulation for effective bs=128

Usage:
  # Phase 1 (data prep, ~30 min):
  python experiments/exp139_massive_train.py --phase download --target-positions 10000000
  
  # Phase 2 (training, ~40-55 hours):
  python experiments/exp139_massive_train.py --phase train --epochs 1 --lr 2e-5
  
  # Phase 3 (eval after training):
  python experiments/exp139_massive_train.py --phase eval
  
  # All phases:
  python experiments/exp139_massive_train.py --phase all --target-positions 10000000
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
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
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

from chess_transformer_factory import build_model, ChessTransformerConfig
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, move_to_index, legal_move_mask
from data_loader import (
    get_hf_dataset_layout, _tokenize_parquet, _hf_token,
    pretokenize_parquet_to_shards, ShardedChessLoader,
    board_array_to_fused, ep_square_to_file, compute_wdl,
    get_eval_batch_input, compute_phase,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp139_massive_train"
SHARD_DIR = OUTPUT_DIR / "shards"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
INIT_CHECKPOINT = ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt"
HF_REPO = "avewright/chess-positions-lichess-sf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_PATH = None
SHUTDOWN = False


def _signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    log(f"\n[SIGNAL] Shutdown requested. Will save checkpoint and exit.")

signal.signal(signal.SIGINT, _signal_handler)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Download + Pretokenize
# ═══════════════════════════════════════════════════════════════════════

def phase_download(target_positions=10_000_000, min_depth=10):
    """Download parquets from HF and pretokenize into shards."""
    from huggingface_hub import hf_hub_download

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check existing shards
    existing = sorted(SHARD_DIR.glob("shard_*.pt"))
    if existing:
        total_existing = 0
        for sf in existing:
            data = torch.load(sf, map_location="cpu", weights_only=True)
            total_existing += data["board_array"].shape[0]
            del data
        log(f"Found {len(existing)} existing shards with {total_existing:,} positions")
        if total_existing >= target_positions:
            log(f"Already have enough data ({total_existing:,} >= {target_positions:,})")
            return total_existing
        log(f"Need {target_positions - total_existing:,} more positions")
    
    # Get file list from HF
    log(f"Fetching layout from {HF_REPO}...")
    layout = get_hf_dataset_layout(HF_REPO)
    token = _hf_token()
    
    # Use src_train parquets (smaller files, faster download)
    src_files = layout.get("train_src", [])
    if not src_files:
        # Fall back to main train files
        src_files = layout.get("train", [])
    log(f"Available: {len(src_files)} parquet files")
    
    # Download parquets to temp dir, then pretokenize
    temp_dir = OUTPUT_DIR / "temp_parquets"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    total_positions = 0
    shard_positions_per = 1_000_000  # 1M per shard
    
    # Accumulators for current shard
    all_ba, all_turn, all_cast, all_ep, all_midx, all_cp, all_mate = (
        [], [], [], [], [], [], [])
    current_count = 0
    shard_idx = len(existing)
    
    # Track eval data
    eval_fens = []
    eval_ba, eval_turn, eval_cast, eval_ep, eval_midx, eval_cp, eval_mate = (
        [], [], [], [], [], [], [])
    n_eval_target = 5000
    eval_saved = (OUTPUT_DIR / "shards" / "eval.pt").exists()
    
    t0 = time.time()
    
    for fi, pq_name in enumerate(src_files):
        if total_positions >= target_positions:
            break
            
        try:
            local_path = hf_hub_download(
                HF_REPO, pq_name, repo_type="dataset",
                token=token, revision=layout["revision"],
            )
        except Exception as e:
            log(f"  Skip {pq_name}: {e}")
            continue
        
        raw = _tokenize_parquet(local_path)
        if raw is None:
            continue
        
        n = raw["board_array"].shape[0]
        
        # Filter by min_depth if available
        if min_depth > 0 and "depth" in raw and raw["depth"] is not None:
            depth = raw["depth"]
            mask = depth >= min_depth
            if hasattr(mask, 'numpy'):
                mask = mask.numpy()
            n_before = n
            for key in raw:
                if raw[key] is not None and hasattr(raw[key], '__len__') and len(raw[key]) == n_before:
                    raw[key] = raw[key][mask]
            n = raw["board_array"].shape[0]
        
        # Siphon eval data first
        if not eval_saved and len(eval_fens) < n_eval_target:
            take = min(n_eval_target - len(eval_fens), n)
            eval_ba.append(raw["board_array"][:take])
            eval_turn.append(raw["turn"][:take])
            eval_cast.append(raw["castling"][:take])
            eval_ep.append(raw["ep_square"][:take])
            eval_midx.append(raw["move_idx"][:take])
            eval_cp.append(raw["cp"][:take])
            eval_mate.append(raw["mate"][:take])
            if "fen" in raw and raw["fen"] is not None:
                eval_fens.extend(raw["fen"][:take])
            else:
                eval_fens.extend([""] * take)  # placeholder
            
            # Remove eval rows from training data
            if take < n:
                for key in raw:
                    if raw[key] is not None and hasattr(raw[key], '__len__') and len(raw[key]) == n:
                        raw[key] = raw[key][take:]
                n = raw["board_array"].shape[0]
            else:
                n = 0
        
        if n > 0:
            all_ba.append(raw["board_array"])
            all_turn.append(raw["turn"])
            all_cast.append(raw["castling"])
            all_ep.append(raw["ep_square"])
            all_midx.append(raw["move_idx"])
            all_cp.append(raw["cp"])
            all_mate.append(raw["mate"])
            current_count += n
            total_positions += n
        
        del raw
        
        # Flush shard when we have enough
        while current_count >= shard_positions_per:
            _flush_accumulated_shard(
                SHARD_DIR, shard_idx, 
                all_ba, all_turn, all_cast, all_ep, all_midx, all_cp, all_mate,
                shard_positions_per
            )
            # Keep remainder
            remainder = current_count - shard_positions_per
            if remainder > 0:
                all_ba = [np.concatenate(all_ba)[shard_positions_per:]]
                all_turn = [np.concatenate(all_turn)[shard_positions_per:]]
                all_cast = [np.concatenate(all_cast)[shard_positions_per:]]
                all_ep = [np.concatenate(all_ep)[shard_positions_per:]]
                all_midx = [np.concatenate(all_midx)[shard_positions_per:]]
                all_cp = [np.concatenate(all_cp)[shard_positions_per:]]
                all_mate = [np.concatenate(all_mate)[shard_positions_per:]]
            else:
                all_ba, all_turn, all_cast, all_ep, all_midx, all_cp, all_mate = (
                    [], [], [], [], [], [], [])
            current_count = remainder
            shard_idx += 1
            
            elapsed = time.time() - t0
            rate = total_positions / max(elapsed, 1)
            eta = (target_positions - total_positions) / max(rate, 1)
            log(f"  Shard {shard_idx}: {total_positions:,}/{target_positions:,} "
                f"({elapsed:.0f}s, {rate:.0f} pos/s, ETA {timedelta(seconds=int(eta))})")
        
        # Progress
        if (fi + 1) % 5 == 0:
            elapsed = time.time() - t0
            log(f"  File {fi+1}/{len(src_files)}: {total_positions:,} positions "
                f"({elapsed:.0f}s)")
    
    # Flush remaining
    if current_count > 0:
        _flush_accumulated_shard(
            SHARD_DIR, shard_idx,
            all_ba, all_turn, all_cast, all_ep, all_midx, all_cp, all_mate,
            current_count
        )
        shard_idx += 1
        log(f"  Final shard {shard_idx}: {current_count:,} positions")
    
    # Save eval data
    if not eval_saved and eval_fens:
        n_eval = len(eval_fens)
        eval_path = SHARD_DIR / "eval.pt"
        torch.save({
            "board_array": torch.from_numpy(np.concatenate(eval_ba)[:n_eval]),
            "turn": torch.from_numpy(np.concatenate(eval_turn)[:n_eval]),
            "castling": torch.from_numpy(np.concatenate(eval_cast)[:n_eval]),
            "ep_square": torch.from_numpy(np.concatenate(eval_ep)[:n_eval]),
            "move_idx": torch.from_numpy(np.concatenate(eval_midx)[:n_eval]),
            "cp": torch.from_numpy(np.concatenate(eval_cp)[:n_eval]),
            "mate": torch.from_numpy(np.concatenate(eval_mate)[:n_eval]),
            "fen": eval_fens[:n_eval],
        }, eval_path)
        log(f"  Eval: {n_eval:,} positions saved")
    
    elapsed = time.time() - t0
    log(f"\nDownload complete: {total_positions:,} positions in {shard_idx} shards "
        f"({elapsed:.0f}s)")
    return total_positions


def _flush_accumulated_shard(shard_dir, shard_idx, 
                             ba_list, turn_list, cast_list, ep_list,
                             midx_list, cp_list, mate_list, n):
    """Concatenate accumulated arrays and save as a shard."""
    shard_path = shard_dir / f"shard_{shard_idx:05d}.pt"
    tmp_path = shard_dir / f"shard_{shard_idx:05d}.pt.tmp"
    
    ba = np.concatenate(ba_list)[:n]
    turn = np.concatenate(turn_list)[:n]
    castling = np.concatenate(cast_list)[:n]
    ep = np.concatenate(ep_list)[:n]
    midx = np.concatenate(midx_list)[:n]
    cp = np.concatenate(cp_list)[:n]
    mate = np.concatenate(mate_list)[:n]
    
    torch.save({
        "board_array": torch.from_numpy(ba),
        "turn": torch.from_numpy(turn),
        "castling": torch.from_numpy(castling),
        "ep_square": torch.from_numpy(ep),
        "move_idx": torch.from_numpy(midx),
        "cp": torch.from_numpy(cp),
        "mate": torch.from_numpy(mate),
        "depth": torch.zeros(n, dtype=torch.int16),  # placeholder
    }, tmp_path)
    os.replace(str(tmp_path), str(shard_path))


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Training
# ═══════════════════════════════════════════════════════════════════════

def phase_train(args):
    """Train 204M model on sharded data with gradient checkpointing."""
    import chess
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model from best checkpoint
    log(f"Loading model from {INIT_CHECKPOINT}...")
    model = build_model()
    
    # Check for resume checkpoint first
    resume_path = CHECKPOINT_DIR / "latest.pt"
    start_step = 0
    start_epoch = 0
    best_acc = 0.0
    
    if resume_path.exists() and not args.fresh:
        log(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        log(f"  Resumed at step {start_step}, epoch {start_epoch}, best_acc={best_acc:.2%}")
    elif INIT_CHECKPOINT.exists():
        ckpt = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        log(f"  Loaded init checkpoint")
    else:
        log(f"WARNING: No checkpoint found, training from scratch")
    
    model.to(DEVICE)
    
    # Enable gradient checkpointing — patch model forward to checkpoint each layer
    if args.grad_checkpoint:
        model.transformer.enable_nested_tensor = False
        _orig_transformer_forward = model.transformer.forward
        
        def _checkpointed_transformer_forward(src, mask=None, src_key_padding_mask=None):
            output = src
            for layer in model.transformer.layers:
                output = grad_checkpoint(
                    layer, output, mask, src_key_padding_mask,
                    use_reentrant=False
                )
            if model.transformer.norm is not None:
                output = model.transformer.norm(output)
            return output
        
        model.transformer.forward = _checkpointed_transformer_forward
        log(f"  Gradient checkpointing ENABLED (16 layers)")
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"  Model: {n_params:.1f}M parameters")
    
    # Set up data loader
    log(f"Setting up ShardedChessLoader from {SHARD_DIR}...")
    loader = ShardedChessLoader(
        SHARD_DIR, 
        batch_size=args.batch_size,
        encoder_type="fused",
        device=DEVICE,
        seed=42,
    )
    total_positions = loader.total_positions
    steps_per_epoch = len(loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    
    log(f"  {total_positions:,} positions, bs={args.batch_size}, "
        f"accum={args.accum_steps} -> eff_bs={args.batch_size * args.accum_steps}")
    log(f"  {steps_per_epoch:,} steps/epoch, {total_steps:,} total steps")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler('cuda')
    
    # LR schedule: linear warmup + cosine decay 
    warmup_steps = min(500, total_steps // 20)
    
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)
    
    # Load eval data
    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = _load_eval_data(eval_path)
        log(f"  Eval: {len(eval_data)} positions")
    
    # Training loop
    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epoch(s), LR={args.lr}, warmup={warmup_steps}")
    log(f"{'='*60}")
    
    model.train()
    step = start_step
    accum_loss_policy = 0.0
    accum_loss_value = 0.0
    accum_batches = 0
    positions_seen = step * args.batch_size * args.accum_steps
    t0 = time.time()
    t_last_log = t0
    
    for epoch in range(start_epoch, args.epochs):
        loader.set_epoch(epoch)
        
        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN:
                _save_checkpoint(model, optimizer, step, epoch, best_acc, 
                               CHECKPOINT_DIR / "latest.pt")
                log(f"Shutdown checkpoint saved at step {step}")
                return
            
            # Forward pass with mixed precision
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                
                # Policy loss (cross-entropy)
                policy_loss = F.cross_entropy(
                    result["policy_logits"], move_targets)
                
                # Value loss (cross-entropy on WDL)
                value_loss = F.cross_entropy(
                    result["value_logits"], wdl_targets)
                
                loss = policy_loss + args.value_weight * value_loss
                loss = loss / args.accum_steps
            
            scaler.scale(loss).backward()
            
            accum_loss_policy += policy_loss.item()
            accum_loss_value += value_loss.item()
            accum_batches += 1
            positions_seen += move_targets.shape[0]
            
            if accum_batches >= args.accum_steps:
                # Gradient step
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                step += 1
                
                # Update LR
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                
                # Logging
                if step % args.log_interval == 0:
                    avg_p = accum_loss_policy / accum_batches
                    avg_v = accum_loss_value / accum_batches
                    elapsed = time.time() - t0
                    pos_per_s = positions_seen / max(elapsed, 1)
                    eta = (total_positions * args.epochs - positions_seen) / max(pos_per_s, 1)
                    
                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p_loss={avg_p:.4f} v_loss={avg_v:.4f} | "
                        f"lr={lr:.2e} | {pos_per_s:.0f} pos/s | "
                        f"pos={positions_seen:,} | "
                        f"ETA {timedelta(seconds=int(eta))}")
                    
                    accum_loss_policy = 0.0
                    accum_loss_value = 0.0
                    accum_batches = 0
                
                # Eval
                if step % args.eval_interval == 0 and eval_data:
                    acc, top3, val_acc = _run_eval(model, eval_data, eval_tensors)
                    log(f"  EVAL step {step}: acc={acc:.2%} top3={top3:.2%} val={val_acc:.2%}")
                    if acc > best_acc:
                        best_acc = acc
                        _save_checkpoint(model, optimizer, step, epoch, best_acc,
                                       CHECKPOINT_DIR / "best_model.pt")
                        log(f"  ** New best! acc={best_acc:.2%}")
                    model.train()
                
                # Save checkpoint
                if step % args.save_interval == 0:
                    _save_checkpoint(model, optimizer, step, epoch, best_acc,
                                   CHECKPOINT_DIR / "latest.pt")
                
                # Reset accumulator counter
                accum_batches = 0
                accum_loss_policy = 0.0
                accum_loss_value = 0.0
        
        log(f"\nEpoch {epoch+1} complete. positions_seen={positions_seen:,}")
    
    # Final save
    _save_checkpoint(model, optimizer, step, start_epoch + args.epochs, best_acc,
                    CHECKPOINT_DIR / "latest.pt")
    _save_checkpoint(model, optimizer, step, start_epoch + args.epochs, best_acc,
                    CHECKPOINT_DIR / "best_model.pt")
    
    elapsed = time.time() - t0
    log(f"\nTraining complete: {step:,} steps, {positions_seen:,} positions, "
        f"{elapsed:.0f}s ({positions_seen/max(elapsed,1):.0f} pos/s)")
    log(f"Best accuracy: {best_acc:.2%}")


def _load_eval_data(eval_path):
    """Load eval.pt and build eval data/tensors."""
    import chess
    
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    
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
        if ev >= 0 and ev < 64:
            ep_str = chr(ord('a') + ev % 8) + str(ev // 8 + 1)
        else:
            ep_str = "-"
        return f"{board_str} {turn_str} {castle_str} {ep_str} 0 1"
    
    eval_data = []
    surviving = []
    wdl = compute_wdl(raw["cp"], raw["mate"])
    fens = raw.get("fen", [])
    
    for i in range(raw["board_array"].shape[0]):
        try:
            # Try stored FEN first, else reconstruct from board_array
            fen = fens[i] if i < len(fens) and fens[i] else None
            if not fen:
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
                "phase": "unknown",
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


def _run_eval(model, eval_data, eval_tensors, batch_size=128):
    """Quick eval: top-1 acc, top-3 acc, value acc."""
    model.eval()
    correct = top3 = val_correct = total = 0
    
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
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk:
                    top3 += 1
                
                # Value accuracy
                vp = F.softmax(value_logits[j], dim=-1)
                pred_class = vp.argmax().item()
                wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                
                total += 1
    
    model.train()
    return correct / max(total, 1), top3 / max(total, 1), val_correct / max(total, 1)


def _save_checkpoint(model, optimizer, step, epoch, best_acc, path):
    """Save checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_acc": best_acc,
    }, tmp)
    os.replace(str(tmp), str(path))


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Evaluation
# ═══════════════════════════════════════════════════════════════════════

def phase_eval(args):
    """Run ELO gauntlet with trained checkpoint."""
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "latest.pt"
    if not ckpt_path.exists():
        log("No checkpoint found for eval")
        return
    
    log(f"ELO evaluation with {ckpt_path}")
    log("Run: python play.py --checkpoint <path> --sf-elo 1900 --games 32")
    

# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="exp139: Large-scale training")
    ap.add_argument("--phase", default="all", choices=["download", "train", "eval", "all"])
    ap.add_argument("--target-positions", type=int, default=10_000_000)
    ap.add_argument("--min-depth", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--fresh", action="store_true", help="Don't resume from latest checkpoint")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True,
                    help="Enable gradient checkpointing (saves VRAM, allows larger batch)")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    args = ap.parse_args()
    
    global LOG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"
    
    log(f"exp139: Large-scale training on {args.target_positions:,} positions")
    log(f"  device: {DEVICE}")
    log(f"  phase: {args.phase}")
    
    if args.phase in ("download", "all"):
        phase_download(args.target_positions, args.min_depth)
    
    if args.phase in ("train", "all"):
        phase_train(args)
    
    if args.phase in ("eval", "all"):
        phase_eval(args)


if __name__ == "__main__":
    main()
