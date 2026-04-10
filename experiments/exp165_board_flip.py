"""exp165: Board-Flip to Side-to-Move Perspective + Compact + HL-Gauss.

Key insight (Monroe 2024 / ChessFormer): Always orient the board from the
side-to-move's perspective. When Black plays, flip the board vertically and
swap piece colors so the model always sees "my pieces" at the bottom.

Benefits:
  1. Halves the effective state space (no separate White/Black patterns)
  2. Stronger inductive bias — model learns unified "from my perspective" view
  3. Perfect data augmentation (every Black position teaches about White patterns)
  4. Removes need for turn embedding (always "my" turn)

Implementation: `flip_batch()` from board_flip.py transforms Black-to-move
positions before the model forward pass. No model architecture changes needed.

Base: exp161 (compact vocab + 128-bin HL-Gauss), from scratch.
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
from board_flip import flip_batch, build_flip_move_table

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp165_flip"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG

N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS


def _signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_remap_tensor():
    mapping = legacy_to_compact_map()
    t = torch.full((max(mapping.keys()) + 1,), -1, dtype=torch.long)
    for old, new in mapping.items():
        t[old] = new
    return t


def cp_to_win_percent(cp, mate):
    win_pct = torch.sigmoid(cp.float() / 111.0)
    mate_mask = mate != 0
    if mate_mask.any():
        win_pct = win_pct.clone()
        win_pct[mate_mask & (mate > 0)] = 1.0
        win_pct[mate_mask & (mate < 0)] = 0.0
    return win_pct.clamp(0.001, 0.999)


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


def _board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
    piece_map = {0: None, 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
                 7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'}
    rows = []
    for rank in range(7, -1, -1):
        row_str = ''
        empty = 0
        for file in range(8):
            sq = rank * 8 + file
            p = piece_map.get(ba_row[sq].item(), None)
            if p is None:
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += p
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)
    fen = '/'.join(rows)
    fen += ' w ' if turn_val == 0 else ' b '
    c = ''
    cv = castling_val
    if cv & 8: c += 'K'
    if cv & 4: c += 'Q'
    if cv & 2: c += 'k'
    if cv & 1: c += 'q'
    fen += c if c else '-'
    ev = int(ep_val)
    if 0 <= ev < 64:
        fen += f' {chr(ord("a") + ev % 8)}{ev // 8 + 1}'
    else:
        fen += ' -'
    fen += ' 0 1'
    return fen


def load_eval_data(eval_path, remap_tensor):
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    ba = raw["board_array"]
    turns = raw["turn"]
    castling = raw["castling"]
    ep = raw["ep_square"]
    move_idx = raw["move_idx"]
    cp = raw["cp"]
    mate = raw["mate"]

    fused = board_array_to_fused(ba)
    ep_file = ep_square_to_file(ep)

    eval_data = []
    for i in range(len(ba)):
        fen = _board_array_to_fen(ba[i], turns[i].item(), castling[i].item(), ep[i].item())
        board = chess.Board(fen)
        legacy_idx = move_idx[i].item()
        compact_idx = remap_tensor[legacy_idx].item() if legacy_idx < len(remap_tensor) else -1
        if compact_idx < 0:
            continue
        eval_data.append({
            "board": board,
            "move": IDX_TO_UCI[compact_idx],
            "compact_idx": compact_idx,
            "cp": cp[i].item(),
            "mate": mate[i].item(),
            "turn": turns[i].item(),
        })

    eval_tensors = {
        "fused_ids": fused[:len(eval_data)],
        "turn": turns[:len(eval_data)],
        "castling": castling[:len(eval_data)],
        "ep_file": ep_file[:len(eval_data)],
    }
    log(f"  Loaded {len(eval_data)} eval positions")
    return eval_data, eval_tensors


def run_eval(model, eval_data, eval_tensors, flip_move_table, batch_size=32):
    """Eval with board flip — flip Black positions before model, unflip predictions."""
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
                "turn": eval_tensors["turn"][idx].long().to(DEVICE),
                "castling": eval_tensors["castling"][idx].long().to(DEVICE),
                "ep_file": eval_tensors["ep_file"][idx].long().to(DEVICE),
            }

            # For eval, we need to track which positions were flipped
            turn = batch_input["turn"]
            black_mask = (turn == 1)

            ba = batch_input["fused_ids"]
            non_king = ((ba >= 1) & (ba <= 5)) | ((ba >= 7) & (ba <= 11))
            piece_counts = non_king.sum(dim=1)

            # Create dummy targets for flip_batch
            dummy_targets = torch.zeros(n, dtype=torch.long, device=DEVICE)
            flipped_input, _ = flip_batch(batch_input, dummy_targets, flip_move_table)

            with autocast('cuda', dtype=torch.float16):
                result = model(flipped_input)

            logits = result["policy_logits"].float()
            value_logits = result["value_logits"].float()
            pred_win_pct = value_logits_to_win_pct(value_logits)

            # For Black positions, the value is from "my" perspective (Black's)
            # We need to invert: Black's win% = 1 - model's win%
            if black_mask.any():
                pred_win_pct = pred_win_pct.clone()
                pred_win_pct[black_mask] = 1.0 - pred_win_pct[black_mask]

            for j, d in enumerate(chunk):
                board = d["board"]
                true_idx = d["compact_idx"]
                is_black = d["turn"] == 1

                l = logits[j].clone()

                # For Black: unflip the legal move mask and move indexing
                if is_black:
                    # Get legal moves for the flipped board representation
                    # The model output moves are in "flipped" space
                    # We need to unflip: model's move in flipped space → real move
                    # So we compare the model's top moves against the flipped target
                    flipped_target = flip_move_table[true_idx].item()

                    # Build legal mask in flipped space
                    # Legal moves from the original board, but flip each to get flipped indices
                    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=DEVICE)
                    for move in board.legal_moves:
                        uci = move.uci()
                        if uci in COMPACT_UCI_TO_IDX:
                            orig_idx = COMPACT_UCI_TO_IDX[uci]
                            flipped_idx = flip_move_table[orig_idx].item()
                            mask[flipped_idx] = True
                    l[~mask] = float("-inf")

                    hit = l.argmax().item() == flipped_target
                    topk = l.topk(min(3, l.shape[0])).indices.tolist()
                    in_top3 = flipped_target in topk
                else:
                    mask = legal_move_mask(board).to(DEVICE)
                    l[~mask] = float("-inf")
                    hit = l.argmax().item() == true_idx
                    topk = l.topk(min(3, l.shape[0])).indices.tolist()
                    in_top3 = true_idx in topk

                if hit:
                    correct += 1
                if in_top3:
                    top3 += 1

                pc = piece_counts[j].item()
                phase = 0 if pc >= 14 else (2 if pc < 6 else 1)
                phase_total[phase] += 1
                if hit:
                    phase_correct[phase] += 1

                # Value MAE
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
    state = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_acc": best_acc,
    }
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--compile", action="store_true")
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
    log("exp165: BOARD FLIP TO SIDE-TO-MOVE + Compact + HL-Gauss")
    log(f"  device: {DEVICE}")
    log(f"  vocab: {VOCAB_SIZE} moves (compact)")
    log(f"  value: {N_VALUE_BINS}-bin HL-Gauss (σ={SIGMA_HL_GAUSS:.4f})")
    log(f"  board_flip: ENABLED (Black positions flipped to White perspective)")
    log(f"  config: {MODEL_CONFIG}")

    remap_tensor = build_remap_tensor()
    log(f"  remap: {(remap_tensor >= 0).sum().item()} legacy→compact mappings")

    # Build flip move table
    flip_move_table = build_flip_move_table()
    log(f"  flip_table: {len(flip_move_table)} move mappings")

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
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors, flip_move_table.to(DEVICE))
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
        model = torch.compile(model)
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

    warmup_steps = min(2000, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump({"model": MODEL_CONFIG.to_dict(), "training": {
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "eff_bs": eff_bs, "lr": args.lr, "epochs": args.epochs,
            "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "init": "random", "vocab": "compact", "vocab_size": VOCAB_SIZE,
            "n_value_bins": N_VALUE_BINS, "sigma_hl_gauss": SIGMA_HL_GAUSS,
            "value_weight": args.value_weight, "board_flip": True,
        }}, f, indent=2)

    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path, remap_tensor)
        log(f"  Eval: {len(eval_data)} positions")

    remap_device = remap_tensor.to(DEVICE)
    flip_device = flip_move_table.to(DEVICE)

    if eval_data and start_step == 0:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors, flip_device)
        log(f"  RANDOM INIT: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        best_acc = acc

    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epochs, LR={args.lr}, warmup={warmup_steps}")
    log(f"  value_weight={args.value_weight}, label_smoothing={args.label_smoothing}")
    log(f"  BOARD FLIP ENABLED — all positions normalized to side-to-move perspective")
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

            # Remap legacy → compact
            move_targets = remap_device[move_targets_legacy]
            valid = move_targets >= 0
            if not valid.all():
                skipped = (~valid).sum().item()
                skipped_moves += skipped
                move_targets = move_targets.clamp(min=0)

            # Extract cp/mate for value targets
            cp_vals = batch_input.pop("cp").to(DEVICE)
            mate_vals = batch_input.pop("mate").to(DEVICE)
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            # BOARD FLIP: normalize to side-to-move perspective
            # For Black positions, flip board + swap colors + flip move targets
            # Value targets: flip win% for Black (their win is our loss)
            turn = batch_input["turn"]
            black_mask = (turn == 1)
            batch_input, move_targets = flip_batch(batch_input, move_targets, flip_device)

            # Flip value targets for Black positions
            if black_mask.any():
                win_pct = win_pct.clone()
                win_pct[black_mask] = 1.0 - win_pct[black_mask]

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing,
                    ignore_index=-1)
                v_loss = hl_gauss_loss(result["value_logits"], win_pct)
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

                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

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

                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"ETA {timedelta(seconds=int(eta))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0

                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")
                    if step % 10000 == 0 and step > 0:
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / f"step_{step}.pt")

                if step % args.eval_interval == 0 and eval_data:
                    torch.cuda.empty_cache()
                    acc, top3, val_mae = run_eval(model, eval_data, eval_tensors, flip_device)
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

        log(f"\nEpoch {epoch+1}/{args.epochs} complete. positions_seen={positions_seen:,}")
        if skipped_moves > 0:
            log(f"  Skipped {skipped_moves} moves with no compact mapping")
            skipped_moves = 0

        save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                       OUTPUT_DIR / f"epoch_{epoch+1}.pt")

        if eval_data:
            torch.cuda.empty_cache()
            acc, top3, val_mae = run_eval(model, eval_data, eval_tensors, flip_device)
            log(f"  EPOCH EVAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                              OUTPUT_DIR / "best_model.pt")
                log(f"  ** New best! top1={best_acc:.2%}")

    save_checkpoint(model, optimizer, scaler, step, args.epochs, best_acc,
                   OUTPUT_DIR / "best_model.pt")
    log(f"\nTraining complete. Best top1={best_acc:.2%}")


if __name__ == "__main__":
    main()
