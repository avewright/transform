"""exp164: Auxiliary Losses for Trunk Regularization.

Adds lightweight auxiliary supervision heads from the CLS token:
  1. Material balance prediction (Huber loss on normalized piece values)
  2. Game phase classification (3-class CE on opening/middlegame/endgame)

Hypothesis: Forcing the trunk to encode basic chess properties (material, phase)
through auxiliary gradient signal improves both policy and value representations.
The auxiliary targets are computed on-the-fly from fused_ids — no data pipeline changes.

References:
  - Czech et al. 2023: +100 Elo from better input features
  - AlphaZero improvements doc: auxiliary losses (material, phase) as dense supervision
  - Multi-task learning: orthogonal auxiliaries regularize shared representations

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

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp164_aux"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

MODEL_CONFIG = DEFAULT_200M_CONFIG

N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS

# Piece values for material balance computation (indexed by fused_id 0-12)
# 0=empty, 1-6=White P/N/B/R/Q/K, 7-12=Black P/N/B/R/Q/K
PIECE_VALUES = torch.tensor(
    [0, 100, 320, 330, 500, 900, 0, -100, -320, -330, -500, -900, 0],
    dtype=torch.float32
)
MATERIAL_SCALE = 900.0  # normalize by queen value

# Phase class weights (inverse frequency from training data analysis):
# opening=79.5%, mid=7.0%, end=13.5% → reweight to balance gradient signal
PHASE_WEIGHTS = torch.tensor([1.0 / 0.795, 1.0 / 0.070, 1.0 / 0.135], dtype=torch.float32)
PHASE_WEIGHTS = PHASE_WEIGHTS / PHASE_WEIGHTS.sum() * 3.0  # normalize so mean weight = 1


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


# ── Auxiliary Heads ────────────────────────────────────────────────────

class AuxiliaryHeads(nn.Module):
    """Lightweight auxiliary heads from CLS token for trunk regularization."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Material balance: predict normalized piece value sum
        self.material_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Game phase: 3-class classification (opening/middlegame/endgame)
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cls_hidden: torch.Tensor):
        material = self.material_head(cls_hidden).squeeze(-1)  # (B,)
        phase = self.phase_head(cls_hidden)  # (B, 3)
        return material, phase


def compute_aux_targets(fused_ids: torch.Tensor, device: torch.device):
    """Compute material balance and game phase from fused_ids.

    Args:
        fused_ids: (B, 64) int tensor, values 0-12
    Returns:
        material_target: (B,) float, normalized material balance
        phase_target: (B,) long, 0=opening, 1=middlegame, 2=endgame
    """
    pv = PIECE_VALUES.to(device)
    material_target = pv[fused_ids].sum(dim=1) / MATERIAL_SCALE  # (B,)

    # Count non-king, non-empty pieces (ids 1-5 for white, 7-11 for black)
    non_king = ((fused_ids >= 1) & (fused_ids <= 5)) | ((fused_ids >= 7) & (fused_ids <= 11))
    piece_count = non_king.sum(dim=1)  # (B,)

    # Phase: >=14 pieces = opening(0), 6-13 = mid(1), <6 = end(2)
    phase_target = torch.where(
        piece_count >= 14, torch.tensor(0, device=device),
        torch.where(piece_count >= 6, torch.tensor(1, device=device),
                     torch.tensor(2, device=device))
    )
    return material_target.float(), phase_target.long()


# ── Model construction ─────────────────────────────────────────────────

def build_model_with_aux(config):
    """Build model with compact vocab + HL-Gauss + auxiliary heads."""
    model = build_model(config)
    # Replace 3-class WDL value head → 128-bin distributional
    hidden_dim = config.hidden_dim
    old_head = model.value_head
    assert isinstance(old_head, nn.Sequential) and len(old_head) == 3
    value_hidden = old_head[0].out_features  # 512
    model.value_head = nn.Sequential(
        nn.Linear(hidden_dim, value_hidden),
        nn.ReLU(),
        nn.Linear(value_hidden, N_VALUE_BINS),
    )
    for layer in model.value_head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # Add auxiliary heads
    model.aux_heads = AuxiliaryHeads(hidden_dim)

    return model


# ── Eval (same as exp161) ──────────────────────────────────────────────

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
    if cv & 1: c += 'K'
    if cv & 2: c += 'Q'
    if cv & 4: c += 'k'
    if cv & 8: c += 'q'
    fen += c if c else '-'
    if ep_val > 0:
        file_char = chr(ord('a') + ep_val - 1)
        rank_char = '6' if turn_val == 0 else '3'
        fen += f' {file_char}{rank_char}'
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
        legal = list(board.legal_moves)
        target_uci = IDX_TO_UCI.get(compact_idx, "")
        eval_data.append({
            "board": board,
            "move": target_uci,
            "compact_idx": compact_idx,
            "cp": cp[i].item(),
            "mate": mate[i].item(),
        })

    eval_tensors = {
        "fused_ids": fused[:len(eval_data)],
        "turn": turns[:len(eval_data)],
        "castling": castling[:len(eval_data)],
        "ep_file": ep_file[:len(eval_data)],
    }
    log(f"  Loaded {len(eval_data)} eval positions")
    return eval_data, eval_tensors


def run_eval(model, eval_data, eval_tensors, batch_size=32):
    model.eval()
    correct = top3 = total = 0
    total_value_mae = 0.0
    phase_correct = [0, 0, 0]
    phase_total = [0, 0, 0]
    # Track auxiliary accuracy
    mat_ae_sum = 0.0
    phase_aux_correct = 0

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

            # Compute auxiliary targets
            mat_tgt, phase_tgt = compute_aux_targets(batch_input["fused_ids"], DEVICE)

            ba = batch_input["fused_ids"]
            non_king = ((ba >= 1) & (ba <= 5)) | ((ba >= 7) & (ba <= 11))
            piece_counts = non_king.sum(dim=1)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)

            logits = result["policy_logits"].float()
            value_logits = result["value_logits"].float()
            pred_win_pct = value_logits_to_win_pct(value_logits)

            # Auxiliary predictions
            if hasattr(model, 'aux_heads'):
                cls_h = result["cls_hidden"].float()
                mat_pred, phase_pred = model.aux_heads(cls_h)
                mat_ae_sum += (mat_pred - mat_tgt).abs().sum().item()
                phase_aux_correct += (phase_pred.argmax(dim=1) == phase_tgt).sum().item()

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

    # Log auxiliary metrics
    if hasattr(model, 'aux_heads') and total > 0:
        mat_mae = mat_ae_sum / total
        phase_acc = phase_aux_correct / total
        log(f"    aux: mat_mae={mat_mae:.4f} phase_acc={phase_acc:.1%}")

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
    ap.add_argument("--aux-weight", type=float, default=0.1,
                    help="Weight for auxiliary losses (material + phase)")
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
    log("exp164: AUXILIARY LOSSES — Material + Phase from CLS token")
    log(f"  device: {DEVICE}")
    log(f"  vocab: {VOCAB_SIZE} moves (compact)")
    log(f"  value: {N_VALUE_BINS}-bin HL-Gauss (σ={SIGMA_HL_GAUSS:.4f})")
    log(f"  aux_weight: {args.aux_weight}")
    log(f"  config: {MODEL_CONFIG}")

    remap_tensor = build_remap_tensor()
    log(f"  remap: {(remap_tensor >= 0).sum().item()} legacy→compact mappings")

    model = build_model_with_aux(MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in model.aux_heads.parameters())
    log(f"  params: {n_params/1e6:.1f}M (aux heads: {aux_params:,})")

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
        log("  Compiling model with torch.compile...")
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
            "value_weight": args.value_weight, "aux_weight": args.aux_weight,
        }}, f, indent=2)

    eval_data, eval_tensors = None, None
    eval_path = SHARD_DIR / "eval.pt"
    if eval_path.exists():
        eval_data, eval_tensors = load_eval_data(eval_path, remap_tensor)
        log(f"  Eval: {len(eval_data)} positions")

    remap_device = remap_tensor.to(DEVICE)

    if eval_data and start_step == 0:
        torch.cuda.empty_cache()
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  RANDOM INIT: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        best_acc = acc

    log(f"\n{'='*60}")
    log(f"Training: {args.epochs} epochs, LR={args.lr}, warmup={warmup_steps}")
    log(f"  value_weight={args.value_weight}, aux_weight={args.aux_weight}")
    log(f"  label_smoothing={args.label_smoothing}")
    log(f"  {N_VALUE_BINS}-bin HL-Gauss value, {VOCAB_SIZE}-move compact policy")
    log(f"  Auxiliary: material_balance (Huber) + game_phase (CE)")
    log(f"{'='*60}")

    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_aux_loss = 0.0
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

            move_targets = remap_device[move_targets_legacy]
            valid = move_targets >= 0
            if not valid.all():
                skipped = (~valid).sum().item()
                skipped_moves += skipped
                move_targets = move_targets.clamp(min=0)

            cp_vals = batch_input.pop("cp").to(DEVICE)
            mate_vals = batch_input.pop("mate").to(DEVICE)
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            # Compute auxiliary targets from fused_ids BEFORE model forward
            mat_tgt, phase_tgt = compute_aux_targets(batch_input["fused_ids"], DEVICE)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(
                    result["policy_logits"], move_targets,
                    label_smoothing=args.label_smoothing,
                    ignore_index=-1)
                v_loss = hl_gauss_loss(result["value_logits"], win_pct)

                # Auxiliary losses from CLS token
                cls_h = result["cls_hidden"]
                mat_pred, phase_pred = model.aux_heads(cls_h)
                mat_loss = F.huber_loss(mat_pred.float(), mat_tgt)
                phase_loss = F.cross_entropy(phase_pred.float(), phase_tgt,
                                             weight=PHASE_WEIGHTS.to(DEVICE))
                a_loss = mat_loss + phase_loss

                loss = (p_loss + args.value_weight * v_loss +
                        args.aux_weight * a_loss) / args.accum_steps

            scaler.scale(loss).backward()

            if torch.isnan(p_loss) or torch.isnan(v_loss):
                log(f"NaN detected at step {step}! Saving and aborting.")
                save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                              OUTPUT_DIR / "latest_nan.pt")
                return

            accum_p_loss += p_loss.item()
            accum_v_loss += v_loss.item()
            accum_aux_loss += a_loss.item()
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
                    avg_a = accum_aux_loss / accum_count
                    avg_gn = grad_norm_accum / args.log_interval
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_s = positions_seen / max(elapsed, 1) if start_step == 0 else \
                             (positions_seen - start_step * eff_bs) / max(elapsed, 1)
                    remaining_steps = total_steps - step
                    remaining_pos = remaining_steps * eff_bs
                    eta = remaining_pos / max(pos_s, 1)

                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} a={avg_a:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"ETA {timedelta(seconds=int(eta))}")

                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_aux_loss = 0.0
                    accum_count = 0

                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                  OUTPUT_DIR / "latest.pt")
                    if step % 10000 == 0 and step > 0:
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / f"step_{step}.pt")

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
                accum_aux_loss = 0.0

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
            acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
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
