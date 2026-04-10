"""exp163: Attention Policy Head — From/To Scaled Dot-Product.

Hypothesis: Replacing SpatialPolicyHead's element-wise multiply + ReLU + linear
with a scaled dot-product attention between from-square and to-square representations
will improve policy prior quality. This is the approach used by ChessFormer (Monroe
et al. 2024), which matched Ruoss et al.'s 270M model at 30x fewer FLOPS.

Architecture change:
  Current SpatialPolicyHead:
    score(m) = W_score * ReLU(Q_from[from(m)] * K_to[to(m)] + global + promo)
    Parameters: ~2.1M (from_proj + to_proj + global_proj + score_proj + promo_embed)

  Attention Policy Head:
    score(m) = (Q_from[from(m)] · K_to[to(m)]) / sqrt(d) + promo_bias[p(m)]
    Multi-head variant: avg over H heads, each with d/H dimensions
    Parameters: ~2.1M (Q_proj + K_proj + promo_bias), comparable param count

Why attention > element-wise multiply:
  1. Dot product naturally measures "compatibility" between from/to squares
  2. Multi-head attention captures different move patterns simultaneously
     (e.g., one head for rook moves, another for bishop diagonals, another for knight jumps)
  3. ChessFormer showed this outperforms standard policy heads empirically
  4. Gradient flows directly through the dot product — no ReLU bottleneck

Protocol:
  1. Train from scratch with compact vocab + 128-bin HL-Gauss (same as exp161)
  2. Quick 5K step ablation to compare against SpatialPolicyHead
  3. If positive: extend to full training run

Usage:
  python experiments/exp163_attention_policy.py --max-steps 5000
  python experiments/exp163_attention_policy.py --epochs 1 --output-dir outputs/exp163_full
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

from chess_transformer_factory import (
    ChessTransformerConfig, ChessTransformer, FusedBoardEncoder,
    _build_move_square_indices, DEFAULT_200M_CONFIG,
)
from move_vocab import (
    VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask,
    LEGACY_UCI_TO_IDX, COMPACT_UCI_TO_IDX, legacy_to_compact_map,
)
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp163_attn_policy"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None
SHUTDOWN = False

N_VALUE_BINS = 128
SIGMA_HL_GAUSS = 0.75 / N_VALUE_BINS


def _signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    log("SHUTDOWN requested.")

signal.signal(signal.SIGINT, _signal_handler)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── Attention Policy Head ──────────────────────────────────────────────

class AttentionPolicyHead(nn.Module):
    """Scaled dot-product attention policy head (ChessFormer-style).
    
    For each move (from_sq → to_sq), computes:
        score = sum_h (Q_h[from_sq] · K_h[to_sq]) / sqrt(d_head) + promo_bias
    
    Multi-head attention captures different move patterns:
    - Some heads might specialize in rook/bishop rays
    - Others in knight jumps or pawn pushes
    """
    
    def __init__(self, hidden_size: int, n_ctx_tokens: int = 4,
                 head_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.num_heads = num_heads
        self.d_head = head_dim // num_heads
        self.scale = self.d_head ** -0.5
        
        # Q projection for from-squares, K projection for to-squares
        self.q_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        
        # Promotion type bias (additive, not multiplicative)
        self.promo_bias = nn.Embedding(5, 1)  # 0=none, 1=q, 2=r, 3=b, 4=n
        nn.init.zeros_(self.promo_bias.weight)
        
        # Global context bias from CLS token
        self.global_gate = nn.Linear(hidden_size, num_heads, bias=True)
        
        # Move index buffers
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)
    
    def forward(self, hidden_states: torch.Tensor, cls_hidden: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]  # (B, 64, H)
        
        # Project all 64 squares to Q and K spaces
        Q = self.q_proj(sq_hidden)  # (B, 64, head_dim)
        K = self.k_proj(sq_hidden)  # (B, 64, head_dim)
        
        # Reshape for multi-head: (B, 64, num_heads, d_head)
        Q = Q.view(B, 64, self.num_heads, self.d_head)
        K = K.view(B, 64, self.num_heads, self.d_head)
        
        # Gather from/to features for each move
        V = VOCAB_SIZE
        Q_from = Q[:, self.from_sqs, :, :]  # (B, V, num_heads, d_head)
        K_to = K[:, self.to_sqs, :, :]      # (B, V, num_heads, d_head)
        
        # Scaled dot-product per head: (B, V, num_heads)
        dots = (Q_from * K_to).sum(dim=-1) * self.scale  # (B, V, num_heads)
        
        # Global gate: CLS token modulates head contributions
        gate = self.global_gate(cls_hidden)  # (B, num_heads)
        dots = dots + gate.unsqueeze(1)      # (B, V, num_heads)
        
        # Sum over heads → (B, V)
        scores = dots.sum(dim=-1)
        
        # Add promotion bias
        scores = scores + self.promo_bias(self.promo_types).squeeze(-1).unsqueeze(0)
        
        return scores


# ── Model construction ─────────────────────────────────────────────────

def build_attn_policy_model(config: ChessTransformerConfig):
    """Build model with attention policy head + 128-bin distributional value."""
    model = ChessTransformer(config)
    
    # Replace SpatialPolicyHead with AttentionPolicyHead
    model.policy_head = AttentionPolicyHead(
        config.hidden_dim,
        n_ctx_tokens=config.n_ctx_tokens,
        head_dim=config.policy_head_dim,
        num_heads=8,
    )
    
    # Replace 3-class WDL value head with 128-bin distributional
    model.value_head = nn.Sequential(
        nn.Linear(config.hidden_dim, config.value_hidden),
        nn.ReLU(),
        nn.Linear(config.value_hidden, N_VALUE_BINS),
    )
    for layer in model.value_head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    return model


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


# ── Move remap ─────────────────────────────────────────────────────────

def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t


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
            fen = _board_array_to_fen(raw["board_array"][i], raw["turn"][i],
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
            eval_data.append({"board": board, "move": move, "compact_idx": compact_idx,
                              "cp": cp_val, "mate": mate_val})
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
        "config": DEFAULT_200M_CONFIG.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step, "epoch": epoch, "best_acc": best_acc,
        "vocab_version": "compact", "n_value_bins": N_VALUE_BINS,
        "policy_head": "attention",
    }, tmp)
    os.replace(str(tmp), str(path))


# ── Main ───────────────────────────────────────────────────────────────

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
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--num-heads", type=int, default=8,
                    help="Number of attention heads in policy head")
    args = ap.parse_args()

    global LOG_PATH, OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"

    config = DEFAULT_200M_CONFIG
    log("=" * 60)
    log("exp163: Attention Policy Head + Compact Vocab + Distributional Value")
    log(f"  device: {DEVICE}")
    log(f"  vocab: {VOCAB_SIZE} moves (compact)")
    log(f"  value: {N_VALUE_BINS}-bin HL-Gauss")
    log(f"  policy: AttentionPolicyHead (num_heads={args.num_heads})")

    remap_tensor = build_remap_tensor()

    model = build_attn_policy_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    policy_params = sum(p.numel() for p in model.policy_head.parameters())
    log(f"  params: {n_params/1e6:.1f}M (policy head: {policy_params/1e6:.2f}M)")

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
        json.dump({"model": config.to_dict(), "training": {
            "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "eff_bs": eff_bs, "lr": args.lr, "epochs": args.epochs,
            "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "init": "random", "vocab": "compact", "vocab_size": VOCAB_SIZE,
            "n_value_bins": N_VALUE_BINS, "policy_head": "attention",
            "policy_head_num_heads": args.num_heads,
            "value_weight": args.value_weight,
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
    log(f"  value_weight={args.value_weight}, label_smoothing={args.label_smoothing}")
    log(f"  AttentionPolicyHead(heads={args.num_heads}), {N_VALUE_BINS}-bin HL-Gauss")
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

            move_targets = remap_device[move_targets_legacy]
            valid = move_targets >= 0
            if not valid.all():
                skipped_moves += (~valid).sum().item()
                move_targets = move_targets.clamp(min=0)

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

            if torch.isnan(p_loss) or torch.isnan(v_loss):
                log(f"NaN at step {step}!")
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
                    remaining = (total_steps - step) * eff_bs / max(pos_s, 1)

                    log(f"  step {step:,}/{total_steps:,} | "
                        f"p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={lr:.2e} gn={avg_gn:.2f} | {pos_s:.0f} pos/s | "
                        f"ETA {timedelta(seconds=int(remaining))}")

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
                    acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
                    log(f"  EVAL step {step}: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        save_checkpoint(model, optimizer, scaler, step, epoch, best_acc,
                                      OUTPUT_DIR / "best_model.pt")
                        log(f"  ** New best! top1={best_acc:.2%}")
                    model.train()

                if args.max_steps > 0 and step >= args.max_steps:
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
            acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
            log(f"  EPOCH EVAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, optimizer, scaler, step, epoch + 1, best_acc,
                              OUTPUT_DIR / "best_model.pt")
                log(f"  ** New best! top1={best_acc:.2%}")

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
                print(f"CUDA error (attempt {_attempt+1}/{MAX_RETRIES}): {e}")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(10)
            else:
                traceback.print_exc()
                break
    else:
        print("All retry attempts exhausted.")
