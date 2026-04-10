"""exp168: Distributional Value Head Surgery on exp100 (2077 Elo baseline).

MOTIVATION:
  exp100 is the current best Elo model (2077 @ 800 MCTS sims) with a 3-class WDL
  value head. Value head quality is THE bottleneck for MCTS performance — the WDL 
  head can't distinguish +50cp from +900cp (both "Win").

  This experiment does VALUE HEAD SURGERY:
  - Load exp100's fully-trained trunk and policy head (legacy 5504 vocab)
  - Replace Linear(512, 3) → Linear(512, 128) distributional value head
  - Fine-tune ONLY the value head (freeze trunk + policy head initially)
  - Then unfreeze everything for a final joint fine-tune phase
  
  Conservative, no architectural changes beyond the value head. Should preserve
  exp100's calibrated policy/trunk while adding distributional value granularity.

EXPECTED GAIN: +100-200 Elo from better MCTS value quality (at 200-800 sims).

Usage:
  python experiments/exp168_exp100_dist_value.py --max-steps 5000   # quick test
  python experiments/exp168_exp100_dist_value.py --max-steps 20000  # full
  python experiments/exp168_exp100_dist_value.py --eval-only
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

from chess_transformer_factory import build_model, ChessTransformerConfig
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp168_exp100_dist"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
SOURCE_CKPT = ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt"
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


# ── HL-Gauss value ──

def cp_to_win_percent(cp, mate):
    N = cp.shape[0]
    win_pct = torch.zeros(N, dtype=torch.float32, device=cp.device)
    win_pct[mate > 0] = 1.0
    win_pct[mate < 0] = 0.0
    no_mate = mate == 0
    if no_mate.any():
        win_pct[no_mate] = torch.sigmoid(cp[no_mate].float() / 111.7)
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
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    eval_data = []
    surviving = []
    for i in range(raw["board_array"].shape[0]):
        try:
            fen = _board_array_to_fen(raw["board_array"][i], raw["turn"][i],
                                       raw["castling"][i], raw["ep_square"][i])
            board = chess.Board(fen)
            move_idx = raw["move_idx"][i].item()
            uci = IDX_TO_UCI[move_idx]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            cp_val = raw["cp"][i].item() if "cp" in raw else 0
            mate_val = raw["mate"][i].item() if "mate" in raw else 0
            eval_data.append({"board": board, "move": move, "move_idx": move_idx,
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


def run_eval(model, eval_data, eval_tensors, batch_size=64):
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
                l = logits[j].clone()
                mask = legal_move_mask(d["board"]).to(DEVICE)
                l[~mask] = float("-inf")
                true_idx = d["move_idx"]
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
                total_value_mae += abs(pred_win_pct[j].item() - true_wp)
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


def save_checkpoint(model, optimizer, scaler, step, best_acc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "best_acc": best_acc,
        "n_value_bins": N_VALUE_BINS,
        "source": "exp100_diverse_training",
    }, tmp)
    os.replace(str(tmp), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="LR for trunk (value head gets 5x)")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--policy-weight", type=float, default=0.1,
                     help="Low policy weight — this is mostly value head training")
    ap.add_argument("--freeze-steps", type=int, default=2000,
                     help="Steps to freeze trunk+policy, training only value head")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    global LOG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "training.log"

    log("=" * 60)
    log("exp168: Distributional Value Surgery on exp100 (2077 Elo)")
    log(f"  source: {SOURCE_CKPT}")
    log(f"  device: {DEVICE}")
    log(f"  value: {N_VALUE_BINS}-bin HL-Gauss")
    log(f"  freeze_steps: {args.freeze_steps} (train value head only)")
    log(f"  policy_weight: {args.policy_weight} (preserve calibrated policy)")

    # Build model with standard vocab (legacy 5504)
    model = build_model(ChessTransformerConfig())

    # Load exp100 checkpoint
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Load everything except value head final layer
    model.load_state_dict(sd, strict=False)
    log("  Loaded exp100 trunk + policy head")

    # Replace value head: Linear(512, 3) → Linear(512, 128)
    old_head = model.value_head
    hidden_dim = old_head[0].out_features  # 512
    model.value_head = nn.Sequential(
        nn.Linear(model.value_head[0].in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, N_VALUE_BINS),
    )
    # Copy weights for the first layer (preserves learned features)
    model.value_head[0].weight.data.copy_(old_head[0].weight.data)
    model.value_head[0].bias.data.copy_(old_head[0].bias.data)
    # Xavier init for the new output layer
    nn.init.xavier_uniform_(model.value_head[2].weight)
    nn.init.zeros_(model.value_head[2].bias)
    log(f"  Value head: 3-class → {N_VALUE_BINS}-bin HL-Gauss")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params/1e6:.1f}M")

    model.to(DEVICE)

    if args.eval_only:
        eval_data, eval_tensors = load_eval_data(SHARD_DIR / "eval.pt")
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  EVAL: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        return

    # Optimizer with differential LR: value head gets 5x
    trunk_params = []
    value_params = []
    for name, p in model.named_parameters():
        if "value_head" in name:
            value_params.append(p)
        else:
            trunk_params.append(p)

    optimizer = AdamW([
        {"params": trunk_params, "lr": args.lr},
        {"params": value_params, "lr": args.lr * 5},
    ], weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    start_step = 0
    best_acc = 0.0

    if args.resume:
        resume_path = OUTPUT_DIR / "latest.pt"
        if resume_path.exists():
            rckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(rckpt["model_state_dict"])
            model.to(DEVICE)
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            scaler.load_state_dict(rckpt["scaler_state_dict"])
            start_step = rckpt.get("step", 0)
            best_acc = rckpt.get("best_acc", 0.0)
            log(f"  Resumed from step {start_step}")

    # Data loader
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=args.batch_size,
        encoder_type="fused", device=DEVICE, seed=42,
        include_cp=True, include_mate=True,
    )
    total_steps = args.max_steps
    eff_bs = args.batch_size * args.accum_steps

    log(f"  {loader.total_positions:,} positions, eff_bs={eff_bs}")
    log(f"  {total_steps} steps, freeze_steps={args.freeze_steps}")

    # LR schedule
    warmup = min(200, total_steps // 10)

    def get_lr(step, base_lr):
        if step < warmup:
            return base_lr * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))

    # Eval data
    eval_data, eval_tensors = load_eval_data(SHARD_DIR / "eval.pt")
    log(f"  Eval: {len(eval_data)} positions")

    # Initial eval (exp100 with random value head)
    if start_step == 0:
        acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
        log(f"  INIT (random value): top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
        best_acc = acc

    # Training
    log(f"\n{'='*60}")
    log(f"Phase 1: Freeze trunk+policy ({args.freeze_steps} steps) → value head only")
    log(f"Phase 2: Unfreeze all ({total_steps - args.freeze_steps} steps) → joint fine-tune")
    log(f"{'='*60}")

    model.train()
    step = start_step
    accum_p_loss = 0.0
    accum_v_loss = 0.0
    accum_count = 0
    positions_seen = step * eff_bs
    t0 = time.time()
    grad_norm_accum = 0.0
    frozen = step < args.freeze_steps

    # Freeze trunk+policy for Phase 1
    if frozen:
        for name, p in model.named_parameters():
            if "value_head" not in name:
                p.requires_grad = False
        log("  Trunk+policy FROZEN")

    for epoch in range(3):  # enough epochs to cover max_steps
        loader.set_epoch(epoch)

        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN or step >= total_steps:
                save_checkpoint(model, optimizer, scaler, step, best_acc,
                              OUTPUT_DIR / "latest.pt")
                break

            # Unfreeze after freeze_steps
            if frozen and step >= args.freeze_steps:
                for p in model.parameters():
                    p.requires_grad = True
                frozen = False
                log(f"  Step {step}: UNFREEZING trunk+policy for joint fine-tune")

            # Extract cp/mate for value targets
            cp_vals = batch_input.pop("cp").to(DEVICE)
            mate_vals = batch_input.pop("mate").to(DEVICE)
            win_pct = cp_to_win_percent(cp_vals, mate_vals)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                p_loss = F.cross_entropy(result["policy_logits"], move_targets)
                v_loss = hl_gauss_loss(result["value_logits"], win_pct)
                pw = args.policy_weight if frozen else 0.5  # low policy weight during freeze
                loss = (pw * p_loss + args.value_weight * v_loss) / args.accum_steps

            scaler.scale(loss).backward()

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
                for pg_idx, pg in enumerate(optimizer.param_groups):
                    base = args.lr if pg_idx == 0 else args.lr * 5
                    pg["lr"] = get_lr(step, base)

                if step % args.log_interval == 0:
                    avg_p = accum_p_loss / accum_count
                    avg_v = accum_v_loss / accum_count
                    avg_gn = grad_norm_accum / args.log_interval
                    grad_norm_accum = 0.0
                    elapsed = time.time() - t0
                    pos_s = positions_seen / max(elapsed, 1)
                    eta = (total_steps - step) * eff_bs / max(pos_s, 1)
                    phase = "FROZEN" if frozen else "JOINT"
                    log(f"  [{phase}] step {step}/{total_steps} | p={avg_p:.4f} v={avg_v:.4f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
                        f"gn={avg_gn:.2f} | {pos_s:.0f}pos/s | ETA {timedelta(seconds=int(eta))}")
                    accum_p_loss = 0.0
                    accum_v_loss = 0.0
                    accum_count = 0

                if step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, scaler, step, best_acc,
                                  OUTPUT_DIR / "latest.pt")

                if step % args.eval_interval == 0:
                    torch.cuda.empty_cache()
                    acc, top3, val_mae = run_eval(model, eval_data, eval_tensors)
                    log(f"  EVAL step {step}: top1={acc:.2%} top3={top3:.2%} mae={val_mae:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        save_checkpoint(model, optimizer, scaler, step, best_acc,
                                      OUTPUT_DIR / "best_model.pt")
                        log(f"  ** New best! top1={best_acc:.2%}")
                    model.train()

                if step >= total_steps:
                    break

                accum_count = 0
                accum_p_loss = 0.0
                accum_v_loss = 0.0

        if step >= total_steps or SHUTDOWN:
            break

    # Final save
    save_checkpoint(model, optimizer, scaler, step, best_acc, OUTPUT_DIR / "best_model.pt")
    elapsed = time.time() - t0
    log(f"\nDone: {step} steps, {positions_seen:,} positions, {timedelta(seconds=int(elapsed))}")
    log(f"Best top1: {best_acc:.2%}")


if __name__ == "__main__":
    main()
