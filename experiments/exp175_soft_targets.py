"""exp175: Soft policy targets from Stockfish Multi-PV — from-scratch ablation.

Hypothesis: Training with soft policy targets (teacher distribution over top-8
SF candidate moves) gives MORE information per position than hard single-move
supervision. Per Ruoss et al. 2024, action-value supervision can be ~30x more
informative than behavioral cloning. Our soft targets are a weaker form of this
(probability over top-8, not full action distribution), but should still help.

Data: exp162_soft_data/shard_shard{0-4}_sf.pt — 5 shards × ~100K = ~500K positions
      Each shard has board data + hard label (move_idx) + soft targets (soft_indices, soft_probs)
      Soft targets: top-8 SF candidate moves with softmax(cp/tau=120) probabilities

Loss: (1-alpha) * hard_CE(policy, best_move) + alpha * soft_CE(policy, teacher_dist)
      + value_weight * CE(value, wdl)

Where soft_CE = -sum_k(teacher_prob_k * log_softmax(policy)[move_k])

Variants:
  O: alpha=0.0 — hard targets only on 500K positions (CONTROL)
  P: alpha=0.3 — 30% soft, 70% hard
  Q: alpha=0.5 — balanced hard + soft

All variants use same 50.1M model and 10K steps on 500K positions.
Eval on held-out shard 9 for comparison with exp174 N (8M positions, hard only).

Expected outcome: If soft targets help, P or Q should exceed O despite identical data.
Bonus: if P/Q MATCH N (8M hard), soft targets effectively multiply data efficiency ~16x.

Usage:
  python experiments/exp175_soft_targets.py --variant O   # hard-only control
  python experiments/exp175_soft_targets.py --variant P   # alpha=0.3
  python experiments/exp175_soft_targets.py --variant Q   # alpha=0.5
  python experiments/exp175_soft_targets.py --all         # run all 3
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MOVE_VOCAB_VERSION'] = 'compact'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import (
    ChessTransformerConfig, build_model, count_parameters,
)
from move_vocab import VOCAB_SIZE, LEGACY_UCI_TO_IDX, legacy_to_compact_map
from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl

ROOT = Path(__file__).resolve().parent.parent
SOFT_DIR = ROOT / "outputs" / "exp162_soft_data"
HARD_SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Compact vocab remap (for eval shard which uses legacy indices) ──
def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t

REMAP = build_remap_tensor()


# ── Model config: same 50.1M as exp173 L / exp174 N ──
MODEL_CONFIG = dict(
    encoder_dim=256, hidden_dim=640, num_layers=10, num_heads=10,
    ffn_ratio=4, dropout=0.05, policy_head_dim=256,
    value_hidden=256, use_pos_embed=True, n_ctx_tokens=4,
    value_head_type="cls", n_value_classes=3,
    use_swiglu=True, use_rel_bias=True,
)


# ── Variants ──
VARIANTS = {
    "O": {
        "name": "SOFT_CONTROL_HARD_ONLY",
        "alpha": 0.0,
        "shards": 5, "steps": 10000,
        "peak_lr": 2e-4, "min_lr": 1e-5, "warmup": 500,
    },
    "P": {
        "name": "SOFT_ALPHA_0.3",
        "alpha": 0.3,
        "shards": 5, "steps": 10000,
        "peak_lr": 2e-4, "min_lr": 1e-5, "warmup": 500,
    },
    "Q": {
        "name": "SOFT_ALPHA_0.5",
        "alpha": 0.5,
        "shards": 5, "steps": 10000,
        "peak_lr": 2e-4, "min_lr": 1e-5, "warmup": 500,
    },
}


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
    log_probs = F.log_softmax(logits.float(), dim=-1)

    valid = (soft_indices >= 0) & (soft_probs > 0)
    safe_indices = soft_indices.clamp(min=0).long()
    gathered = log_probs.gather(1, safe_indices)
    gathered = gathered * valid.float()
    weighted = soft_probs.float() * gathered

    return -weighted.sum(dim=-1).mean()


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, peak_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine)
    return LambdaLR(optimizer, lr_lambda)


# ── Data loading ──
def load_soft_shards(n_shards):
    """Load converted soft target shards from exp162_soft_data."""
    shards = []
    for i in range(n_shards):
        path = SOFT_DIR / f"shard_shard{i}_sf.pt"
        shards.append(torch.load(path, weights_only=False, map_location="cpu"))
    keys = list(shards[0].keys())
    combined = {}
    for k in keys:
        if isinstance(shards[0][k], torch.Tensor):
            combined[k] = torch.cat([s[k] for s in shards], dim=0)
    n = combined["board_array"].shape[0]
    print(f"  Loaded {n_shards} soft shards: {n:,} positions")
    del shards
    gc.collect()
    return combined


def load_eval_shard(shard_idx=9):
    """Load hard eval shard (legacy format, needs remap)."""
    path = HARD_SHARD_DIR / f"shard_{shard_idx:05d}.pt"
    return torch.load(path, weights_only=True, map_location="cpu")


def prepare_soft_batch(data, indices, device):
    """Prepare a training batch from soft shard data (already compact format)."""
    ba = data["board_array"][indices]
    turn = data["turn"][indices]
    castling = data["castling"][indices]
    ep = data["ep_square"][indices]
    move_idx = data["move_idx"][indices]
    cp = data["cp"][indices]
    mate = data["mate"][indices]
    soft_indices = data["soft_indices"][indices]
    soft_probs = data["soft_probs"][indices]

    fused_ids = board_array_to_fused(ba)
    ep_file = ep_square_to_file(ep)
    wdl = compute_wdl(cp, mate)

    # move_idx is already compact in these shards
    compact_move = move_idx.long()
    valid = compact_move >= 0
    compact_move = compact_move.clamp(min=0)

    board_input = {
        "fused_ids": fused_ids.to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_file.long().to(device),
    }
    return (board_input, compact_move.to(device), wdl.to(device), valid.to(device),
            soft_indices.to(device), soft_probs.to(device))


def prepare_eval_batch(data, indices, device):
    """Prepare an eval batch from hard shard (legacy format, needs remap)."""
    ba = data["board_array"][indices]
    turn = data["turn"][indices]
    castling = data["castling"][indices]
    ep = data["ep_square"][indices]
    move_idx = data["move_idx"][indices]
    cp = data["cp"][indices]
    mate = data["mate"][indices]

    fused_ids = board_array_to_fused(ba)
    ep_file = ep_square_to_file(ep)
    wdl = compute_wdl(cp, mate)

    compact_move = REMAP[move_idx.long()]
    valid = compact_move >= 0
    compact_move = compact_move.clamp(min=0)

    board_input = {
        "fused_ids": fused_ids.to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_file.long().to(device),
    }
    return board_input, compact_move.to(device), wdl.to(device), valid.to(device)


@torch.no_grad()
def evaluate(model, eval_data, device, num_samples=5000):
    model.eval()
    bs = 256
    correct1 = correct3 = total = 0
    total_loss = 0.0
    n_batches = 0
    N = min(num_samples, eval_data["board_array"].shape[0])

    for start in range(0, N, bs):
        end = min(start + bs, N)
        idx = torch.arange(start, end)
        board_input, target_move, wdl, valid = prepare_eval_batch(eval_data, idx, device)

        with autocast("cuda", dtype=torch.float16):
            out = model(board_input)
            policy_logits = out["policy_logits"]
            loss = F.cross_entropy(policy_logits[valid.bool()], target_move[valid.bool()])

        total_loss += loss.item()
        n_batches += 1
        preds = policy_logits[valid.bool()].topk(3, dim=-1).indices
        targets = target_move[valid.bool()]
        correct1 += (preds[:, 0] == targets).sum().item()
        correct3 += (preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += valid.sum().item()

    model.train()
    return {
        "top1": correct1 / max(total, 1),
        "top3": correct3 / max(total, 1),
        "loss": total_loss / max(n_batches, 1),
    }


def run_variant(variant_key, eval_every, batch_size, seed):
    v = VARIANTS[variant_key]
    alpha = v["alpha"]
    n_shards = v["shards"]
    steps = v["steps"]
    peak_lr = v["peak_lr"]
    min_lr = v["min_lr"]
    warmup = v["warmup"]

    torch.manual_seed(seed)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_mem:.1f} GB")
    print(f"Compact vocab size: {VOCAB_SIZE}")

    print(f"\n{'='*60}")
    print(f"  VARIANT {variant_key}: {v['name']}")
    print(f"  alpha={alpha} shards={n_shards} steps={steps} bs={batch_size}")
    print(f"  LR: warmup {warmup} -> peak {peak_lr}, cosine -> {min_lr}")
    print(f"{'='*60}\n")

    # Build model
    cfg = ChessTransformerConfig(**MODEL_CONFIG)
    model = build_model(cfg).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    if DEVICE.type == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after model: {vram_used:.2f} / {total_mem:.1f} GB")

    # Load soft target training data
    train_data = load_soft_shards(n_shards)
    n_train = train_data["board_array"].shape[0]

    # Load eval data (shard 9, held out)
    print(f"  Loading eval data (shard 9, held out)...")
    eval_data = load_eval_shard(9)

    opt = AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(opt, warmup, steps, peak_lr, min_lr)
    scaler = GradScaler("cuda")
    model.train()

    results = []
    t0 = time.time()
    total_pos = 0

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch_size,))
        (board_input, target_move, wdl, valid,
         soft_idx, soft_prob) = prepare_soft_batch(train_data, idx, DEVICE)

        with autocast("cuda", dtype=torch.float16):
            out = model(board_input)
            policy_logits = out["policy_logits"]

            # Hard CE loss
            if alpha < 1.0:
                p_loss_hard = F.cross_entropy(
                    policy_logits[valid.bool()], target_move[valid.bool()])
            else:
                p_loss_hard = torch.tensor(0.0, device=DEVICE)

            # Soft CE loss
            if alpha > 0.0:
                p_loss_soft = soft_policy_loss(policy_logits, soft_idx, soft_prob)
            else:
                p_loss_soft = torch.tensor(0.0, device=DEVICE)

            # Combined policy loss
            p_loss = (1 - alpha) * p_loss_hard + alpha * p_loss_soft

            # Value loss
            v_loss = F.cross_entropy(out["value_logits"], wdl)
            loss = p_loss + v_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        total_pos += batch_size

        if step % 100 == 0:
            elapsed = time.time() - t0
            pos_s = total_pos / elapsed
            cur_lr = opt.param_groups[0]["lr"]
            hard_v = p_loss_hard.item() if alpha < 1.0 else 0.0
            soft_v = p_loss_soft.item() if alpha > 0.0 else 0.0
            print(f"  step={step:5d} hard={hard_v:.3f} soft={soft_v:.3f}"
                  f" v_loss={v_loss.item():.3f} lr={cur_lr:.2e} {pos_s:.0f} pos/s")

        if step % eval_every == 0:
            ev = evaluate(model, eval_data, DEVICE)
            results.append({"step": step, **ev})
            print(f"  >>> EVAL step={step}: top1={ev['top1']:.4f} top3={ev['top3']:.4f}"
                  f" loss={ev['loss']:.3f}")

    elapsed = time.time() - t0
    pos_s = total_pos / elapsed

    ev_final = results[-1] if results else {}
    print(f"\n  Completed {v['name']} in {elapsed:.1f}s ({pos_s:.0f} pos/s)")
    print(f"  Final: top1={ev_final.get('top1',0):.4f} top3={ev_final.get('top3',0):.4f}"
          f" loss={ev_final.get('loss',0):.3f}")

    return {
        "variant": variant_key,
        "name": v["name"],
        "alpha": alpha,
        "params": n_params,
        "shards": n_shards,
        "steps": steps,
        "peak_lr": peak_lr,
        "top1": ev_final.get("top1", 0),
        "top3": ev_final.get("top3", 0),
        "loss": ev_final.get("loss", 0),
        "time_s": elapsed,
        "pos_s": pos_s,
        "eval_history": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=VARIANTS.keys())
    parser.add_argument("--all", action="store_true", help="Run all variants sequentially")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.variant and not args.all:
        parser.error("Either --variant or --all is required")

    variants_to_run = list(VARIANTS.keys()) if args.all else [args.variant]
    all_results = []

    out_path = ROOT / "outputs" / "exp175_soft_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    for vk in variants_to_run:
        result = run_variant(vk, args.eval_every, args.batch_size, args.seed)
        all_results.append(result)

        # Save after each variant
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to {out_path}")

        # Clear GPU memory between variants
        if len(variants_to_run) > 1:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
