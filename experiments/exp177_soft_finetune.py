"""exp177: Fine-tune exp174 N checkpoint (21.20% top-1 @20K) with soft policy targets.

Hypothesis: Soft targets (α=0.5) can refine an already-converged 50.1M model trained on 8M
hard-label positions. exp175 proved α=0.5 gives 16.82% from scratch on 500K soft data;
if fine-tuning lifts the 21.20% baseline, soft targets improve even strong models.

Risk: 500K soft positions is much narrower than the 8M training set. Fine-tuning on a
smaller subset could cause catastrophic forgetting. We use low LR to mitigate this.

Variants:
  R: α=0.5, 5K steps, peak LR 5e-5 (conservative fine-tune)
  S: α=0.5, 5K steps, peak LR 2e-5 (very conservative)

Eval on shard 9 (held out), every 500 steps for fine-grained visibility.
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
CHECKPOINT_PATH = ROOT / "outputs" / "exp174_checkpoints" / "exp174_N_step20000.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t

REMAP = build_remap_tensor()

MODEL_CONFIG = dict(
    encoder_dim=256, hidden_dim=640, num_layers=10, num_heads=10,
    ffn_ratio=4, dropout=0.05, policy_head_dim=256,
    value_hidden=256, use_pos_embed=True, n_ctx_tokens=4,
    value_head_type="cls", n_value_classes=3,
    use_swiglu=True, use_rel_bias=True,
)

VARIANTS = {
    "R": {
        "name": "SOFT_FINETUNE_5e-5",
        "alpha": 0.5,
        "shards": 5, "steps": 5000,
        "peak_lr": 5e-5, "min_lr": 1e-5, "warmup": 200,
    },
    "S": {
        "name": "SOFT_FINETUNE_2e-5",
        "alpha": 0.5,
        "shards": 5, "steps": 5000,
        "peak_lr": 2e-5, "min_lr": 5e-6, "warmup": 200,
    },
}


def soft_policy_loss(logits, soft_indices, soft_probs):
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


def load_soft_shards(n_shards):
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
    path = HARD_SHARD_DIR / f"shard_{shard_idx:05d}.pt"
    return torch.load(path, weights_only=True, map_location="cpu")


def prepare_soft_batch(data, indices, device):
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
    print(f"  Fine-tuning from: {CHECKPOINT_PATH.name}")
    print(f"  alpha={alpha} shards={n_shards} steps={steps} bs={batch_size}")
    print(f"  LR: warmup {warmup} -> peak {peak_lr}, cosine -> {min_lr}")
    print(f"{'='*60}\n")

    # Build model and load checkpoint
    cfg = ChessTransformerConfig(**MODEL_CONFIG)
    model = build_model(cfg).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    ckpt = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=DEVICE)
    model.load_state_dict(ckpt)
    del ckpt
    print(f"  Loaded checkpoint: {CHECKPOINT_PATH.name}")

    if DEVICE.type == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after model: {vram_used:.2f} / {total_mem:.1f} GB")

    # Load soft target training data
    train_data = load_soft_shards(n_shards)
    n_train = train_data["board_array"].shape[0]

    # Load eval data
    print(f"  Loading eval data (shard 9, held out)...")
    eval_data = load_eval_shard(9)

    # Baseline eval before fine-tuning
    print(f"  Evaluating baseline (before fine-tuning)...")
    baseline = evaluate(model, eval_data, DEVICE)
    print(f"  >>> BASELINE: top1={baseline['top1']:.4f} top3={baseline['top3']:.4f}"
          f" loss={baseline['loss']:.3f}")

    opt = AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(opt, warmup, steps, peak_lr, min_lr)
    scaler = GradScaler("cuda")
    model.train()

    results = [{"step": 0, **baseline}]
    t0 = time.time()
    total_pos = 0

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch_size,))
        (board_input, target_move, wdl, valid,
         soft_idx, soft_prob) = prepare_soft_batch(train_data, idx, DEVICE)

        with autocast("cuda", dtype=torch.float16):
            out = model(board_input)
            policy_logits = out["policy_logits"]

            p_loss_hard = F.cross_entropy(
                policy_logits[valid.bool()], target_move[valid.bool()])
            p_loss_soft = soft_policy_loss(policy_logits, soft_idx, soft_prob)
            p_loss = (1 - alpha) * p_loss_hard + alpha * p_loss_soft
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
            print(f"  step={step:5d} hard={p_loss_hard.item():.3f} soft={p_loss_soft.item():.3f}"
                  f" v_loss={v_loss.item():.3f} lr={cur_lr:.2e} {pos_s:.0f} pos/s")

        if step % eval_every == 0:
            ev = evaluate(model, eval_data, DEVICE)
            results.append({"step": step, **ev})
            delta_t1 = ev['top1'] - baseline['top1']
            delta_loss = ev['loss'] - baseline['loss']
            print(f"  >>> EVAL step={step}: top1={ev['top1']:.4f} top3={ev['top3']:.4f}"
                  f" loss={ev['loss']:.3f}  (Δtop1={delta_t1:+.4f} Δloss={delta_loss:+.3f})")

    elapsed = time.time() - t0
    pos_s = total_pos / elapsed

    ev_final = results[-1]
    print(f"\n  Completed {v['name']} in {elapsed:.1f}s ({pos_s:.0f} pos/s)")
    print(f"  Final: top1={ev_final['top1']:.4f} top3={ev_final['top3']:.4f}"
          f" loss={ev_final['loss']:.3f}")
    delta_t1 = ev_final['top1'] - baseline['top1']
    print(f"  vs baseline: Δtop1={delta_t1:+.4f}")

    return {
        "variant": variant_key,
        "name": v["name"],
        "alpha": alpha,
        "params": n_params,
        "baseline_top1": baseline["top1"],
        "baseline_top3": baseline["top3"],
        "baseline_loss": baseline["loss"],
        "steps": steps,
        "peak_lr": peak_lr,
        "top1": ev_final["top1"],
        "top3": ev_final["top3"],
        "loss": ev_final["loss"],
        "time_s": elapsed,
        "pos_s": pos_s,
        "eval_history": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=VARIANTS.keys())
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.variant and not args.all:
        parser.error("Either --variant or --all is required")

    variants_to_run = list(VARIANTS.keys()) if args.all else [args.variant]
    all_results = []

    out_path = ROOT / "outputs" / "exp177_soft_finetune_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    for vk in variants_to_run:
        result = run_variant(vk, args.eval_every, args.batch_size, args.seed)
        all_results.append(result)

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to {out_path}")

        if len(variants_to_run) > 1:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
