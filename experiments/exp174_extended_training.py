"""exp174: Extended training to exploit data utilization.

exp173 L (50.1M, 10K steps) achieved 18.36% top-1 on held-out shard 9,
decisively beating the 25.9M baseline. But L was still improving at the
end (loss delta -0.013 at 9K->10K), and data utilization was only 8%
(640K positions seen out of 8M available).

Hypothesis: Extending training from 10K -> 30K steps will unlock
significantly more accuracy. The cosine LR schedule spreads decay over
3x more steps, keeping LR high longer. At 30K steps, the model will see
~24% of available data (vs 8% at 10K).

LR comparison at step 10K:
  - 10K-step schedule: LR ≈ 1e-5 (near minimum, barely learning)
  - 30K-step schedule: LR ≈ 1.53e-4 (76% of peak, still learning fast)

Variants:
  N. L architecture (50.1M), 30K steps — cheap falsification of extended training
  O. To be added based on M results — may use 86M arch if M shows benefit

Comparison targets:
  - exp173 L (50.1M, 10K steps, shard 9): 18.36%/42.58%/2.759
  - exp173 M (86.1M, 10K steps, shard 9): running

Usage:
  python experiments/exp174_extended_training.py --variant N
  python experiments/exp174_extended_training.py --variant N --eval-every 2000
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
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Compact vocab remap ──
def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t

REMAP = build_remap_tensor()


# ── Variants ──
VARIANTS = {
    "N": {
        "name": "EXTENDED_50M_30K",
        "shards": 8, "steps": 30000,
        "peak_lr": 2e-4, "min_lr": 1e-5, "warmup": 500,
        "checkpoint_every": 10000,
        "model": dict(
            encoder_dim=256, hidden_dim=640, num_layers=10, num_heads=10,
            ffn_ratio=4, dropout=0.05, policy_head_dim=256,
            value_hidden=256, use_pos_embed=True, n_ctx_tokens=4,
            value_head_type="cls", n_value_classes=3,
            use_swiglu=True, use_rel_bias=True,
        ),
    },
}


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, peak_lr, min_lr):
    """Linear warmup then cosine decay to min_lr."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine)
    return LambdaLR(optimizer, lr_lambda)


# ── Data ──
def load_shard(shard_idx=0):
    path = SHARD_DIR / f"shard_{shard_idx:05d}.pt"
    return torch.load(path, weights_only=True, map_location="cpu")


def prepare_batch(data, indices, device):
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
        board_input, target_move, wdl, valid = prepare_batch(eval_data, idx, device)

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
    n_shards = v["shards"]
    steps = v["steps"]
    peak_lr = v["peak_lr"]
    min_lr = v["min_lr"]
    warmup = v["warmup"]
    checkpoint_every = v.get("checkpoint_every", 10000)

    torch.manual_seed(seed)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Compact vocab size: {VOCAB_SIZE}")

    print(f"\n{'='*60}")
    print(f"  ABLATION {variant_key}: {v['name']}")
    print(f"  shards={n_shards} steps={steps} bs={batch_size}")
    print(f"  LR: warmup {warmup} steps -> peak {peak_lr}, cosine -> {min_lr}")
    print(f"  Checkpoints every {checkpoint_every} steps")
    print(f"{'='*60}\n")

    # Build model
    cfg = ChessTransformerConfig(**v["model"])
    model = build_model(cfg).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Check VRAM after model load
    if DEVICE.type == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM after model: {vram_used:.2f} / {vram_total:.1f} GB")

    # Load training shards
    print(f"  Loading {n_shards} training shards...")
    train_data_list = []
    for i in range(n_shards):
        train_data_list.append(load_shard(i))
    train_data = {k: torch.cat([d[k] for d in train_data_list], dim=0)
                  for k in train_data_list[0].keys()}
    n_train = train_data["board_array"].shape[0]
    print(f"  Training positions: {n_train:,}")
    del train_data_list
    gc.collect()

    # Eval data: shard 9 (held out from training shards 0-7)
    EVAL_SHARD = 9
    print(f"  Loading eval data (shard {EVAL_SHARD}, held out)...")
    eval_data = load_shard(EVAL_SHARD)

    # Checkpoint directory
    ckpt_dir = ROOT / "outputs" / "exp174_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt = AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(opt, warmup, steps, peak_lr, min_lr)
    scaler = GradScaler("cuda")
    model.train()

    results = []
    t0 = time.time()
    total_pos = 0

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_train, (batch_size,))
        board_input, target_move, wdl, valid = prepare_batch(train_data, idx, DEVICE)

        with autocast("cuda", dtype=torch.float16):
            out = model(board_input)
            p_loss = F.cross_entropy(out["policy_logits"][valid.bool()],
                                     target_move[valid.bool()])
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
            print(f"  step={step:5d} p_loss={p_loss.item():.3f} v_loss={v_loss.item():.3f}"
                  f" lr={cur_lr:.2e} {pos_s:.0f} pos/s")
            sys.stdout.flush()

        if step % eval_every == 0:
            ev = evaluate(model, eval_data, DEVICE)
            results.append({"step": step, **ev})
            print(f"  >>> EVAL step={step}: top1={ev['top1']:.4f} top3={ev['top3']:.4f}"
                  f" loss={ev['loss']:.3f}")
            sys.stdout.flush()

        # Save checkpoint
        if step % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"exp174_{variant_key}_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> CHECKPOINT saved: {ckpt_path.name}")
            sys.stdout.flush()

    # Save final checkpoint if not already saved
    if steps % checkpoint_every != 0:
        ckpt_path = ckpt_dir / f"exp174_{variant_key}_step{steps}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >>> CHECKPOINT saved: {ckpt_path.name}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    pos_s = total_pos / elapsed

    ev_final = results[-1] if results else {}
    print(f"\n  Completed {v['name']} in {elapsed:.1f}s ({pos_s:.0f} pos/s)")

    return {
        "variant": variant_key,
        "name": v["name"],
        "params": n_params,
        "shards": n_shards,
        "steps": steps,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "warmup": warmup,
        "top1": ev_final.get("top1", 0),
        "top3": ev_final.get("top3", 0),
        "loss": ev_final.get("loss", 0),
        "time_s": elapsed,
        "pos_s": pos_s,
        "eval_history": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=VARIANTS.keys())
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_variant(args.variant, args.eval_every, args.batch_size, args.seed)

    out_path = ROOT / "outputs" / "exp174_extended_results.json"
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.append(result)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
