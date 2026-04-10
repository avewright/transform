"""exp176: 86M model with gradient accumulation to fit in 8.6GB VRAM.

exp173 M (86.1M, 12L/768d/12H) OOM'd at batch 64 around step 2K.
exp174 N (50.1M, 10K→20K steps) hit a capacity ceiling at ~21.2% top-1.

This experiment reruns the 86M architecture with:
- batch_size=32, gradient_accumulation_steps=2 → effective batch 64
- 20K optimizer steps (extended training, proven critical by exp174)
- 8 shards (8M positions), cosine LR (peak 1.5e-4 → min 1e-5)
- Checkpoints every 5K steps, eval every 2K steps

Hypothesis: The 86M model has ~72% more capacity than 50M. If the 50M
ceiling was due to model capacity (not data), 86M should break through
21.2% with sufficient training. exp174 showed extended training is
critical — the 50M model didn't surpass its 10K baseline until step 14K.

Expected: 86M should reach ~22-24% top-1 by step 20K if capacity matters.

Comparison targets:
  - exp174 N (50.1M, 20K steps, 8 shards): 21.20% top-1 (ceiling)
  - exp173 L (50.1M, 10K steps, 8 shards): 18.36% top-1
  - exp173 M (86.1M, crashed ~2K): 10.40% top-1 at 2K eval

Usage:
  python experiments/exp176_86m_grad_accum.py --eval-every 2000
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

# ── Config ──
MODEL_CONFIG = dict(
    encoder_dim=256, hidden_dim=768, num_layers=12, num_heads=12,
    ffn_ratio=4, dropout=0.05, policy_head_dim=256,
    value_hidden=256, use_pos_embed=True, n_ctx_tokens=4,
    value_head_type="cls", n_value_classes=3,
    use_swiglu=True, use_rel_bias=True,
)

TRAIN_CONFIG = dict(
    shards=8,
    steps=20000,
    micro_batch=32,
    grad_accum=2,        # effective batch = 32 * 2 = 64
    peak_lr=1.5e-4,
    min_lr=1e-5,
    warmup=500,
    weight_decay=0.01,
    grad_clip=1.0,
    checkpoint_every=5000,
)


# ── Compact vocab remap ──
def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t

REMAP = build_remap_tensor()


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, peak_lr, min_lr):
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
    bs = 128  # smaller eval batch for 86M model
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


def run(eval_every, seed):
    cfg_t = TRAIN_CONFIG
    micro_bs = cfg_t["micro_batch"]
    grad_accum = cfg_t["grad_accum"]
    eff_bs = micro_bs * grad_accum
    steps = cfg_t["steps"]
    peak_lr = cfg_t["peak_lr"]
    min_lr = cfg_t["min_lr"]
    warmup = cfg_t["warmup"]

    torch.manual_seed(seed)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_mem:.1f} GB")
    print(f"Compact vocab size: {VOCAB_SIZE}")

    print(f"\n{'='*60}")
    print(f"  exp176: 86M MODEL WITH GRADIENT ACCUMULATION")
    print(f"  micro_bs={micro_bs} grad_accum={grad_accum} eff_bs={eff_bs}")
    print(f"  shards={cfg_t['shards']} steps={steps}")
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

    # Load training shards
    print(f"  Loading {cfg_t['shards']} training shards...")
    train_data_list = []
    for i in range(cfg_t["shards"]):
        train_data_list.append(load_shard(i))
    train_data = {k: torch.cat([d[k] for d in train_data_list], dim=0)
                  for k in train_data_list[0].keys()}
    n_train = train_data["board_array"].shape[0]
    print(f"  Training positions: {n_train:,}")
    del train_data_list
    gc.collect()

    # Eval data: shard 9 (held out)
    print(f"  Loading eval data (shard 9, held out)...")
    eval_data = load_shard(9)

    opt = AdamW(model.parameters(), lr=peak_lr, weight_decay=cfg_t["weight_decay"])
    scheduler = cosine_schedule_with_warmup(opt, warmup, steps, peak_lr, min_lr)
    scaler = GradScaler("cuda")
    model.train()

    # Checkpointing
    ckpt_dir = ROOT / "outputs" / "exp176_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    total_pos = 0

    for step in range(1, steps + 1):
        # Gradient accumulation: N micro-batches per optimizer step
        opt.zero_grad(set_to_none=True)
        accum_p_loss = 0.0
        accum_v_loss = 0.0

        for micro in range(grad_accum):
            idx = torch.randint(0, n_train, (micro_bs,))
            board_input, target_move, wdl, valid = prepare_batch(train_data, idx, DEVICE)

            with autocast("cuda", dtype=torch.float16):
                out = model(board_input)
                p_loss = F.cross_entropy(out["policy_logits"][valid.bool()],
                                         target_move[valid.bool()])
                v_loss = F.cross_entropy(out["value_logits"], wdl)
                loss = (p_loss + v_loss) / grad_accum  # scale for accumulation

            scaler.scale(loss).backward()
            accum_p_loss += p_loss.item()
            accum_v_loss += v_loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_t["grad_clip"])
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        total_pos += eff_bs
        accum_p_loss /= grad_accum
        accum_v_loss /= grad_accum

        if step % 100 == 0:
            elapsed = time.time() - t0
            pos_s = total_pos / elapsed
            cur_lr = opt.param_groups[0]["lr"]
            if DEVICE.type == "cuda":
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
            else:
                vram_peak = 0
            print(f"  step={step:5d} p_loss={accum_p_loss:.3f} v_loss={accum_v_loss:.3f}"
                  f" lr={cur_lr:.2e} {pos_s:.0f} pos/s peak={vram_peak:.1f}GB")
            sys.stdout.flush()

        if step % eval_every == 0:
            ev = evaluate(model, eval_data, DEVICE)
            results.append({"step": step, **ev})
            print(f"  >>> EVAL step={step}: top1={ev['top1']:.4f} top3={ev['top3']:.4f}"
                  f" loss={ev['loss']:.3f}")
            sys.stdout.flush()

        if cfg_t["checkpoint_every"] and step % cfg_t["checkpoint_every"] == 0:
            ckpt_path = ckpt_dir / f"exp176_R_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path.name}")
            sys.stdout.flush()

    elapsed = time.time() - t0
    pos_s = total_pos / elapsed

    ev_final = results[-1] if results else {}
    print(f"\n  Completed 86M training in {elapsed:.1f}s ({pos_s:.0f} pos/s)")
    print(f"  Final: top1={ev_final.get('top1',0):.4f} top3={ev_final.get('top3',0):.4f}"
          f" loss={ev_final.get('loss',0):.3f}")

    return {
        "variant": "R",
        "name": "86M_GRAD_ACCUM",
        "params": n_params,
        "shards": cfg_t["shards"],
        "steps": steps,
        "micro_batch": micro_bs,
        "grad_accum": grad_accum,
        "eff_batch": eff_bs,
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
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(args.eval_every, args.seed)

    out_path = ROOT / "outputs" / "exp176_86m_results.json"
    with open(out_path, "w") as f:
        json.dump(result, indent=2, fp=f)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
