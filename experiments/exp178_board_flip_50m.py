"""exp178: Board-Flip 50M from scratch — A/B test vs exp174 N.

Hypothesis: Normalizing all positions to side-to-move perspective (board flip)
effectively halves the state space and doubles training signal per position.
ChessFormer (Monroe 2024) showed this is a strong inductive bias.

This is exp174 N (50M, 20K steps) with exactly ONE change: board_flip enabled.
Black-to-move positions are flipped so the model always sees "my pieces at bottom".

Comparison target:
  - exp174 N (50M, no flip, 20K): 21.20% top-1, 46.58% top-3, loss 2.604

Usage:
  python experiments/exp178_board_flip_50m.py --eval-every 1000
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
from board_flip import flip_batch, build_flip_move_table

ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config: identical to exp174 N ──
MODEL_CONFIG = dict(
    encoder_dim=256, hidden_dim=640, num_layers=10, num_heads=10,
    ffn_ratio=4, dropout=0.05, policy_head_dim=256,
    value_hidden=256, use_pos_embed=True, n_ctx_tokens=4,
    value_head_type="cls", n_value_classes=3,
    use_swiglu=True, use_rel_bias=True,
)
N_SHARDS = 8
TOTAL_STEPS = 20000
PEAK_LR = 2e-4
MIN_LR = 1e-5
WARMUP = 500
CHECKPOINT_EVERY = 5000


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
    return board_input, compact_move.to(device), wdl.to(device), valid.to(device), turn.long().to(device)


def apply_board_flip(board_input, move_targets, wdl, flip_table_device):
    """Flip Black positions to side-to-move perspective.
    
    Also flips WDL: for Black, swap W and L columns (model sees from Black's view,
    so White's win becomes Black's loss).
    """
    black_mask = (board_input["turn"] == 1)
    
    # Flip board, castling, move targets for Black positions
    flipped_input, flipped_moves = flip_batch(board_input, move_targets, flip_table_device)
    
    # Flip WDL for Black: [W, D, L] → [L, D, W]  
    if black_mask.any():
        flipped_wdl = wdl.clone()
        flipped_wdl[black_mask, 0] = wdl[black_mask, 2]  # W ← L
        flipped_wdl[black_mask, 2] = wdl[black_mask, 0]  # L ← W
    else:
        flipped_wdl = wdl
    
    return flipped_input, flipped_moves, flipped_wdl


@torch.no_grad()
def evaluate(model, eval_data, device, flip_table_device, num_samples=5000):
    model.eval()
    bs = 256
    correct1 = correct3 = total = 0
    total_loss = 0.0
    n_batches = 0
    N = min(num_samples, eval_data["board_array"].shape[0])

    for start in range(0, N, bs):
        end = min(start + bs, N)
        idx = torch.arange(start, end)
        board_input, target_move, wdl, valid, turn = prepare_batch(eval_data, idx, device)

        # Apply board flip for eval too
        flipped_input, flipped_moves, _ = apply_board_flip(
            board_input, target_move, wdl, flip_table_device
        )

        with autocast("cuda", dtype=torch.float16):
            out = model(flipped_input)
            policy_logits = out["policy_logits"]
            loss = F.cross_entropy(
                policy_logits[valid.bool()], flipped_moves[valid.bool()]
            )

        total_loss += loss.item()
        n_batches += 1
        preds = policy_logits[valid.bool()].topk(3, dim=-1).indices
        targets = flipped_moves[valid.bool()]
        correct1 += (preds[:, 0] == targets).sum().item()
        correct3 += (preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += valid.sum().item()

    model.train()
    return {
        "top1": correct1 / max(total, 1),
        "top3": correct3 / max(total, 1),
        "loss": total_loss / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
    print(f"Compact vocab size: {VOCAB_SIZE}")

    print(f"\n{'='*60}")
    print(f"  exp178: 50M BOARD FLIP FROM SCRATCH")
    print(f"  shards={N_SHARDS} steps={TOTAL_STEPS} bs={args.batch_size}")
    print(f"  LR: warmup {WARMUP} -> peak {PEAK_LR}, cosine -> {MIN_LR}")
    print(f"  BOARD FLIP: ENABLED (Black positions normalized to STM view)")
    print(f"  Comparison: exp174 N (no flip) = 21.20% top-1 @ 20K")
    print(f"{'='*60}\n")

    # Build model — identical to exp174 N
    cfg = ChessTransformerConfig(**MODEL_CONFIG)
    model = build_model(cfg).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    if DEVICE.type == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM after model: {vram_used:.2f} / {vram_total:.1f} GB")

    # Build flip move table
    flip_table = build_flip_move_table()
    flip_table_device = flip_table.to(DEVICE)
    print(f"  Flip table: {len(flip_table)} move mappings")

    # Load training shards
    print(f"  Loading {N_SHARDS} training shards...")
    train_data_list = []
    for i in range(N_SHARDS):
        train_data_list.append(load_shard(i))
    train_data = {k: torch.cat([d[k] for d in train_data_list], dim=0)
                  for k in train_data_list[0].keys()}
    n_train = train_data["board_array"].shape[0]
    print(f"  Training positions: {n_train:,}")
    del train_data_list
    gc.collect()

    # Eval data: shard 9
    EVAL_SHARD = 9
    print(f"  Loading eval data (shard {EVAL_SHARD}, held out)...")
    eval_data = load_shard(EVAL_SHARD)

    # Checkpoint directory
    ckpt_dir = ROOT / "outputs" / "exp178_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)
    scheduler = cosine_schedule_with_warmup(opt, WARMUP, TOTAL_STEPS, PEAK_LR, MIN_LR)
    scaler = GradScaler("cuda")
    model.train()

    results = []
    t0 = time.time()
    total_pos = 0

    # Initial eval
    ev0 = evaluate(model, eval_data, DEVICE, flip_table_device)
    print(f"  >>> EVAL step=0: top1={ev0['top1']:.4f} top3={ev0['top3']:.4f}"
          f" loss={ev0['loss']:.3f}")
    sys.stdout.flush()

    for step in range(1, TOTAL_STEPS + 1):
        idx = torch.randint(0, n_train, (args.batch_size,))
        board_input, target_move, wdl, valid, turn = prepare_batch(
            train_data, idx, DEVICE
        )

        # BOARD FLIP: normalize to side-to-move perspective
        flipped_input, flipped_moves, flipped_wdl = apply_board_flip(
            board_input, target_move, wdl, flip_table_device
        )

        with autocast("cuda", dtype=torch.float16):
            out = model(flipped_input)
            p_loss = F.cross_entropy(
                out["policy_logits"][valid.bool()], flipped_moves[valid.bool()]
            )
            v_loss = F.cross_entropy(out["value_logits"], flipped_wdl)
            loss = p_loss + v_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        total_pos += args.batch_size

        if step % 100 == 0:
            elapsed = time.time() - t0
            pos_s = total_pos / elapsed
            cur_lr = opt.param_groups[0]["lr"]
            peak_vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0
            print(f"  step={step:5d} p_loss={p_loss.item():.3f} v_loss={v_loss.item():.3f}"
                  f" lr={cur_lr:.2e} {pos_s:.0f} pos/s peak={peak_vram:.1f}GB")
            sys.stdout.flush()

        if step % args.eval_every == 0:
            ev = evaluate(model, eval_data, DEVICE, flip_table_device)
            results.append({"step": step, **ev})
            print(f"  >>> EVAL step={step}: top1={ev['top1']:.4f} top3={ev['top3']:.4f}"
                  f" loss={ev['loss']:.3f}")
            sys.stdout.flush()

        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = ckpt_dir / f"exp178_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> CHECKPOINT saved: {ckpt_path.name}")
            sys.stdout.flush()

    # Final checkpoint
    if TOTAL_STEPS % CHECKPOINT_EVERY != 0:
        ckpt_path = ckpt_dir / f"exp178_step{TOTAL_STEPS}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >>> CHECKPOINT saved: {ckpt_path.name}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    pos_s = total_pos / elapsed
    ev_final = results[-1] if results else {}

    print(f"\n  Completed exp178 board flip in {elapsed:.1f}s ({pos_s:.0f} pos/s)")
    print(f"  Final: top1={ev_final.get('top1', 0):.4f} top3={ev_final.get('top3', 0):.4f}"
          f" loss={ev_final.get('loss', 0):.3f}")

    # Save results
    out_path = ROOT / "outputs" / "exp178_board_flip_results.json"
    result = {
        "experiment": "exp178_board_flip_50m",
        "params": n_params,
        "board_flip": True,
        "shards": N_SHARDS,
        "steps": TOTAL_STEPS,
        "peak_lr": PEAK_LR,
        "min_lr": MIN_LR,
        "warmup": WARMUP,
        "batch_size": args.batch_size,
        "top1": ev_final.get("top1", 0),
        "top3": ev_final.get("top3", 0),
        "loss": ev_final.get("loss", 0),
        "time_s": elapsed,
        "pos_s": pos_s,
        "eval_history": results,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
