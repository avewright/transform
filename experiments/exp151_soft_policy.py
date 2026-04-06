"""exp151: Soft policy targets from Stockfish multi-PV.

Hypothesis: Training with soft policy targets (distributing probability across
top-K Stockfish moves weighted by cp) will improve top-1 accuracy vs. hard
targets with uniform label smoothing. The intuition is that label_smoothing=0.1
distributes 10% probability UNIFORMLY across all 4507 moves, while soft targets
concentrate that mass on the 2-5 moves Stockfish actually considers good.

Protocol:
  1. Resume from exp149 checkpoint (epoch_1.pt or best_model.pt)
  2. Train for 5K steps on shard 0 (which has _soft.pt companion)
  3. Compare against control (hard targets, same settings as exp149)
  4. Eval on 5K eval set every 500 steps

Soft target construction:
  - soft_indices (N, K) and soft_cp (N, K) loaded from shard_00000_soft.pt
  - cp values converted to probabilities: softmax(cp / temperature)
  - Loss: (1-alpha) * CE(logits, hard_target) + alpha * soft_CE(logits, soft_dist)
  - Temperature and alpha are hyperparameters to ablate

Ablation matrix (run sequentially, 5K steps each, ~35 min):
  control : hard targets, label_smoothing=0.1 (exp149 baseline)
  soft_A  : alpha=0.5, temp=100, label_smoothing=0.0
  soft_B  : alpha=1.0, temp=100, label_smoothing=0.0 (fully soft)
  soft_C  : alpha=0.5, temp=50,  label_smoothing=0.0
  soft_D  : alpha=0.3, temp=100, label_smoothing=0.05 (hybrid)

Usage:
  python experiments/exp151_soft_policy.py --checkpoint outputs/exp149_scratch_204m/epoch_1.pt
  python experiments/exp151_soft_policy.py --checkpoint outputs/exp149_scratch_204m/best_model.pt --ablation soft_A
  python experiments/exp151_soft_policy.py --checkpoint outputs/exp149_scratch_204m/epoch_1.pt --ablation all
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_transformer_factory import ChessTransformerConfig, build_model
from data_loader import (
    ShardedChessLoader, board_array_to_fused, ep_square_to_file, compute_wdl,
    load_eval_data, run_eval,
)
from move_vocab import VOCAB_SIZE

OUTPUT_DIR = ROOT / "outputs" / "exp151_soft_policy"
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"

MODEL_CONFIG = ChessTransformerConfig(
    encoder_dim=256, hidden_dim=1024, num_layers=16, num_heads=16,
    ffn_ratio=4, num_encoder_blocks=4, policy_head_dim=512,
    spatial_policy=True,
)

SHUTDOWN = False
def _handler(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True
signal.signal(signal.SIGINT, _handler)
signal.signal(signal.SIGTERM, _handler)


# ── Ablation configs ──

ABLATIONS = {
    "control": {
        "alpha": 0.0,       # no soft targets
        "temperature": 100,
        "label_smoothing": 0.1,
        "desc": "hard targets, ls=0.1 (exp149 baseline)",
    },
    "soft_A": {
        "alpha": 0.5,
        "temperature": 100,
        "label_smoothing": 0.0,
        "desc": "50% soft (temp=100), no uniform smoothing",
    },
    "soft_B": {
        "alpha": 1.0,
        "temperature": 100,
        "label_smoothing": 0.0,
        "desc": "100% soft (temp=100), no uniform smoothing",
    },
    "soft_C": {
        "alpha": 0.5,
        "temperature": 50,
        "label_smoothing": 0.0,
        "desc": "50% soft (temp=50, sharper), no uniform smoothing",
    },
    "soft_D": {
        "alpha": 0.3,
        "temperature": 100,
        "label_smoothing": 0.05,
        "desc": "30% soft (temp=100) + mild uniform smoothing",
    },
}


# ── Soft target utilities ──

def load_soft_targets(shard_path):
    """Load soft target companion file for a shard."""
    soft_path = shard_path.parent / f"{shard_path.stem}_soft.pt"
    if not soft_path.exists():
        return None
    data = torch.load(soft_path, map_location="cpu", weights_only=False)
    return {
        "soft_indices": data["soft_indices"],  # (N, K) int16
        "soft_cp": data["soft_cp"],            # (N, K) int16
    }


def compute_soft_loss(logits, soft_indices, soft_cp, temperature):
    """Compute soft cross-entropy loss from Stockfish multi-PV targets.

    Args:
        logits: (B, VOCAB_SIZE) raw policy logits
        soft_indices: (B, K) move vocabulary indices (-1 = padding)
        soft_cp: (B, K) centipawn values for each top move
        temperature: temperature for cp → probability conversion

    Returns:
        loss: scalar soft cross-entropy
    """
    B, K = soft_indices.shape
    device = logits.device

    # Convert cp to float and mask padding
    cp_float = soft_cp.float().to(device)
    mask = (soft_indices >= 0).to(device)
    cp_float[~mask] = float("-inf")

    # Compute soft probabilities over top-K moves
    soft_probs_k = F.softmax(cp_float / temperature, dim=-1)  # (B, K)

    # Scatter into full vocabulary distribution
    soft_target = torch.zeros(B, VOCAB_SIZE, device=device)
    valid_indices = soft_indices.clamp(min=0).long().to(device)
    soft_target.scatter_(1, valid_indices, soft_probs_k)

    # Soft cross-entropy: -sum(soft_target * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_target * log_probs).sum(dim=-1).mean()

    return loss


def combined_policy_loss(logits, move_targets, soft_indices, soft_cp,
                         alpha, temperature, label_smoothing):
    """Combined policy loss: (1-alpha)*hard_CE + alpha*soft_CE.

    alpha=0.0 → pure hard (with label_smoothing)
    alpha=1.0 → pure soft (label_smoothing ignored for hard part)
    """
    if alpha <= 0.0 or soft_indices is None:
        return F.cross_entropy(logits, move_targets,
                               label_smoothing=label_smoothing)

    if alpha >= 1.0:
        return compute_soft_loss(logits, soft_indices, soft_cp, temperature)

    hard_loss = F.cross_entropy(logits, move_targets,
                                label_smoothing=label_smoothing)
    soft_loss = compute_soft_loss(logits, soft_indices, soft_cp, temperature)
    return (1.0 - alpha) * hard_loss + alpha * soft_loss


# ── Training ──

def log(msg, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def run_ablation(checkpoint_path, ablation_name, config, args):
    """Run one ablation: load checkpoint, train 5K steps, eval, return results."""
    global SHUTDOWN
    log_path = OUTPUT_DIR / f"{ablation_name}.log"
    log(f"\n{'='*60}", log_path)
    log(f"Ablation: {ablation_name} — {config['desc']}", log_path)
    log(f"  alpha={config['alpha']}, temp={config['temperature']}, "
        f"ls={config['label_smoothing']}", log_path)
    log(f"{'='*60}", log_path)

    # Build model + load checkpoint
    model = build_model(MODEL_CONFIG)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.cuda().train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Model: {n_params/1e6:.1f}M params", log_path)

    # Optimizer (fresh — short ablation, don't need resume)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    # Data: single shard (shard_00000) with soft targets
    shard_path = SHARD_DIR / "shard_00000.pt"
    shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
    n = shard_data["board_array"].shape[0]

    # Load soft targets if alpha > 0
    soft_data = None
    if config["alpha"] > 0:
        soft_data = load_soft_targets(shard_path)
        if soft_data is None:
            log(f"  WARNING: No soft targets found, falling back to hard targets",
                log_path)
            config = dict(config, alpha=0.0)
        else:
            soft_n = soft_data["soft_indices"].shape[0]
            if soft_n < n:
                log(f"  Soft targets cover {soft_n:,}/{n:,} positions "
                    f"(using first {soft_n:,})", log_path)
                n = soft_n

    # Precompute derived fields
    fused = board_array_to_fused(shard_data["board_array"][:n])
    turn = shard_data["turn"][:n].long()
    castling = shard_data["castling"][:n].long()
    ep_file = ep_square_to_file(shard_data["ep_square"][:n].long())
    move_idx = shard_data["move_idx"][:n].long()
    wdl = compute_wdl(shard_data["cp"][:n], shard_data["mate"][:n])
    del shard_data

    if soft_data:
        soft_indices = soft_data["soft_indices"][:n]
        soft_cp = soft_data["soft_cp"][:n]
    else:
        soft_indices = None
        soft_cp = None

    # Eval data
    eval_path = SHARD_DIR / "eval_20k.pt"
    if not eval_path.exists():
        eval_path = SHARD_DIR / "eval.pt"
    eval_data, eval_tensors = load_eval_data(eval_path)
    log(f"  Eval: {len(eval_data)} positions from {eval_path.name}", log_path)

    # Initial eval (before training)
    torch.cuda.empty_cache()
    acc0, top3_0, val0 = run_eval(model, eval_data, eval_tensors)
    log(f"  Pre-train: acc={acc0:.2%} top3={top3_0:.2%} val={val0:.2%}", log_path)

    best_acc = acc0
    results_history = [{"step": 0, "acc": acc0, "top3": top3_0, "val": val0}]

    # Training loop
    bs = args.batch_size
    accum_steps = args.accum_steps
    max_steps = args.max_steps
    total_loss_p = 0.0
    total_loss_v = 0.0
    accum_count = 0
    step = 0
    positions_seen = 0
    t0 = time.time()

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng)

    batch_start = 0

    while step < max_steps and not SHUTDOWN:
        # Get batch indices
        if batch_start + bs > n:
            # Reshuffle
            perm = torch.randperm(n, generator=rng)
            batch_start = 0
        idx = perm[batch_start:batch_start + bs]
        batch_start += bs

        batch_input = {
            "fused_ids": fused[idx].cuda(),
            "turn": turn[idx].cuda(),
            "castling": castling[idx].cuda(),
            "ep_file": ep_file[idx].cuda(),
        }
        mt = move_idx[idx].cuda()
        wt = wdl[idx].float().cuda()

        # Get soft targets for this batch
        batch_soft_idx = None
        batch_soft_cp = None
        if soft_indices is not None:
            batch_soft_idx = soft_indices[idx]
            batch_soft_cp = soft_cp[idx]

        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)

            p_loss = combined_policy_loss(
                result["policy_logits"], mt,
                batch_soft_idx, batch_soft_cp,
                config["alpha"], config["temperature"],
                config["label_smoothing"])

            v_loss = F.cross_entropy(result["value_logits"], wt)
            loss = (p_loss + 0.5 * v_loss) / accum_steps

        scaler.scale(loss).backward()

        total_loss_p += p_loss.item()
        total_loss_v += v_loss.item()
        accum_count += 1
        positions_seen += bs

        if accum_count >= accum_steps:
            scaler.unscale_(optimizer)
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            accum_count = 0

            # Log
            if step % 100 == 0:
                avg_p = total_loss_p / (accum_steps * 100)
                avg_v = total_loss_v / (accum_steps * 100)
                elapsed = time.time() - t0
                rate = positions_seen / elapsed
                log(f"  step {step}/{max_steps} | p={avg_p:.4f} v={avg_v:.4f} | "
                    f"{rate:.0f} pos/s", log_path)
                total_loss_p = 0.0
                total_loss_v = 0.0

            # Eval
            if step % args.eval_interval == 0:
                torch.cuda.empty_cache()
                acc, top3, val_acc = run_eval(model, eval_data, eval_tensors)
                log(f"  EVAL step {step}: acc={acc:.2%} top3={top3:.2%} "
                    f"val={val_acc:.2%}", log_path)
                results_history.append({
                    "step": step, "acc": acc, "top3": top3, "val": val_acc})
                if acc > best_acc:
                    best_acc = acc
                    log(f"  ** New best! acc={best_acc:.2%}", log_path)
                model.train()

    # Final eval
    torch.cuda.empty_cache()
    acc_f, top3_f, val_f = run_eval(model, eval_data, eval_tensors)
    log(f"  Final: acc={acc_f:.2%} top3={top3_f:.2%} val={val_f:.2%}", log_path)
    results_history.append({"step": step, "acc": acc_f, "top3": top3_f, "val": val_f})

    elapsed = time.time() - t0
    result_summary = {
        "ablation": ablation_name,
        "config": config,
        "pre_train": {"acc": acc0, "top3": top3_0, "val": val0},
        "final": {"acc": acc_f, "top3": top3_f, "val": val_f},
        "best_acc": best_acc,
        "delta_acc": acc_f - acc0,
        "steps": step,
        "elapsed_s": elapsed,
        "history": results_history,
    }

    log(f"\n  Summary: pre={acc0:.2%} → final={acc_f:.2%} "
        f"(Δ={acc_f-acc0:+.2%}), best={best_acc:.2%}, "
        f"time={timedelta(seconds=int(elapsed))}", log_path)

    return result_summary


def main():
    parser = argparse.ArgumentParser(description="exp151: soft policy targets")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to starting checkpoint")
    parser.add_argument("--ablation", default="all",
                        help="Which ablation(s) to run: all, control, soft_A, ...")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Training steps per ablation (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than exp149 for fine-tuning)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=500)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which ablations to run
    if args.ablation == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [a.strip() for a in args.ablation.split(",")]
        for name in ablation_names:
            if name not in ABLATIONS:
                print(f"Unknown ablation: {name}")
                print(f"Available: {list(ABLATIONS.keys())}")
                return

    log(f"exp151: Soft Policy Targets")
    log(f"  Checkpoint: {args.checkpoint}")
    log(f"  Ablations: {ablation_names}")
    log(f"  Steps per ablation: {args.max_steps}")

    all_results = []

    for abl_name in ablation_names:
        config = ABLATIONS[abl_name]
        result = run_ablation(args.checkpoint, abl_name, config, args)
        all_results.append(result)

        if SHUTDOWN:
            log("Shutdown requested — stopping")
            break

    # Print comparison table
    log(f"\n{'='*70}")
    log(f"{'Ablation':<12} {'Alpha':>6} {'Temp':>5} {'LS':>5} "
        f"{'Pre':>7} {'Final':>7} {'Δ':>7} {'Best':>7}")
    log(f"{'-'*70}")
    for r in all_results:
        c = r["config"]
        log(f"{r['ablation']:<12} {c['alpha']:>6.2f} {c['temperature']:>5} "
            f"{c['label_smoothing']:>5.2f} "
            f"{r['pre_train']['acc']:>6.2%} {r['final']['acc']:>6.2%} "
            f"{r['delta_acc']:>+6.2%} {r['best_acc']:>6.2%}")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
