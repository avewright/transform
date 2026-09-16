#!/usr/bin/env python3
"""99M recurrence experiment: untouched sweep, then compute-matched FT arms.

No downloads, uploads, automatic promotion, or edits to the source checkpoint.
See docs/exp283_recurrent_depth.md for the experiment protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch
import torch.nn.functional as F

from chess_inference import load_checkpoint
from chess_squares64 import Squares64RecurrentTransformer, average_recurrent_grads, count_parameters
from autoresearch_8gb.pipeline import (
    attach_static_targets, prepare_soft_batch, position_hashes, soft_policy_loss,
    _row_to_board,
)
from data_loader import PIECE_MAP
from move_vocab import VOCAB_SIZE, legal_move_mask


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def depth_schedule(steps, depths, seed):
    """Shuffle complete cycles: same average depth as fixed-three control."""
    if not depths or min(depths) < 1 or sum(depths) != 3 * len(depths):
        raise ValueError("Training depths must be positive and average exactly 3")
    if steps < 1 or steps % len(depths):
        raise ValueError("steps must be positive and divisible by the depth-cycle length")
    rng = random.Random(seed)
    schedule = []
    for _ in range(steps // len(depths)):
        cycle = list(depths)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = ("board_array", "turn", "castling", "ep_square", "move_idx",
                "cp", "mate", "soft_indices", "soft_probs")
    n = len(data["turn"])
    if not n or any(k not in data or len(data[k]) != n for k in required):
        raise ValueError(f"Empty or malformed soft cache: {path}")
    if "policy_mask" in data:
        keep = data["policy_mask"].reshape(-1).bool()
        data = subset(data, keep)
    if len(data["turn"]) == 0:
        raise ValueError("No policy-valid rows")
    return attach_static_targets(data)


def subset(data, indices):
    n = len(data["turn"])
    return {k: v[indices] if torch.is_tensor(v) and v.ndim and len(v) == n else v
            for k, v in data.items()}


def exclude_overlap(train, evaluation):
    keep = ~np.isin(position_hashes(train), position_hashes(evaluation))
    if not keep.any():
        raise ValueError("All training rows overlap evaluation")
    return subset(train, torch.from_numpy(keep)), int((~keep).sum())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def model_from(path, device):
    model = load_checkpoint(path, device=device)
    if not isinstance(model, Squares64RecurrentTransformer):
        raise ValueError("Expected a squares64 recurrent checkpoint")
    if count_parameters(model) != 98_971_224 or model.config.recurrent_unrolls != 3:
        raise ValueError("Expected the original 98,971,224-parameter three-pass 99M model")
    if model.config.use_history or VOCAB_SIZE != 1968:
        raise ValueError("This experiment requires compact vocab and board-only 99M")
    return model


@torch.no_grad()
def sweep(model, data, depths, device, batch_size):
    """Identical rows at each depth; timed forward excludes masks/data loading."""
    model.eval()
    n = len(data["turn"])
    id_to_piece = {v: k for k, v in PIECE_MAP.items()}
    masks = torch.stack([legal_move_mask(_row_to_board(data, i, id_to_piece))
                         for i in range(n)])
    if not masks.any(dim=1).all():
        raise ValueError("Evaluation includes terminal boards")
    records = []
    for depth in depths:
        # Warm each shape/depth before timing; report throughput, not move latency.
        bi, *_ = prepare_soft_batch(data, torch.arange(min(batch_size, n)), device)
        model(bi, recurrent_unrolls=depth)
        sync(device)
        correct, loss_sum, seconds, chosen = 0, 0., 0., []
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n))
            bi, hard, *_ = prepare_soft_batch(data, idx, device)
            sync(device)
            t0 = time.perf_counter()
            logits = model(bi, recurrent_unrolls=depth)["policy_logits"].float()
            sync(device)
            seconds += time.perf_counter() - t0
            loss_sum += F.cross_entropy(logits, hard, reduction="sum").item()
            pred = logits.masked_fill(~masks[idx].to(device), -torch.inf).argmax(-1)
            correct += (pred == hard).sum().item()
            chosen.extend(pred.cpu().tolist())
        record = dict(loops=depth, effective_layers=model.config.prefix_layers +
                      depth * model.config.recurrent_layers + model.config.suffix_layers,
                      n=n, legal_top1=correct / n, hard_ce=loss_sum / n,
                      forward_positions_per_second=n / seconds, predictions=chosen)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "predictions"}), flush=True)
    base = next((r for r in records if r["loops"] == 3), None)
    if base:
        truth = data["move_idx"].tolist()
        for r in records:
            pairs = list(zip(base["predictions"], r["predictions"], truth))
            r["rescued_vs_3"] = sum(a != y and b == y for a, b, y in pairs)
            r["broken_vs_3"] = sum(a == y and b != y for a, b, y in pairs)
    return records


def save_model(model, path, metadata, depth=3):
    payload = dict(arch="squares64", config=replace(model.config, recurrent_unrolls=depth).to_dict(),
                   model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   experiment=metadata)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_arm(model, data, schedule, args, out):
    """Both arms use fresh AdamW, same row RNG, loss and update count."""
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    geometry_lr = getattr(args, "geometry_lr", args.lr)
    base, geometry = [], []
    for name, parameter in model.named_parameters():
        (geometry if ".gab." in name or ".shaw_" in name else base).append(parameter)
    groups = [{"params": base, "lr": args.lr, "initial_lr": args.lr}]
    if geometry:
        groups.append({"params": geometry, "lr": geometry_lr, "initial_lr": geometry_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=getattr(args, "weight_decay", 0.01))
    model.train()
    device = next(model.parameters()).device
    layers = 0
    t0 = time.monotonic()
    for step, depth in enumerate(schedule, 1):
        warmup = getattr(args, "warmup", 0)
        factor = min(step / warmup, 1.) if warmup else 1.
        if getattr(args, "cosine_decay", False) and step > warmup:
            fraction = (step - warmup) / max(len(schedule) - warmup, 1)
            floor = getattr(args, "min_lr_fraction", 0.1)
            factor = floor + (1. - floor) * 0.5 * (1. + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor
        idx = torch.randint(len(data["turn"]), (args.batch_size,), generator=rng)
        bi, hard, wdl, si, sp = prepare_soft_batch(data, idx, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(bi, recurrent_unrolls=depth)
        alpha = getattr(args, "soft_alpha", 0.55)
        policy = (1. - alpha) * F.cross_entropy(result["policy_logits"], hard)
        policy = policy + alpha * soft_policy_loss(result["policy_logits"], si, sp)
        valid = data["value_valid"][idx].to(device).bool()
        value = (F.cross_entropy(result["value_logits"][valid], wdl[valid])
                 if valid.any() else result["value_logits"].sum() * 0)
        loss = policy + getattr(args, "value_weight", 0.15) * value
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        average_recurrent_grads(model, unrolls=depth)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        layers += args.batch_size * (model.config.prefix_layers +
                                    depth * model.config.recurrent_layers + model.config.suffix_layers)
        if step == 1 or step % 25 == 0 or step == len(schedule):
            row = dict(step=step, loops=depth, loss=loss.item(), grad_norm=float(norm),
                       examples=step * args.batch_size, block_position_evaluations=layers,
                       elapsed_s=time.monotonic() - t0,
                       learning_rates=[g["lr"] for g in optimizer.param_groups])
            with (out / "train.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == len(schedule):
            save_model(model, out / "latest.pt", dict(step=step, arm=args.arm,
                       seed=args.seed, block_position_evaluations=layers))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["sweep", "train"])
    p.add_argument("--ckpt", type=Path, default=ROOT / "outputs/hf_100m_squares64/latest.pt")
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--train-cache", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=["fixed", "variable"], default="variable")
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--train-depths", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-rows", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=283)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--export-depths", action="store_true", help="Write one inference checkpoint per depth for harness.elo")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    if min(args.depths) < 1 or min(args.batch_size, args.eval_rows, args.save_every) < 1 or args.lr <= 0:
        p.error("Depths, sizes, save interval and LR must be positive")
    schedule = depth_schedule(args.steps, args.train_depths, args.seed)
    if args.mode == "train" and args.train_cache is None:
        p.error("train requires --train-cache")
    # A fresh experiment directory avoids accidental checkpoint replacement.
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be new or empty")
    device = torch.device(args.device)
    full_eval = load_data(args.eval_cache)
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_eval["turn"]), generator=rng)[:args.eval_rows]
    evaluation = subset(full_eval, indices)
    train, removed = None, 0
    if args.mode == "train":
        train, removed = exclude_overlap(load_data(args.train_cache), full_eval)
    model = model_from(args.ckpt, device)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(checkpoint_sha256=fingerprint(args.ckpt), eval_sha256=fingerprint(args.eval_cache),
                    eval_indices=indices.tolist(), removed_train_overlap=removed,
                    train_sha256=fingerprint(args.train_cache) if train is not None else None,
                    train_rows=len(train["turn"]) if train is not None else 0,
                    config=model.config.to_dict(), torch_version=torch.__version__,
                    note="Eval provenance may overlap historical pretraining; not claimed globally unseen.")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.mode == "train":
        train_arm(model, train, [3] * args.steps if args.arm == "fixed" else schedule, args, args.out)
    report = sweep(model, evaluation, args.depths, device, args.batch_size)
    (args.out / "sweep.json").write_text(json.dumps(report, indent=2))
    if args.export_depths:
        for depth in args.depths:
            save_model(model, args.out / f"loops_{depth}.pt", manifest, depth)


if __name__ == "__main__":
    main()
