#!/usr/bin/env python3
"""exp188: Finetune HF 200M (legacy 5504) onto compact vocab + soft MultiPV harvest.

SpatialPolicyHead has no vocab-sized linear — only from_sqs/to_sqs/promo buffers.
Loading drops those buffers and rebuilds them for VOCAB_SIZE=1968 under
MOVE_VOCAB_VERSION=compact. Soft cache from exp186 is already compact-indexed.

Usage:
  python experiments/exp188_200m_compact_soft.py --go
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import (
    ChessTransformerConfig,
    DEFAULT_200M_CONFIG,
    build_model,
    count_parameters,
)
from data_loader import board_array_to_fused, compute_wdl, ep_square_to_file, stream_hf_batches
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = ROOT / "outputs" / "hf_checkpoint" / "best_model.pt"
DEFAULT_SOFT = ROOT / "outputs" / "exp186_sf_multipv" / "soft_cache_merged.pt"
SHUTDOWN = False
LOG_PATH: Path | None = None


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _handle_sig(*_):
    global SHUTDOWN
    SHUTDOWN = True
    log("shutdown requested")


def soft_policy_loss(logits, soft_indices, soft_probs):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = (soft_indices >= 0) & (soft_probs > 0)
    safe = soft_indices.clamp(min=0).long()
    gathered = log_probs.gather(1, safe) * valid.float()
    return -(soft_probs.float() * gathered).sum(dim=-1).mean()


def load_200m_compact(ckpt_path: Path):
    """Build DEFAULT_200M under compact vocab; load weights, skip legacy buffers."""
    assert VOCAB_SIZE == 1968, f"expected compact vocab, got {VOCAB_SIZE}"
    model = build_model(DEFAULT_200M_CONFIG)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    skip = {k for k in sd if k.endswith(("from_sqs", "to_sqs", "promo_types"))}
    filtered = {k: v for k, v in sd.items() if k not in skip}
    msg = model.load_state_dict(filtered, strict=False)
    unexpected = [k for k in msg.unexpected_keys if not k.endswith(("from_sqs", "to_sqs", "promo_types"))]
    missing = [k for k in msg.missing_keys if not k.endswith(("from_sqs", "to_sqs", "promo_types"))]
    if unexpected or missing:
        log(f"  load warn missing={missing[:5]} unexpected={unexpected[:5]}")
    log(f"  skipped legacy buffers: {sorted(skip)}")
    log(f"  policy buffers now {tuple(model.policy_head.from_sqs.shape)} (compact)")
    return model


def prepare_soft_batch(data, indices, device):
    ba = data["board_array"][indices]
    turn = data["turn"][indices]
    castling = data["castling"][indices]
    ep = data["ep_square"][indices]
    move_idx = data["move_idx"][indices]
    cp = data["cp"][indices]
    mate = data["mate"][indices]
    soft_i = data["soft_indices"][indices]
    soft_p = data["soft_probs"][indices]
    board_input = {
        "fused_ids": board_array_to_fused(ba).to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_square_to_file(ep).long().to(device),
    }
    wdl = compute_wdl(cp, mate).to(device)
    return board_input, move_idx.long().to(device), wdl, soft_i.to(device), soft_p.to(device)


@torch.no_grad()
def eval_soft_top1(model, soft_data, device, n=5000):
    model.eval()
    N = soft_data["board_array"].shape[0]
    start = max(0, N - n)
    idx = torch.arange(start, N)
    bs = 256
    correct = 0
    soft_loss_sum = 0.0
    total = 0
    for i in range(0, len(idx), bs):
        batch_idx = idx[i:i + bs]
        bi, hard, wdl, si, sp = prepare_soft_batch(soft_data, batch_idx, device)
        with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(bi)
            logits = out["policy_logits"]
            soft_loss_sum += soft_policy_loss(logits, si, sp).item() * batch_idx.numel()
            pred = logits.argmax(dim=-1)
            correct += (pred == hard).sum().item()
            total += batch_idx.numel()
    model.train()
    return {"top1": correct / max(total, 1), "soft_loss": soft_loss_sum / max(total, 1), "n": total}


def save_checkpoint(model, optimizer, scaler, step, best_metric, path, args):
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "step": step,
        "best_metric": best_metric,
        "config": DEFAULT_200M_CONFIG.to_dict(),
        "vocab": "compact",
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)


def main():
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--soft-cache", default=str(DEFAULT_SOFT))
    ap.add_argument("--output-dir", default="outputs/exp188_200m_compact_soft")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--soft-frac", type=float, default=0.85,
                    help="Fraction of steps using harvest soft cache (rest = deep HF hard)")
    ap.add_argument("--soft-alpha", type=float, default=0.25,
                    help="Within soft batches: weight on MultiPV CE vs best-move hard CE")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume weights from a prior exp188 checkpoint (fresh opt unless --resume-opt)")
    ap.add_argument("--resume-opt", action="store_true")
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--value-weight", type=float, default=0.15)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--min-depth", type=int, default=12)
    ap.add_argument("--shuffle-buffer", type=int, default=2048)
    args = ap.parse_args()

    if not args.go:
        print("DRY RUN. Pass --go to train.")
        return
    if args.smoke:
        args.steps = 40
        args.log_interval = 5
        args.save_interval = 20
        args.eval_interval = 20
        args.warmup = 5
        args.batch_size = min(args.batch_size, 64)
        args.accum_steps = 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out / "training.log"
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    log("=" * 64)
    log("exp188: 200M → compact vocab + soft MultiPV finetune")
    log(f"  device={DEVICE} vocab={VOCAB_SIZE}")
    log(f"  ckpt={args.checkpoint}")
    log(f"  soft_cache={args.soft_cache}")
    log(f"  soft_frac={args.soft_frac} soft_alpha={args.soft_alpha} lr={args.lr}")

    soft_data = torch.load(args.soft_cache, map_location="cpu", weights_only=False)
    n_soft = soft_data["board_array"].shape[0]
    eval_holdout = min(5000, max(1024, n_soft // 20))
    train_soft_n = max(1, n_soft - eval_holdout)
    log(f"  soft train={train_soft_n:,} holdout={eval_holdout:,}")

    if args.resume:
        log(f"  resume weights from {args.resume}")
        model = build_model(DEFAULT_200M_CONFIG)
        rckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"], strict=False)
        model = model.to(DEVICE)
        bm = float(rckpt.get("best_metric", 0.0))
        # Older runs stored soft_loss (>1); new runs store top-1 in (0,1]
        start_top1 = bm if 0.0 < bm <= 1.0 else 0.0
    else:
        rckpt = None
        model = load_200m_compact(Path(args.checkpoint)).to(DEVICE)
        start_top1 = 0.0
    n_params = count_parameters(model)
    log(f"  params={n_params/1e6:.1f}M config={DEFAULT_200M_CONFIG}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume and args.resume_opt and rckpt is not None and "optimizer_state_dict" in rckpt:
        try:
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            log("  resumed optimizer state")
        except Exception as e:
            log(f"  optimizer resume skipped: {e}")
    scaler = GradScaler("cuda", enabled=False)  # bf16
    base_lr = args.lr

    def lr_scale(s: int) -> float:
        if s < args.warmup:
            return (s + 1) / max(args.warmup, 1)
        progress = (s - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine

    hard_iter = None
    if args.soft_frac < 1.0:
        hard_iter = iter(stream_hf_batches(
            batch_size=args.batch_size, device=DEVICE, seed=42,
            shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
        ))

    with open(out / "config.json", "w") as f:
        json.dump({
            "model": DEFAULT_200M_CONFIG.to_dict(),
            "training": vars(args),
            "n_params": n_params,
            "n_soft": n_soft,
            "vocab": "compact",
        }, f, indent=2)

    eff_bs = args.batch_size * args.accum_steps
    log(f"  batch={args.batch_size} accum={args.accum_steps} eff_bs={eff_bs}")
    log(f"  steps={args.steps:,}")
    log("=" * 64)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    best_top1 = float(start_top1)
    if args.resume and best_top1 <= 0:
        # Seed from known aborted eval if metric wasn't stored as top-1
        best_top1 = 0.20
        log(f"  seeding best_top1={best_top1*100:.1f}% from prior eval")
    t0 = time.time()
    positions = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    accum_p = accum_v = accum_soft = accum_hard = 0.0
    accum_n = soft_steps = hard_steps = 0
    # Only abort if top-1 collapses badly below zero-shot (~6%)
    top1_abort = 0.03

    if DEVICE.type == "cuda":
        log(f"  vram allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")

    while step < args.steps:
        if SHUTDOWN:
            save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args)
            log(f"Saved on shutdown at step {step}")
            return

        use_soft = torch.rand(1, generator=rng).item() < args.soft_frac
        for _ in range(args.accum_steps):
            if use_soft:
                idx = torch.randint(0, train_soft_n, (args.batch_size,), generator=rng)
                bi, hard, wdl, si, sp = prepare_soft_batch(soft_data, idx, DEVICE)
                with autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out_m = model(bi)
                    logits = out_m["policy_logits"]
                    hard_ce = F.cross_entropy(logits, hard, label_smoothing=args.label_smoothing)
                    soft_ce = soft_policy_loss(logits, si, sp)
                    p_loss = (1.0 - args.soft_alpha) * hard_ce + args.soft_alpha * soft_ce
                    v_loss = F.cross_entropy(out_m["value_logits"], wdl)
                    loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
                loss.backward()
                accum_p += p_loss.item()
                accum_v += v_loss.item()
                accum_soft += soft_ce.item()
                accum_hard += hard_ce.item()
                soft_steps += 1
            else:
                try:
                    bi, move_t, wdl_t = next(hard_iter)
                except StopIteration:
                    hard_iter = iter(stream_hf_batches(
                        batch_size=args.batch_size, device=DEVICE, seed=43 + step,
                        shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
                    ))
                    bi, move_t, wdl_t = next(hard_iter)
                with autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out_m = model(bi)
                    p_loss = F.cross_entropy(out_m["policy_logits"], move_t, label_smoothing=args.label_smoothing)
                    v_loss = F.cross_entropy(out_m["value_logits"], wdl_t)
                    loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
                loss.backward()
                accum_p += p_loss.item()
                accum_v += v_loss.item()
                hard_steps += 1
            accum_n += 1
            positions += args.batch_size

        gn = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * lr_scale(step + 1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_interval == 0:
            elapsed = max(time.time() - t0, 1e-6)
            vram = f" | vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB" if DEVICE.type == "cuda" else ""
            log(
                f"step {step:,}/{args.steps:,} | "
                f"p={accum_p/accum_n:.4f} v={accum_v/accum_n:.4f} "
                f"soft={accum_soft/max(soft_steps,1):.4f} hardCE={accum_hard/max(soft_steps,1):.4f} | "
                f"mix soft_steps={soft_steps} hard_steps={hard_steps} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} gn={float(gn):.2f} | "
                f"{positions/elapsed:.0f} pos/s{vram}"
            )
            accum_p = accum_v = accum_soft = accum_hard = 0.0
            accum_n = soft_steps = hard_steps = 0

        if step % args.eval_interval == 0:
            metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
            log(f"  eval holdout top1={metrics['top1']*100:.2f}% soft_loss={metrics['soft_loss']:.4f} n={metrics['n']}")
            if metrics["top1"] < top1_abort:
                log(f"ABORT top1={metrics['top1']*100:.2f}% < {top1_abort*100:.0f}% — model collapsed")
                save_checkpoint(model, optimizer, scaler, step, best_top1, out / "aborted.pt", args)
                return
            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                save_checkpoint(model, optimizer, scaler, step, best_top1, out / "best.pt", args)
                log(f"  new best top1={best_top1*100:.2f}%")

        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args)
            save_checkpoint(model, optimizer, scaler, step, best_top1, out / f"step_{step:06d}.pt", args)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
    log(f"Done. step={step:,} top1={metrics['top1']*100:.2f}% soft_loss={metrics['soft_loss']:.4f} best_top1={best_top1*100:.2f}%")
    save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args)
    if metrics["top1"] > best_top1:
        save_checkpoint(model, optimizer, scaler, step, metrics["top1"], out / "best.pt", args)


if __name__ == "__main__":
    main()
