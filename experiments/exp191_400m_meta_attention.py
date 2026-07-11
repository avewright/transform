#!/usr/bin/env python3
"""exp191: ≥400M meta-factored attention from scratch (A40 / NorMuon).

Architecture (DEFAULT_400M_META_CONFIG):
  - Separate content / position streams through attention
  - 4-term scores: cc + ss + cs + sc (DeBERTa-style)
  - Shaw relative vectors on ss only (Δfile, Δrank) — no handcrafted rays
  - Removed: ChessRelativeBias, sequence pos_embed, full_dim_attention

A40 RunPod defaults (wise rental use):
  - NorMuon on ≥2D trunk weights + tiny AdamW aux (embeds/heads/norms)
  - Large micro-batch, accum=1, bf16, TF32, cudnn.benchmark
  - Grad checkpoint OFF by default (throughput); --grad-checkpoint if OOM
  - Soft MultiPV + hard HF ballast; optional deep soft mix

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp191_400m_meta_attention.py --go --smoke
  bash scripts/run_exp191_a40.sh
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
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import (
    DEFAULT_400M_META_CONFIG,
    build_model,
    count_parameters,
)
from data_loader import (
    board_array_to_fused,
    compute_wdl,
    ep_square_to_file,
    hflip_board_array,
    hflip_ep_square,
    hflip_move_idx,
    stream_hf_batches,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SOFT = ROOT / "outputs" / "exp186_sf_multipv" / "soft_cache.pt"
SHUTDOWN = False
LOG_PATH: Path | None = None

# AdamW aux group: embeds, norms, biases, heads, Shaw tables, CLS tokens.
# Everything else ≥2D (QKV/FFN/projections) goes to NorMuon.
ADAM_NAME_HINTS = (
    "embed", "policy_head", "value_head", "cls_token", "cls_pos",
    "pos_embed", "norm", "bn", "shaw_",
)


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


def build_normuon_optimizer(model: nn.Module, muon_lr: float, adam_lr: float, weight_decay: float):
    from normuon import SingleDeviceNorMuonWithAuxAdam

    muon_params, adam_params = [], []
    muon_n = adam_n = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if any(h in name for h in ADAM_NAME_HINTS) or param.ndim < 2:
            adam_params.append(param)
            adam_n += n
        else:
            muon_params.append(param)
            muon_n += n

    opt = SingleDeviceNorMuonWithAuxAdam([
        dict(
            params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay,
            momentum=0.95, beta2=0.95,
        ),
        dict(
            params=adam_params, use_muon=False, lr=adam_lr, betas=(0.9, 0.95),
            weight_decay=weight_decay,
        ),
    ])
    return opt, muon_n, adam_n


def prepare_soft_batch(data, indices, device, hflip_p: float = 0.0, rng: torch.Generator | None = None):
    ba = data["board_array"][indices].clone()
    turn = data["turn"][indices].clone()
    castling = data["castling"][indices].clone()
    ep = data["ep_square"][indices].clone()
    move_idx = data["move_idx"][indices].clone()
    cp = data["cp"][indices]
    mate = data["mate"][indices]
    soft_i = data["soft_indices"][indices].clone()
    soft_p = data["soft_probs"][indices].clone()

    if hflip_p > 0:
        flip_mask = torch.rand(ba.shape[0], generator=rng) < hflip_p
        if flip_mask.any():
            ba[flip_mask] = hflip_board_array(ba[flip_mask])
            move_idx[flip_mask] = hflip_move_idx(move_idx[flip_mask]).to(move_idx.dtype)
            castling[flip_mask] = 0
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
            si = soft_i[flip_mask]
            valid = si >= 0
            si_flat = si.clone()
            if valid.any():
                si_flat[valid] = hflip_move_idx(si[valid]).to(si.dtype)
            soft_i[flip_mask] = si_flat

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
    bs = 128
    correct = 0
    soft_loss_sum = 0.0
    total = 0
    for i in range(0, len(idx), bs):
        batch_idx = idx[i:i + bs]
        bi, hard, wdl, si, sp = prepare_soft_batch(soft_data, batch_idx, device, hflip_p=0.0)
        with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(bi)
            logits = out["policy_logits"]
            soft_loss_sum += soft_policy_loss(logits, si, sp).item() * batch_idx.numel()
            pred = logits.argmax(dim=-1)
            correct += (pred == hard).sum().item()
            total += batch_idx.numel()
    model.train()
    return {"top1": correct / max(total, 1), "soft_loss": soft_loss_sum / max(total, 1), "n": total}


def save_checkpoint(model, optimizer, scaler, step, best_metric, path, args, config):
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "step": step,
        "best_metric": best_metric,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "vocab": "compact",
        "vocab_version": "compact",
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)


def main():
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--soft-cache", default=str(DEFAULT_SOFT))
    ap.add_argument("--deep-soft-cache", type=str, default=None)
    ap.add_argument("--deep-mix-frac", type=float, default=0.40)
    ap.add_argument("--output-dir", default="outputs/exp191_400m_meta_attention")
    # A40-oriented defaults: large micro-batch, no accum waste, NorMuon LRs
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--soft-frac", type=float, default=0.72)
    ap.add_argument("--soft-alpha", type=float, default=0.40)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--hflip-p", type=float, default=0.5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-opt", action="store_true")
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--value-weight", type=float, default=0.08)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--min-depth", type=int, default=15)
    ap.add_argument("--shuffle-buffer", type=int, default=4096)
    ap.add_argument(
        "--grad-checkpoint", action="store_true",
        help="Enable activation checkpointing (slower, less VRAM). Default OFF for A40 throughput.",
    )
    ap.add_argument(
        "--adamw", action="store_true",
        help="Fallback to AdamW instead of NorMuon (debug only).",
    )
    args = ap.parse_args()

    if not args.go:
        print("DRY RUN. Pass --go to train.")
        print(f"Config: {DEFAULT_400M_META_CONFIG}")
        print("A40 tip: bash scripts/run_exp191_a40.sh")
        return
    if args.smoke:
        args.steps = 20
        args.log_interval = 5
        args.save_interval = 20
        args.eval_interval = 20
        args.warmup = 2
        args.batch_size = min(args.batch_size, 8)
        args.accum_steps = 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out / "training.log"
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    assert VOCAB_SIZE == 1968, f"expected compact vocab, got {VOCAB_SIZE}"

    model_config = replace(
        DEFAULT_400M_META_CONFIG,
        gradient_checkpointing=bool(args.grad_checkpoint),
    )

    log("=" * 64)
    log("exp191: 400M+ meta-factored attention (A40 / NorMuon)")
    log(f"  device={DEVICE} vocab={VOCAB_SIZE}")
    log(f"  soft_cache={args.soft_cache}")
    log(f"  deep_soft_cache={args.deep_soft_cache} deep_mix_frac={args.deep_mix_frac}")
    log(f"  soft_frac={args.soft_frac} soft_alpha={args.soft_alpha}")
    log(f"  muon_lr={args.muon_lr} adam_lr={args.adam_lr} grad_ckpt={model_config.gradient_checkpointing}")
    log(f"  removed: rel_bias, pos_embed, full_dim_attention")
    log(f"  enabled: meta_attention + shaw_on_pos")

    soft_path = Path(args.soft_cache)
    if not soft_path.exists():
        if args.smoke:
            log(f"  smoke: soft cache missing ({soft_path}) — synthetic batch only")
            soft_data = None
            train_soft_n = eval_holdout = 0
        else:
            raise FileNotFoundError(f"soft cache not found: {soft_path}")
    else:
        soft_data = torch.load(soft_path, map_location="cpu", weights_only=False)
        n_soft = soft_data["board_array"].shape[0]
        eval_holdout = min(8000, max(2048, n_soft // 40))
        train_soft_n = max(1, n_soft - eval_holdout)
        log(f"  soft train={train_soft_n:,} holdout={eval_holdout:,}")

    deep_data = None
    train_deep_n = 0
    deep_holdout = 0
    if args.deep_soft_cache:
        deep_path = Path(args.deep_soft_cache)
        if not deep_path.exists():
            raise FileNotFoundError(f"deep soft cache not found: {deep_path}")
        deep_data = torch.load(deep_path, map_location="cpu", weights_only=False)
        n_deep = deep_data["board_array"].shape[0]
        deep_holdout = min(4000, max(512, n_deep // 20))
        train_deep_n = max(1, n_deep - deep_holdout)
        log(f"  deep soft train={train_deep_n:,} holdout={deep_holdout:,}")

    if args.resume:
        log(f"  resume from {args.resume}")
        model = build_model(model_config)
        rckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"], strict=False)
        start_top1 = float(rckpt.get("best_metric", 0.0))
        if not (0.0 < start_top1 <= 1.0):
            start_top1 = 0.0
    else:
        rckpt = None
        model = build_model(model_config)
        start_top1 = 0.0

    model = model.to(DEVICE)
    n_params = count_parameters(model)
    log(f"  params={n_params/1e6:.1f}M (>=400 required)")
    if n_params < 400_000_000:
        raise RuntimeError(f"param count {n_params} < 400M — bump config")
    log(f"  config={model_config}")

    if args.adamw:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay)
        log(f"  optimizer=AdamW lr={args.adam_lr}")
    else:
        try:
            optimizer, muon_n, adam_n = build_normuon_optimizer(
                model, args.muon_lr, args.adam_lr, args.weight_decay,
            )
            log(f"  optimizer=NorMuon ({muon_n/1e6:.1f}M) + AdamW aux ({adam_n/1e6:.1f}M)")
        except ImportError:
            log("ERROR: pip install git+https://github.com/zichongli5/NorMuon.git")
            return

    if args.resume and args.resume_opt and rckpt is not None and "optimizer_state_dict" in rckpt:
        try:
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            log("  resumed optimizer state")
        except Exception as e:
            log(f"  optimizer resume skipped: {e}")
    scaler = GradScaler("cuda", enabled=False)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_scale(s: int) -> float:
        if s < args.warmup:
            return (s + 1) / max(args.warmup, 1)
        progress = (s - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine

    def set_lrs(s: int) -> None:
        scale = lr_scale(s)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    hard_iter = None
    if args.soft_frac < 1.0 and not args.smoke:
        hard_iter = iter(stream_hf_batches(
            batch_size=args.batch_size, device=DEVICE, seed=42,
            shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
        ))

    with open(out / "config.json", "w") as f:
        json.dump({
            "model": model_config.to_dict(),
            "training": vars(args),
            "n_params": n_params,
            "vocab": "compact",
            "goal": "break_1700_wall_via_meta_attention_a40_normuon",
            "removed": ["use_rel_bias", "use_pos_embed", "full_dim_attention"],
            "optimizer": "adamw" if args.adamw else "normuon",
        }, f, indent=2)

    eff_bs = args.batch_size * args.accum_steps
    log(f"  batch={args.batch_size} accum={args.accum_steps} eff_bs={eff_bs}")
    log(f"  steps={args.steps:,}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  gpu={torch.cuda.get_device_name(0)} vram={props.total_memory/1e9:.1f}GB")
    log("=" * 64)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    best_top1 = float(start_top1)
    t0 = time.time()
    positions = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    accum_p = accum_v = accum_soft = accum_hard = 0.0
    accum_n = soft_steps = hard_steps = deep_steps = shallow_steps = 0

    if DEVICE.type == "cuda":
        log(f"  vram allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")

    if soft_data is not None and not args.smoke:
        metrics0 = eval_soft_top1(model, soft_data, DEVICE, n=min(eval_holdout, 2048))
        log(f"  zero-shot shallow top1={metrics0['top1']*100:.2f}% soft_loss={metrics0['soft_loss']:.4f}")

    while step < args.steps:
        if SHUTDOWN:
            save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args, model_config)
            log(f"Saved on shutdown at step {step}")
            return

        use_soft = (
            soft_data is not None
            and (args.smoke or torch.rand(1, generator=rng).item() < args.soft_frac)
        )
        for _ in range(args.accum_steps):
            if use_soft and soft_data is not None:
                use_deep = (
                    deep_data is not None
                    and train_deep_n > 0
                    and torch.rand(1, generator=rng).item() < args.deep_mix_frac
                )
                if use_deep:
                    idx = torch.randint(0, train_deep_n, (args.batch_size,), generator=rng)
                    bi, hard, wdl, si, sp = prepare_soft_batch(
                        deep_data, idx, DEVICE, hflip_p=args.hflip_p, rng=rng,
                    )
                    deep_steps += 1
                else:
                    idx = torch.randint(0, train_soft_n, (args.batch_size,), generator=rng)
                    bi, hard, wdl, si, sp = prepare_soft_batch(
                        soft_data, idx, DEVICE, hflip_p=args.hflip_p, rng=rng,
                    )
                    shallow_steps += 1
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
            elif args.smoke:
                # Synthetic board batch when no soft cache
                bi = {
                    "fused_ids": torch.randint(0, 13, (args.batch_size, 64), device=DEVICE),
                    "turn": torch.zeros(args.batch_size, dtype=torch.long, device=DEVICE),
                    "castling": torch.zeros(args.batch_size, dtype=torch.long, device=DEVICE),
                    "ep_file": torch.zeros(args.batch_size, dtype=torch.long, device=DEVICE),
                }
                hard = torch.randint(0, VOCAB_SIZE, (args.batch_size,), device=DEVICE)
                wdl = torch.randint(0, 3, (args.batch_size,), device=DEVICE)
                with autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out_m = model(bi)
                    p_loss = F.cross_entropy(out_m["policy_logits"], hard)
                    v_loss = F.cross_entropy(out_m["value_logits"], wdl)
                    loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
                loss.backward()
                accum_p += p_loss.item()
                accum_v += v_loss.item()
                hard_steps += 1
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
                    p_loss = F.cross_entropy(
                        out_m["policy_logits"], move_t, label_smoothing=args.label_smoothing,
                    )
                    v_loss = F.cross_entropy(out_m["value_logits"], wdl_t)
                    loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
                loss.backward()
                accum_p += p_loss.item()
                accum_v += v_loss.item()
                hard_steps += 1
            accum_n += 1
            positions += args.batch_size

        gn = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        set_lrs(step + 1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_interval == 0:
            elapsed = max(time.time() - t0, 1e-6)
            vram = f" | vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB" if DEVICE.type == "cuda" else ""
            log(
                f"step {step:,}/{args.steps:,} | "
                f"p={accum_p/max(accum_n,1):.4f} v={accum_v/max(accum_n,1):.4f} "
                f"soft={accum_soft/max(soft_steps,1):.4f} hardCE={accum_hard/max(soft_steps,1):.4f} | "
                f"mix soft={soft_steps}(deep={deep_steps},shallow={shallow_steps}) hard={hard_steps} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} gn={float(gn):.2f} | "
                f"{positions/elapsed:.0f} pos/s{vram}"
            )
            accum_p = accum_v = accum_soft = accum_hard = 0.0
            accum_n = soft_steps = hard_steps = deep_steps = shallow_steps = 0

        if soft_data is not None and step % args.eval_interval == 0:
            metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
            log(f"  eval shallow top1={metrics['top1']*100:.2f}% soft_loss={metrics['soft_loss']:.4f}")
            track = metrics["top1"]
            if deep_data is not None:
                dm = eval_soft_top1(model, deep_data, DEVICE, n=deep_holdout)
                log(f"  eval deep top1={dm['top1']*100:.2f}%")
                track = 0.5 * metrics["top1"] + 0.5 * dm["top1"]
            if track > best_top1:
                best_top1 = track
                save_checkpoint(model, optimizer, scaler, step, best_top1, out / "best.pt", args, model_config)
                log(f"  new best track={best_top1*100:.2f}%")

        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args, model_config)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args, model_config)
    log(f"Done. step={step:,} best_track={best_top1*100:.2f}% params={n_params/1e6:.1f}M")


if __name__ == "__main__":
    main()
