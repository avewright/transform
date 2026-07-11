#!/usr/bin/env python3
"""exp189: Max-Elo pure next-move finetune of HF 200M compact-soft.

Continues avewright/chess-transformer-200m-compact-soft on the full
avewright/exp186-sf-multipv-2m soft cache, mixed with deep hard labels
from lichess-sf. No MCTS — policy argmax only.

Recipe (policy strength without search):
  - Resume from compact-soft best (already ~33.8% soft holdout top-1)
  - Soft MultiPV CE + hard best-move CE on 2M SF labels
  - Deep hard stream (depth>=15) so shallow MultiPV doesn't dominate
  - 50% horizontal flip aug on soft batches (free diversity)
  - Gentle LR, long schedule, SWA-friendly step checkpoints

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp189_200m_maxelo_policy.py --go
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import (
    DEFAULT_200M_CONFIG,
    build_model,
    count_parameters,
)
from data_loader import (
    board_array_to_fused,
    compute_wdl,
    ep_square_to_file,
    hflip_board_array,
    hflip_castling,
    hflip_ep_square,
    hflip_move_idx,
    stream_hf_batches,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = ROOT / "outputs" / "hf_checkpoint" / "best_model.pt"
DEFAULT_SOFT = ROOT / "outputs" / "exp186_sf_multipv" / "soft_cache.pt"
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
            # Castling after hflip is inconsistent (king not on e-file) — zero it
            castling[flip_mask] = 0
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
            # Mirror each soft MultiPV candidate; keep probs
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
    bs = 256
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
        "vocab_version": "compact",
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
    ap.add_argument("--deep-soft-cache", type=str, default=None,
                    help="Optional exp190-style deep MultiPV cache to mix in")
    ap.add_argument("--deep-mix-frac", type=float, default=0.40,
                    help="Among soft steps, fraction drawn from deep-soft-cache")
    ap.add_argument("--output-dir", default="outputs/exp189_200m_maxelo_policy")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch-size", type=int, default=448)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--soft-frac", type=float, default=0.72,
                    help="Soft-cache fraction; rest = deep HF hard (Elo ballast)")
    ap.add_argument("--soft-alpha", type=float, default=0.35,
                    help="Within soft batches: MultiPV CE vs best-move hard CE")
    ap.add_argument("--lr", type=float, default=6e-6)
    ap.add_argument("--hflip-p", type=float, default=0.5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-opt", action="store_true")
    ap.add_argument("--min-lr-frac", type=float, default=0.08)
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

    assert VOCAB_SIZE == 1968, f"expected compact vocab, got {VOCAB_SIZE}"

    log("=" * 64)
    log("exp189: 200M max-Elo pure policy (no MCTS)")
    log(f"  device={DEVICE} vocab={VOCAB_SIZE}")
    log(f"  ckpt={args.checkpoint}")
    log(f"  soft_cache={args.soft_cache}")
    log(f"  deep_soft_cache={args.deep_soft_cache} deep_mix_frac={args.deep_mix_frac}")
    log(f"  soft_frac={args.soft_frac} soft_alpha={args.soft_alpha} lr={args.lr}")
    log(f"  hflip_p={args.hflip_p} min_depth={args.min_depth}")

    soft_data = torch.load(args.soft_cache, map_location="cpu", weights_only=False)
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
        if "phase" in deep_data:
            ph = deep_data["phase"]
            log(
                f"  deep phases o={(ph == 0).sum().item():,} "
                f"m={(ph == 1).sum().item():,} e={(ph == 2).sum().item():,}"
            )
        if "label_depth" in deep_data:
            log(f"  deep depth mean={deep_data['label_depth'].float().mean():.1f}")

    init_path = Path(args.resume) if args.resume else Path(args.checkpoint)
    log(f"  loading weights from {init_path}")
    model = build_model(DEFAULT_200M_CONFIG)
    rckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(rckpt["model_state_dict"], strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        log(f"  load warn missing={msg.missing_keys[:5]} unexpected={msg.unexpected_keys[:5]}")
    model = model.to(DEVICE)
    bm = float(rckpt.get("best_metric", 0.0))
    start_top1 = bm if 0.0 < bm <= 1.0 else 0.0
    n_params = count_parameters(model)
    log(f"  params={n_params/1e6:.1f}M start_top1={start_top1*100:.2f}% config={DEFAULT_200M_CONFIG}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_opt and "optimizer_state_dict" in rckpt:
        try:
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            log("  resumed optimizer state")
        except Exception as e:
            log(f"  optimizer resume skipped: {e}")
    scaler = GradScaler("cuda", enabled=False)
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
            "goal": "max_elo_next_move_no_mcts",
        }, f, indent=2)

    eff_bs = args.batch_size * args.accum_steps
    soft_epochs = (args.steps * args.soft_frac * eff_bs) / max(train_soft_n, 1)
    log(f"  batch={args.batch_size} accum={args.accum_steps} eff_bs={eff_bs}")
    log(f"  steps={args.steps:,} (~{soft_epochs:.1f} soft epochs)")
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
    top1_abort = 0.05

    if DEVICE.type == "cuda":
        log(f"  vram allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Zero-shot eval before training
    metrics0 = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
    log(f"  zero-shot shallow holdout top1={metrics0['top1']*100:.2f}% soft_loss={metrics0['soft_loss']:.4f}")
    if metrics0["top1"] > best_top1:
        best_top1 = metrics0["top1"]
    if deep_data is not None:
        d0 = eval_soft_top1(model, deep_data, DEVICE, n=deep_holdout)
        log(f"  zero-shot deep holdout top1={d0['top1']*100:.2f}% soft_loss={d0['soft_loss']:.4f}")
        # Reset best tracker to blended metric so deep gains can win checkpoints
        best_top1 = 0.5 * metrics0["top1"] + 0.5 * d0["top1"]
        log(f"  blend track seed={best_top1*100:.2f}% (0.5 shallow + 0.5 deep)")

    while step < args.steps:
        if SHUTDOWN:
            save_checkpoint(model, optimizer, scaler, step, best_top1, out / "latest.pt", args)
            log(f"Saved on shutdown at step {step}")
            return

        use_soft = torch.rand(1, generator=rng).item() < args.soft_frac
        for _ in range(args.accum_steps):
            if use_soft:
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
                f"mix soft={soft_steps}(deep={deep_steps},shallow={shallow_steps}) hard={hard_steps} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} gn={float(gn):.2f} | "
                f"{positions/elapsed:.0f} pos/s{vram}"
            )
            accum_p = accum_v = accum_soft = accum_hard = 0.0
            accum_n = soft_steps = hard_steps = deep_steps = shallow_steps = 0

        if step % args.eval_interval == 0:
            metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
            log(f"  eval shallow holdout top1={metrics['top1']*100:.2f}% soft_loss={metrics['soft_loss']:.4f} n={metrics['n']}")
            if deep_data is not None:
                dm = eval_soft_top1(model, deep_data, DEVICE, n=deep_holdout)
                log(f"  eval deep holdout top1={dm['top1']*100:.2f}% soft_loss={dm['soft_loss']:.4f} n={dm['n']}")
                # Track best on deep when mixing — that's the new teacher
                track = 0.5 * metrics["top1"] + 0.5 * dm["top1"]
            else:
                track = metrics["top1"]
            if metrics["top1"] < top1_abort:
                log(f"ABORT top1={metrics['top1']*100:.2f}% < {top1_abort*100:.0f}% — model collapsed")
                save_checkpoint(model, optimizer, scaler, step, best_top1, out / "aborted.pt", args)
                return
            if track > best_top1:
                best_top1 = track
                save_checkpoint(model, optimizer, scaler, step, best_top1, out / "best.pt", args)
                log(f"  new best track={best_top1*100:.2f}%")

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

    # SWA over late step checkpoints if we have enough
    step_ckpts = sorted(out.glob("step_*.pt"))
    if len(step_ckpts) >= 3:
        late = step_ckpts[-min(5, len(step_ckpts)):]
        log(f"Building SWA from {[p.name for p in late]}")
        try:
            from swa_checkpoint import average_checkpoints
            avg_state, config, steps = average_checkpoints([str(p) for p in late], decay=0.9)
            swa_path = out / "swa.pt"
            torch.save({
                "model_state_dict": avg_state,
                "config": config,
                "vocab": "compact",
                "vocab_version": "compact",
                "swa_steps": steps,
                "best_metric": best_top1,
            }, swa_path)
            log(f"  wrote {swa_path}")
        except Exception as e:
            log(f"  SWA skipped: {e}")


if __name__ == "__main__":
    main()
