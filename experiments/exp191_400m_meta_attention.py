#!/usr/bin/env python3
"""exp191: ≥400M meta-factored attention from scratch (A40 / NorMuon).

Architecture (DEFAULT_400M_META_CONFIG):
  - Separate content / position streams through attention
  - 4-term scores: cc + ss + cs + sc (DeBERTa-style)
  - Shaw relative vectors on ss only (Δfile, Δrank) — no handcrafted rays
  - Removed: ChessRelativeBias, sequence pos_embed, full_dim_attention

A40 RunPod defaults (wise rental use):
  - NorMuon on ≥2D trunk weights + tiny AdamW aux (embeds/heads/norms)
  - Optional Polar Express orthogonalization + cautious WD (modded-nanogpt)
  - Optional torch.compile (--compile) for ~20–40% step throughput
  - Large micro-batch, accum=1, bf16, TF32, cudnn.benchmark
  - Grad checkpoint OFF by default (throughput); --grad-checkpoint if OOM
  - Soft MultiPV + hard ballast from local RAM cache (no mid-step HF stream)
  - Async CPU batch prep (double-buffer) + pin_memory H2D
  - Eval every 1500 steps on ~2k holdout (not 8k / every 500)

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp191_400m_meta_attention.py --go --smoke
  bash scripts/run_exp191_a40.sh
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
from torch.amp import autocast

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
    PrefetchIterator,
    board_array_to_fused,
    compute_wdl,
    ep_square_to_file,
    hard_ballast_cache_path,
    hflip_board_array,
    hflip_castling,
    hflip_ep_square,
    hflip_move_idx,
    load_or_build_hard_ballast,
    stream_hf_batches,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SOFT = ROOT / "outputs" / "exp186_sf_multipv" / "soft_cache.pt"
DEFAULT_HARD = ROOT / "outputs" / "data_cache" / "hard_ballast_d15_n2000000_s42.pt"
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


def _split_muon_adam_params(model: nn.Module):
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
    return muon_params, adam_params, muon_n, adam_n


def _strip_orig_mod_prefix(state: dict) -> dict:
    """torch.compile checkpoints may prefix keys with _orig_mod."""
    if not any(isinstance(k, str) and k.startswith("_orig_mod.") for k in state):
        return state
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}


def build_normuon_optimizer(
    model: nn.Module,
    muon_lr: float,
    adam_lr: float,
    weight_decay: float,
    *,
    polar: bool = False,
    cautious_wd: bool = True,
    compile_polar: bool = True,
):
    muon_params, adam_params, muon_n, adam_n = _split_muon_adam_params(model)
    groups = [
        dict(
            params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay,
            momentum=0.95, beta2=0.95,
        ),
        dict(
            params=adam_params, use_muon=False, lr=adam_lr, betas=(0.9, 0.95),
            weight_decay=weight_decay,
        ),
    ]
    if polar:
        from polar_normuon import SingleDeviceNorMuonPolarWithAuxAdam

        opt = SingleDeviceNorMuonPolarWithAuxAdam(
            groups, cautious_wd=cautious_wd, compile_polar=compile_polar,
        )
    else:
        from normuon import SingleDeviceNorMuonWithAuxAdam

        opt = SingleDeviceNorMuonWithAuxAdam(groups)
    return opt, muon_n, adam_n


def _to_device(t: torch.Tensor, device, non_blocking: bool = True) -> torch.Tensor:
    # Small chess batches: pin_memory setup cost > benefit; non_blocking still helps.
    if device.type == "cuda" and t.device.type == "cpu":
        return t.to(device, non_blocking=non_blocking)
    return t.to(device)


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
            castling[flip_mask] = hflip_castling(castling[flip_mask]).to(castling.dtype)
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
            si = soft_i[flip_mask]
            valid = si >= 0
            si_flat = si.clone()
            if valid.any():
                si_flat[valid] = hflip_move_idx(si[valid]).to(si.dtype)
            soft_i[flip_mask] = si_flat

    board_input = {
        "fused_ids": _to_device(board_array_to_fused(ba), device),
        "turn": _to_device(turn.long(), device),
        "castling": _to_device(castling.long(), device),
        "ep_file": _to_device(ep_square_to_file(ep).long(), device),
    }
    wdl = _to_device(compute_wdl(cp, mate), device)
    return (
        board_input,
        _to_device(move_idx.long(), device),
        wdl,
        _to_device(soft_i, device),
        _to_device(soft_p, device),
    )


def prepare_hard_batch(data, indices, device, hflip_p: float = 0.0, rng: torch.Generator | None = None):
    """Hard CE ballast from a local RAM cache (same encoding as soft)."""
    ba = data["board_array"][indices].clone()
    turn = data["turn"][indices].clone()
    castling = data["castling"][indices].clone()
    ep = data["ep_square"][indices].clone()
    move_idx = data["move_idx"][indices].clone()
    cp = data["cp"][indices]
    mate = data["mate"][indices]

    if hflip_p > 0:
        flip_mask = torch.rand(ba.shape[0], generator=rng) < hflip_p
        if flip_mask.any():
            ba[flip_mask] = hflip_board_array(ba[flip_mask])
            move_idx[flip_mask] = hflip_move_idx(move_idx[flip_mask]).to(move_idx.dtype)
            castling[flip_mask] = hflip_castling(castling[flip_mask]).to(castling.dtype)
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)

    board_input = {
        "fused_ids": _to_device(board_array_to_fused(ba), device),
        "turn": _to_device(turn.long(), device),
        "castling": _to_device(castling.long(), device),
        "ep_file": _to_device(ep_square_to_file(ep).long(), device),
    }
    wdl = _to_device(compute_wdl(cp, mate, turn.long()), device)
    return board_input, _to_device(move_idx.long(), device), wdl


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


def save_checkpoint(model, optimizer, step, best_metric, path, args, config, select_metric="soft_loss"):
    from polar_normuon import unwrap_compiled

    tmp = Path(str(path) + ".tmp")
    raw = unwrap_compiled(model)
    torch.save({
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "select_metric": select_metric,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "vocab": "compact",
        "vocab_version": "compact",
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)


class EpochSampler:
    """Sample indices without replacement; reshuffle when an epoch is exhausted."""

    def __init__(self, n: int, rng: torch.Generator):
        self.n = int(n)
        self.rng = rng
        self.perm = torch.empty(0, dtype=torch.long)
        self.cursor = 0

    def take(self, k: int) -> torch.Tensor:
        out = []
        need = k
        while need > 0:
            if self.cursor >= self.perm.numel():
                self.perm = torch.randperm(self.n, generator=self.rng)
                self.cursor = 0
            take_n = min(need, self.n - self.cursor)
            out.append(self.perm[self.cursor:self.cursor + take_n])
            self.cursor += take_n
            need -= take_n
        return torch.cat(out, dim=0)


def main():
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--soft-cache", default=str(DEFAULT_SOFT))
    ap.add_argument("--deep-soft-cache", type=str, default=None)
    ap.add_argument(
        "--deep-stream-lichess", action="store_true",
        help="Stream Lichess cloud-eval soft MultiPV online (no deep RAM cache).",
    )
    ap.add_argument(
        "--deep-stream-shard-start", type=int, default=4,
        help="Skip first N local parquet shards when streaming (default 4 = virgin vs FT3/12M).",
    )
    ap.add_argument(
        "--deep-stream-buffer-fens", type=int, default=8192,
        help="Online FEN grouping buffer size for stream collator.",
    )
    ap.add_argument("--deep-mix-frac", type=float, default=0.20,
                    help="Frac of soft steps drawn from deep cache (keep low if deep << shallow).")
    ap.add_argument("--output-dir", default="outputs/exp191_400m_meta_attention")
    # A40-oriented defaults: large micro-batch, no accum waste, NorMuon LRs
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--soft-frac", type=float, default=0.72)
    ap.add_argument("--soft-alpha", type=float, default=0.40)
    ap.add_argument("--muon-lr", type=float, default=0.015)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--hflip-p", type=float, default=0.5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-opt", action="store_true")
    ap.add_argument(
        "--init-checkpoint", type=str, default=None,
        help="Load weights only; reset step=0 with fresh optimizer (soft finetune).",
    )
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--value-weight", type=float, default=0.08)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=1500)
    ap.add_argument("--eval-n", type=int, default=2048,
                    help="Max holdout positions for shallow/deep eval (smaller = faster).")
    ap.add_argument("--min-depth", type=int, default=15)
    ap.add_argument("--shuffle-buffer", type=int, default=4096)
    ap.add_argument("--hard-cache", type=str, default=str(DEFAULT_HARD),
                    help="Local hard ballast .pt (built from HF train_main if missing).")
    ap.add_argument("--hard-n", type=int, default=2_000_000,
                    help="Hard ballast size when building cache.")
    ap.add_argument("--prefetch-depth", type=int, default=3,
                    help="Async CPU batch prep queue depth (soft+hard RAM path).")
    ap.add_argument("--no-hard-cache", action="store_true",
                    help="Force legacy HF mid-step streaming (slow; debug only).")
    ap.add_argument(
        "--select-metric", choices=("soft_loss", "top1"), default="soft_loss",
        help="Checkpoint selection: soft_loss (lower better) matches soft training objective.",
    )
    ap.add_argument(
        "--grad-checkpoint", action="store_true",
        help="Enable activation checkpointing (slower, less VRAM). Default OFF for A40 throughput.",
    )
    ap.add_argument(
        "--adamw", action="store_true",
        help="Fallback to AdamW instead of NorMuon (debug only).",
    )
    ap.add_argument(
        "--compile", action="store_true",
        help="torch.compile the model (inductor). Expect compile latency on first steps.",
    )
    ap.add_argument(
        "--polar", action="store_true",
        help="Polar Express orthogonalization instead of Newton–Schulz (modded-nanogpt).",
    )
    ap.add_argument(
        "--cautious-wd", action=argparse.BooleanOptionalAction, default=True,
        help="Cautious weight decay when using --polar (default: on).",
    )
    ap.add_argument(
        "--compile-polar", action=argparse.BooleanOptionalAction, default=True,
        help="torch.compile the Polar Express kernel (default: on with --polar).",
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
    log(f"  deep_soft_cache={args.deep_soft_cache} deep_stream={args.deep_stream_lichess} "
        f"deep_mix_frac={args.deep_mix_frac}")
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
        eval_holdout = min(int(args.eval_n), max(512, n_soft // 200))
        train_soft_n = max(1, n_soft - eval_holdout)
        log(f"  soft train={train_soft_n:,} holdout={eval_holdout:,}")

    deep_data = None
    train_deep_n = 0
    deep_holdout = 0
    deep_stream = None
    deep_stream_prefetch = None
    if args.deep_stream_lichess:
        from lichess_soft_stream import iter_lichess_soft_batches

        def _deep_stream_factory(seed=42):
            return iter_lichess_soft_batches(
                batch_size=args.batch_size,
                min_depth=max(22, args.min_depth),
                min_knodes=5000,
                buffer_fens=args.deep_stream_buffer_fens,
                shard_start=args.deep_stream_shard_start,
                seed=seed,
                infinite=True,
            )

        deep_stream_prefetch = PrefetchIterator(
            _deep_stream_factory, depth=max(2, args.prefetch_depth), name="lichess-deep-stream",
        )
        deep_stream = deep_stream_prefetch
        train_deep_n = 10**12
        log(
            f"  deep soft=STREAM lichess shards[{args.deep_stream_shard_start}:] "
            f"buffer_fens={args.deep_stream_buffer_fens}"
        )
    elif args.deep_soft_cache:
        deep_path = Path(args.deep_soft_cache)
        if not deep_path.exists():
            raise FileNotFoundError(f"deep soft cache not found: {deep_path}")
        deep_data = torch.load(deep_path, map_location="cpu", weights_only=False)
        n_deep = deep_data["board_array"].shape[0]
        deep_holdout = min(int(args.eval_n), max(256, n_deep // 50))
        train_deep_n = max(1, n_deep - deep_holdout)
        log(f"  deep soft train={train_deep_n:,} holdout={deep_holdout:,}")

    if args.init_checkpoint and args.resume:
        raise SystemExit("Pass only one of --init-checkpoint / --resume")

    if args.init_checkpoint:
        log(f"  init from {args.init_checkpoint} (fresh optimizer, step=0)")
        model = build_model(model_config)
        rckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip_orig_mod_prefix(rckpt["model_state_dict"]), strict=False)
        start_step = 0
        start_best = float("inf") if args.select_metric == "soft_loss" else 0.0
        prior = rckpt.get("best_metric")
        log(f"  loaded weights (prior_metric={prior}, select={rckpt.get('select_metric')})")
    elif args.resume:
        log(f"  resume from {args.resume}")
        model = build_model(model_config)
        rckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(_strip_orig_mod_prefix(rckpt["model_state_dict"]), strict=False)
        start_step = int(rckpt.get("step", 0) or 0)
        raw_best = float(rckpt.get("best_metric", float("inf")))
        # Legacy checkpoints stored top1 in (0,1]; soft_loss selection uses lower-is-better.
        prev_select = rckpt.get("select_metric", "top1")
        if args.select_metric == "soft_loss":
            if prev_select == "soft_loss" and raw_best > 1.0:
                start_best = raw_best
            else:
                start_best = float("inf")  # re-compete under soft_loss
        else:
            start_best = raw_best if 0.0 < raw_best <= 1.0 else 0.0
        log(f"  resume step={start_step:,} prior_metric={raw_best} ({prev_select}) → track={start_best}")
    else:
        rckpt = None
        model = build_model(model_config)
        start_step = 0
        start_best = float("inf") if args.select_metric == "soft_loss" else 0.0

    model = model.to(DEVICE)
    n_params = count_parameters(model)
    log(f"  params={n_params/1e6:.1f}M (>=400 required)")
    if n_params < 400_000_000:
        raise RuntimeError(f"param count {n_params} < 400M — bump config")
    log(f"  config={model_config}")

    if args.compile and DEVICE.type == "cuda":
        log("  torch.compile(model) … (first steps will be slow)")
        model = torch.compile(model)
        log("  torch.compile: ok")

    if args.adamw:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay)
        log(f"  optimizer=AdamW lr={args.adam_lr}")
    else:
        try:
            optimizer, muon_n, adam_n = build_normuon_optimizer(
                model, args.muon_lr, args.adam_lr, args.weight_decay,
                polar=args.polar,
                cautious_wd=args.cautious_wd,
                compile_polar=args.compile_polar,
            )
            if args.polar:
                log(
                    f"  optimizer=NorMuon+PolarExpress ({muon_n/1e6:.1f}M) + AdamW aux "
                    f"({adam_n/1e6:.1f}M) cautious_wd={args.cautious_wd}"
                )
                if args.compile_polar and DEVICE.type == "cuda":
                    from polar_normuon import get_polar_express
                    log("  warming Polar Express compile…")
                    fn = get_polar_express(compile_polar=True)
                    _ = fn(torch.randn(64, 32, device=DEVICE))
                    torch.cuda.synchronize()
                    log("  Polar Express compile: ok")
            else:
                log(f"  optimizer=NorMuon ({muon_n/1e6:.1f}M) + AdamW aux ({adam_n/1e6:.1f}M)")
        except ImportError as e:
            log(f"ERROR: optimizer import failed: {e}")
            log("  pip install git+https://github.com/zichongli5/NorMuon.git")
            return

    # Capture unscaled peak LRs BEFORE loading optimizer state (resume-opt stores scaled LRs).
    base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]

    if args.resume and args.resume_opt and rckpt is not None and "optimizer_state_dict" in rckpt:
        try:
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            log("  resumed optimizer state (base_lrs kept from fresh schedule peaks)")
        except Exception as e:
            log(f"  optimizer resume skipped: {e}")

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

    hard_data = None
    hard_iter = None
    hard_prefetch = None
    train_hard_n = 0
    if args.soft_frac < 1.0 and not args.smoke:
        if not args.no_hard_cache:
            try:
                hard_path = Path(args.hard_cache) if args.hard_cache else hard_ballast_cache_path(
                    args.hard_n, args.min_depth, 42,
                )
                log(f"  hard ballast: loading/building {hard_path} (n={args.hard_n:,})")
                hard_data, hard_path = load_or_build_hard_ballast(
                    n_total=args.hard_n,
                    min_depth=args.min_depth,
                    seed=42,
                    path=hard_path,
                )
                train_hard_n = int(hard_data["board_array"].shape[0])
                log(f"  hard ballast RAM={train_hard_n:,} path={hard_path}")
            except Exception as e:
                log(f"  hard ballast failed ({e}); falling back to prefetched HF stream")
                hard_data = None
        if hard_data is None:
            seed0 = 42 + start_step

            def _hard_factory(seed=seed0):
                return stream_hf_batches(
                    batch_size=args.batch_size, device="cpu", seed=seed,
                    shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
                )

            hard_prefetch = PrefetchIterator(_hard_factory, depth=max(2, args.prefetch_depth), name="hard-hf")
            hard_iter = hard_prefetch
            log(f"  hard path=prefetched HF stream depth={args.prefetch_depth}")

    with open(out / "config.json", "w") as f:
        json.dump({
            "model": model_config.to_dict(),
            "training": vars(args),
            "n_params": n_params,
            "vocab": "compact",
            "goal": "break_1700_wall_via_meta_attention_a40_normuon",
            "removed": ["use_rel_bias", "use_pos_embed", "full_dim_attention"],
            "optimizer": "adamw" if args.adamw else "normuon",
            "select_metric": args.select_metric,
            "start_step": start_step,
            "hard_ballast_n": train_hard_n,
            "eval_holdout": eval_holdout if soft_data is not None else 0,
        }, f, indent=2)

    eff_bs = args.batch_size * args.accum_steps
    log(f"  batch={args.batch_size} accum={args.accum_steps} eff_bs={eff_bs}")
    log(f"  steps={args.steps:,} start_step={start_step:,} select={args.select_metric}")
    log(f"  warmup={args.warmup} muon_lr={args.muon_lr} grad_clip={args.grad_clip} deep_mix={args.deep_mix_frac}")
    log(f"  eval_interval={args.eval_interval} eval_n={args.eval_n} prefetch_depth={args.prefetch_depth}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  gpu={torch.cuda.get_device_name(0)} vram={props.total_memory/1e9:.1f}GB")
    log("=" * 64)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    best_metric = float(start_best)
    t0 = time.time()
    window_t0 = t0
    window_positions = 0
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42 + start_step)
    accum_p = accum_v = accum_soft = accum_hard = 0.0
    accum_n = soft_steps = hard_steps = deep_steps = shallow_steps = 0

    deep_sampler = EpochSampler(train_deep_n, rng) if (train_deep_n > 0 and deep_data is not None) else None
    shallow_sampler = EpochSampler(train_soft_n, rng) if train_soft_n > 0 else None
    hard_sampler = EpochSampler(train_hard_n, rng) if train_hard_n > 0 else None
    use_deep_stream = deep_stream is not None

    prep_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(2, args.prefetch_depth)),
        thread_name_prefix="batchprep",
    )

    def _hflip_rng() -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item()))
        return g

    def _plan_microbatch():
        """Decide next microbatch on the main thread (keeps samplers/RNG ordered)."""
        use_soft = (
            soft_data is not None
            and (args.smoke or torch.rand(1, generator=rng).item() < args.soft_frac)
        )
        if use_soft and soft_data is not None:
            use_deep = (
                (deep_sampler is not None or use_deep_stream)
                and torch.rand(1, generator=rng).item() < args.deep_mix_frac
            )
            if use_deep and use_deep_stream:
                return {"kind": "soft_stream", "tag": "deep"}
            if use_deep and deep_sampler is not None:
                return {
                    "kind": "soft",
                    "tag": "deep",
                    "data": deep_data,
                    "idx": deep_sampler.take(args.batch_size),
                    "rng": _hflip_rng(),
                }
            return {
                "kind": "soft",
                "tag": "shallow",
                "data": soft_data,
                "idx": shallow_sampler.take(args.batch_size),
                "rng": _hflip_rng(),
            }
        if args.smoke:
            return {"kind": "smoke"}
        if hard_sampler is not None:
            return {
                "kind": "hard_ram",
                "data": hard_data,
                "idx": hard_sampler.take(args.batch_size),
                "rng": _hflip_rng(),
            }
        return {"kind": "hard_stream"}

    def _prep_microbatch(plan: dict):
        """CPU-side prep only — H2D happens on the main thread after .result()."""
        kind = plan["kind"]
        if kind == "soft":
            packed = prepare_soft_batch(
                plan["data"], plan["idx"], torch.device("cpu"),
                hflip_p=args.hflip_p, rng=plan["rng"],
            )
            return kind, plan["tag"], packed
        if kind == "soft_stream":
            from lichess_soft_stream import batch_dict_to_train_tensors
            try:
                raw = next(deep_stream)
            except StopIteration:
                # PrefetchIterator should restart; pull once more
                raw = next(deep_stream)
            # Keep on CPU; _to_gpu moves. Reuse prepare path shape via stream helper on CPU.
            packed = batch_dict_to_train_tensors(raw, torch.device("cpu"))
            return "soft", plan["tag"], packed
        if kind == "hard_ram":
            packed = prepare_hard_batch(
                plan["data"], plan["idx"], torch.device("cpu"),
                hflip_p=args.hflip_p, rng=plan["rng"],
            )
            return kind, None, packed
        if kind == "smoke":
            bi = {
                "fused_ids": torch.randint(0, 13, (args.batch_size, 64)),
                "turn": torch.zeros(args.batch_size, dtype=torch.long),
                "castling": torch.zeros(args.batch_size, dtype=torch.long),
                "ep_file": torch.zeros(args.batch_size, dtype=torch.long),
            }
            hard = torch.randint(0, VOCAB_SIZE, (args.batch_size,))
            wdl = torch.randint(0, 3, (args.batch_size,))
            return kind, None, (bi, hard, wdl)
        # hard_stream: pull CPU batch from prefetch
        try:
            bi_cpu, move_t, wdl_t = next(hard_iter)
        except StopIteration:
            restart = iter(stream_hf_batches(
                batch_size=args.batch_size, device="cpu", seed=43 + step,
                shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
            ))
            bi_cpu, move_t, wdl_t = next(restart)
        return kind, None, (bi_cpu, move_t.long(), wdl_t.float())

    def _to_gpu(kind: str, packed):
        if kind == "soft":
            bi, hard, wdl, si, sp = packed
            bi = {k: _to_device(v, DEVICE) for k, v in bi.items()}
            return bi, _to_device(hard, DEVICE), _to_device(wdl, DEVICE), _to_device(si, DEVICE), _to_device(sp, DEVICE)
        bi, move_t, wdl_t = packed
        bi = {k: _to_device(v, DEVICE) for k, v in bi.items()}
        return bi, _to_device(move_t, DEVICE), _to_device(wdl_t, DEVICE)

    if DEVICE.type == "cuda":
        log(f"  vram allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")

    if soft_data is not None and not args.smoke:
        metrics0 = eval_soft_top1(model, soft_data, DEVICE, n=min(eval_holdout, args.eval_n))
        log(f"  zero-shot shallow top1={metrics0['top1']*100:.2f}% soft_loss={metrics0['soft_loss']:.4f}")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Double-buffer: plan+prep next microbatch while GPU runs current one.
    pending_plan = _plan_microbatch()
    pending_fut = prep_pool.submit(_prep_microbatch, pending_plan)

    try:
        while step < args.steps:
            if SHUTDOWN:
                save_checkpoint(
                    model, optimizer, step, best_metric, out / "latest.pt", args, model_config,
                    select_metric=args.select_metric,
                )
                log(f"Saved on shutdown at step {step}")
                return

            for _ in range(args.accum_steps):
                kind, tag, packed = pending_fut.result()
                packed = _to_gpu(kind, packed)
                # Kick off next CPU prep immediately (overlaps with forward/backward).
                pending_plan = _plan_microbatch()
                pending_fut = prep_pool.submit(_prep_microbatch, pending_plan)

                if kind == "soft":
                    bi, hard, wdl, si, sp = packed
                    if tag == "deep":
                        deep_steps += 1
                    else:
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
                    bi, move_t, wdl_t = packed
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
                window_positions += args.batch_size

            gn = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            set_lrs(step + 1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0:
                elapsed = max(time.time() - window_t0, 1e-6)
                vram = (
                    f" | vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
                    if DEVICE.type == "cuda" else ""
                )
                log(
                    f"step {step:,}/{args.steps:,} | "
                    f"p={accum_p/max(accum_n,1):.4f} v={accum_v/max(accum_n,1):.4f} "
                    f"soft={accum_soft/max(soft_steps,1):.4f} hardCE={accum_hard/max(soft_steps,1):.4f} | "
                    f"mix soft={soft_steps}(deep={deep_steps},shallow={shallow_steps}) hard={hard_steps} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} gn={float(gn):.2f} | "
                    f"{window_positions/elapsed:.0f} pos/s{vram}"
                )
                accum_p = accum_v = accum_soft = accum_hard = 0.0
                accum_n = soft_steps = hard_steps = deep_steps = shallow_steps = 0
                window_t0 = time.time()
                window_positions = 0

            if soft_data is not None and step % args.eval_interval == 0:
                try:
                    metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
                    log(
                        f"  eval shallow top1={metrics['top1']*100:.2f}% "
                        f"soft_loss={metrics['soft_loss']:.4f}"
                    )
                    soft_track = metrics["soft_loss"]
                    top1_track = metrics["top1"]
                    if deep_data is not None:
                        dm = eval_soft_top1(model, deep_data, DEVICE, n=deep_holdout)
                        log(
                            f"  eval deep top1={dm['top1']*100:.2f}% "
                            f"soft_loss={dm['soft_loss']:.4f}"
                        )
                        soft_track = 0.5 * metrics["soft_loss"] + 0.5 * dm["soft_loss"]
                        top1_track = 0.5 * metrics["top1"] + 0.5 * dm["top1"]
                    if args.select_metric == "soft_loss":
                        improved = soft_track < best_metric
                        candidate = soft_track
                        log(
                            f"  track soft_loss={soft_track:.4f} "
                            f"(best={best_metric:.4f}) top1={top1_track*100:.2f}%"
                        )
                    else:
                        improved = top1_track > best_metric
                        candidate = top1_track
                        log(
                            f"  track top1={top1_track*100:.2f}% "
                            f"(best={best_metric*100:.2f}%) soft_loss={soft_track:.4f}"
                        )
                    if improved:
                        best_metric = candidate
                        save_checkpoint(
                            model, optimizer, step, best_metric, out / "best.pt",
                            args, model_config, select_metric=args.select_metric,
                        )
                        if args.select_metric == "soft_loss":
                            log(f"  new best soft_loss={best_metric:.4f}")
                        else:
                            log(f"  new best track={best_metric*100:.2f}%")
                except Exception as e:
                    log(f"  eval skipped due to error: {e}")

            if step % args.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, best_metric, out / "latest.pt", args, model_config,
                    select_metric=args.select_metric,
                )
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        prep_pool.shutdown(wait=False, cancel_futures=True)
        if hard_prefetch is not None:
            hard_prefetch.close()
        if deep_stream_prefetch is not None:
            deep_stream_prefetch.close()

    save_checkpoint(
        model, optimizer, step, best_metric, out / "latest.pt", args, model_config,
        select_metric=args.select_metric,
    )
    if args.select_metric == "soft_loss":
        log(f"Done. step={step:,} best_soft_loss={best_metric:.4f} params={n_params/1e6:.1f}M")
    else:
        log(f"Done. step={step:,} best_track={best_metric*100:.2f}% params={n_params/1e6:.1f}M")


if __name__ == "__main__":
    main()
