"""exp182: Pretrain deep-narrow ChessTransformer with Muon optimizer.

Default: 96L/512d (~309M) — fits 8GB laptop GPU with gradient checkpointing.
Cloud:   63L/960d (~705M) via --cloud (needs A100-class GPU).
A100:    --a100 preset tunes batch/accum/shards for 80GB (705M default).

Encoder: StrengthenedBoardEncoder (STM flip + 384d embed + 2-layer conv stem).
Optimizer: Muon on all ≥2D trunk weights; tiny AdamW aux group for embeddings
            and heads only (~5% of params — required by Muon, not full-Adam training).

Safety: dry-run by default. Pass --go to train. Never auto-starts.

Usage:
  python experiments/exp182_pretrain_700m.py
  python experiments/exp182_pretrain_700m.py --go --smoke
  python experiments/exp182_pretrain_700m.py --go --shard-dir PATH
  python experiments/exp182_pretrain_700m.py --go --a100
  python experiments/exp182_pretrain_700m.py --go --a100 --a100-309m
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

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board_flip import build_flip_move_table, flip_move_targets
from chess_transformer_factory import (
    DEFAULT_8GB_CONFIG,
    DEFAULT_700M_CONFIG,
    DEFAULT_A100_309M_CONFIG,
    DEFAULT_A100_700M_CONFIG,
    ChessTransformerConfig,
    build_model,
    count_parameters,
)
from data_loader import ShardedChessLoader, stream_hf_batches

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH: Path | None = None
SHUTDOWN = False
FLIP_TABLE = build_flip_move_table()

SIGMA_HL_GAUSS = 0.04
# AdamW aux group: embeddings, heads, norms only (~5% of params).
ADAM_NAME_HINTS = (
    "embed", "policy_head", "value_head", "cls_token", "pos_embed",
    "norm", "bn", "rel_bias",
)


def build_muon_optimizer(
    model: nn.Module, muon_lr: float, adam_lr: float, weight_decay: float,
):
    """Muon on ≥2D trunk weights; minimal AdamW aux for embeddings/heads."""
    from muon import SingleDeviceMuonWithAuxAdam

    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
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

    opt = SingleDeviceMuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
        dict(
            params=adam_params, use_muon=False, lr=adam_lr,
            betas=(0.9, 0.95), weight_decay=weight_decay,
        ),
    ])
    return opt, muon_n, adam_n


def estimate_vram_gb(n_params: int, muon_frac: float = 0.93) -> float:
    """Muon hybrid: bf16 weights/grads + Muon momentum + small AdamW aux."""
    weights = n_params * 2 / 1e9
    grads = n_params * 2 / 1e9
    n_muon = int(n_params * muon_frac)
    n_adam = n_params - n_muon
    muon_state = n_muon * 4 / 1e9      # momentum buffer
    adam_state = n_adam * 8 / 1e9      # Adam m+v on aux params only
    return weights + grads + muon_state + adam_state + 0.8


def hl_gauss_target(win_pct: torch.Tensor, n_bins: int) -> torch.Tensor:
    bin_centers = torch.linspace(
        0.5 / n_bins, 1 - 0.5 / n_bins, n_bins, device=win_pct.device,
    )
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)
    return F.softmax(-0.5 * (diff / SIGMA_HL_GAUSS) ** 2, dim=-1)


def value_loss(logits: torch.Tensor, wdl: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes == 3:
        return F.cross_entropy(logits, wdl.argmax(dim=-1))
    win_pct = wdl[:, 0] + 0.5 * wdl[:, 1]
    targets = hl_gauss_target(win_pct, n_classes)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def save_checkpoint(model, optimizer, scaler, step, best_metric, path: Path, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "best_metric": best_metric,
    }, tmp)
    os.replace(str(tmp), str(path))


def train_step(model, batch_input, move_targets, wdl_targets, args, scaler, n_value_classes):
    move_targets = flip_move_targets(move_targets, batch_input["turn"], FLIP_TABLE)
    amp_dtype = torch.bfloat16 if getattr(args, "use_bf16", False) else torch.float16
    with autocast("cuda", dtype=amp_dtype, enabled=DEVICE.type == "cuda"):
        out = model(batch_input)
        p_loss = F.cross_entropy(
            out["policy_logits"], move_targets,
            label_smoothing=args.label_smoothing,
        )
        v_loss = value_loss(out["value_logits"], wdl_targets, n_value_classes)
        loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
    scaler.scale(loss).backward()
    return p_loss.item(), v_loss.item()


def iter_training_batches(args, device):
    shard_dir = Path(args.shard_dir) if args.shard_dir else DEFAULT_SHARD_DIR
    if shard_dir.exists() and any(shard_dir.glob("shard_*.pt")):
        log(f"Data: ShardedChessLoader ({shard_dir})")
        loader = ShardedChessLoader(
            shard_dir, batch_size=args.batch_size,
            encoder_type="fused", device=device, seed=42,
        )
        while True:
            yield from loader
            loader.set_epoch(loader.epoch + 1)
    else:
        log(f"Data: HF streaming (batch={args.batch_size}, buffer={args.shuffle_buffer})")
        while True:
            yield from stream_hf_batches(
                batch_size=args.batch_size, device=device, seed=42,
                shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
            )


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def apply_a100_preset(args) -> ChessTransformerConfig:
    """Tune training defaults for A100 80GB."""
    args.force = True
    args.shard_dir = args.shard_dir or str(DEFAULT_SHARD_DIR)
    args.log_interval = max(args.log_interval, 25)
    args.save_interval = max(args.save_interval, 1000)
    args.shuffle_buffer = max(args.shuffle_buffer, 4096)
    args.use_bf16 = True

    if args.a100_309m:
        args.cloud = False
        args.batch_size = 128 if args.batch_size == 8 else args.batch_size
        args.accum_steps = 1 if args.accum_steps == 8 else args.accum_steps
        if args.output_dir is None:
            args.output_dir = "outputs/exp182_pretrain_a100_309m"
        return DEFAULT_A100_309M_CONFIG

    args.cloud = True
    args.batch_size = 64 if args.batch_size == 8 else args.batch_size
    args.accum_steps = 2 if args.accum_steps == 8 else args.accum_steps
    if args.output_dir is None:
        args.output_dir = "outputs/exp182_pretrain_a100"
    return DEFAULT_A100_700M_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true",
                        help="Required: actually run training")
    parser.add_argument("--cloud", action="store_true",
                        help="Use 705M config (needs A100+, not 8GB laptop)")
    parser.add_argument("--a100", action="store_true",
                        help="A100 80GB preset: 705M, bs=64, accum=2, local shards")
    parser.add_argument("--a100-309m", action="store_true",
                        help="With --a100: 309M fast path (no grad ckpt, bs=128)")
    parser.add_argument("--force", action="store_true", help="Skip VRAM preflight")
    parser.add_argument("--smoke", action="store_true", help="100-step smoke test")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--shard-dir", type=str, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    args.use_bf16 = False
    if args.a100_309m and not args.a100:
        parser.error("--a100-309m requires --a100")

    if args.a100:
        model_config = apply_a100_preset(args)
    elif args.cloud:
        model_config = DEFAULT_700M_CONFIG
    else:
        model_config = DEFAULT_8GB_CONFIG

    output_dir = Path(args.output_dir or (
        "outputs/exp182_pretrain_700m" if args.cloud else "outputs/exp182_pretrain_8gb"
    ))

    if not args.go:
        print("DRY RUN — no training.")
        print("  Laptop:  python experiments/exp182_pretrain_700m.py --go --smoke")
        print("  Cloud:   python experiments/exp182_pretrain_700m.py --go --cloud")
        print("  A100:    python experiments/exp182_pretrain_700m.py --go --a100")
        print("  A100 fast: python experiments/exp182_pretrain_700m.py --go --a100 --a100-309m")
        return

    if args.smoke:
        args.steps = 100
        args.log_interval = 5
        args.save_interval = 50

    global LOG_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = output_dir / "training.log"

    preset = "A100 705M" if args.a100 and not args.a100_309m else (
        "A100 309M" if args.a100 else ("705M cloud" if args.cloud else "309M 8GB")
    )
    log("=" * 60)
    log(f"exp182: pretrain ({preset})")
    log(f"  device: {DEVICE}")
    log(f"  config: {model_config}")
    if args.a100:
        log(f"  a100 preset: shard_dir={args.shard_dir} bf16={args.use_bf16}")

    model = build_model(model_config)
    n_params = count_parameters(model)
    log(f"  params: {n_params / 1e6:.1f}M")

    est_gb = estimate_vram_gb(n_params)
    log(f"  est. VRAM (Muon hybrid): ~{est_gb:.1f} GB")
    if DEVICE.type == "cuda" and est_gb > 7.2 and not args.force and not args.cloud:
        log("  ABORT: est. VRAM > 7.2GB for 8GB GPU. Use --force or reduce --batch-size.")
        return

    model.to(DEVICE)
    model.train()

    try:
        optimizer, muon_n, adam_n = build_muon_optimizer(
            model, args.muon_lr, args.adam_lr, args.weight_decay,
        )
        log(f"  optimizer: Muon ({muon_n/1e6:.1f}M params) + AdamW aux ({adam_n/1e6:.1f}M)")
    except ImportError:
        log("  ERROR: pip install git+https://github.com/KellerJordan/Muon")
        return

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    step = 0
    best_metric = float("inf")
    resume_path = output_dir / "latest.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        best_metric = ckpt.get("best_metric", float("inf"))
        log(f"  resumed step={step}")

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

    eff_bs = args.batch_size * args.accum_steps
    log(f"  batch={args.batch_size} accum={args.accum_steps} eff_bs={eff_bs}")
    log(f"  steps={args.steps:,} muon_lr={args.muon_lr} adam_aux_lr={args.adam_lr}")
    log("=" * 60)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_config.to_dict(), "training": vars(args)}, f, indent=2)

    batch_iter = iter_training_batches(args, DEVICE)
    accum_p = accum_v = 0.0
    accum_n = 0
    t0 = time.time()
    positions = step * eff_bs

    while step < args.steps:
        if SHUTDOWN:
            save_checkpoint(model, optimizer, scaler, step, best_metric, resume_path, model_config)
            log(f"Shutdown at step {step}")
            return

        try:
            batch_input, move_targets, wdl_targets = next(batch_iter)
        except StopIteration:
            batch_iter = iter_training_batches(args, DEVICE)
            continue

        p, v = train_step(
            model, batch_input, move_targets, wdl_targets,
            args, scaler, model_config.n_value_classes,
        )
        accum_p += p
        accum_v += v
        accum_n += 1
        positions += move_targets.shape[0]

        if accum_n < args.accum_steps:
            continue

        scaler.unscale_(optimizer)
        gn = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        set_lrs(step)

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            pos_s = positions / max(elapsed, 1)
            vram = ""
            if DEVICE.type == "cuda":
                vram = f" | vram={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            log(
                f"step {step:,}/{args.steps:,} | "
                f"p={accum_p/accum_n:.4f} v={accum_v/accum_n:.4f} | "
                f"muon_lr={optimizer.param_groups[0]['lr']:.2e} gn={gn:.2f} | "
                f"{pos_s:.1f} pos/s{vram}"
            )
            accum_p = accum_v = 0.0
            accum_n = 0

        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, step, best_metric, resume_path, model_config)
            save_checkpoint(
                model, optimizer, scaler, step, best_metric,
                output_dir / f"step_{step:06d}.pt", model_config,
            )
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, scaler, step, best_metric, resume_path, model_config)
    log(f"Done. {step:,} steps, {positions:,} positions.")


if __name__ == "__main__":
    main()
