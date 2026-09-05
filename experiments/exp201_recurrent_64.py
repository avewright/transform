#!/usr/bin/env python3
"""exp201: ~100M squares-only recurrent transformer (compact vocab).

Architecture (see chess_squares64.py):
  - 64×64 attention only (no ctx tokens in the sequence)
  - Fused piece-color embeds (WQ ≠ BQ)
  - Turn / castling / EP as FiLM transforms on the square stream
  - Trunk: prefix 4 + bank 7×3 unrolls + suffix 4  (29 effective depth)
  - After backward: average_recurrent_grads() before optimizer.step()

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp201_recurrent_64.py --go
  MOVE_VOCAB_VERSION=compact python experiments/exp201_recurrent_64.py --smoke
  MOVE_VOCAB_VERSION=compact python experiments/exp201_recurrent_64.py --smoke --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chess_squares64 import (
    DEFAULT_100M_SQUARES64_CONFIG,
    average_recurrent_grads,
    build_squares64,
    count_parameters,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp201_recurrent_64"
DEFAULT_SOFT = ROOT / "outputs" / "hf_elo_mix" / "soft_cache.pt"
DEFAULT_DEEP = ROOT / "outputs" / "hf_elo_mix" / "deep_cache.pt"


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(
            f"Expected compact vocab 1968, got {VOCAB_SIZE}. "
            "Export MOVE_VOCAB_VERSION=compact."
        )


def smoke(device: torch.device) -> dict:
    cfg = DEFAULT_100M_SQUARES64_CONFIG
    model = build_squares64(cfg).to(device)
    n = count_parameters(model)
    print(
        f"params={n:,} ({n/1e6:.1f}M)  unique_layers={cfg.unique_layers}  "
        f"effective_depth={cfg.effective_depth}  "
        f"bank={cfg.recurrent_layers}×{cfg.recurrent_unrolls}"
    )

    B = 4
    board_input = {
        "fused_ids": torch.randint(0, 13, (B, 64), device=device),
        "turn": torch.randint(0, 2, (B,), device=device),
        "castling": torch.randint(0, 16, (B,), device=device),
        "ep_file": torch.randint(0, 9, (B,), device=device),
    }
    # Distinct WQ / BQ slots
    board_input["fused_ids"][:, 3] = 5
    board_input["fused_ids"][:, 59] = 11

    model.train()
    out = model(board_input)
    assert out["square_hidden"].shape == (B, 64, cfg.hidden_dim), out["square_hidden"].shape
    assert out["policy_logits"].shape[-1] == VOCAB_SIZE

    loss = out["policy_logits"].float().pow(2).mean() + out["value_logits"].float().pow(2).mean()
    loss.backward()

    # Recurrent grads should be non-None; averaging must be safe.
    bank_grads_before = [
        p.grad.detach().abs().mean().item()
        for p in model.recurrent_parameters() if p.grad is not None
    ]
    average_recurrent_grads(model)
    bank_grads_after = [
        p.grad.detach().abs().mean().item()
        for p in model.recurrent_parameters() if p.grad is not None
    ]
    ratio = (
        sum(bank_grads_after) / max(sum(bank_grads_before), 1e-12)
        if bank_grads_before else float("nan")
    )

    enc = model.encoder
    wq = enc.piece_color_embed.weight[5].detach()
    bq = enc.piece_color_embed.weight[11].detach()
    cos = torch.nn.functional.cosine_similarity(wq, bq, dim=0).item()

    summary = {
        "params": n,
        "params_m": round(n / 1e6, 2),
        "vocab_size": VOCAB_SIZE,
        "config": cfg.to_dict(),
        "unique_layers": cfg.unique_layers,
        "effective_depth": cfg.effective_depth,
        "policy_shape": list(out["policy_logits"].shape),
        "square_hidden_shape": list(out["square_hidden"].shape),
        "wq_bq_cosine": round(cos, 4),
        "recurrent_grad_scale_after_avg": round(ratio, 4),
        "smoke_loss": float(loss.detach().cpu()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "smoke.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    print(
        "Train step pattern:\n"
        "  loss.backward()\n"
        "  average_recurrent_grads(model)  # /3 on bank grads\n"
        "  optimizer.step()"
    )
    return summary


def ensure_hf_mix(soft: Path, deep: Path, *, soft_n: int, syzygy_n: int) -> None:
    if soft.exists() and deep.exists():
        print(f"using existing mix soft={soft} deep={deep}", flush=True)
        return
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "build_hf_elo_mix.py"),
        "--go",
        "--output-dir", str(soft.parent),
        "--soft-n", str(soft_n),
        "--syzygy-n", str(syzygy_n),
    ]
    print(f"building HF soft mix: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"build_hf_elo_mix failed rc={rc}")
    if not soft.exists():
        raise SystemExit(f"mix build did not write {soft}")


def trial_config() -> dict:
    model = DEFAULT_100M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp201_recurrent_64",
        "arch": "squares64",
        "desc": "~100M squares64 recurrent + avewright HF soft mix",
        "model": model,
        "train": {
            "batch_size": 256,
            "min_batch_size": 16,
            "max_batch_size": 1024,
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.55,
            "soft_temp": 4.0,
            "soft_temp_weight": 0.4,
            "deep_mix_frac": 0.4,
            "use_swa": True,
            "swa_start_frac": 0.75,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
            "optimizer": "polar_normuon",
            "compile_polar": True,
            "muon_lr": 0.02,
            "adam_lr": 0.0003,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup": 500,
            "min_lr_frac": 0.05,
            "torch_compile": True,
            "grad_checkpoint": False,
            "fill_vram": True,
            "max_vram_gb": 14.5,
            "save_every_steps": 250,
            "keep_step_every": 500,
            "keep_last_ckpts": 4,
        },
    }


def train(args: argparse.Namespace) -> dict:
    from autoresearch_8gb.train_trial import train_trial

    soft = Path(args.soft_cache)
    deep = Path(args.deep_cache)
    if not args.skip_mix:
        ensure_hf_mix(soft, deep, soft_n=args.soft_n, syzygy_n=args.syzygy_n)
    if not soft.exists():
        raise SystemExit(
            f"missing soft cache: {soft}\n"
            "Run: python scripts/build_hf_elo_mix.py --go"
        )
    if not deep.exists():
        print(f"warn: deep cache missing ({deep}); training without syzygy mix", flush=True)
        deep = None

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trial = trial_config()
    resume = Path(args.resume) if args.resume else None
    if resume is not None and not resume.exists():
        raise SystemExit(f"resume ckpt missing: {resume}")
    trial["train"]["deep_mix_frac"] = float(args.deep_mix_frac)
    if args.continue_min_lr:
        train = trial["train"]
        train["muon_lr"] = float(train.get("muon_lr", 0.02)) * float(train.get("min_lr_frac", 0.05))
        train["adam_lr"] = float(train.get("adam_lr", 3e-4)) * float(train.get("min_lr_frac", 0.05))
        train["min_lr_frac"] = 1.0
        train["warmup"] = 0
    result = train_trial(
        trial,
        out,
        soft_cache=soft,
        deep_cache=deep,
        max_steps=args.max_steps,
        max_minutes=args.train_minutes,
        smoke=False,
        resume_ckpt=resume,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="Train 100M recurrent on avewright HF soft mix")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--soft-cache", default=str(DEFAULT_SOFT))
    ap.add_argument("--deep-cache", default=str(DEFAULT_DEEP))
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--train-minutes", type=float, default=1440.0)
    ap.add_argument("--soft-n", type=int, default=1_500_000)
    ap.add_argument("--syzygy-n", type=int, default=400_000)
    ap.add_argument("--skip-mix", action="store_true")
    ap.add_argument(
        "--resume",
        default=None,
        help="Resume training. Full resume if optimizer/RNG present; otherwise a labeled weights-only warm start.",
    )
    ap.add_argument(
        "--continue-min-lr",
        action="store_true",
        help="Hold cosine-floor LR (for post-30k continuation)",
    )
    ap.add_argument(
        "--deep-mix-frac",
        type=float,
        default=0.4,
        help="Fraction of batches drawn from Syzygy deep_cache",
    )
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_100M_SQUARES64_CONFIG
    print("DEFAULT_100M_SQUARES64_CONFIG")
    print(
        f"  {cfg.hidden_dim}d / {cfg.num_heads}H | "
        f"prefix={cfg.prefix_layers} bank={cfg.recurrent_layers}×{cfg.recurrent_unrolls} "
        f"suffix={cfg.suffix_layers} | effective={cfg.effective_depth} unique={cfg.unique_layers}"
    )
    print("  attention=64×64  side-info=FiLM  embeds=fused piece×color  vocab=compact")

    if args.go:
        train(args)
        return

    if not args.smoke:
        n = count_parameters(build_squares64(cfg))
        print(f"  params≈{n:,} ({n/1e6:.1f}M) — pass --go to train or --smoke to check")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    smoke(device)


if __name__ == "__main__":
    main()
