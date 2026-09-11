#!/usr/bin/env python3
"""exp270: ~270M squares-only recurrent pretrain (fresh init).

Width-only scale of the 99M incumbent. Same 4+7×3+4 trunk, compact 1968 vocab,
squares-only attention. Measured params: 268,552,344 @ 1216d / 16H.

This is a new architecture. Do NOT load the 99M optimizer. Fresh weights only.
The 99M incumbent stays the untouched benchmark.

Usage (prepare / smoke only until you launch on RunPod):
  MOVE_VOCAB_VERSION=compact python experiments/exp270_squares64_pretrain.py
  MOVE_VOCAB_VERSION=compact python experiments/exp270_squares64_pretrain.py --smoke --device cpu
  MOVE_VOCAB_VERSION=compact python experiments/exp270_squares64_pretrain.py --go   # train

RunPod entry: scripts/runpod_exp270.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chess_squares64 import (
    DEFAULT_100M_SQUARES64_CONFIG,
    DEFAULT_270M_SQUARES64_CONFIG,
    EXPECTED_270M_PARAMS,
    average_recurrent_grads,
    build_squares64,
    count_parameters,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp270_squares64_pretrain"
DEFAULT_SOFT = ROOT / "outputs" / "exp270_mix_v1" / "soft_cache.pt"
DEFAULT_DEEP = ROOT / "outputs" / "exp270_mix_v1" / "deep_cache.pt"
RECIPE = ROOT / "chess_master" / "recipes" / "exp270_75_20_5.json"
INCUMBENT_MARKERS = (
    "exp201_recurrent_64",
    "hf_100m_squares64",
    "chess-transformer-100m",
    "overnight_20260908",
    "sf19_ft/run2",
    "sf19_ft/overnight",
    "DEFAULT_100M",
)

# Match the 99M continuation phase (Polar-NorMuon + constant LRs).
# 99M pretrain peak 0.02 / 3e-4, scaled by ~270/99.
MUON_LR = 0.02 / 2.7
ADAM_LR = 3e-4 / 2.7
DEFAULT_BONUS = ROOT / "outputs" / "exp270_mix_v1" / "bonus_cache.pt"
DEFAULT_QUALITY = ROOT / "outputs" / "exp270_mix_v1" / "sf19_eco_train.pt"
DEFAULT_PUZZLE = ROOT / "outputs" / "exp270_mix_v1" / "puzzle_cache.pt"


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(
            f"Expected compact vocab 1968, got {VOCAB_SIZE}. "
            "Export MOVE_VOCAB_VERSION=compact."
        )


def refuse_incumbent_resume(path: Path | None) -> None:
    """Fresh-init experiment: reject 99M optimizer / silent partial copies."""
    if path is None:
        return
    text = str(path.resolve())
    for marker in INCUMBENT_MARKERS:
        if marker in text:
            raise SystemExit(
                f"refusing resume from incumbent path {path}\n"
                "exp270 is a new architecture. Fresh init only. "
                "Do not load the 99M optimizer. If you later want a "
                "documented weight expansion, implement it as an explicit "
                "transform with output-preservation tests — do not pass "
                "this checkpoint to --resume."
            )


def trial_config() -> dict:
    model = DEFAULT_270M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp270_squares64_pretrain",
        "arch": "squares64",
        "desc": (
            "~270M squares64 recurrent, fresh init, "
            "PolarNorMuon, soft_alpha=0.55, 20% Syzygy + 8% correction + 5% SF19 ECO + 5% puzzle play per batch."
        ),
        "incumbent_benchmark": {
            "name": "99M squares64",
            "params": 98_971_224,
            "hidden_dim": DEFAULT_100M_SQUARES64_CONFIG.hidden_dim,
            "num_heads": DEFAULT_100M_SQUARES64_CONFIG.num_heads,
            "note": "Different training history. First night is a learning-curve run.",
        },
        "recipe": str(RECIPE),
        "model": model,
        "train": {
            "batch_size": 48,
            "min_batch_size": 8,
            "max_batch_size": 80,
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.55,
            "soft_temp": 4.0,
            "soft_temp_weight": 0.4,
            "deep_mix_frac": 0.20,
            "deep_in_each_batch": True,
            "bonus_mix_frac": 0.08,
            "quality_mix_frac": 0.05,
            "puzzle_mix_frac": 0.05,
            "use_swa": False,
            "swa_start_frac": 0.75,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
            "optimizer": "polar_normuon",
            "compile_polar": True,
            "force_lr": True,
            "muon_lr": MUON_LR,
            "adam_lr": ADAM_LR,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup": 0,
            "min_lr_frac": 1.0,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": True,
            "max_vram_gb": 14.2,
            "save_every_steps": 500,
            "keep_step_every": 1000,
            "keep_last_ckpts": 4,
            "val_every_steps": 1000,
            "val_eval_n": 256,
            "elo_every_steps": 0,
        },
    }


def smoke(device: torch.device) -> dict:
    cfg = DEFAULT_270M_SQUARES64_CONFIG
    model = build_squares64(cfg).to(device)
    n = count_parameters(model)
    if n != EXPECTED_270M_PARAMS:
        raise SystemExit(f"param count {n} != {EXPECTED_270M_PARAMS}")
    print(
        f"params={n:,} ({n/1e6:.3f}M)  hidden={cfg.hidden_dim} heads={cfg.num_heads} "
        f"head_dim={cfg.hidden_dim // cfg.num_heads}  "
        f"unique={cfg.unique_layers} effective={cfg.effective_depth}  "
        f"bank={cfg.recurrent_layers}×{cfg.recurrent_unrolls}"
    )

    B = 2
    board_input = {
        "fused_ids": torch.randint(0, 13, (B, 64), device=device),
        "turn": torch.randint(0, 2, (B,), device=device),
        "castling": torch.randint(0, 16, (B,), device=device),
        "ep_file": torch.randint(0, 9, (B,), device=device),
    }
    board_input["fused_ids"][:, 3] = 5
    board_input["fused_ids"][:, 59] = 11

    model.train()
    out = model(board_input)
    assert out["square_hidden"].shape == (B, 64, cfg.hidden_dim)
    assert out["policy_logits"].shape[-1] == VOCAB_SIZE
    loss = out["policy_logits"].float().pow(2).mean() + out["value_logits"].float().pow(2).mean()
    loss.backward()
    before = [
        p.grad.detach().abs().mean().item()
        for p in model.recurrent_parameters() if p.grad is not None
    ]
    average_recurrent_grads(model)
    after = [
        p.grad.detach().abs().mean().item()
        for p in model.recurrent_parameters() if p.grad is not None
    ]
    ratio = sum(after) / max(sum(before), 1e-12) if before else float("nan")
    summary = {
        "params": n,
        "params_m": round(n / 1e6, 3),
        "expected_params": EXPECTED_270M_PARAMS,
        "vocab_size": VOCAB_SIZE,
        "config": cfg.to_dict(),
        "unique_layers": cfg.unique_layers,
        "effective_depth": cfg.effective_depth,
        "policy_shape": list(out["policy_logits"].shape),
        "square_hidden_shape": list(out["square_hidden"].shape),
        "recurrent_grad_scale_after_avg": round(ratio, 4),
        "fresh_init": True,
        "incumbent_untouched": True,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "smoke.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    return summary


def train(args: argparse.Namespace) -> dict:
    from autoresearch_8gb.train_trial import train_trial

    refuse_incumbent_resume(Path(args.resume) if args.resume else None)
    soft = Path(args.soft_cache)
    deep = Path(args.deep_cache)
    if not soft.exists():
        raise SystemExit(
            f"missing mix {soft}\n"
            "On the pod: python scripts/pack_exp270_all.py"
        )
    if not deep.exists():
        raise SystemExit(f"missing deep cache {deep}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trial = trial_config()
    train_cfg = trial["train"]
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
        train_cfg["max_batch_size"] = max(int(train_cfg.get("max_batch_size", 0)), int(args.batch_size))
    if args.accum_steps is not None:
        train_cfg["accum_steps"] = int(args.accum_steps)
    if args.muon_lr is not None:
        train_cfg["muon_lr"] = float(args.muon_lr)
    if args.adam_lr is not None:
        train_cfg["adam_lr"] = float(args.adam_lr)
    if args.grad_checkpoint:
        train_cfg["grad_checkpoint"] = True
        trial["model"]["gradient_checkpointing"] = True
    if args.max_vram_gb is not None:
        train_cfg["max_vram_gb"] = float(args.max_vram_gb)
    if args.warmup is not None:
        train_cfg["warmup"] = int(args.warmup)
    if args.elo_every is not None:
        train_cfg["elo_every_steps"] = int(args.elo_every)
    if args.save_every is not None:
        train_cfg["save_every_steps"] = int(args.save_every)
    if args.val_every is not None:
        train_cfg["val_every_steps"] = int(args.val_every)
    if args.torch_compile is not None:
        train_cfg["torch_compile"] = bool(args.torch_compile)
    if args.optimizer is not None:
        train_cfg["optimizer"] = str(args.optimizer)
    if args.compile_mode is not None:
        train_cfg["compile_mode"] = str(args.compile_mode)
    if args.deep_mix_frac is not None:
        train_cfg["deep_mix_frac"] = float(args.deep_mix_frac)
    if args.deep_in_each_batch is not None:
        train_cfg["deep_in_each_batch"] = bool(args.deep_in_each_batch)
    if args.soft_alpha is not None:
        train_cfg["soft_alpha"] = float(args.soft_alpha)
    if args.bonus_mix_frac is not None:
        train_cfg["bonus_mix_frac"] = float(args.bonus_mix_frac)
    if args.quality_mix_frac is not None:
        train_cfg["quality_mix_frac"] = float(args.quality_mix_frac)
    if args.puzzle_mix_frac is not None:
        train_cfg["puzzle_mix_frac"] = float(args.puzzle_mix_frac)
    if args.force_lr:
        train_cfg["force_lr"] = True
    if args.compile_polar is not None:
        train_cfg["compile_polar"] = bool(args.compile_polar)
    if args.fill_vram is not None:
        train_cfg["fill_vram"] = bool(args.fill_vram)
    if args.external_eval:
        spec = {}
        for item in args.external_eval:
            if "=" not in item:
                raise SystemExit(f"--external-eval needs NAME=PATH, got {item}")
            name, path = item.split("=", 1)
            spec[name] = str(Path(path).resolve())
        train_cfg["external_eval"] = spec
    if args.block_manifest:
        train_cfg["block_manifests"] = [str(Path(p).resolve()) for p in args.block_manifest]
    extras = [Path(p) for p in (args.extra_soft_cache or [])]
    extras = [p if p.is_absolute() else ROOT / p for p in extras]

    resume = Path(args.resume) if args.resume else None
    if resume is not None and not resume.exists():
        raise SystemExit(f"resume ckpt missing: {resume}")

    bonus = Path(args.bonus_cache) if args.bonus_cache else None
    if bonus is not None and not bonus.is_absolute():
        bonus = ROOT / bonus
    if bonus is not None and not bonus.exists():
        bonus = None

    quality = Path(args.quality_cache) if args.quality_cache else None
    if quality is not None and not quality.is_absolute():
        quality = ROOT / quality
    if quality is not None and not quality.exists():
        quality = None

    puzzle = Path(args.puzzle_cache) if args.puzzle_cache else None
    if puzzle is not None and not puzzle.is_absolute():
        puzzle = ROOT / puzzle
    if puzzle is not None and not puzzle.exists():
        puzzle = None

    result = train_trial(
        trial,
        out,
        soft_cache=soft,
        deep_cache=deep,
        max_steps=args.max_steps,
        max_minutes=args.train_minutes,
        smoke=False,
        resume_ckpt=resume,
        extra_soft_caches=extras or None,
        bonus_cache=bonus,
        quality_cache=quality,
        puzzle_cache=puzzle,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="Start fresh-init 270M pretrain")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--soft-cache", default=str(DEFAULT_SOFT))
    ap.add_argument("--deep-cache", default=str(DEFAULT_DEEP))
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--train-minutes", type=float, default=840.0, help="Default 14h")
    ap.add_argument("--resume", default=None, help="Resume THIS run only (exp270 latest.pt). Not the 99M incumbent.")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--muon-lr", type=float, default=None)
    ap.add_argument("--adam-lr", type=float, default=None)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--max-vram-gb", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--elo-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--optimizer", default=None, help="normuon | polar_normuon | adamw")
    ap.add_argument("--compile-mode", default=None, help="default | reduce-overhead | max-autotune")
    ap.add_argument("--deep-mix-frac", type=float, default=None)
    ap.add_argument("--deep-in-each-batch", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--soft-alpha", type=float, default=None)
    ap.add_argument("--bonus-mix-frac", type=float, default=None)
    ap.add_argument("--bonus-cache", default=str(DEFAULT_BONUS), help="Correction / disagreement soft cache")
    ap.add_argument("--quality-mix-frac", type=float, default=None)
    ap.add_argument(
        "--quality-cache",
        default=str(DEFAULT_QUALITY),
        help="Oversampled SF19 MultiPV ECO soft cache (split=0 only)",
    )
    ap.add_argument("--puzzle-mix-frac", type=float, default=None)
    ap.add_argument(
        "--puzzle-cache",
        default=str(DEFAULT_PUZZLE),
        help="Lichess puzzle play-through (solver plies, one-hot)",
    )
    ap.add_argument("--force-lr", action="store_true", help="Hold muon/adam LRs constant")
    ap.add_argument("--compile-polar", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--fill-vram", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--external-eval", action="append", default=[])
    ap.add_argument("--block-manifest", action="append", default=[])
    ap.add_argument(
        "--extra-soft-cache",
        action="append",
        default=[],
        help="Extra disjoint soft .pt shards (remaining Lichess). Repeatable.",
    )
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_270M_SQUARES64_CONFIG
    print("DEFAULT_270M_SQUARES64_CONFIG")
    print(
        f"  {cfg.hidden_dim}d / {cfg.num_heads}H (head_dim={cfg.hidden_dim // cfg.num_heads}) | "
        f"prefix={cfg.prefix_layers} bank={cfg.recurrent_layers}×{cfg.recurrent_unrolls} "
        f"suffix={cfg.suffix_layers} | effective={cfg.effective_depth} unique={cfg.unique_layers}"
    )
    print(f"  expected_params={EXPECTED_270M_PARAMS:,}  fresh_init  mix=75/20/5  incumbent=99M benchmark only")

    if args.go:
        train(args)
        return
    if args.smoke:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        smoke(device)
        return
    n = count_parameters(build_squares64(cfg))
    print(f"  params={n:,} ({n/1e6:.3f}M) — pass --smoke or --go")


if __name__ == "__main__":
    main()
