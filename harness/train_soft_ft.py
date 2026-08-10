#!/usr/bin/env python3
"""Elo-safe soft FT launcher — wraps exp191 with top1 selection + best.pt aliases."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.common import (
    DEFAULT_SEED_CKPT,
    ROOT,
    ensure_best_aliases,
    git_sha,
)


EXP191 = ROOT / "experiments" / "exp191_400m_meta_attention.py"


def build_cmd(args: argparse.Namespace) -> list[str]:
    out = Path(args.out)
    cmd = [
        sys.executable,
        "-u",
        str(EXP191),
        "--go",
        "--init-checkpoint",
        str(args.init),
        "--soft-cache",
        str(args.soft_cache),
        "--output-dir",
        str(out),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--soft-frac",
        str(args.soft_frac),
        "--soft-alpha",
        str(args.soft_alpha),
        "--deep-mix-frac",
        str(args.deep_mix_frac),
        "--select-metric",
        "top1",
        "--hflip-p",
        str(args.hflip_p),
        "--value-weight",
        str(args.value_weight),
        "--warmup",
        str(args.warmup),
        "--min-depth",
        str(args.min_depth),
        "--log-interval",
        "25",
        "--save-interval",
        str(args.save_interval),
        "--eval-interval",
        str(args.eval_interval),
    ]
    if args.deep_soft_cache:
        cmd.extend(["--deep-soft-cache", str(args.deep_soft_cache)])
    if args.no_hard_cache:
        cmd.append("--no-hard-cache")
    if args.smoke:
        cmd.append("--smoke")
    if args.compile:
        cmd.append("--compile")
    if args.polar:
        cmd.append("--polar")
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Soft FT via exp191 (top1 only)")
    ap.add_argument("--init", default=str(DEFAULT_SEED_CKPT), help="Init checkpoint")
    ap.add_argument("--soft-cache", required=True)
    ap.add_argument("--deep-soft-cache", default=None)
    ap.add_argument("--out", required=True, help="Output dir under outputs/")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--soft-frac", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=0.38)
    ap.add_argument("--deep-mix-frac", type=float, default=0.42)
    ap.add_argument("--hflip-p", type=float, default=0.5)
    ap.add_argument("--value-weight", type=float, default=0.08)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-depth", type=int, default=15)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--no-hard-cache", action="store_true", default=True)
    ap.add_argument("--hard-cache", action="store_true", help="Allow hard ballast cache")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--polar", action="store_true")
    args = ap.parse_args(argv)

    if args.hard_cache:
        args.no_hard_cache = False

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    args.out = out
    out.mkdir(parents=True, exist_ok=True)

    if not Path(args.init).exists():
        raise SystemExit(f"init checkpoint missing: {args.init}")
    if not Path(args.soft_cache).exists():
        raise SystemExit(f"soft cache missing: {args.soft_cache}")

    run_meta = {
        "init": str(args.init),
        "soft_cache": str(args.soft_cache),
        "deep_soft_cache": args.deep_soft_cache,
        "out": str(out),
        "steps": args.steps,
        "select_metric": "top1",
        "soft_frac": args.soft_frac,
        "soft_alpha": args.soft_alpha,
        "deep_mix_frac": args.deep_mix_frac,
        "git_sha": git_sha(),
        "cmd": build_cmd(args),
    }
    (out / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("MOVE_VOCAB_VERSION", "compact")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    cmd = build_cmd(args)
    print("=== harness.train_soft_ft ===", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    ensure_best_aliases(out)
    best = out / "best.pt"
    if best.exists():
        print(f"best -> {best}", flush=True)
    else:
        print(f"WARNING: no best.pt in {out}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
