#!/usr/bin/env python3
"""Evaluate one or more checkpoints on the same frozen holdout caches."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from autoresearch_8gb.pipeline import attach_static_targets, cheap_eval_losses  # noqa: E402
from chess_inference import load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--eval", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--write", default="")
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--microbatch", type=int, default=64)
    args = ap.parse_args()
    if not args.ckpt or not args.eval:
        raise SystemExit("need --ckpt NAME=PATH and --eval NAME=PATH")

    evals = {}
    for item in args.eval:
        name, path = item.split("=", 1)
        data = torch.load(path, map_location="cpu", weights_only=False)
        attach_static_targets(data)
        evals[name] = data

    report = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for item in args.ckpt:
        name, path = item.split("=", 1)
        model = load_checkpoint(path, device=device)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.eval()
        report[name] = {}
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            for ev_name, data in evals.items():
                n = min(args.val_n, int(data["board_array"].shape[0]))
                take = torch.arange(n, dtype=torch.int64)
                report[name][ev_name] = cheap_eval_losses(
                    raw, data, take, device, microbatch=args.microbatch,
                )
                print(f"{name}/{ev_name} " + " ".join(
                    f"{k}={v:.4f}" for k, v in report[name][ev_name].items()
                ), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.write:
        Path(args.write).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.write}", flush=True)


if __name__ == "__main__":
    main()
