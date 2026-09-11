#!/usr/bin/env python3
"""exp272: 270M vs SF19 mistake mine. Collect + verify only; no auto-train.

Usage:
  MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 \\
    python -u experiments/exp272_sf19_corrections.py go \\
      --out-dir outputs/exp272_sf19_corrections

  # smoke (tiny schedule + shallow teacher)
  python -u experiments/exp272_sf19_corrections.py go --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from exp272_mine import (  # noqa: E402
    assemble,
    default_ckpt,
    freeze_players,
    pack_examples,
    run_analyze,
    run_collect,
    run_train,
    write_reports,
)

DEFAULT_OUT = ROOT / "outputs" / "exp272_sf19_corrections"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="go",
                    choices=["freeze", "collect", "analyze", "assemble", "report", "pack", "train", "go"])
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-minutes", type=float, default=180.0)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-openings", type=int, default=50)
    ap.add_argument("--epsilon", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=272)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--target", type=int, default=100_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt()
    n_open = 4 if args.smoke else args.n_openings
    if args.smoke:
        args.target = min(args.target, 200)
    if args.cmd in {"freeze", "go"}:
        freeze_players(out, ckpt, n_openings=n_open, epsilon=args.epsilon, seed=args.seed, smoke=args.smoke)
    if args.cmd in {"collect", "go"}:
        run_collect(
            out,
            device=args.device,
            workers=args.workers,
            target=args.target,
            epsilon=args.epsilon,
            seed=args.seed,
        )
    if args.cmd in {"analyze"}:
        run_analyze(out, workers=args.workers, smoke=args.smoke)
    if args.cmd in {"assemble", "go"}:
        assemble(out)
    if args.cmd in {"report"}:
        write_reports(out)
    if args.cmd in {"pack", "train"}:
        pack_examples(out)
    if args.cmd in {"train"}:
        run_train(out, max_steps=args.max_steps, batch_size=args.batch_size, minutes=args.train_minutes)


if __name__ == "__main__":
    main()
