#!/usr/bin/env python3
"""CPU-only prepare_soft_batch timing. Does not touch the GPU."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import chess
import torch

from autoresearch_8gb.pipeline import (
    attach_static_targets,
    board_to_cache_row,
    prepare_soft_batch,
    stack_rows,
)


def main() -> None:
    rows = []
    b = chess.Board()
    for _ in range(192):
        rows.append(board_to_cache_row(b, chess.Move.from_uci("e2e4")))
    data = stack_rows(rows)
    attach_static_targets(data)
    idx = torch.arange(192)
    dev = torch.device("cpu")
    # warmup
    for _ in range(5):
        prepare_soft_batch(data, idx, dev, hflip_p=0.5)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        prepare_soft_batch(data, idx, dev, hflip_p=0.5)
    dt = (time.perf_counter() - t0) / n
    print(f"prepare_soft_batch CPU: {dt*1e3:.2f} ms/batch  ({192/dt:.0f} pos/s prep-only)")
    print("live train is ~1076 ms/step @ 178 pos/s; CPU prep is not the limiter unless it grows.")


if __name__ == "__main__":
    main()
