#!/usr/bin/env python3
"""CPU tests for the 99M Syzygy 80/20 packer."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "scripts")]

import torch

from exp273_puzzle_finetune import drop_eval_overlap
from exp274_syzygy_finetune import SOURCE_SYZYGY, TRAIN_PCT, split_of, to_onehot


def test_split_is_stable():
    assert split_of("123") == split_of("123")
    assert split_of("123") in (0, 1)


def test_split_is_roughly_80_20():
    counts = Counter(split_of(str(i)) for i in range(10_000))
    train = counts[0] / 10_000
    assert 0.78 <= train <= 0.82, train
    assert TRAIN_PCT == 80


def test_overlap_drop_and_value_mask():
    n = 4
    base = {
        "board_array": torch.zeros(n, 64, dtype=torch.int8),
        "turn": torch.zeros(n, dtype=torch.int8),
        "castling": torch.zeros(n, dtype=torch.int8),
        "ep_square": torch.full((n,), -1, dtype=torch.int8),
        "move_idx": torch.zeros(n, dtype=torch.int64),
        "cp": torch.zeros(n, dtype=torch.int32),
        "mate": torch.zeros(n, dtype=torch.int32),
        "soft_indices": torch.full((n, 8), -1, dtype=torch.int64),
        "soft_probs": torch.zeros(n, 8, dtype=torch.float32),
        "source": torch.full((n,), SOURCE_SYZYGY, dtype=torch.int8),
        "value_valid": torch.zeros(n, dtype=torch.int8),
        "split": torch.zeros(n, dtype=torch.int8),
    }
    base["board_array"][0, 0] = 1
    base["board_array"][1, 0] = 2
    ev = {k: v[:1].clone() for k, v in base.items()}
    ev["split"][:] = 1
    cleaned, dropped = drop_eval_overlap(base, ev)
    assert dropped == 1
    assert int(cleaned["turn"].shape[0]) == 3
    assert int(cleaned["value_valid"].max()) == 0
    assert int(cleaned["source"][0]) == SOURCE_SYZYGY


def test_to_onehot_uses_move_idx():
    n = 3
    table = {
        "move_idx": torch.tensor([4, 7, 2], dtype=torch.int64),
        "soft_indices": torch.tensor([[1, 2, -1, -1, -1, -1, -1, -1]] * n),
        "soft_probs": torch.full((n, 8), 0.125),
    }
    out = to_onehot(table)
    assert out["soft_indices"][:, 0].tolist() == [4, 7, 2]
    assert out["soft_probs"][:, 0].tolist() == [1.0, 1.0, 1.0]
    assert float(out["soft_probs"][:, 1:].sum()) == 0.0
