"""CPU tests for exp275 endgame pack split and stratified val."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import numpy as np
import torch

from exp273_puzzle_finetune import drop_eval_overlap
from exp275_endgame_finetune import TRAIN_PCT, split_of, stratified_indices


def test_split_stable_and_80_20():
    assert split_of("99") == split_of("99")
    counts = Counter(split_of(str(i)) for i in range(10_000))
    assert 0.78 <= counts[0] / 10_000 <= 0.82
    assert TRAIN_PCT == 80


def test_stratified_covers_piece_counts():
    pcs = np.concatenate([np.full(200, p) for p in range(6, 14)])
    rng = np.random.default_rng(275)
    idx = stratified_indices(pcs, 800, rng)
    hist = Counter(pcs[idx].tolist())
    assert len(idx) == 800
    for p in range(6, 14):
        assert hist[p] >= 90, hist


def test_overlap_drop():
    n = 4
    base = {
        "board_array": torch.zeros(n, 64, dtype=torch.int8),
        "turn": torch.zeros(n, dtype=torch.int8),
        "castling": torch.zeros(n, dtype=torch.int8),
        "ep_square": torch.full((n,), -1, dtype=torch.int8),
        "move_idx": torch.arange(n, dtype=torch.int64),
        "cp": torch.zeros(n, dtype=torch.int32),
        "mate": torch.zeros(n, dtype=torch.int32),
        "soft_indices": torch.full((n, 8), -1),
        "soft_probs": torch.zeros(n, 8),
    }
    base["board_array"][0, 0] = 1
    train = {k: torch.cat([v, v[:1]], 0) for k, v in base.items()}
    ev = {k: v[:1] for k, v in base.items()}
    cleaned, dropped = drop_eval_overlap(train, ev)
    assert dropped >= 1
    assert int(cleaned["turn"].shape[0]) < int(train["turn"].shape[0])
