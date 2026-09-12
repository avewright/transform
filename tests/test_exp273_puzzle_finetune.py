#!/usr/bin/env python3
"""CPU tests for the 99M Lichess puzzle finetune packer."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from exp273_puzzle_finetune import (  # noqa: E402
    SOURCE_PUZZLE,
    TRAIN_PCT,
    _stack_rows,
    drop_eval_overlap,
    play_puzzle,
    split_of,
)
from move_vocab import UCI_TO_IDX


LICHESS_00008 = {
    "PuzzleId": "00008",
    "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
    "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
    "Rating": 1914,
}


def test_split_is_stable_and_puzzle_level():
    a = split_of("00008")
    b = split_of("00008")
    assert a == b
    assert a in (0, 1)
    assert split_of("00008", seed=1) == split_of("00008", seed=1)


def test_split_is_roughly_80_20():
    counts = Counter(split_of(f"p{i}") for i in range(10_000))
    train = counts[0] / 10_000
    assert 0.78 <= train <= 0.82, train
    assert counts[0] + counts[1] == 10_000
    assert TRAIN_PCT == 80


def test_play_puzzle_labels_solver_plies_only():
    rows = play_puzzle(LICHESS_00008)
    assert len(rows) == 3
    moves = [r[4] for r in rows]
    assert moves == [UCI_TO_IDX["e6e7"], UCI_TO_IDX["b3c1"], UCI_TO_IDX["h6c1"]]
    # After opponent setup (f2g3), White is to move.
    assert rows[0][1] == 0


def test_play_puzzle_skips_short_or_bad_rating():
    assert play_puzzle({**LICHESS_00008, "Moves": "f2g3"}) == []
    assert play_puzzle({**LICHESS_00008, "Rating": 50}, min_rating=400) == []


def test_stack_and_overlap_drop():
    rows = play_puzzle(LICHESS_00008)
    train = _stack_rows(rows, 0)
    ev = _stack_rows(rows[:1], 1)
    assert train is not None and ev is not None
    assert int(train["source"][0]) == SOURCE_PUZZLE
    assert int(train["value_valid"][0]) == 0
    assert int(train["soft_probs"][0, 0]) == 1
    cleaned, dropped = drop_eval_overlap(train, ev)
    assert dropped == 1
    assert int(cleaned["turn"].shape[0]) == 2
