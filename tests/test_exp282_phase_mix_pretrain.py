"""CPU tests for the exp282 4-way phase mix."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import torch

from exp273_puzzle_finetune import SOURCE_PUZZLE
from exp282_phase_mix_pretrain import (
    INCUMBENT_REPO,
    NAMED_FRACS,
    SOURCE_END,
    SOURCE_MID,
    SOURCE_OPENING,
    assemble_mix,
    n_rows,
    refuse_incumbent,
    resolve_fracs,
    tag_source,
    write_init_ckpt,
)


def _table(n: int, source: int, *, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    boards = torch.zeros(n, 64, dtype=torch.int8)
    boards[:, 0] = 6
    boards[:, 4] = 12
    idx = torch.arange(n)
    boards[:, 8] = (idx % 12 + 1).to(torch.int8)
    boards[:, 16] = ((idx // 12) % 12 + 1).to(torch.int8)
    boards[:, 24] = ((idx // 144) % 6 + 1).to(torch.int8)
    boards[:, 32] = ((idx + seed * 17) % 12 + 1).to(torch.int8)
    mid = (torch.arange(n) % 100 + 10).to(torch.int64)
    si = torch.full((n, 8), -1, dtype=torch.int64)
    si[:, 0] = mid
    sp = torch.zeros(n, 8, dtype=torch.float32)
    sp[:, 0] = 1.0
    return {
        "board_array": boards,
        "turn": torch.zeros(n, dtype=torch.int8),
        "castling": torch.zeros(n, dtype=torch.int8),
        "ep_square": torch.full((n,), -1, dtype=torch.int8),
        "move_idx": mid,
        "cp": torch.randint(-50, 50, (n,), generator=g).to(torch.int32),
        "mate": torch.zeros(n, dtype=torch.int32),
        "soft_indices": si,
        "soft_probs": sp,
        "source": torch.full((n,), source, dtype=torch.int8),
        "value_valid": torch.ones(n, dtype=torch.int8),
        "label_depth": torch.full((n,), 20, dtype=torch.int16),
        "phase": torch.zeros(n, dtype=torch.int8),
    }


def test_resolve_fracs_spreads_leftover_to_25_each():
    fracs = resolve_fracs(NAMED_FRACS)
    assert NAMED_FRACS == {
        "opening": 0.20,
        "middlegame": 0.20,
        "endgame": 0.20,
        "puzzles": 0.20,
    }
    assert abs(sum(fracs.values()) - 1.0) < 1e-9
    for v in fracs.values():
        assert abs(v - 0.25) < 1e-9


def test_assemble_mix_is_equal_share_and_tagged():
    parts = {
        "opening": _table(400, SOURCE_OPENING, seed=1),
        "middlegame": _table(400, SOURCE_MID, seed=2),
        "endgame": _table(400, SOURCE_END, seed=3),
        "puzzles": _table(400, SOURCE_PUZZLE, seed=4),
    }
    train, ev, report = assemble_mix(parts, train_n=200, eval_n=20, seed=282)
    assert n_rows(train) == 200
    assert n_rows(ev) == 80
    counts = {name: int((train["source"] == sid).sum()) for name, sid in (
        ("opening", SOURCE_OPENING),
        ("middlegame", SOURCE_MID),
        ("endgame", SOURCE_END),
        ("puzzles", SOURCE_PUZZLE),
    )}
    assert counts == {"opening": 50, "middlegame": 50, "endgame": 50, "puzzles": 50}
    assert int((train["value_valid"][train["source"] == SOURCE_PUZZLE] == 0).all())
    assert int((train["value_valid"][train["source"] != SOURCE_PUZZLE] == 1).all())
    assert report["leftover_rule"].startswith("spread leftover")
    assert report["overlap_dropped"] == 0


def test_tag_source_masks_puzzle_value():
    t = tag_source(_table(3, 0, seed=9), SOURCE_PUZZLE, None)
    assert int(t["source"][0]) == SOURCE_PUZZLE
    assert int(t["value_valid"].sum()) == 0


def test_write_init_is_weights_only(tmp_path: Path):
    src = tmp_path / "src.pt"
    dest = tmp_path / "init.pt"
    torch.save({"model_state_dict": {"w": torch.ones(2)}, "steps": 99, "optimizer_state_dict": {"x": 1}}, src)
    write_init_ckpt(src, dest)
    ckpt = torch.load(dest, map_location="cpu", weights_only=False)
    assert ckpt["eval_only"] is True
    assert ckpt["steps"] == 0
    assert "optimizer_state_dict" not in ckpt
    assert torch.equal(ckpt["model_state_dict"]["w"], torch.ones(2))


def test_refuse_incumbent_repo():
    try:
        refuse_incumbent(INCUMBENT_REPO)
        raise AssertionError("should refuse")
    except SystemExit as e:
        assert "incumbent" in str(e)
    refuse_incumbent("avewright/phase-mix-scratch")
