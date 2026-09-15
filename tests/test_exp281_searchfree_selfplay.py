#!/usr/bin/env python3
"""CPU tests for search-free self-play: train on the winner's moves."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import chess
import torch
import torch.nn as nn

from move_vocab import VOCAB_SIZE, move_to_index
from rl_selfplay.config import searchfree_99m_config
from rl_selfplay.searchfree import (
    filter_student_win_positions,
    filter_winner_positions,
    sample_policy_moves,
)
from rl_selfplay.train import train_on_positions


def _ply(fen: str, uci: str, stm_white: bool) -> dict:
    return {
        "fen": fen,
        "chosen_move": move_to_index(chess.Move.from_uci(uci)),
        "stm_white": stm_white,
    }


def _scholar_traj() -> list[dict]:
    # Minimal alternating sides; indices only need to be legal vocab ids.
    start = chess.Board()
    return [
        _ply(start.fen(), "e2e4", True),
        _ply("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e5", False),
        _ply("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "d1h5", True),
        _ply("rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2", "b8c6", False),
        _ply("r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3", "f1c4", True),
        _ply("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3", "g8f6", False),
        _ply("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", True),
    ]


def test_searchfree_99m_defaults():
    cfg = searchfree_99m_config()
    assert cfg.use_search is False
    assert cfg.sample_temp == 0.7
    assert cfg.winner_only is True
    assert cfg.vs_incumbent is True
    assert cfg.incumbent_temp == 1.0
    assert cfg.value_weight == 0.0


def test_white_win_keeps_white_moves_only():
    traj = _scholar_traj()
    kept = filter_winner_positions(traj, 1.0, game_id=0)
    assert len(kept) == 4
    assert all(p["stm_white"] for p in kept)
    assert all(p["source"] == "winner" for p in kept)
    assert kept[0]["chosen_move"] == move_to_index(chess.Move.from_uci("e2e4"))


def test_black_win_keeps_black_moves_only():
    traj = _scholar_traj()
    kept = filter_winner_positions(traj, 0.0, game_id=3)
    assert len(kept) == 3
    assert all(not p["stm_white"] for p in kept)
    assert kept[0]["game_id"] == 3


def test_draw_keeps_nothing():
    traj = _scholar_traj()
    assert filter_winner_positions(traj, 0.5, game_id=1) == []


def test_student_win_only_keeps_student_moves():
    traj = _scholar_traj()
    assert len(filter_student_win_positions(traj, 1.0, 0, student_white=True)) == 4
    assert filter_student_win_positions(traj, 1.0, 0, student_white=False) == []
    assert len(filter_student_win_positions(traj, 0.0, 1, student_white=False)) == 3
    assert filter_student_win_positions(traj, 0.0, 1, student_white=True) == []
    assert filter_student_win_positions(traj, 0.5, 2, student_white=True) == []
    assert filter_student_win_positions(traj, 0.5, 2, student_white=False) == []


class _DummyPolicy(nn.Module):
    def forward(self, batch):
        b = batch["fused_ids"].shape[0]
        return {
            "policy_logits": torch.zeros(b, VOCAB_SIZE),
            "value_logits": torch.zeros(b, 3),
        }


def test_sample_temperature_is_legal():
    torch.manual_seed(0)
    board = chess.Board()
    model = _DummyPolicy()
    (move, idx), = sample_policy_moves(model, [board], torch.device("cpu"), temperature=0.7)
    assert move in board.legal_moves
    assert idx == move_to_index(move)


class _TinyTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(1, VOCAB_SIZE)
        self.value = nn.Linear(1, 3)
        self.config = type("C", (), {"n_value_classes": 3, "use_history": False})()

    def forward(self, batch):
        b = batch["fused_ids"].shape[0]
        x = self.policy.weight.new_zeros(b, 1)
        return {"policy_logits": self.policy(x), "value_logits": self.value(x)}


def test_train_on_winner_positions():
    traj = _scholar_traj()
    positions = filter_winner_positions(traj, 1.0, 0) + filter_winner_positions(traj, 0.0, 1)
    assert positions
    cfg = searchfree_99m_config(
        train_epochs=1, train_batch_size=4, train_lr=1e-3, use_bf16=False, use_fp16=False,
    )
    model = _TinyTrain()
    metrics = train_on_positions(model, positions, torch.device("cpu"), cfg, 3, log_fn=lambda *_: None)
    assert metrics["loss"] == metrics["loss"]
    assert metrics["policy"] >= 0
