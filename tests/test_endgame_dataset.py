"""CPU tests for the 6–12 piece SF endgame harvest."""
from __future__ import annotations

import random

import chess

from scripts.sf19_soft_dataset import (
    ENDGAME_MAX_PIECES,
    ENDGAME_MIN_PIECES,
    build_endgame_game_spec,
    in_piece_window,
    pick_fast_forward_move,
)


def test_window_is_under_14():
    assert ENDGAME_MIN_PIECES == 6
    assert ENDGAME_MAX_PIECES == 13
    assert not in_piece_window(5)
    assert in_piece_window(6)
    assert in_piece_window(13)
    assert not in_piece_window(14)


def test_fast_forward_prefers_captures():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    hits = 0
    for i in range(40):
        mv = pick_fast_forward_move(board, random.Random(i), capture_p=1.0)
        assert mv is not None
        assert board.is_capture(mv)
        hits += 1
    assert hits == 40


def test_fast_forward_always_legal_without_captures():
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    mv = pick_fast_forward_move(board, random.Random(0), capture_p=1.0)
    assert mv in board.legal_moves
    assert not board.is_capture(mv)


def test_endgame_spec_sets_window():
    spec = build_endgame_game_spec(
        3,
        [{"fen": chess.STARTING_FEN, "eco": "A00", "name": "Start"}],
        seed=19,
        holdout_frac=0.05,
    )
    assert spec["endgame"] is True
    assert spec["label_min_pieces"] == 6
    assert spec["label_max_pieces"] == 13
    assert spec["start_fen"] == chess.STARTING_FEN


def test_endgame_spec_uses_seed_fen_and_epsilon():
    fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    spec = build_endgame_game_spec(
        5,
        [{"fen": chess.STARTING_FEN, "eco": "A00", "name": "Start"}],
        seed=19,
        holdout_frac=0.05,
        seed_fens=[fen],
    )
    assert spec["start_fen"] == fen
    assert spec["book_noise"] == 0
    assert spec["epsilon"] in (0.25, 0.40, 0.55, 0.70)
