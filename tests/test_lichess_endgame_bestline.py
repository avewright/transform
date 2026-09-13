"""CPU tests: <14-piece filter and one-hot best-line selection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_lichess_evals_soft_cache import (  # noqa: E402
    acc_for_materialize,
    pieces_fen,
    update_acc,
    white_score_to_stm,
)
from move_vocab import UCI_TO_IDX  # noqa: E402


def test_pieces_fen_under_14():
    rook_end = "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -"
    assert pieces_fen(rook_end) == 9
    five = "6k1/6p1/8/4K3/4NN2/8/8/8 w - -"
    assert pieces_fen(five) == 5
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    assert pieces_fen(start) == 32
    assert pieces_fen(start) > 13


def test_bestline_prefers_deeper_then_score():
    acc: dict = {}
    fen = "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -"
    # shallower winning-looking Ra7
    assert update_acc(acc, fen, "e7a7", 29, 1_000, 8308, target=10, one_hot=True)
    # deeper mate-in-1 Re8 must replace it
    mate_s = white_score_to_stm(None, 1, turn_black=False)
    assert mate_s is not None and mate_s > 8308
    assert update_acc(acc, fen, "e7e8", 245, 157, mate_s, target=10, one_hot=True)
    d, k, s, mv = acc[fen]
    assert mv == "e7e8"
    assert d == 245
    # worse score at same search does not win
    assert update_acc(acc, fen, "e7a7", 245, 157, 100, target=10, one_hot=True)
    assert acc[fen][3] == "e7e8"


def test_onehot_materialize_is_single_move():
    fen = "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -"
    acc = {fen: (40, 8000, 99_999, "e7e8")}
    moves = acc_for_materialize(acc, one_hot=True)
    assert list(moves[fen]) == ["e7e8"]
    assert "e7e8" in UCI_TO_IDX
