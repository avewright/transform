import os
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import numpy as np

from chess_chessbot import CHESSBOT_UCI_TO_IDX, CHESSFENS_POLICY_SIZE
from move_vocab import UCI_TO_IDX
from scripts.export_chessfens_policy import compact_to_lc0, convert_row, puzzle_to_soft, soft_to_chessfens_policy


def _array(board):
    arr = np.zeros(64, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        arr[sq] = piece.piece_type + (0 if piece.color else 6)
    castle = (8 * board.has_kingside_castling_rights(True)
              + 4 * board.has_queenside_castling_rights(True)
              + 2 * board.has_kingside_castling_rights(False)
              + board.has_queenside_castling_rights(False))
    ep = -1 if board.ep_square is None else board.ep_square
    return arr, 0 if board.turn else 1, castle, ep


def test_compact_e2e4_lands_in_lc0_prefix():
    idx = compact_to_lc0(UCI_TO_IDX["e2e4"])
    assert idx == CHESSBOT_UCI_TO_IDX["e2e4"]
    assert 0 <= idx < CHESSFENS_POLICY_SIZE


def test_soft_policy_is_1858_with_minus_one_fill():
    e2e4 = UCI_TO_IDX["e2e4"]
    d2d4 = UCI_TO_IDX["d2d4"]
    pol = soft_to_chessfens_policy(e2e4, [e2e4, d2d4, -1], [0.7, 0.3, 0.0])
    assert len(pol) == CHESSFENS_POLICY_SIZE
    assert abs(pol[CHESSBOT_UCI_TO_IDX["e2e4"]] - 0.7) < 1e-6
    assert abs(pol[CHESSBOT_UCI_TO_IDX["d2d4"]] - 0.3) < 1e-6
    assert sum(1 for x in pol if x < 0) == CHESSFENS_POLICY_SIZE - 2


def test_convert_row_emits_fen_and_no_wdl():
    arr, turn, castle, ep = _array(chess.Board())
    rec = dict(board_array=arr, turn=turn, castling=castle, ep_square=ep,
               move_idx=UCI_TO_IDX["e2e4"],
               soft_indices=[UCI_TO_IDX["e2e4"], -1, -1, -1, -1, -1, -1, -1],
               soft_probs=[1.0, 0, 0, 0, 0, 0, 0, 0])
    row, err = convert_row(rec, "avewright/test")
    assert err is None
    assert set(row) == {"fen", "policy", "source"}
    assert "wdl" not in row
    assert row["fen"].split()[:2] == ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "w"]
    assert len(row["policy"]) == 1858


def test_king_e1h1_is_castle_but_rook_e1h1_stays_a_slide():
    king = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    rook = chess.Board("4k3/8/8/8/8/8/8/R3R3 w - - 0 1")
    idx = UCI_TO_IDX["e1h1"]
    assert compact_to_lc0(idx, king) == CHESSBOT_UCI_TO_IDX["e1g1"]
    assert compact_to_lc0(idx, rook) == CHESSBOT_UCI_TO_IDX["e1h1"]
    arr, turn, castle, ep = _array(rook)
    row, err = convert_row(dict(
        board_array=arr, turn=turn, castling=castle, ep_square=ep,
        move_idx=idx, soft_indices=[idx], soft_probs=[1.0],
    ), "avewright/test")
    assert err is None
    assert row["policy"][CHESSBOT_UCI_TO_IDX["e1h1"]] == 1.0
    assert row["policy"][CHESSBOT_UCI_TO_IDX["e1g1"]] < 0


def test_lichess_puzzle_without_rating_is_dropped():
    assert puzzle_to_soft({
        "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
    }) is None


def test_lichess_puzzle_becomes_onehot_policy():
    puzzle = {
        "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
        "Rating": 1939,
    }
    rec = puzzle_to_soft(puzzle)
    assert rec is not None
    row, err = convert_row(rec, "Lichess/chess-puzzles")
    assert err is None
    assert len(row["policy"]) == 1858
    assert abs(sum(x for x in row["policy"] if x > 0) - 1.0) < 1e-6
    assert row["fen"].split()[1] == "w"  # after opponent's first forced move
