"""Exploration must diversify trajectories without changing greedy error labels."""
import random

import chess
import numpy as np
import torch

from scripts.harvest_exp201_lapses import select_policy_moves, play_one


def test_board_array_fen_roundtrip():
    from data_loader import _fast_parse_fen
    from scripts.harvest_exp201_lapses import board_array_to_fen
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    arr = np.zeros(64, dtype=np.int8)
    _fast_parse_fen(fen, arr)
    out = board_array_to_fen(arr, 1, 15, -1)
    board = chess.Board(out)
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.E4).symbol() == "P"
    assert board.piece_at(chess.E5).symbol() == "p"


def test_exploration_is_seeded_legal_and_top_k():
    logits = torch.tensor([99.0, 2.0, 1.9, 1.8, -9.0])
    mask = torch.tensor([False, True, True, True, True])
    def sample(seed):
        rng = random.Random(seed)
        return [select_policy_moves(logits, mask, epsilon=1, temperature=0.8,
                                    top_k=2, rng=rng) for _ in range(100)]
    a = sample(17)
    assert a == sample(17)
    assert all(g == 1 and p in (1, 2) for g, p in a)
    assert any(p == 2 for _, p in a)
    assert select_policy_moves(logits, mask, epsilon=0) == (1, 1)


def test_single_legal_move_and_low_temperature():
    mask = torch.tensor([False, True])
    assert select_policy_moves(torch.tensor([100., 0.]), mask, epsilon=1,
                               top_k=4, rng=random.Random(1)) == (1, 1)
    assert select_policy_moves(torch.tensor([2., 1.]), torch.tensor([True, True]),
                               epsilon=1, temperature=0.001, rng=random.Random(2)) == (0, 0)


def test_trajectory_explores_but_teacher_labels_greedy(monkeypatch):
    import scripts.harvest_exp201_lapses as h
    from types import SimpleNamespace
    greedy = chess.Move.from_uci("e2e4")
    played = chess.Move.from_uci("d2d4")
    epsilons = []
    def fake_move(*args, **kwargs):
        epsilons.append(kwargs["epsilon"])
        return greedy, played
    monkeypatch.setattr(h, "model_move", fake_move)
    monkeypatch.setattr(h, "analyze_multipv", lambda *a, **k: {
        "ucis": ["d2d4", "e2e4"], "cps": [100, -100], "mates": [0, 0],
        "best_cp": 100, "best_mate": 0, "best_uci": "d2d4", "depth": 16,
        "probs": [0.9, 0.1],
    })
    class Opponent:
        def play(self, board, limit):
            assert board.peek() == played
            return SimpleNamespace(move=chess.Move.from_uci("d7d5"))
    rows, meta = play_one(None, None, Opponent(), None, model_color=chess.WHITE,
                         opening=[], opp_label="test", nodes=100, movetime=0,
                         tau=120, ply_cap=2, sf_movetime=0.01, unlimited_opp=False,
                         explore_epsilon=0.15, explore_seed=12)
    assert epsilons == [0.15]
    assert meta["n_exploratory_moves"] == 1
    assert rows[0]["_model_uci"] == "e2e4"
    assert rows[0]["_played_uci"] == "d2d4"
    assert rows[0]["_model_cp"] == -100


def test_book_noise_changes_start_without_seed_fen(monkeypatch):
    import scripts.harvest_exp201_lapses as h
    from types import SimpleNamespace

    monkeypatch.setattr(h, "model_move", lambda *a, **k: (
        chess.Move.from_uci("e2e4"), chess.Move.from_uci("e2e4"),
    ))
    monkeypatch.setattr(h, "analyze_multipv", lambda *a, **k: None)

    class Opponent:
        def play(self, board, limit):
            return SimpleNamespace(move=next(iter(board.legal_moves)))

    _, meta = play_one(
        None, None, Opponent(), None, model_color=chess.WHITE,
        opening=["e2e4"], opp_label="test", nodes=1, movetime=0, tau=120,
        ply_cap=4, sf_movetime=0.01, unlimited_opp=True, explore_seed=7,
        book_noise_plies=2,
    )
    assert meta["start_fen"] == ""
    assert meta["opening"] == "e2e4"


def test_as_fixed_respects_record_batch_slice():
    import pyarrow as pa
    from scripts.harvest_hf100m_bulk import _as_fixed

    n, w = 8, 64
    flat = np.arange(n * w, dtype=np.int8)
    boards = pa.FixedSizeListArray.from_arrays(pa.array(flat), w)
    turn = pa.array(np.arange(n, dtype=np.int8))
    batch = pa.RecordBatch.from_arrays([boards, turn], names=["board_array", "turn"])
    sliced = batch.slice(3, 2)
    got = _as_fixed(sliced, "board_array", w, np.int8)
    assert got.shape == (2, w)
    assert got[0, 0] == flat[3 * w]
    assert got[1, 0] == flat[4 * w]
