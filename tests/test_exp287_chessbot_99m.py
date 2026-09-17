import json
import os
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import numpy as np
import pytest
import torch

from chess_chessbot import (
    CHESSBOT_POLICY,
    CHESSBOT_UCI_TO_IDX,
    CHESSBOT_VOCAB_SIZE,
    CHESSFENS_POLICY_SIZE,
    DEFAULT_99M_CHESSBOT_CONFIG,
    EXPECTED_99M_PARAMS,
    ChessBot99Config,
    average_recurrent_grads,
    build_chessbot99,
    count_parameters,
    fen_to_planes,
    flip_board_opposite_color,
    flip_planes_opposite_color,
    flip_policy_vector,
    move_to_policy_index,
    stm_wdl_to_chessbot,
)
from chess_inference import load_checkpoint
from experiments.exp287_chessbot_99m import (
    collate_rows,
    pad_policy,
    policy_losses,
    smoke,
    save_model,
    synthetic_rows,
)


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(287)
    return build_chessbot99(ChessBot99Config(
        d_model=32, d_ff=64, num_heads=4, dropout=0,
        prefix_layers=1, recurrent_layers=1, suffix_layers=1, recurrent_unrolls=2,
    ))


def chessbot_fen_to_tensor(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 19), dtype=np.float32)
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        tensor[7 - rank, file, piece_map[piece.symbol()]] = 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor[7 - rank, file, 13] = 1.0
    tensor[:, :, 14] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[:, :, 15] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[:, :, 16] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[:, :, 17] = float(board.has_queenside_castling_rights(chess.BLACK))
    tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    return tensor.reshape(64, 19)


def test_vocab_matches_chessbot_and_chessfens():
    assert CHESSBOT_VOCAB_SIZE == 1929
    assert CHESSFENS_POLICY_SIZE == 1858
    assert CHESSBOT_POLICY[0] == "a1b1"
    assert CHESSBOT_POLICY[1857] == "h7h8b"
    assert CHESSBOT_POLICY[1858] == "a2a1q"
    assert CHESSBOT_POLICY[-1] == "padding_token"
    assert move_to_policy_index(chess.Move.from_uci("e2e4")) == CHESSBOT_UCI_TO_IDX["e2e4"]


def test_planes_match_chessbot_fen_to_tensor():
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 12 20",
        "8/8/8/8/8/8/8/4K3 b - - 0 1",
    ]
    for fen in fens:
        ours = fen_to_planes(fen).numpy()
        theirs = chessbot_fen_to_tensor(fen)
        np.testing.assert_array_equal(ours[0], theirs)
    start = fen_to_planes(chess.STARTING_FEN)[0]
    assert start[0, 9] == 1.0  # token 0 is a8, black rook
    assert start[56, 3] == 1.0  # token 56 is a1, white rook
    assert float(start[:, 12].max()) == 1.0


def test_wdl_color_conversion():
    stm = torch.tensor([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]])
    turn = torch.tensor([True, False])
    out = stm_wdl_to_chessbot(stm, turn)
    torch.testing.assert_close(out[0], torch.tensor([0.1, 0.2, 0.7]))
    torch.testing.assert_close(out[1], torch.tensor([0.7, 0.2, 0.1]))


def test_rank_flip_maps_e2e4_and_inverts_turn():
    board = chess.Board()
    flipped = flip_board_opposite_color(board)
    assert flipped.turn == chess.BLACK
    assert flipped.piece_at(chess.E7).symbol() == "p"
    policy = torch.full((CHESSFENS_POLICY_SIZE,), -1.0)
    policy[CHESSBOT_UCI_TO_IDX["e2e4"]] = 1.0
    moved = flip_policy_vector(pad_policy(policy))
    assert CHESSBOT_POLICY[int(moved.argmax())] == "e7e5"
    torch.testing.assert_close(
        flip_planes_opposite_color(fen_to_planes(board.fen())),
        fen_to_planes(flipped.fen()),
    )


def test_tiny_forward_recurrence_and_grads():
    model = tiny().train()
    x = model.prepare_input(chess.Board(), torch.device("cpu"))
    a = model(x, recurrent_unrolls=2)
    b = model(x, recurrent_unrolls=3)
    assert a["policy_logits"].shape[-1] == 1929
    assert a["value_logits"].shape[-1] == 3
    assert not torch.allclose(a["policy_logits"], b["policy_logits"])
    loss = a["policy_logits"].square().mean()
    loss.backward()
    before = [p.grad.detach().abs().sum().item() for p in model.recurrent_parameters() if p.grad is not None]
    average_recurrent_grads(model, unrolls=2)
    after = [p.grad.detach().abs().sum().item() for p in model.recurrent_parameters() if p.grad is not None]
    assert after and sum(after) < sum(before)


def test_select_move_is_legal_and_checkpoint_reloads(tmp_path):
    model = tiny().eval()
    board = chess.Board()
    move, info = model.select_move(board, torch.device("cpu"), 0.0)
    assert move in board.legal_moves
    assert "wdl" in info
    save_model(model, tmp_path / "m.pt", {"test": True})
    restored = load_checkpoint(tmp_path / "m.pt", device="cpu")
    x = model.prepare_input(board, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(model(x)["policy_logits"], restored(x)["policy_logits"])


def test_soft_policy_ignores_illegal_minus_one():
    logits = torch.zeros(1, CHESSBOT_VOCAB_SIZE)
    idx = CHESSBOT_UCI_TO_IDX["e2e4"]
    logits[0, idx] = 4.0
    policy = torch.full((1, CHESSFENS_POLICY_SIZE), -1.0)
    policy[0, idx] = 1.0
    policy[0, 0] = -1.0
    loss, _, valid = policy_losses(logits, policy, soft_alpha=1.0)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([idx]))
    assert bool(valid.all())
    torch.testing.assert_close(loss, expected)


def test_collate_flip_keeps_batch_shapes():
    rows = synthetic_rows(4)
    planes, policy, wdl, turn, flips = collate_rows(rows, hflip_p=1.0, generator=torch.Generator().manual_seed(1))
    assert planes.shape == (4, 64, 19)
    assert policy.shape == (4, 1929)
    assert wdl.shape == (4, 3)
    assert turn.tolist() == [False] * 4
    assert all(flips)


def test_default_config_and_validate():
    cfg = DEFAULT_99M_CHESSBOT_CONFIG
    assert cfg.unique_layers == 15
    assert cfg.effective_depth == 29
    assert cfg.use_swiglu and cfg.use_qk_norm
    cfg.validate()
    with pytest.raises(ValueError):
        ChessBot99Config(num_heads=7).validate()


def test_smoke_tiny(tmp_path):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/exp287_chessbot_99m.json").read_text())
    result = smoke(cfg, tmp_path, torch.device("cpu"), tiny=True)
    assert (tmp_path / "latest.pt").exists()
    assert result["step"] == 2


def test_99m_param_count():
    n = count_parameters(build_chessbot99())
    assert n == EXPECTED_99M_PARAMS
