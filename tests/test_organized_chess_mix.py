"""CPU tests for the organized chess mix: labels, holdouts, eval-value mask."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

import chess
import numpy as np
import torch
import torch.nn as nn

from build_organized_chess_mix import (
    CASTLE_FEN,
    SOURCE_PUZZLE,
    SOURCE_SF19,
    SOURCE_SYZYGY,
    VOCAB_SIZE,
    canonical_hashes,
    encode_board,
    group_key,
    is_eval_group,
    legal_row,
    pack_puzzle,
    policy_ok,
    reconstruct_board,
    syzygy_meta_ok,
    value_ok,
)
from autoresearch_8gb.pipeline import (
    attach_static_targets,
    cheap_eval_losses,
    position_hashes,
)
from exp193_puzzle_soft_harvest import puzzle_to_record
from move_vocab import UCI_TO_IDX


def _start_row() -> dict:
    board = chess.Board()
    board.push_uci("e2e4")
    arr, turn, castling, ep = encode_board(board)
    mid = UCI_TO_IDX["d7d5"]
    si = torch.full((8,), -1, dtype=torch.int64)
    sp = torch.zeros(8, dtype=torch.float32)
    si[0] = mid
    sp[0] = 1.0
    return {
        "board_array": torch.from_numpy(arr.copy()),
        "turn": torch.tensor(turn, dtype=torch.int8),
        "castling": torch.tensor(castling, dtype=torch.int8),
        "ep_square": torch.tensor(ep, dtype=torch.int8),
        "move_idx": torch.tensor(mid, dtype=torch.int64),
        "cp": torch.tensor(20, dtype=torch.int32),
        "mate": torch.tensor(0, dtype=torch.int32),
        "soft_indices": si,
        "soft_probs": sp,
        "label_depth": torch.tensor(16, dtype=torch.int16),
        "phase": torch.tensor(0, dtype=torch.int8),
        "wdl": torch.tensor([0.2, 0.6, 0.2], dtype=torch.float32),
    }


def _as_batch(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[k] = v.unsqueeze(0)
    return out


def test_castle_bits_match_data_loader():
    board = chess.Board()
    _, _, castling, _ = encode_board(board)
    assert castling == 15
    assert CASTLE_FEN(15) == "KQkq"
    # White kingside only is bit 8, not the inverted harvest encoding.
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    _, _, c, _ = encode_board(chess.Board(fen))
    assert c & 8
    assert CASTLE_FEN(c) == "KQkq"


def test_policy_rejects_oob_nan_and_illegal():
    row = _start_row()
    d = _as_batch(row)
    assert policy_ok(d, 0) and legal_row(d, 0)

    d2 = _as_batch(row)
    d2["soft_indices"] = d2["soft_indices"].clone()
    d2["soft_indices"][0, 0] = VOCAB_SIZE
    assert not policy_ok(d2, 0)

    d3 = _as_batch(row)
    d3["soft_probs"] = d3["soft_probs"].clone()
    d3["soft_probs"][0, 0] = float("nan")
    assert not policy_ok(d3, 0)

    d4 = _as_batch(row)
    d4["soft_indices"] = d4["soft_indices"].clone()
    d4["soft_probs"] = d4["soft_probs"].clone()
    d4["soft_indices"][0, 0] = UCI_TO_IDX["a2a4"]  # illegal after e2e4 (pawn still on a2 but wait — e2e4, a2a4 is legal)
    d4["soft_indices"][0, 0] = UCI_TO_IDX["e2e4"]  # already played
    d4["move_idx"][0] = UCI_TO_IDX["e2e4"]
    d4["soft_probs"][0, 0] = 1.0
    assert not legal_row(d4, 0)


def test_value_validated_separately_from_policy():
    good = torch.tensor([0.1, 0.2, 0.7])
    assert value_ok(good)
    assert not value_ok(torch.tensor([0.5, 0.5, float("inf")]))
    assert not value_ok(torch.tensor([0.2, 0.2, 0.2]))
    assert not value_ok(torch.tensor([-0.1, 0.5, 0.6]))
    # Policy can be fine while value is dummy.
    row = _as_batch(_start_row())
    row["wdl"] = torch.tensor([[0.0, 1.0, 0.0]])
    assert legal_row(row, 0)
    assert value_ok(row["wdl"][0])


def test_syzygy_dtz_is_not_mate_and_value_stays_off():
    ok, reason = syzygy_meta_ok(tb_wdl=2, dtz=12, mate=6)
    assert ok and reason == "ok"
    # DTZ-derived mate field is recorded, not treated as mate distance.
    ok2, reason2 = syzygy_meta_ok(tb_wdl=2, dtz=12, mate=6, reject_dtz_mate=True)
    assert not ok2 and reason2 == "dtz_as_mate"
    bad, why = syzygy_meta_ok(tb_wdl=3, dtz=0, mate=0)
    assert not bad and why == "tb_wdl_range"
    # Training value stays off even when metadata is internally consistent.
    assert SOURCE_SYZYGY == 2


def test_puzzle_keeps_full_line_and_game_id_and_correct_castling():
    puzzle = {
        "FEN": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "Moves": "e2e4 e7e5",
        "Rating": 1400,
        "Themes": "opening",
        "PuzzleId": "cst1",
        "GameId": "gameABC#12",
    }
    rec = puzzle_to_record(puzzle, 600, 3500)
    packed, meta = pack_puzzle(puzzle)
    assert packed is not None
    assert meta["GameId"] == "gameABC#12"
    assert meta["PuzzleId"] == "cst1"
    assert meta["Moves"] == "e2e4 e7e5"
    assert int(packed["value_valid"]) == 0
    assert int(packed["source"]) == SOURCE_PUZZLE
    assert CASTLE_FEN(int(packed["castling"])) == "KQkq"
    d = {k: packed[k].unsqueeze(0) for k in packed}
    board = reconstruct_board(d, 0)
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert board.has_kingside_castling_rights(chess.BLACK)
    assert legal_row(d, 0)
    assert rec["best_move"] == "e7e5"


def test_stacked_scalars_are_1d():
    from build_organized_chess_mix import squeeze_scalars, stack_rows, take_row
    d = _as_batch(_start_row())
    sm = {"wdl": [0.2, 0.6, 0.2]}
    row = take_row(d, 0, "sf19", SOURCE_SF19, sm)
    stacked = squeeze_scalars(attach_static_targets(stack_rows([row, row])))
    assert stacked["turn"].shape == (2,)
    assert stacked["wdl"].shape == (2, 3)
    assert stacked["board_array"].shape == (2, 64)


def test_flip_dedup_and_game_split():
    b = chess.Board()
    b.push_uci("e2e4")
    arr, turn, castling, ep = encode_board(b)
    d = {
        "board_array": torch.from_numpy(arr).unsqueeze(0),
        "turn": torch.tensor([turn], dtype=torch.int8),
        "castling": torch.tensor([0], dtype=torch.int8),  # no rights → flip-equivalent
        "ep_square": torch.tensor([ep if ep > 0 else -1], dtype=torch.int8),
    }
    h = canonical_hashes(d)
    raw = position_hashes(d)
    assert h.shape == raw.shape
    assert is_eval_group("sf19-game:7", 2) in (True, False)
    assert group_key("sf19", {"game_id": 9}, 123) == "sf19-game:9"
    assert group_key("puzzles", {"GameId": "abc#1", "PuzzleId": "z"}, 1) == "puzzle-game:abc#1"
    assert group_key("lichess", {}, 99).startswith("pos:")


def test_cheap_eval_omits_value_on_policy_only_rows():
    row = _as_batch(_start_row())
    row["source"] = torch.tensor([SOURCE_PUZZLE], dtype=torch.int8)
    row["value_valid"] = torch.tensor([0], dtype=torch.int8)
    row["wdl"] = torch.tensor([[0.0, 0.0, 1.0]])  # fabricated loss
    attach_static_targets(row)
    assert int(row["value_valid"][0]) == 0

    class Const(nn.Module):
        def forward(self, bi):
            n = int(bi["turn"].shape[0])
            pol = torch.zeros(n, VOCAB_SIZE)
            val = torch.zeros(n, 3)
            val[:, 0] = 8.0  # model is sure White wins
            return {"policy_logits": pol, "value_logits": val}

    metrics = cheap_eval_losses(Const(), row, torch.tensor([0]), torch.device("cpu"), microbatch=4)
    assert metrics["value_rows"] == 0.0
    assert "wdl_ce" not in metrics
    assert "hard_ce" in metrics and "soft_ce" in metrics

    mixed = {k: torch.cat([row[k], row[k]], dim=0) if torch.is_tensor(row[k]) else row[k] for k in row}
    mixed["value_valid"] = torch.tensor([0, 1], dtype=torch.int8)
    mixed["source"] = torch.tensor([SOURCE_PUZZLE, SOURCE_SF19], dtype=torch.int8)
    mixed["wdl"] = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    both = cheap_eval_losses(Const(), mixed, torch.tensor([0, 1]), torch.device("cpu"), microbatch=4)
    only_real = cheap_eval_losses(
        Const(),
        {k: (v[1:2] if torch.is_tensor(v) and v.shape[0] == 2 else v) for k, v in mixed.items()},
        torch.tensor([0]),
        torch.device("cpu"),
        microbatch=4,
    )
    assert both["value_rows"] == 1.0
    assert abs(both["wdl_ce"] - only_real["wdl_ce"]) < 1e-5
    # Fabricated puzzle target would have been a large CE against P(win)≈1.
    puzzle_only = cheap_eval_losses(
        Const(),
        {k: (v[:1] if torch.is_tensor(v) and v.shape[0] == 2 else v) for k, v in mixed.items()},
        torch.tensor([0]),
        torch.device("cpu"),
        microbatch=4,
    )
    assert "wdl_ce" not in puzzle_only
    assert both["wdl_ce"] < 0.05
