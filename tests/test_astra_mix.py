"""CPU tests for the Astra mix builder: shares, puzzle adapter, holdout exclusion."""
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

from build_astra_mix import (
    SHARES,
    SOURCE_PUZZLE,
    SOURCE_SF19,
    SUBSTANTIAL_TAGS,
    apply_value_valid,
    filter_substantial,
    priority_dedupe,
    puzzle_rating_bucket,
    record_to_tensors,
    shares_for_pool,
    soft_internal_counts,
    tag_source,
    _iter_mistake_shards,
)
from harvest_swa_mistakes import TAG_TO_I
from build_hf_elo_mix import filter_excluded, position_hashes
from exp193_puzzle_soft_harvest import puzzle_to_record
from move_vocab import UCI_TO_IDX
from autoresearch_8gb.pipeline import (
    PUZZLE_SOURCE,
    attach_static_targets,
    masked_mean_ce,
    value_valid_rows,
)


def test_shares_sum_to_pool():
    for n in (100, 4_000_000, 5_000_000):
        got = shares_for_pool(n)
        assert sum(got.values()) == n
        assert set(got) == set(SHARES)
        assert abs(got["lichess"] / n - SHARES["lichess"]) < 0.01


def test_soft_internal_is_eighty_percent():
    t = shares_for_pool(4_000_000)
    soft = soft_internal_counts(4_000_000)
    assert soft["lichess"] + soft["sf19"] + soft["puzzles"] == t["lichess"] + t["sf19"] + t["puzzles"]
    assert t["lichess"] + t["sf19"] + t["puzzles"] == int(round(4_000_000 * 0.80)) or abs(
        (t["lichess"] + t["sf19"] + t["puzzles"]) / 4_000_000 - 0.80
    ) < 0.001
    # With bonus=0.15 and deep=0.05, uniform soft draws recover 40/30/10.
    soft_n = t["lichess"] + t["sf19"] + t["puzzles"]
    assert abs(t["lichess"] / soft_n - 0.50) < 1e-6
    assert abs(t["sf19"] / soft_n - 0.375) < 1e-6
    assert abs(t["puzzles"] / soft_n - 0.125) < 1e-6


def test_puzzle_adapter_applies_opponent_setup_move():
    # Official Lichess puzzle FEN is before the opponent's setup move.
    puzzle = {
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4 e7e5",
        "Rating": 1400,
        "Themes": "opening",
        "PuzzleId": "test1",
    }
    rec = puzzle_to_record(puzzle, 0, 3000)
    assert rec is not None
    board = chess.Board(rec["fen"])
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.E4) is not None
    assert rec["best_move"] == "e7e5"
    assert rec["best_move"] in UCI_TO_IDX
    packed = record_to_tensors(rec)
    assert packed is not None
    assert int(packed["turn"]) == 1
    assert int(packed["move_idx"]) == UCI_TO_IDX["e7e5"]
    assert float(packed["soft_probs"][0]) == 1.0
    assert int(packed["value_valid"]) == 0
    assert int(packed["source"]) == SOURCE_PUZZLE == PUZZLE_SOURCE


def test_puzzle_adapter_rejects_too_short_line():
    puzzle = {
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4",
        "Rating": 1400,
        "Themes": "",
        "PuzzleId": "short",
    }
    assert puzzle_to_record(puzzle, 0, 3000) is None


def test_rating_buckets_are_balanced_and_not_hard_only():
    assert puzzle_rating_bucket(900) == 0
    assert puzzle_rating_bucket(1500) == 1
    assert puzzle_rating_bucket(1800) == 2
    assert puzzle_rating_bucket(2200) == 3
    assert puzzle_rating_bucket(2600) == 4
    assert puzzle_rating_bucket(100) is None


def test_filter_excluded_drops_blocked_and_keeps_fresh():
    def row(uci: str) -> dict:
        board = chess.Board()
        board.push_uci(uci)
        arr = [0] * 64
        for sq, piece in board.piece_map().items():
            arr[sq] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
        return {
            "board_array": torch.tensor([arr], dtype=torch.int8),
            "turn": torch.tensor([0 if board.turn else 1], dtype=torch.int8),
            "castling": torch.tensor([15], dtype=torch.int8),
            "ep_square": torch.tensor([0], dtype=torch.int8),
            "move_idx": torch.tensor([0], dtype=torch.int64),
            "cp": torch.tensor([0], dtype=torch.int32),
            "mate": torch.tensor([0], dtype=torch.int32),
            "soft_indices": torch.zeros((1, 8), dtype=torch.int64),
            "soft_probs": torch.zeros((1, 8), dtype=torch.float32),
        }

    a = row("e2e4")
    b = row("d2d4")
    data = {k: torch.cat([a[k], b[k]], dim=0) for k in a}
    blocked = {int(position_hashes(a)[0])}
    kept = filter_excluded(data, blocked)
    assert int(kept["board_array"].shape[0]) == 1
    assert int(kept["turn"][0]) == int(b["turn"][0])
    assert int(np.isin(position_hashes(kept), np.array(list(blocked), dtype=np.uint64)).sum()) == 0


def _row(uci: str) -> dict:
    board = chess.Board()
    board.push_uci(uci)
    arr = [0] * 64
    for sq, piece in board.piece_map().items():
        arr[sq] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
    return {
        "board_array": torch.tensor([arr], dtype=torch.int8),
        "turn": torch.tensor([0 if board.turn else 1], dtype=torch.int8),
        "castling": torch.tensor([15], dtype=torch.int8),
        "ep_square": torch.tensor([0], dtype=torch.int8),
        "move_idx": torch.tensor([0], dtype=torch.int64),
        "cp": torch.tensor([0], dtype=torch.int32),
        "mate": torch.tensor([0], dtype=torch.int32),
        "soft_indices": torch.zeros((1, 8), dtype=torch.int64),
        "soft_probs": torch.zeros((1, 8), dtype=torch.float32),
    }


def test_corrections_win_on_overlap():
    a = _row("e2e4")
    b = _row("d2d4")
    mistakes = {k: a[k].clone() for k in a}
    tag_source(mistakes, 3)
    ordinary = {k: torch.cat([a[k], b[k]], dim=0) for k in a}
    tag_source(ordinary, SOURCE_SF19)
    out, stats = priority_dedupe([("mistakes", mistakes), ("sf19", ordinary)])
    assert int(out["mistakes"]["board_array"].shape[0]) == 1
    assert int(out["sf19"]["board_array"].shape[0]) == 1
    assert stats["sf19"]["vs_seen"] == 1
    mh = int(position_hashes(out["mistakes"])[0])
    assert int(np.isin(position_hashes(out["sf19"]), np.array([mh], dtype=np.uint64)).sum()) == 0


def test_puzzle_value_is_masked():
    rec = puzzle_to_record({
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4 e7e5",
        "Rating": 1400,
        "Themes": "opening",
        "PuzzleId": "v0",
    }, 0, 3000)
    packed = record_to_tensors(rec)
    puz = {
        "board_array": packed["board_array"].unsqueeze(0),
        "turn": packed["turn"].view(1),
        "castling": packed["castling"].view(1),
        "ep_square": packed["ep_square"].view(1),
        "move_idx": packed["move_idx"].view(1),
        "cp": packed["cp"].view(1),
        "mate": packed["mate"].view(1),
        "soft_indices": packed["soft_indices"].unsqueeze(0),
        "soft_probs": packed["soft_probs"].unsqueeze(0),
        "phase": packed["phase"].view(1),
        "label_depth": packed["label_depth"].view(1),
        "source": packed["source"].view(1),
        "value_valid": packed["value_valid"].view(1),
    }
    attach_static_targets(puz)
    assert int(puz["value_valid"][0]) == 0
    logits = torch.zeros(1, 3)
    logits[0, 0] = 10.0
    wdl = torch.tensor([[0.0, 0.0, 1.0]])
    loss = masked_mean_ce(logits, wdl, value_valid_rows(puz, torch.tensor([0])))
    assert float(loss) == 0.0
    mixed_valid = torch.tensor([0, 1], dtype=torch.int8)
    logits2 = torch.zeros(2, 3)
    logits2[0, 0] = 10.0
    logits2[1, 2] = 10.0
    wdl2 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mixed = masked_mean_ce(logits2, wdl2, mixed_valid)
    only_real = masked_mean_ce(logits2[1:], wdl2[1:], None)
    assert abs(float(mixed) - float(only_real)) < 1e-5


def test_filter_substantial_drops_ok_and_off_pv():
    n = 4
    data = {
        "board_array": torch.zeros(n, 64, dtype=torch.int8),
        "tag": torch.tensor([
            TAG_TO_I["ok"],
            TAG_TO_I["inaccuracy"],
            TAG_TO_I["off_pv"],
            TAG_TO_I["blunder"],
        ], dtype=torch.int8),
        "drop_cp": torch.tensor([200, 90, 60, 180], dtype=torch.int32),
    }
    kept = filter_substantial(data)
    assert int(kept["tag"].shape[0]) == 2
    assert set(int(x) for x in kept["tag"].tolist()) <= set(SUBSTANTIAL_TAGS)
    assert TAG_TO_I["ok"] not in set(int(x) for x in kept["tag"].tolist())


def test_iter_mistake_shards_finds_analyzed_inbox():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        hidden = root / "analyzed" / "inbox" / "shard_000000"
        hidden.mkdir(parents=True)
        (hidden / "soft_cache.pt").write_bytes(b"x")
        (hidden / "READY").write_text("n=1\n")
        found = _iter_mistake_shards(root / "analyzed")
        assert found == [hidden / "soft_cache.pt"]
