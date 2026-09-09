"""CPU tests for chess_master identity, contracts, recipes, and resume."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import chess
import numpy as np
import torch

from chess_master.contracts import (
    depth_sentinel,
    legal_policy,
    lichess_cp_is_mate_sentinel,
    qualify_source,
    regret_status,
    syzygy_meta_ok,
    value_wdl_ok,
)
from chess_master.identity import (
    equivalence_key,
    fen_4,
    namespace_game_id,
    position_id,
    position_record,
)
from chess_master.io_util import json_write
from chess_master.recipes import load_recipe, overlap_assignment, requested_counts
from chess_master.rows import from_lichess, from_puzzle_packed, from_sf19, from_syzygy
from chess_master.schema import (
    ANNOTATION_ENGINE_POLICY,
    ANNOTATION_PUZZLE,
    ANNOTATION_TABLEBASE,
    PERSPECTIVE_UNKNOWN,
    PERSPECTIVE_WHITE,
    POLICY_PROBS,
    UNKNOWN,
)
from data_loader import _fast_parse_fen
from exp193_puzzle_soft_harvest import puzzle_to_record
from move_vocab import UCI_TO_IDX


def _encode(board: chess.Board):
    arr = np.zeros(64, dtype=np.int8)
    turn, castling, ep = _fast_parse_fen(board.fen(), arr)
    return arr, int(turn), int(castling), int(ep if ep is not None else -1)


def _soft(uci: str):
    mid = UCI_TO_IDX[uci]
    si = np.full(8, -1, dtype=np.int64)
    sp = np.zeros(8, dtype=np.float32)
    si[0] = mid
    sp[0] = 1.0
    return mid, si, sp


def _batch(board: chess.Board, uci: str, **extra):
    arr, turn, castle, ep = _encode(board)
    mid, si, sp = _soft(uci)
    n = 1
    d = {
        "board_array": arr.reshape(n, 64),
        "turn": np.array([turn], dtype=np.int8),
        "castling": np.array([castle], dtype=np.int8),
        "ep_square": np.array([ep], dtype=np.int8),
        "move_idx": np.array([mid], dtype=np.int64),
        "cp": np.array([extra.get("cp", 20)], dtype=np.int32),
        "mate": np.array([extra.get("mate", 0)], dtype=np.int32),
        "soft_indices": si.reshape(n, 8),
        "soft_probs": sp.reshape(n, 8),
        "label_depth": np.array([extra.get("depth", 16)], dtype=np.int16),
        "phase": np.array([1], dtype=np.int8),
    }
    for k in ("wdl", "tau", "nodes_budget", "nodes", "split", "policy_mask", "bound_skipped", "flags"):
        if k in extra:
            d[k] = extra[k]
    return d


def test_board_identity_excludes_clocks():
    b = chess.Board()
    b.push_uci("e2e4")
    arr, turn, castle, ep = _encode(b)
    a = position_id(arr, turn, castle, ep)
    rec = position_record(arr, turn, castle, ep, halfmove=4, fullmove=2)
    assert rec["position_id"] == a
    assert rec["fen_6"].endswith(" 4 2")
    assert rec["rule_state_available"] is True
    no_clocks = position_record(arr, turn, castle, ep)
    assert no_clocks["position_id"] == a
    assert no_clocks["fen_6"] is None
    assert no_clocks["rule_state_available"] is False
    assert no_clocks["halfmove"] is None


def test_equivalence_only_hflip_without_castling():
    b = chess.Board()
    arr, turn, castle, ep = _encode(b)
    key, method = equivalence_key(arr, turn, castle, ep)
    assert method == "identity"
    assert key.startswith("exact:")
    b2 = chess.Board("8/8/8/8/8/8/4P3/4K3 w - - 0 1")
    arr2, turn2, castle2, ep2 = _encode(b2)
    key2, method2 = equivalence_key(arr2, turn2, castle2, ep2)
    assert method2 == "hflip_no_castling"
    assert key2.startswith("hflip64:")


def test_multiple_annotations_share_position():
    b = chess.Board()
    b.push_uci("e2e4")
    d = _batch(b, "d7d5", wdl=np.array([[0.2, 0.6, 0.2]], dtype=np.float32), tau=np.array([120.0], dtype=np.float32),
               nodes_budget=np.array([100000], dtype=np.int32), nodes=np.array([100100], dtype=np.int32),
               split=np.array([0], dtype=np.int8), policy_mask=np.array([1], dtype=np.int8))
    pos_a, sf = from_sf19(d, 0, source_path="s.parquet", revision="abc", extra={"wdl": [0.2, 0.6, 0.2], "split": 0})
    pos_b, lich = from_lichess(d, 0, source_path="l.parquet", revision="def")
    assert pos_a["position_id"] == pos_b["position_id"]
    assert sf["annotation_id"] != lich["annotation_id"]
    assert sf["annotation_type"] == ANNOTATION_ENGINE_POLICY
    assert sf["value_perspective"] == PERSPECTIVE_WHITE
    assert lich["value_perspective"] == PERSPECTIVE_UNKNOWN
    assert lich["value_eligible"] == 0
    assert sf["policy_kind"] == POLICY_PROBS
    assert "logit" not in (sf["policy_transform"] or "").lower()


def test_puzzle_applies_setup_move_and_keeps_line():
    from build_organized_chess_mix import pack_puzzle

    puzzle = {
        "FEN": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "Moves": "e2e4 e7e5",
        "Rating": 1400,
        "Themes": ["opening"],
        "PuzzleId": "cst1",
        "GameId": "gameABC#12",
    }
    rec = puzzle_to_record(puzzle, 600, 3500)
    packed, meta = pack_puzzle(puzzle)
    d = {k: v.unsqueeze(0) for k, v in packed.items()}
    pos, ann = from_puzzle_packed(d, 0, meta, source_path="p.parquet", revision="r", source_row=0)
    assert rec["best_move"] == "e7e5"
    assert ann["best_uci"] == "e7e5"
    assert ann["puzzle_moves"] == "e2e4 e7e5"
    assert ann["puzzle_setup_fen"].startswith("r3k2r/")
    assert chess.Board(ann["puzzle_solver_fen"]).fen().startswith("r3k2r/pppppppp/8/8/4P3/8/PPPP1PPP/R3K2R")
    assert ann["annotation_type"] == ANNOTATION_PUZZLE
    assert int(ann["value_eligible"]) == 0
    assert pos["legality_status"] == "legal"


def test_syzygy_dtz_is_not_mate():
    b = chess.Board("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1")
    d = _batch(b, "e1d1", cp=0, mate=6, depth=999)
    d["wdl"] = np.array([2], dtype=np.int8)
    d["dtz"] = np.array([12], dtype=np.int16)
    pos, ann = from_syzygy(d, 0, source_path="z.pt", revision="r", extra={"tb_wdl": 2, "dtz": 12})
    assert ann["annotation_type"] == ANNOTATION_TABLEBASE
    assert int(ann["mate_is_dtz_proxy"]) == 1
    assert int(ann["value_eligible"]) == 0
    assert int(ann["tb_wdl"]) == 2
    assert int(ann["tb_dtz"]) == 12
    assert ann["depth_is_sentinel"] is True
    ok, reason = syzygy_meta_ok(2, 12)
    assert ok and reason == "ok"
    assert not syzygy_meta_ok(3, 0)[0]


def test_unknown_regret_when_move_not_in_pv():
    cp, status = regret_status(0, 80)
    assert cp is None and status == "unknown_off_pv"
    cp2, status2 = regret_status(1, 40)
    assert cp2 == 40 and status2 == "verified"
    cp3, status3 = regret_status(None, 99)
    assert cp3 is None and status3 == UNKNOWN


def test_depth_sentinels_and_lichess_cp():
    assert depth_sentinel("syzygy", 999)[0]
    assert depth_sentinel("lichess", 128)[0]
    assert depth_sentinel("sf19", 245)[0]
    assert not depth_sentinel("sf19", 14)[0]
    assert lichess_cp_is_mate_sentinel(90000, 0)
    assert not lichess_cp_is_mate_sentinel(40, 0)


def test_qualify_never_passes_upstream_holdout_or_bad_policy():
    b = chess.Board()
    b.push_uci("e2e4")
    d = _batch(b, "d7d5", depth=14, wdl=np.array([[0.2, 0.6, 0.2]]))
    fields = {
        "board_array": d["board_array"][0], "turn": d["turn"][0], "castling": d["castling"][0],
        "ep_square": d["ep_square"][0], "move_idx": d["move_idx"][0],
        "soft_indices": d["soft_indices"][0], "soft_probs": d["soft_probs"][0],
        "split": 1, "policy_mask": 1, "nodes_budget": 100000, "label_depth": 14,
        "wdl": [0.2, 0.6, 0.2],
    }
    assert qualify_source("sf19", fields) == "upstream_holdout"
    fields["split"] = 0
    fields["nodes_budget"] = 1000
    assert qualify_source("sf19", fields) == "teacher_quality"
    fields["nodes_budget"] = 100000
    fields["soft_probs"] = np.array([0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert qualify_source("sf19", fields) == "invalid_policy"


def test_recipe_overlap_and_no_relax():
    recipe = load_recipe("pilot_45_35_15_5")
    assert recipe["quality_relaxation"] == "never"
    counts = requested_counts(recipe)
    assert counts["sf19"] == 450000
    assert counts["lichess"] == 350000
    assert counts["puzzles"] == 150000
    assert counts["syzygy"] == 50000
    assert overlap_assignment(["puzzles", "sf19"], "exclusive_source_bucket", ["sf19", "lichess", "puzzles"]) == "sf19"
    assert namespace_game_id("puzzles", "abc#1")[1] == "lichess-game:abc#1"
    assert namespace_game_id("lichess", None)[1].endswith("unknown")


def test_writer_resume_does_not_duplicate_positions(tmp_path: Path):
    from chess_master.ingest import Writer
    from chess_master.schema import POSITIONS_SCHEMA

    b = chess.Board()
    arr, turn, castle, ep = _encode(b)
    pos = position_record(arr, turn, castle, ep)
    w = Writer(tmp_path, "mix")
    w.add_position(pos)
    w.close()
    w2 = Writer(tmp_path, "mix")
    w2.load_seen()
    w2.add_position(pos)
    w2.close()
    files = list((tmp_path / "positions").glob("*.parquet"))
    assert files
    import pyarrow.parquet as pq
    n = sum(pq.read_table(p).num_rows for p in files)
    assert n == 1


def test_value_ok_separate_from_policy():
    assert value_wdl_ok([0.1, 0.2, 0.7])
    assert not value_wdl_ok([0.2, 0.2, 0.2])
    b = chess.Board()
    b.push_uci("e2e4")
    arr, turn, castle, ep = _encode(b)
    mid, si, sp = _soft("d7d5")
    ok, why = legal_policy(arr, turn, castle, ep, mid, si, sp)
    assert ok and why == "ok"
    ok2, why2 = legal_policy(arr, turn, castle, ep, UCI_TO_IDX["e2e4"], si, sp)
    assert not ok2
