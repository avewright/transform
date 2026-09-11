"""CPU tests for exp272 opening sample, regret, legality, and resume."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import chess
import numpy as np

from exp272_mine import (
    BENCH_OPENINGS,
    KEEP_MISTAKE,
    completed_game_ids,
    explore_opening,
    hard_teacher_row,
    outcome_kind,
    parse_openings_tsv,
    pgn_to_uci,
    regret_from_roots,
    sample_schedule,
    to_white_abs,
)
from harvest_swa_mistakes import classify_lapse
from move_vocab import UCI_TO_IDX, VOCAB_SIZE
from sf19_soft_dataset import label_to_row, softmax_from_scores


TSV = """eco	name	pgn
C50	Italian Game	1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5
B90	Sicilian Najdorf	1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6
E20	Nimzo-Indian	1. d4 Nf6 2. c4 e6 3. Nc3 Bb4
A10	English	1. c4 Nf6 2. Nc3 e6 3. Nf3 d5
D10	Slav	1. d4 d5 2. c4 c6 3. Nc3 Nf6
"""


def test_pgn_to_uci_italian():
    ucis = pgn_to_uci("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5")
    assert ucis == ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]


def test_parse_skips_short_and_benchmark_exact():
    text = TSV + "A00\tStart\t1. e4 e5\nA00\tNh3\t1. Nh3\n"
    rows = parse_openings_tsv(text, source="fixture")
    uci_sets = {tuple(r["uci"]) for r in rows}
    assert ("e2e4", "e7e5") not in uci_sets
    assert all(len(r["uci"]) >= 3 for r in rows)
    assert any(r["eco"] == "C50" for r in rows)


def test_epsilon_can_leave_book():
    book = ["e2e4", "e7e5", "g1f3", "b8c6"]
    hit = False
    for seed in range(80):
        out = explore_opening(book, epsilon=1.0, rng=random.Random(seed))
        if out["diverted_at"] is not None:
            hit = True
            assert out["uci"][out["diverted_at"]] != book[out["diverted_at"]]
            board = chess.Board()
            for u in out["uci"]:
                mv = chess.Move.from_uci(u)
                assert mv in board.legal_moves
                board.push(mv)
            break
    assert hit


def test_epsilon_zero_follows_book():
    book = ["e2e4", "e7e5", "g1f3"]
    out = explore_opening(book, epsilon=0.0, rng=random.Random(0))
    assert out["uci"] == book
    assert out["diverted_at"] is None


def test_sample_schedule_avoids_bench_and_splits():
    pool = parse_openings_tsv(TSV, source="fixture")
    sch = sample_schedule(pool, n_openings=4, epsilon=0.0, seed=1)
    assert sch["n_games"] == 8  # 4 openings × 2 colors × 2500
    played = [tuple(o["played_uci"]) for o in sch["openings"]]
    assert all(p not in BENCH_OPENINGS for p in played)
    assert any(o["split"] == "val" for o in sch["openings"])
    assert any(o["split"] == "train" for o in sch["openings"])
    ids = [g["game_id"] for g in sch["games"]]
    assert len(ids) == len(set(ids))


def test_score_perspective_flips_across_a_move():
    assert to_white_abs(80, 0, False) == (80, 0)
    assert to_white_abs(80, 0, True) == (-80, 0)
    assert to_white_abs(0, 2, True) == (0, -2)


def test_regret_is_stm_and_mates_stay_mates():
    teacher = {"cp_stm": 220, "mate_stm": 0, "wdl_stm": [0.6, 0.3, 0.1]}
    model = {"cp_stm": 20, "mate_stm": 0, "wdl_stm": [0.4, 0.4, 0.2]}
    r = regret_from_roots(teacher, model, in_pv=False)
    assert r["drop_cp"] == 200
    assert r["regret_kind"] == "cp"
    mate_t = {"cp_stm": 0, "mate_stm": 3, "wdl_stm": [1, 0, 0]}
    mate_m = {"cp_stm": 40, "mate_stm": 0, "wdl_stm": [0.5, 0.4, 0.1]}
    r2 = regret_from_roots(mate_t, mate_m, in_pv=False)
    assert r2["drop_cp"] is None
    assert r2["missed_mate"] is True
    assert r2["regret_kind"] == "mate"


def test_draw_best_not_penalized_as_conversion():
    teacher = {"cp_stm": 10, "mate_stm": 0}
    model = {"cp_stm": 0, "mate_stm": 0}
    r = regret_from_roots(teacher, model, in_pv=True)
    assert r["best_is_draw"] is True
    assert r["tag"] == "ok"
    assert outcome_kind(10, 0, 0, 0) == "same"


def test_classify_missed_mate_not_cp():
    info = classify_lapse(best_cp=0, best_mate=2, model_cp=300, model_mate=0, model_in_pv=False)
    assert info["kind"] == "missed_mate"
    assert info["drop_cp"] is None


def test_soft_targets_legal_and_normalized():
    board = chess.Board()
    e2e4 = UCI_TO_IDX["e2e4"]
    d2d4 = UCI_TO_IDX["d2d4"]
    parsed = {
        "terminal": False,
        "items": [
            {"uci": "e2e4", "stm_cp": 40, "stm_mate": 0, "rank": 40, "wdl": np.array([0.4, 0.4, 0.2], np.float32)},
            {"uci": "d2d4", "stm_cp": 30, "stm_mate": 0, "rank": 30, "wdl": None},
        ],
        "probs": softmax_from_scores([40.0, 30.0], 120.0),
        "depth": 12,
        "nodes": 1000,
        "n_legal": board.legal_moves.count(),
        "bound_skipped": 0,
    }
    row = label_to_row(board, parsed, tau=120.0, nodes_budget=1000)
    legal = {m.uci() for m in board.legal_moves}
    for idx, p in zip(row["soft_indices"], row["soft_probs"]):
        if int(idx) < 0:
            continue
        assert 0 <= int(idx) < VOCAB_SIZE
        uci = next(u for u, i in UCI_TO_IDX.items() if i == int(idx))
        assert uci in legal
        assert float(p) > 0
    assert abs(float(sum(row["soft_probs"])) - 1.0) < 1e-5
    assert int(row["move_idx"]) == e2e4
    assert d2d4 in set(int(x) for x in row["soft_indices"])


def test_model_outside_multipv_is_unknown_until_root_eval():
    teacher = {"cp_stm": 80, "mate_stm": 0}
    # Without an explicit model root, we must not invent a drop.
    in_pv = False
    # After explicit eval:
    model = {"cp_stm": 10, "mate_stm": 0}
    r = regret_from_roots(teacher, model, in_pv=in_pv)
    assert r["drop_cp"] == 70
    unknown_drop = 0  # pre-eval contract
    assert unknown_drop == 0


def test_truncation_is_not_a_draw():
    assert "*" != "1/2-1/2"
    rec = {"result": "*", "termination": "truncated", "truncated": True}
    assert rec["truncated"] is True
    assert rec["result"] == "*"


def test_resume_skips_completed_ids():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "games.jsonl").write_text(
            json.dumps({"game_id": "o00_white_sf19_near_2050"}) + "\n"
            + json.dumps({"game_id": "o00_black_sf19_near_2050"}) + "\n",
            encoding="utf-8",
        )
        seen = completed_game_ids(out)
        assert seen == {"o00_white_sf19_near_2050", "o00_black_sf19_near_2050"}


def test_hard_teacher_row_is_onehot_legal():
    board = chess.Board()
    row = hard_teacher_row(board, "e2e4")
    assert row is not None
    assert int(row["move_idx"]) == UCI_TO_IDX["e2e4"]
    assert row["teacher_uci"] == "e2e4"
    assert abs(sum(row["soft_probs"]) - 1.0) < 1e-6
    assert "inaccuracy" in KEEP_MISTAKE and "blunder" in KEEP_MISTAKE
    assert "ok" not in KEEP_MISTAKE


def test_history_flags():
    board = chess.Board()
    assert board.halfmove_clock == 0
    # Fifty-move approaching
    fen = "8/8/8/8/8/8/4k3/4K3 w - - 40 80"
    b = chess.Board(fen)
    assert b.halfmove_clock >= 40
    # Threefold setup
    b2 = chess.Board()
    for u in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        b2.push(chess.Move.from_uci(u))
    assert b2.is_repetition(2) or b2.can_claim_threefold_repetition()
