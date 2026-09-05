"""CPU tests for exp201 lapse harvest encoding."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chess

from data_loader import CASTLING_MAP
from harvest_exp201_lapses import (
    MISTAKE_TAGS,
    board_row,
    classify_lapse,
    newest_stable_step,
    summarize_lapse_audit,
    tag_lapse,
    write_inbox_shard,
)


def test_board_row_uses_canonical_castling_and_ep():
    board = chess.Board()
    soft = {
        "ucis": ["e2e4"],
        "probs": [1.0],
        "best_uci": "e2e4",
        "best_cp": 30,
        "best_mate": 0,
        "depth": 12,
        "cps": [30],
    }
    row = board_row(board, soft)
    assert row is not None
    assert row["castling"] == CASTLING_MAP["K"] | CASTLING_MAP["Q"] | CASTLING_MAP["k"] | CASTLING_MAP["q"]
    assert row["castling"] == 15
    assert row["ep_square"] == -1
    assert row["turn"] == 0
    assert row["source"] == 3


def test_write_inbox_shard_ready_and_dedup():
    board = chess.Board()
    soft = {
        "ucis": ["e2e4"],
        "probs": [1.0],
        "best_uci": "e2e4",
        "best_cp": 30,
        "best_mate": 0,
        "depth": 12,
        "cps": [30],
    }
    row = board_row(board, soft)
    assert row is not None
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sh, seen, stats = write_inbox_shard([row, row], td, None)
        assert sh is not None
        assert (sh / "READY").exists()
        assert stats["n_out"] == 1
        assert stats["internal_dups"] == 1
        sh2, seen2, stats2 = write_inbox_shard([row], td, seen)
        assert sh2 is None
        assert stats2["n_out"] == 0
        assert stats2["vs_seen"] == 1
        assert seen2 is seen or int(seen2.size) == int(seen.size)


def test_tag_lapse_major_and_conversion():
    assert tag_lapse(drop=320, best_cp=40, model_cp=-280, best_mate=0, model_in_pv=True) == "major"
    assert tag_lapse(drop=210, best_cp=250, model_cp=40, best_mate=0, model_in_pv=True) == "conversion"
    assert tag_lapse(drop=160, best_cp=80, model_cp=-80, best_mate=0, model_in_pv=True) == "blunder"
    assert tag_lapse(drop=90, best_cp=40, model_cp=-50, best_mate=0, model_in_pv=True) == "inaccuracy"
    assert tag_lapse(drop=10, best_cp=40, model_cp=30, best_mate=0, model_in_pv=True) == "ok"
    assert "major" in MISTAKE_TAGS


def test_classify_lapse_separates_mates_from_cp():
    missed = classify_lapse(
        best_cp=99900, best_mate=3, model_cp=40, model_mate=0, model_in_pv=False,
    )
    assert missed["tag"] == "major"
    assert missed["kind"] == "missed_mate"
    assert missed["drop_cp"] is None
    allowed = classify_lapse(
        best_cp=30, best_mate=0, model_cp=-99900, model_mate=-2, model_in_pv=False,
    )
    assert allowed["kind"] == "allowed_mate"
    assert allowed["drop_cp"] is None
    cp = classify_lapse(
        best_cp=80, best_mate=0, model_cp=-80, model_mate=0, model_in_pv=False,
    )
    assert cp["kind"] == "cp"
    assert cp["drop_cp"] == 160
    assert cp["tag"] == "blunder"


def test_summarize_audit_ignores_synthetic_mate_drops():
    items = [
        {"tag": "major", "kind": "missed_mate", "drop_cp": None, "drop": 99900},
        {"tag": "blunder", "kind": "cp", "drop_cp": 180, "drop": 180},
        {"tag": "inaccuracy", "kind": "cp", "drop_cp": 90, "drop": 90},
    ]
    s = summarize_lapse_audit(items)
    assert s["n_mate_fail"] == 1
    assert s["n_cp"] == 2
    assert abs(s["mean_drop_cp"] - 135.0) < 1e-6
    assert s["major_rate"] == 1 / 3


def test_newest_stable_step_skips_fresh_and_latest():
    import time

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        old = td / "step_059000.pt"
        fresh = td / "step_059500.pt"
        (td / "latest.pt").write_bytes(b"x")
        old.write_bytes(b"a")
        fresh.write_bytes(b"b")
        os.utime(old, (time.time() - 120, time.time() - 120))
        os.utime(fresh, (time.time(), time.time()))
        got = newest_stable_step(td, min_age_s=45.0, min_step=0)
        assert got == old
        got2 = newest_stable_step(td, min_age_s=45.0, min_step=59000)
        assert got2 is None


if __name__ == "__main__":
    test_board_row_uses_canonical_castling_and_ep()
    test_write_inbox_shard_ready_and_dedup()
    test_tag_lapse_major_and_conversion()
    test_classify_lapse_separates_mates_from_cp()
    test_summarize_audit_ignores_synthetic_mate_drops()
    test_newest_stable_step_skips_fresh_and_latest()
    print("ok")
