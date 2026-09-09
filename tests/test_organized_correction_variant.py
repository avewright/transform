"""CPU tests for the SF19→correction swap. Does not download HF or touch v1."""
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

import numpy as np
import torch

from build_organized_chess_mix import SOURCE_SF19, encode_board, json_write
from build_organized_correction_variant import (
    SOURCE_MISTAKE,
    freeze_baseline,
    replace_sf19,
    select_corrections,
    wdl_valid_mask,
)
from harvest_swa_mistakes import TAG_TO_I
from move_vocab import UCI_TO_IDX
import chess


def _row(uci="d7d5", after="e2e4") -> dict:
    board = chess.Board()
    board.push_uci(after)
    arr, turn, castling, ep = encode_board(board)
    mid = UCI_TO_IDX[uci]
    si = torch.full((8,), -1, dtype=torch.int64)
    sp = torch.zeros(8, dtype=torch.float32)
    si[0] = mid
    sp[0] = 1.0
    return {
        "board_array": torch.from_numpy(arr.copy()),
        "turn": torch.tensor(turn, dtype=torch.int8),
        "castling": torch.tensor(castling, dtype=torch.int8),
        "ep_square": torch.tensor(ep if ep > 0 else -1, dtype=torch.int8),
        "move_idx": torch.tensor(mid, dtype=torch.int64),
        "cp": torch.tensor(80, dtype=torch.int32),
        "mate": torch.tensor(0, dtype=torch.int32),
        "soft_indices": si,
        "soft_probs": sp,
        "label_depth": torch.tensor(16, dtype=torch.int16),
        "phase": torch.tensor(0, dtype=torch.int8),
        "source": torch.tensor(SOURCE_MISTAKE, dtype=torch.int8),
        "wdl": torch.tensor([0.3, 0.5, 0.2], dtype=torch.float32),
        "value_valid": torch.tensor(1, dtype=torch.int8),
    }


def _stack(rows: list[dict], **extra) -> dict:
    keys = rows[0].keys()
    out = {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}
    out.update(extra)
    return out


def test_wdl_mask_rejects_dummy_and_nan():
    good = torch.tensor([[0.2, 0.6, 0.2], [0.0, 1.0, 0.0]])
    assert wdl_valid_mask(good).tolist() == [True, True]
    bad = torch.tensor([[0.2, 0.2, 0.2], [float("nan"), 0.5, 0.5]])
    assert wdl_valid_mask(bad).tolist() == [False, False]


def test_select_drops_ok_blocked_and_overlap():
    good = _row()
    ok_row = _row()
    blocked_row = _row()
    overlap_row = _row()
    data = _stack(
        [good, ok_row, blocked_row, overlap_row],
        tag=torch.tensor([
            TAG_TO_I["blunder"], TAG_TO_I["ok"],
            TAG_TO_I["major"], TAG_TO_I["inaccuracy"],
        ], dtype=torch.int8),
        drop_cp=torch.tensor([400, 12, 90, 200], dtype=torch.int32),
    )
    from build_organized_chess_mix import canonical_hashes
    hs = canonical_hashes(data)
    # Same start position → same hash. Distinguish by mutating boards for 2/3.
    data["board_array"] = data["board_array"].clone()
    data["board_array"][2, 8] = 4  # illegal-ish piece tweak for a unique hash
    data["board_array"][3, 9] = 4
    hs = canonical_hashes(data)
    kept, report = select_corrections(
        data, n_take=1, blocked=np.asarray([hs[2]], dtype=np.uint64),
        occupied=np.asarray([hs[3]], dtype=np.uint64),
    )
    assert int(kept["turn"].shape[0]) == 1
    assert int(kept["source"][0]) == SOURCE_MISTAKE
    assert report["rejected"]["not_substantial"] == 1
    assert report["rejected"]["blocked"] >= 1
    assert report["rejected"]["overlap_baseline"] >= 1
    assert report["tags"]["blunder"] == 1
    assert int(kept["value_valid"][0]) == 1


def test_replace_keeps_sf19_family_share():
    sf_rows = [_row() for _ in range(4)]
    for r in sf_rows:
        r["source"] = torch.tensor(SOURCE_SF19, dtype=torch.int8)
    corr_rows = [_row() for _ in range(2)]
    sf19 = _stack(sf_rows)
    corr = _stack(corr_rows)
    mixed = replace_sf19(sf19, corr, n_keep_sf19=2, seed=7)
    assert int(mixed["turn"].shape[0]) == 4
    assert int((mixed["source"] == SOURCE_SF19).sum()) == 2
    assert int((mixed["source"] == SOURCE_MISTAKE).sum()) == 2


def test_freeze_is_idempotent_and_blocks_rebuild(tmp_path: Path):
    (tmp_path / "dummy.pt").write_bytes(b"x")
    json_write(tmp_path / "manifest.json", {
        "status": "complete",
        "seed": 1,
        "actual_counts": {"sf19": 1},
    })
    first = freeze_baseline(tmp_path)
    assert (tmp_path / "FROZEN.json").exists()
    man = (tmp_path / "manifest.json").read_text()
    assert '"status": "frozen"' in man
    second = freeze_baseline(tmp_path)
    assert second["frozen_at"] == first["frozen_at"]

    from build_organized_chess_mix import build
    import argparse
    args = argparse.Namespace(
        output=str(tmp_path), rows=10, eval_rows=1, seed=1, force=True,
    )
    try:
        build(args)
        raise AssertionError("frozen mix must refuse rebuild even with --force")
    except SystemExit as e:
        assert "frozen" in str(e)
