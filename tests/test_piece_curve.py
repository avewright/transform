"""Piece-count harvest target is one truncated normal."""
from pathlib import Path

import numpy as np

from scripts.lichess_openings import load_openings, start_positions
from scripts.piece_curve import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    STREAMS,
    curve_summary,
    empty_counts,
    most_deficit_n,
    should_keep,
    target_pmf,
)
from scripts.sf19_soft_dataset import build_piece_curve_game_spec

FIXTURE = Path(__file__).parent / "fixtures" / "openings_sample.tsv"


def test_pmf_is_unimodal_normal():
    p = target_pmf()
    assert p[0] == 0 and p[1] == 0
    sl = p[2:]
    assert abs(sl.sum() - 1.0) < 1e-9
    peak = int(np.argmax(p))
    assert abs(peak - DEFAULT_MU) <= 1
    # single peak: rises then falls
    assert np.all(np.diff(p[2 : peak + 1]) >= -1e-15)
    assert np.all(np.diff(p[peak:]) <= 1e-15)
    xs = np.arange(33)
    mean = float((xs * p).sum())
    std = float(np.sqrt(((xs - mean) ** 2 * p).sum()))
    assert abs(mean - DEFAULT_MU) < 1.5
    assert abs(std - DEFAULT_SIGMA) < 1.5


def test_keep_rejects_overfull_bin():
    p = target_pmf()
    have = empty_counts()
    assert should_keep(17, have, 0, p)
    have[32] = 1000
    assert not should_keep(32, have, 1000, p)
    assert should_keep(17, have, 1000, p)


def test_curve_summary_peak():
    have = empty_counts()
    have[16] = 50
    have[17] = 80
    have[18] = 40
    s = curve_summary(have)
    assert s["peak"] == 17
    assert s["n"] == 170


def test_most_deficit_picks_empty_center():
    p = target_pmf()
    have = empty_counts()
    have[32] = 200
    assert most_deficit_n(have, p) == 17 or abs(most_deficit_n(have, p) - 17) <= 2


def test_copy_paths_see_local_sf19():
    from scripts.piece_curve import local_copy_paths, local_fen_paths
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    copies = local_copy_paths(root)
    fens = local_fen_paths(root)
    assert any("organized_chess_v1/sf19_train.pt" in str(p) for p in copies) or any(
        "eco_1m" in str(p) for p in copies
    )
    assert any("lichess_train.pt" in str(p) or "hf_elo_mix" in str(p) for p in fens)


def test_normalize_row_pads_short_cache():
    from scripts.sf19_soft_dataset import normalize_harvest_row
    row = {
        "board_array": np.zeros(64, dtype=np.int8),
        "turn": 0,
        "castling": 0,
        "ep_square": -1,
        "move_idx": 12,
        "soft_indices": np.arange(8),
        "soft_probs": np.full(8, 0.125, dtype=np.float32),
        "cp": 10,
        "mate": 0,
    }
    row["board_array"][0] = 1
    out = normalize_harvest_row(row)
    assert out is not None
    assert out["soft_cps"].shape == (8,)
    assert int(out["nodes_budget"]) == 100000


def test_specs_rotate_streams():
    starts = start_positions(load_openings(FIXTURE), include_prefixes=False)
    specs = [build_piece_curve_game_spec(i, starts, seed=1, holdout_frac=0.0) for i in range(9)]
    assert [s["stream"] for s in specs] == list(STREAMS) * 3
    assert all(s["piece_curve"] for s in specs)
    assert {s["label_when_pieces_le"] for s in specs if s["stream"] == "endgame"} == {12}
