"""Elo bracket must not invert when scores are non-monotonic."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.elo import estimate_elo


def test_overnight_swa_bracket_is_2050_2200():
    est = estimate_elo([
        {"sf_elo": 2050, "score": 0.6015625, "games": 64},
        {"sf_elo": 2200, "score": 0.4453125, "games": 64},
    ])
    assert est["lower_bound"] == 2050
    assert est["upper_bound"] == 2200
    assert est["lower_bound"] < est["upper_bound"]
    assert 2050 <= est["estimated_elo"] <= 2200


def test_nonmonotonic_does_not_invert_bounds():
    # Control 32-game screen: lost to 1750, beat 1900.
    est = estimate_elo([
        {"sf_elo": 1750, "score": 0.469, "games": 16},
        {"sf_elo": 1900, "score": 0.656, "games": 16},
    ])
    lo, hi = est["lower_bound"], est["upper_bound"]
    assert lo is None or hi is None or lo <= hi
    assert "non-monotonic" in est["note"]


def test_single_level_uses_logistic():
    est = estimate_elo([{"sf_elo": 2050, "score": 0.75, "games": 12}])
    assert est["estimated_elo"] > 2050
    assert est["lower_bound"] == 2050
    assert est["upper_bound"] is None
    est = estimate_elo([{"sf_elo": 2050, "score": 0.25, "games": 12}])
    assert est["estimated_elo"] < 2050
    assert est["upper_bound"] == 2050


def test_unbeaten_top_has_lower_bound_only():
    est = estimate_elo([
        {"sf_elo": 1750, "score": 0.656, "games": 16},
        {"sf_elo": 1900, "score": 0.594, "games": 16},
    ])
    assert est["lower_bound"] == 1900
    assert est["upper_bound"] is None
    assert est["estimated_elo"] == 1900
