"""CPU tests for Mac correction ranking, regret, sampling, and rescan."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from build_organized_chess_mix import SOURCE_LICHESS, SOURCE_PUZZLE, SOURCE_SF19, SOURCE_SYZYGY
from harvest_swa_mistakes import TAG_TO_I
from mac_correction_pipeline import (
    REGRET_KNOWN,
    REGRET_UNKNOWN,
    _annotate_batch,
    eval_bucket,
    rank_score,
    remaining_weaknesses,
    repeat_exposure,
    select_verify_queue,
    teacher_p_and_in_pv,
)


def test_missing_multipv_is_unknown_regret_not_a_penalty():
    pred = np.array([9], dtype=np.int64)
    teacher = np.array([1], dtype=np.int64)
    soft_i = np.array([[1, 2, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    soft_p = np.array([[0.7, 0.3, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    p, in_pv = teacher_p_and_in_pv(soft_i, soft_p, pred)
    assert float(p[0]) == 0.0 and int(in_pv[0]) == 0
    extra = _annotate_batch(
        pred, np.array([0.95], np.float32), np.zeros(1, np.float32),
        teacher, soft_i, soft_p,
    )
    assert int(extra["regret_status"][0]) == REGRET_UNKNOWN
    assert int(extra["drop_cp"][0]) == 0
    assert int(extra["disagree"][0]) == 1


def test_rank_is_not_ce_or_disagreement_only():
    # Same teacher miss, higher confidence + known 200cp drop wins.
    # High CE / disagreement with low confidence loses.
    # Agreement with high CE-looking teacher_p loses.
    scores = rank_score(
        confidence=np.array([0.92, 0.20, 0.92, 0.92], np.float32),
        teacher_p=np.array([0.04, 0.04, 0.04, 0.85], np.float32),
        regret_status=np.array([REGRET_KNOWN, REGRET_KNOWN, REGRET_UNKNOWN, REGRET_KNOWN], np.int8),
        drop_cp=np.array([200, 200, 0, 0], np.int32),
    )
    assert scores[0] > scores[1]
    assert scores[0] > scores[3]
    # Unknown regret is queued (nonzero) but not given a fake 200cp boost.
    assert 0 < scores[2] < scores[0]


def test_verify_queue_keeps_unknown_uncertain_and_control():
    n = 200
    conf = np.full(n, 0.8, np.float32)
    conf[80:100] = 0.25
    teacher_p = np.full(n, 0.9, np.float32)
    teacher_p[:60] = 0.02
    regret = np.full(n, REGRET_KNOWN, np.int8)
    regret[40:70] = REGRET_UNKNOWN
    drop = np.zeros(n, np.int32)
    drop[:40] = 180
    disagree = np.zeros(n, np.int8)
    disagree[:70] = 1
    scan = {
        "turn": torch.zeros(n, dtype=torch.int8),
        "confidence": torch.from_numpy(conf),
        "teacher_p": torch.from_numpy(teacher_p),
        "regret_status": torch.from_numpy(regret),
        "drop_cp": torch.from_numpy(drop),
        "disagree": torch.from_numpy(disagree),
    }
    idx, report = select_verify_queue(scan, n_verify=40, rng=np.random.RandomState(0))
    assert report["informative"] > 0
    assert report["unknown_regret"] > 0
    assert report["uncertain"] > 0
    assert report["control"] > 0
    assert int(idx.size) <= 40
    # Control includes at least one agreement.
    assert (disagree[idx] == 0).any()


def test_eval_buckets_respect_value_mask_and_sentinels():
    assert eval_bucket(80, 0, 0, 1, SOURCE_SF19) == "equal"
    assert eval_bucket(-400, 0, 0, 1, SOURCE_SF19) == "losing"
    assert eval_bucket(-400, 0, 1, 1, SOURCE_SF19) == "winning"  # white-abs, black to move
    assert eval_bucket(0, 3, 0, 1, SOURCE_SF19) == "mate_win"
    assert eval_bucket(20, 0, 0, 0, SOURCE_PUZZLE) == "masked"
    assert eval_bucket(0, 0, 0, 0, SOURCE_SYZYGY) == "masked"
    assert eval_bucket(90_000, 0, 0, 0, SOURCE_LICHESS) == "sentinel"


def test_repeat_exposure_and_remaining_weaknesses():
    cov = repeat_exposure(unique_n=2500, mix_frac=0.12, steps=8000, batch=64)
    assert cov["expected_draws"] == 8000 * 64 * 0.12
    assert abs(cov["repeat_exposure"] - cov["expected_draws"] / 2500) < 1e-9
    focus, rep = remaining_weaknesses(
        pos_hash=np.array([10, 11, 12, 13], dtype=np.int64),
        new_pred=np.array([1, 9, 3, 8]),
        teacher=np.array([1, 2, 3, 4]),
        old_pred=np.array([9, 9, 3, 4]),
        verified_tag=np.array([
            TAG_TO_I["blunder"], TAG_TO_I["major"], TAG_TO_I["ok"], TAG_TO_I["ok"],
        ], dtype=np.int8),
        prev_fixed=np.array([10], dtype=np.int64),
    )
    assert rep["fixed"] == 1
    assert focus[1]  # still wrong substantial
    assert not focus[0]  # already in fixed registry
    assert focus[3]  # newly wrong
    assert not focus[2]
