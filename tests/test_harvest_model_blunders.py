"""Blunder harvest keeps substantial mistakes and unique board hashes."""
import numpy as np

from scripts.harvest_swa_mistakes import TAG_TO_I, classify_lapse
from scripts.harvest_model_blunders import SUBSTANTIAL, attach_meta, row_hash
from scripts.sf19_soft_dataset import SOFT_K


def test_inaccuracy_and_blunder_are_kept():
    inn = classify_lapse(best_cp=120, best_mate=0, model_cp=40, model_mate=0, model_in_pv=True)
    bl = classify_lapse(best_cp=80, best_mate=0, model_cp=-80, model_mate=0, model_in_pv=True)
    ok = classify_lapse(best_cp=40, best_mate=0, model_cp=30, model_mate=0, model_in_pv=True)
    assert inn["tag"] == "inaccuracy" and TAG_TO_I[inn["tag"]] in SUBSTANTIAL
    assert bl["tag"] == "blunder" and TAG_TO_I[bl["tag"]] in SUBSTANTIAL
    assert ok["tag"] == "ok" and TAG_TO_I[ok["tag"]] not in SUBSTANTIAL


def test_hash_is_unique_for_different_boards():
    a = np.zeros(64, dtype=np.int8)
    b = np.zeros(64, dtype=np.int8)
    a[0] = 1
    b[1] = 1
    assert row_hash(a, 0, 0, -1) != row_hash(b, 0, 0, -1)
    assert row_hash(a, 0, 0, -1) == row_hash(a, 0, 0, -1)
    assert row_hash(a, 0, 0, -1) != row_hash(a, 1, 0, -1)


def test_attach_meta_sets_pieces_soft_and_hash():
    ba = np.zeros(64, dtype=np.int8)
    ba[:12] = 1
    row = {
        "board_array": ba,
        "turn": np.int8(0),
        "castling": np.int8(0),
        "ep_square": np.int8(-1),
        "soft_indices": np.array([1, 2, -1, -1, -1, -1, -1, -1], dtype=np.int64),
        "soft_probs": np.array([0.6, 0.4, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }
    out = attach_meta(row)
    assert int(out["n_pieces"]) == 12
    assert 1 <= int(out["n_soft"]) <= SOFT_K
    assert int(out["n_soft"]) == 2
    assert int(out["pos_hash"]) != 0
