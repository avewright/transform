"""CPU tests for exp280 full-MoE general Lichess training."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import torch

from exp280_moe_full import (
    SOURCE_EXPERT,
    concat_soft,
    holdout_split,
    parquet_to_soft,
    source_buckets,
    stratified_pick,
    subsample_rows,
    tag_source,
)


def test_holdout_and_concat():
    n = 20
    data = {
        "board_array": torch.zeros(n, 64, dtype=torch.int8),
        "turn": torch.zeros(n, dtype=torch.int8),
        "move_idx": torch.arange(n),
        "soft_indices": torch.zeros(n, 8, dtype=torch.int64),
        "soft_probs": torch.zeros(n, 8),
    }
    tr, va = holdout_split(data, 4, seed=280)
    assert int(va["move_idx"].shape[0]) == 4
    assert int(tr["move_idx"].shape[0]) == 16
    cat = concat_soft([tr, va])
    assert int(cat["move_idx"].shape[0]) == 20


def test_parquet_to_soft(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 5
    table = pa.table(
        {
            "board_array": [[0] * 64 for _ in range(n)],
            "turn": np.zeros(n, dtype=np.int8),
            "castling": np.zeros(n, dtype=np.int8),
            "ep_square": np.full(n, -1, dtype=np.int8),
            "move_idx": np.arange(n, dtype=np.int64),
            "cp": np.zeros(n, dtype=np.int32),
            "mate": np.zeros(n, dtype=np.int32),
            "soft_indices": [list(range(8)) for _ in range(n)],
            "soft_probs": [[1.0] + [0.0] * 7 for _ in range(n)],
            "label_depth": np.full(n, 20, dtype=np.int16),
            "phase": np.array([0, 1, 2, 0, 1], dtype=np.int8),
        }
    )
    dest = tmp_path / "data-00000.parquet"
    pq.write_table(table, dest)
    rows = parquet_to_soft(dest)
    assert rows["board_array"].shape == (n, 64)
    assert rows["phase"].tolist() == [0, 1, 2, 0, 1]


def test_tag_and_stratify():
    n = 50
    raw = {
        "board_array": torch.zeros(n, 64, dtype=torch.int8),
        "turn": torch.zeros(n, dtype=torch.int8),
        "castling": torch.zeros(n, dtype=torch.int8),
        "ep_square": torch.full((n,), -1, dtype=torch.int8),
        "move_idx": torch.arange(n),
    }
    tagged = tag_source(raw, "opening")
    assert tagged["source_expert"][0].item() == SOURCE_EXPERT["opening"]
    mid = tag_source({k: v.clone() if torch.is_tensor(v) else v for k, v in raw.items()}, "middlegame")
    mix = concat_soft([tagged, mid])
    mix["source_id"] = torch.cat([tagged["source_id"], mid["source_id"]])
    mix["source_expert"] = torch.cat([tagged["source_expert"], mid["source_expert"]])
    buckets = source_buckets(mix["source_id"])
    assert len(buckets) == 2
    pick = stratified_pick(buckets, 16)
    assert int(pick.numel()) == 16
    small = subsample_rows(tagged, 10, 0)
    assert int(small["board_array"].shape[0]) == 10
