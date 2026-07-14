#!/usr/bin/env python3
"""Export soft_cache .pt files to a HuggingFace Dataset (parquet shards).

Schema (per row):
  board_array: list[int8] length 64  (piece ids, STM-normalized encoding)
  turn: int8                         (0 white / 1 black to move)
  castling: int8                     (bitmask)
  ep_square: int8                    (-1 none, else 0..63)
  move_idx: int64                    (hard best move, compact vocab)
  cp: int32                          (STM-centric cp; 0 if mate)
  mate: int32                        (STM-centric mate distance sign; 0 if cp)
  soft_indices: list[int64] len 8    (-1 pad)
  soft_probs: list[float32] len 8
  label_depth: int16                 (optional)
  phase: int8                        (0 open / 1 mid / 2 end; optional)
  source: int8                       (optional; 0 deep / 1 puzzle / 2 syzygy / 3 harvest)
  cache_name: string                 (which local cache this row came from)

Usage:
  HF_TOKEN=... python scripts/export_soft_caches_to_hf.py \\
    --repo avewright/chess-soft-multipv-lichess \\
    --caches outputs/lichess_evals_soft/soft_cache_virgin_6m.pt ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(msg, flush=True)


SCHEMA = pa.schema([
    ("board_array", pa.list_(pa.int8(), 64)),
    ("turn", pa.int8()),
    ("castling", pa.int8()),
    ("ep_square", pa.int8()),
    ("move_idx", pa.int64()),
    ("cp", pa.int32()),
    ("mate", pa.int32()),
    ("soft_indices", pa.list_(pa.int64(), 8)),
    ("soft_probs", pa.list_(pa.float32(), 8)),
    ("label_depth", pa.int16()),
    ("phase", pa.int8()),
    ("source", pa.int8()),
    ("cache_name", pa.string()),
])


def _fixed_list(arr: np.ndarray, value_type: pa.DataType, width: int) -> pa.Array:
    """arr shape (n, width) → FixedSizeListArray."""
    flat = pa.array(arr.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def cache_chunk_table(d: dict, name: str, start: int, end: int) -> pa.Table:
    n = end - start
    ba = d["board_array"][start:end].numpy().astype(np.int8, copy=False)
    si = d["soft_indices"][start:end].numpy().astype(np.int64, copy=False)
    sp = d["soft_probs"][start:end].numpy().astype(np.float32, copy=False)

    if "label_depth" in d:
        depth = d["label_depth"][start:end].numpy().astype(np.int16, copy=False)
    else:
        depth = np.zeros(n, dtype=np.int16)
    if "phase" in d:
        phase = d["phase"][start:end].numpy().astype(np.int8, copy=False)
    else:
        phase = np.full(n, -1, dtype=np.int8)
    if "source" in d:
        source = d["source"][start:end].numpy().astype(np.int8, copy=False)
    else:
        source = np.full(n, -1, dtype=np.int8)

    cols = [
        _fixed_list(ba, pa.int8(), 64),
        pa.array(d["turn"][start:end].numpy().astype(np.int8, copy=False)),
        pa.array(d["castling"][start:end].numpy().astype(np.int8, copy=False)),
        pa.array(d["ep_square"][start:end].numpy().astype(np.int8, copy=False)),
        pa.array(d["move_idx"][start:end].numpy().astype(np.int64, copy=False)),
        pa.array(d["cp"][start:end].numpy().astype(np.int32, copy=False)),
        pa.array(d["mate"][start:end].numpy().astype(np.int32, copy=False)),
        _fixed_list(si, pa.int64(), 8),
        _fixed_list(sp, pa.float32(), 8),
        pa.array(depth),
        pa.array(phase),
        pa.array(source),
        pa.array([name] * n, type=pa.string()),
    ]
    return pa.Table.from_arrays(cols, schema=SCHEMA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id (user/name)")
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-rows-per-cache", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=500_000)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-prefix", default="data", help="parquet filename prefix")
    ap.add_argument("--local-dir", default="outputs/hf_soft_export")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-wipe", action="store_true", help="keep existing local parquet shards")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.push and not token:
        raise SystemExit("HF_TOKEN required for --push")

    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    if not args.no_wipe:
        for old in local.glob(f"{args.shard_prefix}-*.parquet"):
            old.unlink()

    shard_i = int(args.shard_start)

    def write_shard(table: pa.Table) -> None:
        nonlocal shard_i
        out = local / f"{args.shard_prefix}-{shard_i:05d}.parquet"
        pq.write_table(table, out, compression="zstd")
        log(f"wrote {out} n={table.num_rows:,}")
        shard_i += 1

    for cpath in args.caches:
        p = Path(cpath)
        if not p.exists():
            log(f"skip missing {p}")
            continue
        log(f"load {p}")
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = int(d["board_array"].shape[0])
        if args.max_rows_per_cache > 0:
            n = min(n, args.max_rows_per_cache)
        name = p.stem
        log(f"  rows={n:,}")
        for start in range(0, n, args.shard_size):
            end = min(start + args.shard_size, n)
            write_shard(cache_chunk_table(d, name, start, end))
        del d

    readme = local / "README.md"
    readme.write_text(
        f"""---
license: mit
task_categories:
- other
tags:
- chess
- soft-labels
- multipv
- stockfish
---

# {args.repo}

Soft MultiPV policy targets for chess transformers (compact move vocab).

Built from Lichess cloud evaluations + local harvests. Each row is a position with
an 8-wide soft move distribution (`soft_indices` / `soft_probs`) plus hard best move.

## Fields
- `board_array` (64): piece encoding
- `turn`, `castling`, `ep_square`
- `move_idx`, `cp`, `mate`
- `soft_indices[8]`, `soft_probs[8]`
- `label_depth`, `phase`, `source`, `cache_name`
""",
        encoding="utf-8",
    )

    if args.push:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(local),
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
        )
        log(f"pushed https://huggingface.co/datasets/{args.repo}")
    else:
        log(f"local export ready at {local} (pass --push to upload)")


if __name__ == "__main__":
    main()
