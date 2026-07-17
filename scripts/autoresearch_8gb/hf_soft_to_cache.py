"""Download a HF soft parquet dataset into soft_cache.pt for train_trial."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)
OPTIONAL = ("label_depth", "phase", "source", "n_pieces", "wdl", "dtz", "ply")


def _to_tensor(col, dtype: torch.dtype) -> torch.Tensor:
    # Prefer numpy path (fast). Fall back for nested lists.
    try:
        arr = np.asarray(col, dtype=np.float64 if dtype == torch.float32 else None)
        if arr.dtype == object:
            arr = np.stack([np.asarray(x) for x in col])
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)
    except Exception:
        return torch.tensor(col, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="avewright/chess-soft-syzygy")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    print(f"loading {args.repo} split={args.split} ...", flush=True)
    ds = load_dataset(args.repo, split=args.split)
    n_all = len(ds)
    if args.max_rows is not None and args.max_rows < n_all:
        ds = ds.select(range(args.max_rows))
        print(f"rows={len(ds):,} (subsample of {n_all:,})", flush=True)
    else:
        print(f"rows={len(ds):,}", flush=True)

    ds = ds.with_format("numpy")
    out: dict = {}
    for k in CORE:
        if k not in ds.column_names:
            raise SystemExit(f"missing column {k}; have {ds.column_names}")
        print(f"  convert {k} ...", flush=True)
        col = ds[k]
        if k == "board_array":
            out[k] = _to_tensor(col, torch.int8)
        elif k in ("turn", "castling", "ep_square"):
            out[k] = _to_tensor(col, torch.int8).view(-1)
        elif k == "move_idx":
            out[k] = _to_tensor(col, torch.int64).view(-1)
        elif k in ("cp", "mate"):
            out[k] = _to_tensor(col, torch.int32).view(-1)
        elif k == "soft_indices":
            out[k] = _to_tensor(col, torch.int64)
        elif k == "soft_probs":
            out[k] = _to_tensor(col, torch.float32)
    for k in OPTIONAL:
        if k not in ds.column_names:
            continue
        print(f"  convert {k} ...", flush=True)
        col = ds[k]
        if k in ("label_depth", "ply", "dtz"):
            out[k] = _to_tensor(col, torch.int16).view(-1)
        elif k in ("phase", "source", "n_pieces", "wdl"):
            out[k] = _to_tensor(col, torch.int8).view(-1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(
        f"wrote {out_path} n={out['board_array'].shape[0]:,} "
        f"board={tuple(out['board_array'].shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
