#!/usr/bin/env python3
"""Extract deep soft rows from a pool that are NOT in exclude caches (exact board key)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def pack_keys(d: dict) -> np.ndarray:
    ba = np.ascontiguousarray(d["board_array"].numpy(), dtype=np.uint8)
    turn = d["turn"].numpy().astype(np.uint8)
    cast = d["castling"].numpy().astype(np.uint8)
    ep = (d["ep_square"].numpy().astype(np.int16) + 1).astype(np.uint8)
    extra = np.stack([turn, cast, ep], axis=1)
    key = np.ascontiguousarray(np.concatenate([ba, extra], axis=1))
    # Flatten to 1-D void keys (view can otherwise be (N,1) and break indexing).
    return np.ascontiguousarray(key).view(np.dtype((np.void, key.shape[1]))).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="Source soft cache (e.g. lichess 8M)")
    ap.add_argument("--exclude", nargs="+", required=True, help="Caches already trained on")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all unseen")
    args = ap.parse_args()

    log(f"load pool {args.pool}")
    pool = torch.load(args.pool, map_location="cpu", weights_only=False)
    pool_k = pack_keys(pool)
    n_pool = pool_k.shape[0]
    log(f"  pool rows={n_pool:,}")

    excl_parts = []
    for path in args.exclude:
        log(f"load exclude {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
        excl_parts.append(pack_keys(d))
        log(f"  +{excl_parts[-1].shape[0]:,}")
    excl = np.unique(np.concatenate(excl_parts))
    log(f"exclude unique={len(excl):,}")

    # mask: pool keys not in excl (searchsorted)
    # pool may have dups; work row-wise
    idx = np.searchsorted(excl, pool_k)
    in_bounds = idx < len(excl)
    matched = np.zeros(n_pool, dtype=bool)
    matched[in_bounds] = excl[idx[in_bounds]] == pool_k[in_bounds]
    keep = ~matched
    keep_idx = np.nonzero(keep)[0]
    log(f"unseen rows={len(keep_idx):,} / {n_pool:,} ({100*len(keep_idx)/max(n_pool,1):.1f}%)")

    if args.max_rows > 0 and len(keep_idx) > args.max_rows:
        rng = np.random.default_rng(42)
        keep_idx = np.sort(rng.choice(keep_idx, size=args.max_rows, replace=False))
        log(f"subsample → {len(keep_idx):,}")

    keys = [k for k in list(CORE) + ["phase", "label_depth"] if k in pool]
    out = {k: pool[k][keep_idx].contiguous() for k in keys}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".pt.tmp")
    torch.save(out, tmp)
    os.replace(tmp, outp)
    # summary
    if "label_depth" in out:
        log(f"depth mean={out['label_depth'].float().mean():.1f}")
    if "phase" in out:
        ph = out["phase"]
        for i, name in enumerate(["open", "mid", "end"]):
            log(f"  phase {name}={(ph==i).float().mean()*100:.1f}%")
    log(f"wrote {outp} n={out['board_array'].shape[0]:,} ({outp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
