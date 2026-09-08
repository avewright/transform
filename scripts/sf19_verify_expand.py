#!/usr/bin/env python3
"""Verify expanded SF19 shards against the frozen eval split.

Checks:
  - internal / vs-base duplicates (trainer position hashes)
  - frozen eval positions and their horizontal-flip equivalents
  - all new rows stay split=0 (eval split is not rewritten)

Does not invent a new evaluation split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.autoresearch_8gb.pipeline import (  # noqa: E402
    apply_membership,
    hflip_cache_slice,
    make_val_membership,
    position_hashes,
)
from scripts.sf19_soft_dataset import SeenDB, compact_key_bytes  # noqa: E402


def _load_cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def frozen_block_sets(base: dict) -> dict:
    man = make_val_membership(base, n_hold=0, seed=201, source="sf19_base")
    if man.get("method") != "saved_split_v1":
        raise SystemExit(f"expected saved_split_v1, got {man.get('method')}")
    val_h = np.asarray(man["hashes"], dtype=np.uint64)
    blocked = np.asarray(man["blocked_hashes"], dtype=np.uint64)
    return {
        "manifest": man,
        "val_h": val_h,
        "blocked_h": blocked,
    }


def harvest_eval_keys(base: dict) -> list[bytes]:
    """SeenDB keys for eval rows and castling-safe flips (harvest key space)."""
    split = base["split"].view(-1).numpy()
    val = np.flatnonzero(split != 0)
    keys: list[bytes] = []
    for i in val.tolist():
        keys.append(
            compact_key_bytes(
                base["board_array"][i], base["turn"][i], base["castling"][i], base["ep_square"][i]
            )
        )
    cast = base["castling"].view(-1).numpy()
    flip_src = np.flatnonzero((split != 0) & (cast == 0))
    if flip_src.size:
        flipped = hflip_cache_slice(base, torch.from_numpy(flip_src.astype(np.int64)))
        for j in range(int(flipped["board_array"].shape[0])):
            keys.append(
                compact_key_bytes(
                    flipped["board_array"][j],
                    flipped["turn"][j],
                    flipped["castling"][j],
                    flipped["ep_square"][j],
                )
            )
    return keys


def ready_shards(inbox: Path) -> list[Path]:
    out = []
    for sh in sorted(inbox.glob("shard_*")):
        cache = sh / "soft_cache.pt"
        if cache.exists() and (sh / "READY").exists():
            out.append(cache)
    return out


def audit(args: argparse.Namespace) -> dict:
    base = _load_cache(Path(args.base_cache))
    block = frozen_block_sets(base)
    base_h = position_hashes(base).astype(np.uint64)
    seen = set(int(x) for x in np.unique(base_h).tolist())
    shards = ready_shards(Path(args.inbox))
    rows = 0
    split_nonzero = 0
    internal_dups = 0
    vs_base = 0
    vs_prior_expand = 0
    eval_exact = 0
    eval_or_flip = 0
    per_shard = []
    val_set = set(int(x) for x in block["val_h"].tolist())
    blocked_set = set(int(x) for x in block["blocked_h"].tolist())
    for cache in shards:
        data = _load_cache(cache)
        n = int(data["move_idx"].shape[0])
        rows += n
        if "split" in data:
            split_nonzero += int((data["split"].view(-1) != 0).sum().item())
        hs = position_hashes(data).astype(np.uint64)
        uniq, first = np.unique(hs, return_index=True)
        d_int = n - int(first.size)
        internal_dups += d_int
        vs_b = int(np.isin(hs, base_h).sum())
        vs_base += vs_b
        vs_p = 0
        keep_first = np.zeros(n, dtype=np.bool_)
        keep_first[first] = True
        for h in hs[keep_first].tolist():
            ih = int(h)
            if ih in seen:
                vs_p += 1
            else:
                seen.add(ih)
        vs_prior_expand += vs_p
        n_eval = int(np.isin(hs, block["val_h"]).sum())
        n_block = int(np.isin(hs, block["blocked_h"]).sum())
        eval_exact += n_eval
        eval_or_flip += n_block
        rec = {
            "shard": cache.parent.name,
            "n": n,
            "internal_dups": d_int,
            "vs_base": vs_b,
            "vs_prior_expand": vs_p,
            "eval_exact": n_eval,
            "eval_or_flip": n_block,
            "split_nonzero": int((data["split"].view(-1) != 0).sum().item()) if "split" in data else None,
        }
        per_shard.append(rec)
        print(
            f"{cache.parent.name} n={n:,} dups={d_int} vs_base={vs_b} "
            f"eval={n_eval} eval+flip={n_block}",
            flush=True,
        )
        del data
    report = {
        "base_cache": str(Path(args.base_cache).resolve()),
        "inbox": str(Path(args.inbox).resolve()),
        "n_base": int(base_h.size),
        "n_base_unique": int(np.unique(base_h).size),
        "n_eval": int(block["val_h"].size),
        "n_blocked_eval_plus_flip": int(block["blocked_h"].size),
        "split_method": block["manifest"]["method"],
        "n_shards": len(shards),
        "n_expand_rows": rows,
        "split_nonzero_in_expand": split_nonzero,
        "internal_dups": internal_dups,
        "vs_base": vs_base,
        "vs_prior_expand": vs_prior_expand,
        "eval_exact": eval_exact,
        "eval_or_flip": eval_or_flip,
        "shards": per_shard,
        "ok": split_nonzero == 0 and eval_or_flip == 0,
        "note": (
            "Eval split stays in the original cache (split=1). Expand rows must "
            "be train-only. eval_or_flip counts harvest rows that match a frozen "
            "eval position or its castling-safe horizontal flip."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "shards"}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return report


def block_harvest(args: argparse.Namespace) -> None:
    base = _load_cache(Path(args.base_cache))
    keys = harvest_eval_keys(base)
    seen = SeenDB(Path(args.seen))
    added = seen.add_many(keys)
    print(
        f"blocked harvest keys eval+flip={len(keys):,} newly_added={added:,} seen={len(seen):,}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("audit", "block-harvest"))
    ap.add_argument("--base-cache", default="outputs/sf19_ft/soft_cache.pt")
    ap.add_argument("--inbox", default="outputs/sf19_soft/expand1/inbox")
    ap.add_argument("--seen", default="outputs/sf19_soft/expand1/seen.sqlite")
    ap.add_argument("--out", default="outputs/sf19_ft/compare1/expand_audit.json")
    args = ap.parse_args()
    if args.cmd == "audit":
        audit(args)
    else:
        block_harvest(args)


if __name__ == "__main__":
    main()
