#!/usr/bin/env python3
"""Build phase-balanced QUALITY soft mix from UNSEEN positions only.

Pipeline:
  1. Load Lichess pool (e.g. soft_cache_20m.pt)
  2. Drop any exact board keys present in --exclude caches (already trained)
  3. Phase-cap deep Lichess (default 22/48/30 open/mid/end)
  4. Add harvest / puzzle / syzygy spice (also deduped vs exclude), source-capped
  5. Write soft_cache.pt + mix_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse helpers from quality mixer
from scripts.build_quality_deep_soft_mix import (  # noqa: E402
    CORE,
    PHASE_ID,
    PHASE_TARGETS,
    ensure_phase,
    gen_syzygy_soft,
    load_cache,
    log,
    subsample_phase_balanced,
)
from scripts.extract_unseen_soft_cache import pack_keys  # noqa: E402

SRC = {"deep": 0, "puzzle": 1, "syzygy": 2, "harvest": 3}


def filter_unseen(pool: dict, excl_keys: np.ndarray) -> dict:
    pool_k = pack_keys(pool)
    n = pool_k.shape[0]
    if len(excl_keys) == 0:
        keep = np.ones(n, dtype=bool)
    else:
        idx = np.searchsorted(excl_keys, pool_k)
        in_b = idx < len(excl_keys)
        matched = np.zeros(n, dtype=bool)
        matched[in_b] = excl_keys[idx[in_b]] == pool_k[in_b]
        keep = ~matched
    keep_idx = np.nonzero(keep)[0]
    log(f"  unseen {len(keep_idx):,} / {n:,} ({100 * len(keep_idx) / max(n, 1):.1f}%)")
    keys = [k for k in list(CORE) + ["phase", "label_depth"] if k in pool]
    return {k: pool[k][keep_idx].contiguous() for k in keys}


def load_exclude_keys(paths: list[str]) -> np.ndarray:
    parts = []
    for p in paths:
        d = load_cache(Path(p))
        if d is None:
            continue
        parts.append(pack_keys(d))
    if not parts:
        return np.array([], dtype=np.dtype((np.void, 1)))
    return np.unique(np.concatenate(parts))


def take_unseen_chunk(data: dict, excl: np.ndarray, n_max: int, rng: random.Random) -> dict | None:
    if data is None or n_max <= 0:
        return None
    filtered = filter_unseen(data, excl)
    n = filtered["board_array"].shape[0]
    if n == 0:
        return None
    if n > n_max:
        idx = np.sort(rng.sample(range(n), n_max))
        filtered = {k: v[idx].contiguous() for k, v in filtered.items()}
    return filtered


def concat_rows(chunks: list[dict]) -> dict:
    keys = list(CORE)
    for opt in ("phase", "label_depth", "source"):
        if all(opt in c for c in chunks):
            keys.append(opt)
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="Large Lichess soft cache (e.g. 20M)")
    ap.add_argument("--exclude", nargs="+", required=True)
    ap.add_argument("--output-dir", default="outputs/unseen_quality_mix")
    ap.add_argument("--target", type=int, default=4_000_000)
    ap.add_argument("--deep-frac", type=float, default=0.72)
    ap.add_argument("--puzzle-frac", type=float, default=0.12)
    ap.add_argument("--harvest-frac", type=float, default=0.10)
    ap.add_argument("--syzygy-frac", type=float, default=0.06)
    ap.add_argument("--syzygy-dir", default="syzygy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()

    log("load exclude keys…")
    excl = load_exclude_keys(args.exclude)
    log(f"  exclude unique={len(excl):,}")

    log(f"load pool {args.pool}")
    pool = load_cache(Path(args.pool))
    assert pool is not None
    unseen = filter_unseen(pool, excl)
    del pool

    n_deep = int(round(args.target * args.deep_frac))
    log(f"phase-balance deep → {n_deep:,}")
    deep = subsample_phase_balanced(unseen, n_deep, PHASE_TARGETS, rng)
    deep["source"] = torch.full((deep["board_array"].shape[0],), SRC["deep"], dtype=torch.int8)
    del unseen

    chunks = [deep]

    # Harvests / puzzles
    harvest_paths = [
        ("harvest", Path("outputs/exp190_phase_deep/soft_cache.pt")),
        ("harvest", Path("outputs/exp190_phase_deep_continue/soft_cache.pt")),
        ("harvest", Path("outputs/exp192_edge_soft/soft_cache.pt")),
        ("harvest", Path("outputs/exp095_endgame_deep/soft_cache.pt")),
        ("puzzle", Path("outputs/exp193_puzzle_soft/soft_cache.pt")),
    ]
    n_harv = int(round(args.target * args.harvest_frac))
    n_puz = int(round(args.target * args.puzzle_frac))
    harv_budget, puz_budget = n_harv, n_puz
    harv_parts, puz_parts = [], []

    for kind, path in harvest_paths:
        if (kind == "harvest" and harv_budget <= 0) or (kind == "puzzle" and puz_budget <= 0):
            continue
        raw = load_cache(path)
        budget = harv_budget if kind == "harvest" else puz_budget
        part = take_unseen_chunk(raw, excl, budget, rng)
        if part is None:
            continue
        part["source"] = torch.full(
            (part["board_array"].shape[0],), SRC[kind], dtype=torch.int8,
        )
        if "phase" not in part:
            part["phase"] = ensure_phase(part)
        if kind == "harvest":
            harv_parts.append(part)
            harv_budget -= part["board_array"].shape[0]
        else:
            puz_parts.append(part)
            puz_budget -= part["board_array"].shape[0]

    if harv_parts:
        chunks.append(concat_rows(harv_parts))
    if puz_parts:
        chunks.append(concat_rows(puz_parts))

    n_syz = int(round(args.target * args.syzygy_frac))
    syz = gen_syzygy_soft(n_syz, Path(args.syzygy_dir), args.seed + 7, tau=120.0)
    if syz is not None:
        syz["source"] = torch.full((syz["board_array"].shape[0],), SRC["syzygy"], dtype=torch.int8)
        # filter vs exclude
        syz = filter_unseen(syz, excl)
        if syz["board_array"].shape[0] > 0:
            chunks.append(syz)

    # Final concat + light shuffle
    mix = concat_rows(chunks)
    n = mix["board_array"].shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    mix = {k: v[perm].contiguous() for k, v in mix.items()}

    report = {
        "n": n,
        "phase": {
            name: float((mix["phase"] == pid).float().mean())
            for name, pid in PHASE_ID.items()
        },
        "source": {
            name: float((mix["source"] == sid).float().mean())
            for name, sid in SRC.items()
            if "source" in mix
        },
        "depth_mean": float(mix["label_depth"].float().mean()) if "label_depth" in mix else None,
        "exclude_n": int(len(excl)),
        "pool": args.pool,
        "target": args.target,
    }
    out_pt = out_dir / "soft_cache.pt"
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(mix, tmp)
    os.replace(tmp, out_pt)
    (out_dir / "mix_report.json").write_text(json.dumps(report, indent=2))
    log(f"wrote {out_pt} n={n:,} ({out_pt.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    log(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
