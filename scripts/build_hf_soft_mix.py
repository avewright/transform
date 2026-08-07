#!/usr/bin/env python3
"""Build a phase-balanced soft mix from HF packs + local harvest caches.

Sources
  - deep: avewright/chess-soft-multipv-lichess (parquet → soft_cache)
  - syzygy: avewright/chess-soft-syzygy (or local pt)
  - harvest: local MultiPV caches (openings / walks)

Writes outputs/hf_soft_mix/{soft_cache.pt,mix_report.json,deep_soft.pt}.

Usage:
  python scripts/build_hf_soft_mix.py --go --target 1500000
  python scripts/build_hf_soft_mix.py --go --target 5000000 --deep-max-rows 8000000
  python scripts/build_hf_soft_mix.py --go --skip-download  # reuse local pts
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)
META = ("phase", "label_depth", "n_pieces", "ply", "source")
SRC = {"deep": 0, "syzygy": 2, "harvest": 3}
PHASE = {0: "opening", 1: "middlegame", 2: "endgame"}


def log(msg: str) -> None:
    print(msg, flush=True)


def annotate(data: dict, source: int, default_depth: int = 20) -> dict:
    n = int(data["board_array"].shape[0])
    ba = data["board_array"]
    n_pieces = (ba != 0).sum(dim=1).to(torch.int8)
    np_i = n_pieces.to(torch.int16)
    if "phase" in data and data["phase"] is not None:
        phase = data["phase"].to(torch.int8)
    else:
        phase = torch.where(
            np_i >= 26, torch.zeros(n, dtype=torch.int8),
            torch.where(np_i >= 14, torch.ones(n, dtype=torch.int8),
                        torch.full((n,), 2, dtype=torch.int8)),
        )
    ply = (8 + (32 - np_i).clamp(min=0) * 3).to(torch.int16).clamp(max=120)
    out = {k: data[k] for k in CORE if k in data}
    out["phase"] = phase
    if "label_depth" in data:
        out["label_depth"] = data["label_depth"].to(torch.int16)
    else:
        out["label_depth"] = torch.full((n,), default_depth, dtype=torch.int16)
    out["n_pieces"] = n_pieces
    out["ply"] = ply
    out["source"] = torch.full((n,), source, dtype=torch.int8)
    return out


def load_pt(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    log(f"  load {path}: {d['board_array'].shape[0]:,}")
    return d


def filter_depth(data: dict, min_depth: int) -> dict:
    if "label_depth" not in data or min_depth <= 0:
        return data
    mask = data["label_depth"] >= min_depth
    n = int(mask.sum())
    log(f"  depth>={min_depth}: {data['board_array'].shape[0]:,} → {n:,}")
    return {k: v[mask].contiguous() for k, v in data.items()}


def subsample(data: dict, n: int, rng: random.Random) -> dict:
    total = data["board_array"].shape[0]
    if total <= n:
        return data
    idx = torch.tensor(rng.sample(range(total), n), dtype=torch.long)
    return {k: v[idx].contiguous() for k, v in data.items()}


def merge_dedupe(chunks: list[dict]) -> dict:
    """Concat + exact board/state dedupe via packed uint64 words (no weak fingerprint)."""
    keys = [k for k in list(CORE) + list(META) if all(k in c for c in chunks)]
    cat = {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}
    n = cat["board_array"].shape[0]
    ba = np.ascontiguousarray(cat["board_array"].numpy(), dtype=np.uint8)
    if ba.shape[1] != 64:
        raise SystemExit(f"expected board_array [N,64], got {ba.shape}")
    words = ba.view(np.uint64).reshape(n, 8)
    turn = cat["turn"].numpy().astype(np.int8)
    cast = cat["castling"].numpy().astype(np.int8)
    ep = cat["ep_square"].numpy().astype(np.int8)
    rec = np.empty(
        n,
        dtype=[
            ("w0", "u8"), ("w1", "u8"), ("w2", "u8"), ("w3", "u8"),
            ("w4", "u8"), ("w5", "u8"), ("w6", "u8"), ("w7", "u8"),
            ("t", "i1"), ("c", "i1"), ("e", "i1"),
        ],
    )
    for i in range(8):
        rec[f"w{i}"] = words[:, i]
    rec["t"] = turn
    rec["c"] = cast
    rec["e"] = ep
    # keep last occurrence (later sources win)
    _, inv = np.unique(rec[::-1], return_index=True)
    keep = n - 1 - inv
    keep.sort()
    log(f"  dedupe {n:,} → {len(keep):,}")
    return {k: v[keep].contiguous() for k, v in cat.items()}


def phase_quota(data: dict, n_target: int, phase_frac: dict[int, float], rng: random.Random) -> dict:
    """Sample with preferred phase fractions; top up preferring middlegame.

    Uses numpy RNG (not Python random.sample on multi-million lists).
    """
    ph = data["phase"].numpy()
    n = len(ph)
    n_target = min(n_target, n)
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    chosen_parts: list[np.ndarray] = []
    used = np.zeros(n, dtype=bool)

    for pid, frac in phase_frac.items():
        want = int(round(n_target * frac))
        pool = np.flatnonzero(ph == pid)
        if pool.size == 0 or want <= 0:
            continue
        # Allow replacement for scarce phases (esp. middlegame) so quotas can
        # be met when HF deep packs are opening-heavy.
        replace = pool.size < want
        pick = rs.choice(pool, size=want, replace=replace)
        chosen_parts.append(pick)
        if not replace:
            used[pick] = True
        else:
            used[pool] = True

    chosen = np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=np.int64)

    if chosen.size < n_target:
        need = n_target - int(chosen.size)
        mid_pool = np.flatnonzero(ph == 1)
        if mid_pool.size and need:
            pick = rs.choice(mid_pool, size=need, replace=True)
            chosen = np.concatenate([chosen, pick])
            need = 0
        if need > 0:
            rest = np.flatnonzero(~used)
            if rest.size == 0:
                rest = np.arange(n)
            pick = rs.choice(rest, size=min(need, rest.size), replace=rest.size < need)
            chosen = np.concatenate([chosen, pick])

    if chosen.size > n_target:
        chosen = rs.choice(chosen, size=n_target, replace=False)
    # With replacement, indices may duplicate — keep as-is for mid boost.
    chosen.sort()
    idx = torch.from_numpy(chosen.astype(np.int64))
    return {k: v[idx].contiguous() for k, v in data.items()}


def report(data: dict) -> dict:
    n = data["board_array"].shape[0]
    ph = data["phase"].numpy()
    src = data["source"].numpy()
    depth = data["label_depth"].numpy()
    inv_src = {v: k for k, v in SRC.items()}
    return {
        "n": int(n),
        "phase": {PHASE[i]: float((ph == i).mean()) for i in range(3)},
        "phase_counts": {PHASE[i]: int((ph == i).sum()) for i in range(3)},
        "source": {inv_src[i]: float((src == i).mean()) for i in range(4) if i in inv_src and (src == i).any()},
        "source_counts": {inv_src[i]: int((src == i).sum()) for i in range(4) if i in inv_src and (src == i).any()},
        "label_depth": {
            "mean": float(depth.mean()),
            "p50": float(np.percentile(depth, 50)),
            "frac_ge_15": float((depth >= 15).mean()),
            "frac_ge_20": float((depth >= 20).mean()),
        },
    }


def ensure_hf_cache(repo: str, out: Path, max_rows: int | None) -> Path:
    if out.exists():
        log(f"reuse {out}")
        return out
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/autoresearch_8gb/hf_soft_to_cache.py"),
        "--repo", repo, "--out", str(out),
    ]
    if max_rows is not None:
        cmd.extend(["--max-rows", str(max_rows)])
    log(f"download/convert {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--output-dir", default="outputs/hf_soft_mix")
    ap.add_argument("--target", type=int, default=5_000_000,
                    help="Final mix size (default 5M for path-to-2500)")
    ap.add_argument("--deep-max-rows", type=int, default=8_000_000,
                    help="Max rows to convert from multipv-lichess before filtering")
    ap.add_argument("--min-depth", type=int, default=12)
    ap.add_argument("--syzygy-n", type=int, default=500_000)
    ap.add_argument("--harvest-n", type=int, default=750_000,
                    help="Absolute cap on harvest rows before phase quota")
    ap.add_argument("--harvest-frac", type=float, default=0.15,
                    help="Max fraction of final mix from shallow harvest (default 0.15)")
    ap.add_argument("--mid-frac", type=float, default=0.48,
                    help="Target middlegame fraction (default 0.48)")
    ap.add_argument("--open-frac", type=float, default=0.22)
    ap.add_argument("--end-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.go:
        print("Pass --go")
        return

    mid_f = args.mid_frac
    open_f = args.open_frac
    end_f = args.end_frac
    s = open_f + mid_f + end_f
    open_f, mid_f, end_f = open_f / s, mid_f / s, end_f / s

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = ROOT / "outputs" / "hf_soft"
    hf_dir.mkdir(parents=True, exist_ok=True)

    deep_pt = hf_dir / "multipv_lichess_soft.pt"
    syz_pt = hf_dir / "syzygy_soft.pt"

    if not args.skip_download:
        ensure_hf_cache("avewright/chess-soft-syzygy", syz_pt, None)
        ensure_hf_cache("avewright/chess-soft-multipv-lichess", deep_pt, args.deep_max_rows)
    elif not deep_pt.exists():
        raise SystemExit(f"missing {deep_pt}; drop --skip-download")

    chunks: list[dict] = []

    deep = load_pt(deep_pt)
    if deep is None:
        raise SystemExit("deep multipv cache missing")
    deep = annotate(deep, SRC["deep"], default_depth=20)
    deep = filter_depth(deep, args.min_depth)
    # keep most of target as deep (leave room for syzygy + capped harvest)
    deep_keep = min(deep["board_array"].shape[0], int(args.target * 0.80))
    deep = subsample(deep, deep_keep, rng)
    chunks.append(deep)
    torch.save(deep, out_dir / "deep_soft.pt")
    log(f"deep kept {deep['board_array'].shape[0]:,}")

    syz = load_pt(syz_pt)
    if syz is not None:
        syz = annotate(syz, SRC["syzygy"], default_depth=999)
        syz_n = min(args.syzygy_n, syz["board_array"].shape[0], int(args.target * 0.15))
        syz = subsample(syz, syz_n, rng)
        chunks.append(syz)
        log(f"syzygy kept {syz['board_array'].shape[0]:,}")

    harvest_paths = [
        ROOT / "outputs/autoresearch_8gb/soft_cache_merged_550k.pt",
        ROOT / "outputs/autoresearch_8gb/soft_cache_200k.pt",
        ROOT / "outputs/autoresearch_8gb/soft_cache_openings_v2.pt",
    ]
    harvest_chunks = []
    for p in harvest_paths:
        h = load_pt(p)
        if h is not None:
            harvest_chunks.append(annotate(h, SRC["harvest"], default_depth=8))
    if harvest_chunks:
        harvest = merge_dedupe(harvest_chunks) if len(harvest_chunks) > 1 else harvest_chunks[0]
        harvest_cap = min(
            args.harvest_n,
            harvest["board_array"].shape[0],
            int(args.target * args.harvest_frac),
        )
        harvest = subsample(harvest, harvest_cap, rng)
        chunks.append(harvest)
        log(f"harvest kept {harvest['board_array'].shape[0]:,} "
            f"(cap frac={args.harvest_frac})")

    mixed = merge_dedupe(chunks)
    # mid-heavy phase mix
    final = phase_quota(
        mixed,
        min(args.target, mixed["board_array"].shape[0]),
        {0: open_f, 1: mid_f, 2: end_f},
        rng,
    )
    # Enforce harvest ballast ≤ harvest_frac after phase quota
    src = final["source"].numpy()
    harvest_mask = src == SRC["harvest"]
    n_h = int(harvest_mask.sum())
    max_h = int(final["board_array"].shape[0] * args.harvest_frac)
    if n_h > max_h:
        h_idx = np.where(harvest_mask)[0]
        drop = set(rng.sample(h_idx.tolist(), n_h - max_h))
        keep = torch.tensor([i for i in range(len(src)) if i not in drop], dtype=torch.long)
        final = {k: v[keep].contiguous() for k, v in final.items()}
        log(f"harvest post-cap: dropped {n_h - max_h:,} → "
            f"{int((final['source'].numpy() == SRC['harvest']).sum()):,}")

    rep = report(final)
    log("=== MIX REPORT ===")
    log(json.dumps(rep, indent=2))
    mid = rep["phase"].get("middlegame", 0.0)
    if mid < 0.40:
        log(f"WARNING: middlegame frac={mid:.3f} < 0.40 — need more deep mid rows")

    out_pt = out_dir / "soft_cache.pt"
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(final, tmp)
    os.replace(tmp, out_pt)
    (out_dir / "mix_report.json").write_text(json.dumps(rep, indent=2))
    log(f"wrote {out_pt} ({out_pt.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
