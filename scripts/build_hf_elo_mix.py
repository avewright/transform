#!/usr/bin/env python3
"""Build an Elo-oriented soft mix from avewright HF parquet packs.

Pulls a phase-balanced subsample of chess-soft-multipv-lichess (deep MultiPV)
plus the full chess-soft-syzygy pack as deep_cache for train_trial deep_mix.

Usage:
  python scripts/build_hf_elo_mix.py --go --soft-n 1500000
  python scripts/build_hf_elo_mix.py --go --smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOFT_REPO = "avewright/chess-soft-multipv-lichess"
SYZYGY_REPO = "avewright/chess-soft-syzygy"
CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)
META = ("label_depth", "phase", "source")
PHASE_FRAC = {0: 0.22, 1: 0.50, 2: 0.28}  # open / mid / end


def log(msg: str) -> None:
    print(msg, flush=True)


def _to_tensor(col, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(col)
    if arr.dtype == object:
        arr = np.stack([np.asarray(x) for x in col])
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)


def load_parquet_rows(path: Path, columns: list[str]) -> dict:
    table = pq.read_table(path, columns=columns)
    out: dict = {}
    for k in columns:
        col = table.column(k).to_pylist() if k == "board_array" else table.column(k).to_numpy()
        if k == "board_array":
            out[k] = _to_tensor(col, torch.int8)
        elif k in ("turn", "castling", "ep_square", "phase", "source"):
            out[k] = torch.from_numpy(np.ascontiguousarray(col, dtype=np.int8)).view(-1)
        elif k == "move_idx":
            out[k] = torch.from_numpy(np.ascontiguousarray(col, dtype=np.int64)).view(-1)
        elif k in ("cp", "mate"):
            out[k] = torch.from_numpy(np.ascontiguousarray(col, dtype=np.int32)).view(-1)
        elif k == "soft_indices":
            out[k] = _to_tensor(col, torch.int64)
        elif k == "soft_probs":
            out[k] = _to_tensor(col, torch.float32)
        elif k == "label_depth":
            out[k] = torch.from_numpy(np.ascontiguousarray(col, dtype=np.int16)).view(-1)
    return out


def concat(chunks: list[dict]) -> dict:
    keys = [k for k in list(CORE) + list(META) if all(k in c for c in chunks)]
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def phase_subsample(data: dict, n_target: int, rng: random.Random) -> dict:
    ph = data["phase"].numpy()
    want = {pid: int(round(n_target * frac)) for pid, frac in PHASE_FRAC.items()}
    while sum(want.values()) > n_target:
        want[1] -= 1
    while sum(want.values()) < n_target:
        want[1] += 1
    chosen: list[int] = []
    leftovers: list[int] = []
    for pid in range(3):
        idx = np.nonzero(ph == pid)[0].tolist()
        rng.shuffle(idx)
        take = want[pid]
        chosen.extend(idx[:take])
        leftovers.extend(idx[take:])
        log(f"  phase {pid}: want={take:,} avail={len(idx):,}")
    if len(chosen) < n_target:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n_target - len(chosen)])
    rng.shuffle(chosen)
    chosen = chosen[:n_target]
    return {k: v[chosen].contiguous() for k, v in data.items()}


def build_soft(n_target: int, seed: int, max_shards: int | None) -> dict:
    files = [f for f in list_repo_files(SOFT_REPO, repo_type="dataset") if f.endswith(".parquet")]
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_shards is not None:
        files = files[:max_shards]
    # Oversample pool ~1.6x then phase-cap.
    pool_target = int(n_target * 1.6)
    cols = list(CORE) + [c for c in META]
    chunks: list[dict] = []
    n = 0
    t0 = time.time()
    for i, f in enumerate(files):
        local = Path(hf_hub_download(SOFT_REPO, f, repo_type="dataset"))
        chunk = load_parquet_rows(local, cols)
        # Prefer deeper rows when shard is large.
        depth = chunk["label_depth"].numpy()
        keep = np.nonzero(depth >= 12)[0]
        if len(keep) < chunk["board_array"].shape[0]:
            chunk = {k: v[keep] for k, v in chunk.items()}
        chunks.append(chunk)
        n += chunk["board_array"].shape[0]
        log(f"  soft shard {i+1}/{len(files)} +{chunk['board_array'].shape[0]:,} total={n:,}")
        if n >= pool_target:
            break
    if not chunks:
        raise SystemExit("no soft shards loaded")
    data = concat(chunks)
    log(f"soft pool={data['board_array'].shape[0]:,} in {time.time()-t0:.1f}s → subsample {n_target:,}")
    return phase_subsample(data, n_target, rng)


def build_syzygy(max_rows: int | None = None) -> dict:
    files = sorted(f for f in list_repo_files(SYZYGY_REPO, repo_type="dataset") if f.endswith(".parquet"))
    cols = list(CORE) + [c for c in META if True]
    # syzygy has extra cols; only request those that exist in schema
    first = Path(hf_hub_download(SYZYGY_REPO, files[0], repo_type="dataset"))
    names = set(pq.ParquetFile(first).schema_arrow.names)
    cols = [c for c in cols if c in names]
    chunks = []
    for f in files:
        local = Path(hf_hub_download(SYZYGY_REPO, f, repo_type="dataset"))
        chunks.append(load_parquet_rows(local, cols))
    data = concat(chunks)
    if "source" not in data:
        data["source"] = torch.full((data["board_array"].shape[0],), 2, dtype=torch.int8)
    if "phase" not in data:
        n_pieces = (data["board_array"] != 0).sum(dim=1)
        data["phase"] = torch.full((data["board_array"].shape[0],), 2, dtype=torch.int8)
        data["phase"] = torch.where(n_pieces >= 14, torch.ones_like(data["phase"]), data["phase"])
    if max_rows is not None and data["board_array"].shape[0] > max_rows:
        idx = torch.randperm(data["board_array"].shape[0])[:max_rows]
        data = {k: v[idx].contiguous() for k, v in data.items()}
    log(f"syzygy n={data['board_array'].shape[0]:,}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default="outputs/hf_elo_mix")
    ap.add_argument("--soft-n", type=int, default=1_500_000)
    ap.add_argument("--syzygy-n", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-shards", type=int, default=None, help="Cap soft shards (debug)")
    args = ap.parse_args()
    if not args.go:
        print("Pass --go")
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    soft_n = 20_000 if args.smoke else args.soft_n
    syz_n = 5_000 if args.smoke else args.syzygy_n
    max_shards = 1 if args.smoke else args.max_shards

    log(f"building soft n={soft_n:,} from {SOFT_REPO}")
    soft = build_soft(soft_n, args.seed, max_shards)
    soft_path = out / "soft_cache.pt"
    torch.save(soft, soft_path)
    log(f"wrote {soft_path} n={soft['board_array'].shape[0]:,}")

    log(f"building syzygy deep n≤{syz_n:,} from {SYZYGY_REPO}")
    deep = build_syzygy(syz_n)
    deep_path = out / "deep_cache.pt"
    torch.save(deep, deep_path)
    log(f"wrote {deep_path} n={deep['board_array'].shape[0]:,}")

    report = {
        "soft_repo": SOFT_REPO,
        "syzygy_repo": SYZYGY_REPO,
        "soft_n": int(soft["board_array"].shape[0]),
        "deep_n": int(deep["board_array"].shape[0]),
        "soft_depth": {
            "min": int(soft["label_depth"].min()),
            "p50": int(soft["label_depth"].median()),
            "max": int(soft["label_depth"].max()),
        },
        "soft_phase": {
            str(i): int((soft["phase"] == i).sum()) for i in range(3)
        },
        "seed": args.seed,
        "smoke": args.smoke,
    }
    (out / "mix_report.json").write_text(json.dumps(report, indent=2))
    log(f"report {report}")


if __name__ == "__main__":
    main()
