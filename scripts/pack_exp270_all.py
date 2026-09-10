#!/usr/bin/env python3
"""Pack every available HF soft shard for exp270. No quality quotas.

Sources:
  avewright/chess-soft-sf19          (data/ + overnight_20260908/)
  avewright/chess-soft-multipv-lichess
  avewright/chess-soft-syzygy

Train keeps split!=1. split==1 (sf19 shard_000000 frozen eval) is eval-only.
Lichess / Syzygy values stay masked.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from move_vocab import VOCAB_SIZE

OUT_DEFAULT = ROOT / "outputs" / "exp270_mix_v1"
REPOS = {
    "sf19": ("avewright/chess-soft-sf19", 4),
    "lichess": ("avewright/chess-soft-multipv-lichess", 1),
    "syzygy": ("avewright/chess-soft-syzygy", 2),
}
CORE = (
    "board_array", "turn", "castling", "ep_square", "move_idx",
    "cp", "mate", "soft_indices", "soft_probs", "label_depth", "phase",
)
OPTIONAL = ("source", "split", "wdl", "dtz", "n_pieces", "ply")


def _to_numpy(table, name):
    if name not in table.column_names:
        return None
    col = table[name]
    try:
        arr = col.to_numpy(zero_copy_only=False)
    except Exception:
        arr = np.array(col.to_pylist(), dtype=object)
    if arr.dtype == object or getattr(arr, "ndim", 1) == 1 and name in (
        "board_array", "soft_indices", "soft_probs", "wdl",
    ):
        arr = np.stack([np.asarray(x) for x in col.to_pylist()])
    return np.ascontiguousarray(arr)


def read_parquet(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    out = {}
    for k in CORE + OPTIONAL:
        arr = _to_numpy(table, k)
        if arr is not None:
            out[k] = arr
    return out


def policy_ok(move_idx, soft_idx, soft_pr) -> np.ndarray:
    mid = move_idx.reshape(-1)
    mid_ok = (mid >= 0) & (mid < VOCAB_SIZE)
    finite = np.isfinite(soft_pr).all(axis=-1)
    nonneg = (soft_pr >= 0).all(axis=-1)
    sum_ok = np.abs(soft_pr.sum(axis=-1) - 1.0) < 1e-3
    active = soft_pr > 0
    idx = soft_idx.astype(np.int64, copy=False)
    idx_ok = ((idx >= 0) | ~active) & ((idx < VOCAB_SIZE) | ~active)
    idx_ok = idx_ok.all(axis=-1)
    has = active.any(axis=-1)
    return mid_ok & finite & nonneg & sum_ok & idx_ok & has


def as_tensors(rows: dict[str, np.ndarray], source_id: int, value_valid: int) -> dict:
    n = int(rows["turn"].shape[0])
    out = {
        "board_array": torch.from_numpy(np.ascontiguousarray(rows["board_array"]).astype(np.int8)),
        "turn": torch.from_numpy(np.ascontiguousarray(rows["turn"]).reshape(n).astype(np.int8)),
        "castling": torch.from_numpy(np.ascontiguousarray(rows["castling"]).reshape(n).astype(np.int8)),
        "ep_square": torch.from_numpy(np.ascontiguousarray(rows["ep_square"]).reshape(n).astype(np.int8)),
        "move_idx": torch.from_numpy(np.ascontiguousarray(rows["move_idx"]).reshape(n).astype(np.int64)),
        "cp": torch.from_numpy(np.ascontiguousarray(rows["cp"]).reshape(n).astype(np.int32)),
        "mate": torch.from_numpy(np.ascontiguousarray(rows["mate"]).reshape(n).astype(np.int32)),
        "soft_indices": torch.from_numpy(np.ascontiguousarray(rows["soft_indices"]).astype(np.int64)),
        "soft_probs": torch.from_numpy(np.ascontiguousarray(rows["soft_probs"]).astype(np.float32)),
        "source": torch.full((n,), source_id, dtype=torch.int8),
        "value_valid": torch.full((n,), value_valid, dtype=torch.int8),
    }
    if "label_depth" in rows:
        out["label_depth"] = torch.from_numpy(
            np.ascontiguousarray(rows["label_depth"]).reshape(n).astype(np.int16)
        )
    if "phase" in rows:
        out["phase"] = torch.from_numpy(np.ascontiguousarray(rows["phase"]).reshape(n).astype(np.int8))
    return out


def cat_dicts(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in keys}


def pack_repo(name: str, repo: str, source_id: int, local_dir: Path) -> tuple[dict, dict, dict]:
    files = sorted(p for p in local_dir.rglob("*.parquet"))
    train_parts, eval_parts = [], []
    stats = {"files": 0, "rows": 0, "train": 0, "eval": 0, "dropped_policy": 0, "dropped_holdout": 0}
    for fp in files:
        raw = read_parquet(fp)
        n = int(raw["turn"].shape[0])
        stats["files"] += 1
        stats["rows"] += n
        ok = policy_ok(raw["move_idx"], raw["soft_indices"], raw["soft_probs"])
        dropped = int((~ok).sum())
        stats["dropped_policy"] += dropped
        if "split" in raw:
            hold = raw["split"].reshape(-1) != 0
            stats["dropped_holdout"] += int((hold & ok).sum())
            train_m = ok & ~hold
            eval_m = ok & hold
        else:
            train_m = ok
            eval_m = np.zeros(n, dtype=bool)
        def take(mask):
            if not mask.any():
                return None
            sl = {k: v[mask] for k, v in raw.items()}
            vv = 1 if name == "sf19" else 0
            return as_tensors(sl, source_id, vv)
        tr = take(train_m)
        ev = take(eval_m)
        if tr is not None:
            train_parts.append(tr)
            stats["train"] += int(train_m.sum())
        if ev is not None:
            eval_parts.append(ev)
            stats["eval"] += int(eval_m.sum())
        print(
            f"  {name} {fp.relative_to(local_dir)} rows={n} train={int(train_m.sum())} "
            f"eval={int(eval_m.sum())} drop_pol={dropped}",
            flush=True,
        )
    if not train_parts:
        raise SystemExit(f"{name}: no train rows")
    train = cat_dicts(train_parts)
    ev = cat_dicts(eval_parts) if eval_parts else None
    print(f"{name} TOTAL {stats}", flush=True)
    return train, ev, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    trains = {}
    evals = {}
    report = {"repos": {}, "vocab": VOCAB_SIZE, "status": "packing"}
    for name, (repo, source_id) in REPOS.items():
        print(f"download {repo}", flush=True)
        local = Path(snapshot_download(repo, repo_type="dataset"))
        train, ev, stats = pack_repo(name, repo, source_id, local)
        trains[name] = train
        if ev is not None:
            evals[name] = ev
            torch.save(ev, out / f"{name}_eval.pt")
        torch.save(train, out / f"{name}_train.pt")
        report["repos"][name] = stats

    soft = cat_dicts([trains["sf19"], trains["lichess"]])
    deep = trains["syzygy"]
    torch.save(soft, out / "soft_cache.pt")
    torch.save(deep, out / "deep_cache.pt")
    n_soft = int(soft["turn"].shape[0])
    n_deep = int(deep["turn"].shape[0])
    report.update({
        "status": "complete",
        "soft_n": n_soft,
        "deep_n": n_deep,
        "sf19_train": int(trains["sf19"]["turn"].shape[0]),
        "lichess_train": int(trains["lichess"]["turn"].shape[0]),
        "syzygy_train": n_deep,
        "deep_mix_frac": n_deep / max(n_soft + n_deep, 1),
        "note": "All available shards. Holdout=sf19 split!=0 only. No phase/quality quotas.",
    })
    (out / "FROZEN.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "dataset_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
