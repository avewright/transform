#!/usr/bin/env python3
"""Stage more incumbent-vs-SF19 harvest work without touching the GPU.

1. Pack unresolved off-PV rows into an analyze inbox (CPU SF19).
2. Build an unseen SF19 cache for a later GPU scan (positions not in the
   existing tagged harvest).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import list_repo_files

from build_hf_elo_mix import position_hashes
from build_overnight_corr15 import (
    SF19_REPO,
    SF19_REV,
    download,
    load_tagged_mistakes,
    packed_fast,
    take_rows,
)
from harvest_hf100m_bulk import _write_shard
from harvest_swa_mistakes import TAG_TO_I

DISAGREE = ROOT / "outputs/swa_sf19_disagree_v1"
OUT = ROOT / "outputs/swa_harvest_more"
BLOCKED = ROOT / "outputs/overnight_corr15/blocked_manifest.json"
UNSEEN_CAP = 1_000_000


def log(msg: str) -> None:
    print(msg, flush=True)


def load_blocked() -> np.ndarray:
    if not BLOCKED.exists():
        return np.zeros(0, dtype=np.uint64)
    raw = json.loads(BLOCKED.read_text()).get("blocked_hashes") or []
    return np.asarray(raw, dtype=np.uint64)


def pack_analyze_inbox(disagree: dict) -> dict:
    needs = disagree["needs_sf"].numpy() != 0
    n = int(needs.sum())
    inbox = OUT / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    if n == 0:
        return {"n": 0, "shards": 0}
    idx = np.flatnonzero(needs)
    shard_size = 4096
    n_shards = 0
    for start in range(0, n, shard_size):
        sl = idx[start:start + shard_size]
        packed = take_rows(disagree, sl)
        _write_shard(packed, OUT, n_shards)
        # _write_shard writes under OUT/inbox if that's how it works — check
        n_shards += 1
    # harvest _write_shard uses out_dir/inbox/shard_*
    return {"n": n, "shards": n_shards}


def seen_hashes() -> np.ndarray:
    extra = DISAGREE / "seen_all_tagged_hashes.npy"
    if extra.exists():
        return np.load(extra).astype(np.uint64)
    log("hashing prior tagged harvest so we do not rescan ok/disagree rows")
    raw, _ = load_tagged_mistakes()
    hs = np.unique(position_hashes(raw).astype(np.uint64))
    del raw
    np.save(extra, hs)
    return hs


def build_unseen_sf19(seen: np.ndarray, blocked: np.ndarray) -> dict:
    deny = np.unique(np.concatenate([seen, blocked])) if blocked.size else seen
    files = sorted(
        f for f in list_repo_files(SF19_REPO, repo_type="dataset", revision=SF19_REV)
        if f.endswith(".parquet") and (f.startswith("data/") or f.startswith("overnight_20260908/"))
    )
    chunks = []
    stats = {"files": 0, "rows": 0, "split1": 0, "seen_or_blocked": 0, "kept": 0}
    for fn in files:
        if stats["kept"] >= UNSEEN_CAP:
            break
        d = packed_fast(pq.read_table(download(SF19_REPO, fn, repo_type="dataset", revision=SF19_REV)))
        n = int(d["turn"].shape[0])
        stats["files"] += 1
        stats["rows"] += n
        split = d["split"].numpy() if "split" in d else np.zeros(n, dtype=np.int8)
        if fn.endswith("data/shard_000000.parquet"):
            split = np.ones(n, dtype=np.int8)
        hs = position_hashes(d).astype(np.uint64)
        keep = split == 0
        stats["split1"] += int((~keep).sum())
        vs = np.isin(hs, deny)
        stats["seen_or_blocked"] += int((keep & vs).sum())
        keep &= ~vs
        idx = np.flatnonzero(keep)
        room = UNSEEN_CAP - stats["kept"]
        if idx.size > room:
            idx = idx[:room]
        if idx.size:
            chunks.append(take_rows(d, idx))
            stats["kept"] += int(idx.size)
        log(f"  unseen {fn} keep_running={stats['kept']:,}")
        del d
    if not chunks:
        raise SystemExit("no unseen SF19 rows")
    keys = [k for k in chunks[0] if all(k in c and torch.is_tensor(c[k]) for c in chunks)]
    data = {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}
    hs = position_hashes(data).astype(np.uint64)
    _, first = np.unique(hs, return_index=True)
    first.sort()
    data = take_rows(data, first)
    dest = OUT / "unseen_sf19.pt"
    torch.save(data, dest)
    stats["unique"] = int(first.size)
    stats["path"] = str(dest)
    return stats


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    disagree_p = DISAGREE / "all_disagreements.pt"
    if not disagree_p.exists():
        raise SystemExit(f"missing {disagree_p}")
    disagree = torch.load(disagree_p, map_location="cpu", weights_only=False)
    inbox = pack_analyze_inbox(disagree)
    log(f"analyze inbox needs_sf={inbox['n']:,} shards={inbox['shards']}")

    blocked = load_blocked()
    seen = seen_hashes()
    np.save(OUT / "seen_hashes.npy", seen)
    log(f"seen hashes={seen.size:,} blocked={blocked.size:,}")
    unseen = build_unseen_sf19(seen, blocked)
    log(f"unseen sf19 unique={unseen['unique']:,}")

    plan = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ckpt": "outputs/sf19_ft/overnight_20260908/eval_swa.pt",
        "do_not_use_gpu_until_train_done": True,
        "analyze": {
            "n": inbox["n"],
            "shards": inbox["shards"],
            "nodes": 250000,
            "note": "Unresolved off-PV from the existing harvest. CPU only.",
        },
        "scan": {
            "cache": unseen["path"],
            "n": unseen["unique"],
            "cap": UNSEEN_CAP,
            "note": "SF19 positions not in the previous tagged harvest and not holdout/blocked.",
        },
        "seen": int(seen.size),
        "blocked": int(blocked.size),
        "stats": unseen,
    }
    (OUT / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    log("WROTE " + str(OUT / "PLAN.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
