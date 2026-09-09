#!/usr/bin/env python3
"""Write a local incumbent-SWA vs SF19 disagreement dataset.

Does not use the GPU. Uses the frozen corr15 verified set plus the tagged
HF mistake shards already audited. Does not upload.
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from build_hf_elo_mix import position_hashes
from build_overnight_corr15 import (
    CORR_REPO,
    CORR_REV,
    INCUMBENT_SHA,
    SUBSTANTIAL,
    load_tagged_mistakes,
    sha256_file,
)
from export_soft_caches_to_hf import SCHEMA, _fixed_list, cache_chunk_table
from harvest_swa_mistakes import I_TO_TAG, TAG_TO_I

OUT = ROOT / "outputs/swa_sf19_disagree_v1"
BONUS = ROOT / "outputs/overnight_corr15/bonus_cache.pt"
FROZEN = ROOT / "outputs/overnight_corr15/FROZEN.json"
AUDIT = ROOT / "outputs/overnight_corr15/corrections_audit.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def extra_table(d: dict, name: str) -> pa.Table:
    n = int(d["turn"].shape[0])
    base = cache_chunk_table(d, name, 0, n)
    extras = [
        ("tag", pa.int8(), np.int8, 6),
        ("drop_cp", pa.int32(), np.int32, 0),
        ("needs_sf", pa.int8(), np.int8, 0),
        ("model_in_pv", pa.int8(), np.int8, 0),
        ("origin", pa.int8(), np.int8, 0),
        ("model_move_idx", pa.int64(), np.int64, -1),
        ("teacher_move_idx", pa.int64(), np.int64, -1),
    ]
    for key, ptype, ndt, default in extras:
        if key in d:
            col = np.asarray(d[key].cpu().numpy(), dtype=ndt)
        elif key == "teacher_move_idx" and "move_idx" in d:
            col = np.asarray(d["move_idx"].cpu().numpy(), dtype=ndt)
        else:
            col = np.full(n, default, dtype=ndt)
        base = base.append_column(key, pa.array(col, type=ptype))
    if "soft_cps" in d:
        base = base.append_column(
            "soft_cps", _fixed_list(d["soft_cps"].cpu().numpy().astype(np.int32), pa.int32(), 8),
        )
    hs = position_hashes(d).astype(np.uint64)
    base = base.append_column("position_hash", pa.array(hs.astype(np.uint64)))
    tag_name = [
        I_TO_TAG.get(int(x), "unknown")
        for x in (d["tag"].cpu().numpy() if "tag" in d else np.full(n, 6))
    ]
    base = base.append_column("tag_name", pa.array(tag_name, type=pa.string()))
    return base


def take(d: dict, idx: np.ndarray) -> dict:
    n = None
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.ndim and n is None:
            n = int(v.shape[0])
        if torch.is_tensor(v) and n is not None and int(v.shape[0]) == n:
            out[k] = v[idx].contiguous()
    return out


def hist(tags: np.ndarray) -> dict:
    return {I_TO_TAG.get(int(k), str(int(k))): int((tags == k).sum()) for k in np.unique(tags)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not BONUS.exists():
        raise SystemExit(f"missing {BONUS}; corr15 mix must be frozen first")
    verified = torch.load(BONUS, map_location="cpu", weights_only=False)
    n_v = int(verified["turn"].shape[0])
    torch.save(verified, OUT / "verified_substantial.pt")
    pq.write_table(extra_table(verified, "verified_substantial"), OUT / "verified_substantial.parquet", compression="zstd")
    log(f"verified_substantial n={n_v:,}")

    log("reload tagged shards for the full disagreement table")
    raw, shard_meta = load_tagged_mistakes()
    tags = raw["tag"].numpy()
    keep = tags != TAG_TO_I["ok"]
    disagree = take(raw, np.flatnonzero(keep))
    n_d = int(disagree["turn"].shape[0])
    torch.save(disagree, OUT / "all_disagreements.pt")
    pq.write_table(extra_table(disagree, "all_disagreements"), OUT / "all_disagreements.parquet", compression="zstd")
    log(f"all_disagreements n={n_d:,} tags={hist(disagree['tag'].numpy())}")

    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    needs = int((disagree["needs_sf"].numpy() != 0).sum()) if "needs_sf" in disagree else None
    manifest = {
        "name": "swa_sf19_disagree_v1",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "checkpoint": "outputs/sf19_ft/overnight_20260908/eval_swa.pt",
            "sha256": INCUMBENT_SHA,
            "hf": "avewright/chess-transformer-100m-overnight_20260908",
        },
        "teacher": {
            "engine": "Stockfish 19",
            "eval_file": "nn-1a298aa575a0.nnue",
            "source": "avewright/chess-soft-sf19 MultiPV labels",
            "cp_perspective": "white_absolute",
            "soft_cps_perspective": "side_to_move",
        },
        "source_pack": {"repo": CORR_REPO, "revision": CORR_REV, "shards": shard_meta},
        "tables": {
            "verified_substantial": {
                "n": n_v,
                "unique": n_v,
                "file_pt": "verified_substantial.pt",
                "file_parquet": "verified_substantial.parquet",
                "sha256_pt": sha256_file(OUT / "verified_substantial.pt"),
                "include": ["inaccuracy", "blunder", "conversion", "major"],
                "exclude": ["ok", "off_pv", "disagree", "needs_sf!=0", "model==teacher"],
                "tags": frozen.get("corrections", {}).get("tags"),
                "note": "Train-ready. Same frozen set as outputs/overnight_corr15/bonus_cache.pt",
            },
            "all_disagreements": {
                "n": n_d,
                "unique": int(np.unique(position_hashes(disagree).astype(np.uint64)).size),
                "file_pt": "all_disagreements.pt",
                "file_parquet": "all_disagreements.parquet",
                "sha256_pt": sha256_file(OUT / "all_disagreements.pt"),
                "tags": hist(disagree["tag"].numpy()),
                "needs_sf": needs,
                "note": "Every tagged non-ok row. off_pv / needs_sf are unresolved and not for training.",
            },
        },
        "do_not_upload": True,
        "used_by_train_arm": "outputs/overnight_corr15",
        "audit": {
            "perspective": (audit.get("score_perspective") or {}).get("interpretation"),
            "checkpoint_matches_incumbent": (audit.get("checkpoint_provenance") or {}).get("matches_locked_incumbent"),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("WROTE " + str(OUT))
    print(json.dumps(manifest["tables"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
