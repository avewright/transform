#!/usr/bin/env python3
"""Pack Lichess shards NOT already in the lean mix (skip first 8 parquet files)."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from pack_exp270_all import as_tensors, cat_dicts, policy_ok, read_parquet

SKIP = 8
CHUNK = 16
OUT = ROOT / "outputs" / "exp270_mix_v1"
DEFAULT_MAX_ROWS = 30_000_000


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--skip", type=int, default=SKIP)
    ap.add_argument("--shard-offset", type=int, default=0, help="Index of first lichess_rest_XX.pt to write")
    args = ap.parse_args()
    local = Path(snapshot_download("avewright/chess-soft-multipv-lichess", repo_type="dataset"))
    files = sorted(local.rglob("*.parquet"))[args.skip:]
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"remaining files={len(files)} skip_first={args.skip} "
        f"shard_offset={args.shard_offset} chunk={CHUNK} max_rows={args.max_rows:,}",
        flush=True,
    )
    written = []
    total = 0
    for i in range(0, len(files), CHUNK):
        if total >= args.max_rows:
            break
        batch = files[i : i + CHUNK]
        parts = []
        n = 0
        for fp in batch:
            if total + n >= args.max_rows:
                break
            raw = read_parquet(fp)
            ok = policy_ok(raw["move_idx"], raw["soft_indices"], raw["soft_probs"])
            sl = {k: v[ok] for k, v in raw.items()}
            parts.append(as_tensors(sl, 1, 0))
            keep = int(ok.sum())
            n += keep
            print(f"  {fp.name} keep={keep}", flush=True)
        if not parts:
            break
        dest = OUT / f"lichess_rest_{args.shard_offset + i // CHUNK:02d}.pt"
        torch.save(cat_dicts(parts), dest)
        written.append({"path": dest.name, "n": n, "files": [p.name for p in batch]})
        total += n
        print(f"wrote {dest} n={n:,} running={total:,}", flush=True)
        del parts
    report = {
        "status": "lichess_rest_complete",
        "skip": args.skip,
        "max_rows": args.max_rows,
        "shards": written,
        "total": sum(s["n"] for s in written),
    }
    (OUT / "lichess_rest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps({"shards": len(written), "total": report["total"]}), flush=True)


if __name__ == "__main__":
    main()
