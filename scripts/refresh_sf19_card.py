#!/usr/bin/env python3
"""Refresh avewright/chess-soft-sf19 dataset card after the expand upload."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import _hf_token

REPO = "avewright/chess-soft-sf19"
N_TOTAL = 2_010_006
N_BASE = 1_010_000
N_EXPAND = 1_000_006


def main() -> None:
    token = _hf_token()
    if not token:
        raise SystemExit("HF_TOKEN missing")
    from huggingface_hub import HfApi

    text = f"""---
license: mit
tags:
- chess
- stockfish-19
- soft-labels
- multipv
pretty_name: Stockfish 19 soft targets
---

# {REPO}

Official **Stockfish 19** MultiPV soft targets. Not a filter of
`chess-soft-multipv-lichess` or `chess-soft-100m-disagreements`.

**{N_TOTAL:,}** rows. Source id `4`. Vocab `compact` (1968).

## Mix

| split | rows | files | note |
|---|---:|---|---|
| original | {N_BASE:,} | `data/shard_000000.parquet` … | 1M train + 10k frozen eval (`split=1` in shard 0) |
| expand1 | {N_EXPAND:,} | `data/shard_000051.parquet`–`data/shard_000101.parquet` | new games, same teacher, hash-excluded vs original including eval flips |

Honor `split`. Do not invent a new position-hash holdout. Frozen eval is
`saved_split_v1` (unique hashes 9,992; blocked with flips 15,339).

## Teacher

- Stockfish 19, full strength, `Threads=1`, `Hash=32`, `UCI_ShowWDL=true`
- Label budget: **100k nodes / MultiPV=8** / `tau=120`
- `cp` / `mate` / `wdl`: **White-absolute**
- `soft_indices` / `soft_probs`: width 8
- `soft_cps` / `soft_mates` stored so softmax can be rebuilt without SF

## Files

- `data/shard_XXXXXX.parquet`
- `teacher.json`, `summary.json`, `sampling.json`, `audit.json`, `eval_manifest.json` (original release)
"""
    dest = Path("/tmp/chess-soft-sf19-README.md")
    dest.write_text(text, encoding="utf-8")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(dest),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"card: {N_TOTAL:,} rows (original + expand1)",
        token=token,
    )
    print(f"https://huggingface.co/datasets/{REPO} rows={N_TOTAL:,}")


if __name__ == "__main__":
    main()
