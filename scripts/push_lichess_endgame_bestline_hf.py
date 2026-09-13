#!/usr/bin/env python3
"""Push Lichess <14-piece best-line inbox shards to Hugging Face.

  python3 -u scripts/push_lichess_endgame_bestline_hf.py --go
  python3 -u scripts/push_lichess_endgame_bestline_hf.py --go --watch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_soft_caches_to_hf import cache_chunk_table  # noqa: E402
from upload_exp201_hf import load_hf_token  # noqa: E402

DEFAULT_REPO = "avewright/lichess-endgame-bestline"
INBOX = ROOT / "outputs" / "lichess_endgame_bestline" / "inbox"
OUT = ROOT / "outputs" / "lichess_endgame_bestline"
STATE = OUT / "hf_upload.json"
STAGING = OUT / "hf_staging"
BLOCKED = {
    "avewright/chess-transformer-100m-squares64",
    "avewright/puzzle-model",
    "avewright/syzygy-model",
    "avewright/endgame-dataset",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"repo": DEFAULT_REPO, "uploaded": [], "rows": 0}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def inbox_shards(inbox: Path) -> list[Path]:
    found: list[Path] = []
    for sh in sorted(inbox.glob("shard_*")):
        if not (sh / "soft_cache.pt").exists():
            continue
        if not ((sh / "READY").exists() or (sh / "ATTACHED").exists()):
            continue
        found.append(sh)
    return found


def readme_text(repo: str, n_rows: int, n_shards: int) -> str:
    return f"""---
license: mit
tags:
- chess
- lichess
- endgame
- one-hot
- stockfish
pretty_name: Lichess <14-piece best-line
---

# {repo}

**{n_rows:,}** unique positions with **fewer than 14 pieces**, one-hot on the
**best first move** (PV1) from [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations).

Not a simultaneous MultiPV snapshot. Same FEN can appear at many depths in the
source pack; we keep the deepest / highest-knodes / best-score first move.

## Filter

- `2 ≤ n_pieces ≤ 13`
- First UCI token of `line` only
- Compact move vocab (1968)

## Fields

Same soft-cache schema as the other `avewright/chess-soft-*` packs. One-hot:
`soft_indices[0] = move_idx`, `soft_probs[0] = 1`. `cp` / `mate` are
White-relative in the Lichess source and converted to STM in the builder.

## Files

- `data/shard_XXXXXX.parquet` ({n_shards} shards)
- Labels are mixed browser Stockfish, not SF19 @ 100k
"""


def convert_shard(sh: Path, dest: Path) -> int:
    d = torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)
    n = int(d["move_idx"].shape[0])
    if "source" not in d:
        d["source"] = torch.full((n,), 6, dtype=torch.int8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = cache_chunk_table(d, sh.name, 0, n)
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, dest)
    del d
    return n


def push_once(*, repo: str, inbox: Path, watch: bool) -> dict:
    from huggingface_hub import HfApi, create_repo

    if repo in BLOCKED:
        raise SystemExit(f"blocked repo: {repo}")
    token = load_hf_token()
    api = HfApi(token=token)
    create_repo(repo, repo_type="dataset", private=False, exist_ok=True, token=token)
    state = load_state()
    uploaded = set(state.get("uploaded") or [])
    rows = int(state.get("rows") or 0)
    staging = STAGING / "data"
    staging.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for sh in inbox_shards(inbox):
        remote = f"data/{sh.name}.parquet"
        if remote in uploaded:
            continue
        dest = staging / f"{sh.name}.parquet"
        n = convert_shard(sh, dest)
        api.upload_file(
            path_or_fileobj=str(dest),
            path_in_repo=remote,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add {remote} n={n:,}",
            token=token,
        )
        dest.unlink(missing_ok=True)
        uploaded.add(remote)
        rows += n
        n_new += n
        state = {"repo": repo, "uploaded": sorted(uploaded), "rows": rows}
        save_state(state)
        log(f"uploaded {remote} n={n:,} total={rows:,}")
    card = STAGING / "README.md"
    card.write_text(readme_text(repo, rows, len(uploaded)), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"card: {rows:,} rows",
        token=token,
    )
    log(f"https://huggingface.co/datasets/{repo} rows={rows:,} new={n_new:,} watch={watch}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--inbox", default=str(INBOX))
    ap.add_argument("--poll", type=float, default=60.0)
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")
    inbox = Path(args.inbox)
    while True:
        push_once(repo=args.repo, inbox=inbox, watch=args.watch)
        if not args.watch:
            break
        time.sleep(float(args.poll))


if __name__ == "__main__":
    main()
