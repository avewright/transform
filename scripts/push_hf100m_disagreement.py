#!/usr/bin/env python3
"""Create/update the public 100M-policy disagreement dataset on Hugging Face.

Converts harvest inbox shards (soft_cache.pt) to parquet and uploads incrementally.
Safe to rerun; already-uploaded shards are skipped.

Usage:
  MOVE_VOCAB_VERSION=compact python -u scripts/push_hf100m_disagreement.py --go
  MOVE_VOCAB_VERSION=compact python -u scripts/push_hf100m_disagreement.py --go --watch
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

from data_loader import _hf_token  # noqa: E402
from export_soft_caches_to_hf import cache_chunk_table  # noqa: E402

DEFAULT_INBOX = ROOT / "outputs" / "hf100m_bulk" / "20260905_175038_bulk"
DEFAULT_REPO = "avewright/chess-soft-100m-disagreements"
DEFAULT_CKPT = "avewright/chess-transformer-100m-squares64"
SOURCE_REPO = "avewright/chess-soft-multipv-lichess"


def log(msg: str) -> None:
    print(msg, flush=True)


def _state_path(out_dir: Path) -> Path:
    return out_dir / "hf_upload.json"


def load_state(out_dir: Path, repo: str) -> dict:
    p = _state_path(out_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"repo": repo, "uploaded": {}, "n_uploaded": 0}


def save_state(out_dir: Path, state: dict) -> None:
    tmp = _state_path(out_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(_state_path(out_dir))


def list_ready_shards(inbox: Path) -> list[Path]:
    found = []
    for sh in sorted(inbox.glob("shard_*")):
        if (sh / "soft_cache.pt").exists():
            found.append(sh)
    return found


def write_readme(
    path: Path,
    *,
    repo: str,
    n_rows: int,
    n_shards: int,
    stream_rows: int = 0,
    sf19_rows: int = 0,
    sf19_shards: int = 0,
) -> None:
    extra = ""
    if stream_rows or sf19_rows:
        extra = f"""
## Mix

| split | rows | shards | labels |
|---|---:|---:|---|
| `data/shard_*.parquet` | {stream_rows:,} | {n_shards - sf19_shards} | teacher MultiPV from `{SOURCE_REPO}` |
| `data/sf19/*.parquet` | {sf19_rows:,} | {sf19_shards} | **Stockfish 19** max-Elo MultiPV from new games |

Stream rows are an argmax filter of existing boards. SF19 rows are **new games**
(100M vs unlimited SF19; hash-excluded against the stream). Soft targets on the
SF19 split are live SF19 MultiPV, not the original teacher pack.
"""
    path.write_text(
        f"""---
license: mit
task_categories:
- other
tags:
- chess
- soft-labels
- multipv
- disagreement
- policy
- stockfish-19
pretty_name: 100M squares64 policy disagreements
---

# {repo}

Positions where the greedy policy of
[`{DEFAULT_CKPT}`](https://huggingface.co/avewright/chess-transformer-100m-squares64)
disagrees with a strong teacher best move (`move_idx`).

Current upload: **{n_rows:,}** rows in **{n_shards}** shards.
{extra}
## Columns

Same schema as `{SOURCE_REPO}`:

- `board_array[64]`, `turn`, `castling`, `ep_square`
- `move_idx`, `cp`, `mate` — teacher hard best + eval
- `soft_indices[8]`, `soft_probs[8]` — teacher soft policy
- `label_depth`, `phase`, `source` (3 = harvest), `cache_name`

## License

MIT. Stream labels inherit from the source MultiPV pack. SF19 labels are
Stockfish 19 analysis from this harvest.
""",
        encoding="utf-8",
    )


def convert_shard(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    d = torch.load(src / "soft_cache.pt", map_location="cpu", weights_only=False)
    n = int(d["move_idx"].shape[0])
    table = cache_chunk_table(d, src.name, 0, n)
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)
    del d
    return n


def convert_shards_packed(srcs: list[Path], dest: Path, cache_name: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tables = []
    n = 0
    for src in srcs:
        d = torch.load(src / "soft_cache.pt", map_location="cpu", weights_only=False)
        take = int(d["move_idx"].shape[0])
        tables.append(cache_chunk_table(d, cache_name, 0, take))
        n += take
        del d
    if not tables:
        return 0
    import pyarrow as pa
    table = pa.concat_tables(tables)
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)
    return n


def push_pending(
    out_dir: Path,
    repo: str,
    *,
    staging: Path,
    token: str,
    remote_dir: str = "data",
    stream_rows: int = 0,
    pack: bool = False,
) -> dict:
    from huggingface_hub import HfApi, create_repo

    inbox = out_dir / "inbox"
    state = load_state(out_dir, repo)
    api = HfApi(token=token)
    create_repo(repo, repo_type="dataset", private=False, exist_ok=True, token=token)

    shards = list_ready_shards(inbox)
    pending = [sh for sh in shards if sh.name not in state["uploaded"]]
    log(f"repo={repo} ready={len(shards)} uploaded={len(state['uploaded'])} pending={len(pending)}")
    if not pending and state["n_uploaded"] == 0:
        write_readme(staging / "README.md", repo=repo, n_rows=0, n_shards=0)
        api.upload_file(
            path_or_fileobj=str(staging / "README.md"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="init public disagreement dataset",
            token=token,
        )

    if pack and pending:
        pack_i = int(state.get("pack_i", 0))
        pack_name = f"pack_{pack_i:06d}"
        parquet = staging / f"{pack_name}.parquet"
        n = convert_shards_packed(pending, parquet, pack_name)
        remote = f"{remote_dir.rstrip('/')}/{pack_name}.parquet"
        api.upload_file(
            path_or_fileobj=str(parquet),
            path_in_repo=remote,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add {pack_name} n={n:,} from {len(pending)} inbox shards",
            token=token,
        )
        for sh in pending:
            state["uploaded"][sh.name] = {"path": remote, "n": 0, "packed_into": pack_name}
        state["uploaded"][pack_name] = {"path": remote, "n": n}
        state["n_uploaded"] = int(sum(v.get("n", 0) for v in state["uploaded"].values()))
        state["pack_i"] = pack_i + 1
        save_state(out_dir, state)
        log(f"uploaded {pack_name} n={n:,} total={state['n_uploaded']:,} packed={len(pending)}")
        try:
            parquet.unlink()
        except OSError:
            pass
        write_readme(
            staging / "README.md",
            repo=repo,
            n_rows=stream_rows + int(state["n_uploaded"]),
            n_shards=(83 if stream_rows else 0) + int(state.get("pack_i", 0)),
            stream_rows=stream_rows,
            sf19_rows=int(state["n_uploaded"]),
            sf19_shards=int(state.get("pack_i", 0)),
        )
        api.upload_file(
            path_or_fileobj=str(staging / "README.md"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"card: stream {stream_rows:,} + sf19 {state['n_uploaded']:,}",
            token=token,
        )
        return state

    for i, sh in enumerate(pending, 1):
        parquet = staging / f"{sh.name}.parquet"
        n = convert_shard(sh, parquet)
        remote = f"{remote_dir.rstrip('/')}/{sh.name}.parquet"
        api.upload_file(
            path_or_fileobj=str(parquet),
            path_in_repo=remote,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add {sh.name} n={n:,}",
            token=token,
        )
        state["uploaded"][sh.name] = {"path": remote, "n": n}
        state["n_uploaded"] = int(sum(v["n"] for v in state["uploaded"].values()))
        save_state(out_dir, state)
        log(f"uploaded {sh.name} n={n:,} total={state['n_uploaded']:,} ({i}/{len(pending)})")
        try:
            parquet.unlink()
        except OSError:
            pass
        if i == 1 or i == len(pending) or i % 10 == 0:
            sf19_rows = int(state["n_uploaded"])
            write_readme(
                staging / "README.md",
                repo=repo,
                n_rows=stream_rows + sf19_rows,
                n_shards=(83 if stream_rows else 0) + len(state["uploaded"]),
                stream_rows=stream_rows,
                sf19_rows=sf19_rows,
                sf19_shards=len(state["uploaded"]),
            )
            api.upload_file(
                path_or_fileobj=str(staging / "README.md"),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
                commit_message=(
                    f"card: stream {stream_rows:,} + sf19 {sf19_rows:,}"
                    if stream_rows else f"card: {sf19_rows:,} rows"
                ),
                token=token,
            )
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--inbox-root", default=str(DEFAULT_INBOX))
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--watch", action="store_true", help="Keep uploading new harvest shards")
    ap.add_argument("--poll-s", type=float, default=90.0)
    ap.add_argument("--remote-dir", default="data", help="HF path prefix for parquet shards")
    ap.add_argument("--stream-rows", type=int, default=0,
                    help="Already-uploaded stream-filter rows to show on the card")
    ap.add_argument("--pack", action="store_true",
                    help="Merge pending inbox shards into one parquet per watch cycle")
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")

    token = _hf_token()
    if not token:
        raise SystemExit("HF_TOKEN / HUGGING_FACE_HUB_TOKEN missing")

    out_dir = Path(args.inbox_root)
    staging = out_dir / "hf_staging"
    staging.mkdir(parents=True, exist_ok=True)

    while True:
        state = push_pending(
            out_dir, args.repo, staging=staging, token=token,
            remote_dir=args.remote_dir, stream_rows=args.stream_rows,
            pack=args.pack,
        )
        log(
            f"https://huggingface.co/datasets/{args.repo} "
            f"rows={state['n_uploaded']:,} shards={len(state['uploaded'])} "
            f"remote={args.remote_dir}"
        )
        if not args.watch:
            break
        time.sleep(max(args.poll_s, 15.0))


if __name__ == "__main__":
    main()
