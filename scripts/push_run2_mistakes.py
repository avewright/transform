#!/usr/bin/env python3
"""Upload run2 mistake shards to avewright/chess-soft-100m-run2-mistakes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import _hf_token  # noqa: E402
from export_soft_caches_to_hf import _fixed_list, cache_chunk_table  # noqa: E402
from harvest_swa_mistakes import I_TO_TAG  # noqa: E402

REPO = "avewright/chess-soft-100m-swa-mistakes"


def log(msg: str) -> None:
    print(msg, flush=True)


def convert(src: Path, dest: Path) -> int:
    d = torch.load(src / "soft_cache.pt", map_location="cpu", weights_only=False)
    n = int(d["move_idx"].shape[0])
    table = cache_chunk_table(d, src.name, 0, n)
    extras = [
        ("origin", pa.int8(), np.int8, 0),
        ("model_move_idx", pa.int64(), np.int64, 0),
        ("tag", pa.int8(), np.int8, 6),
        ("drop_cp", pa.int32(), np.int32, 0),
        ("model_in_pv", pa.int8(), np.int8, 0),
        ("needs_sf", pa.int8(), np.int8, 0),
        ("teacher_move_idx", pa.int64(), np.int64, -1),
    ]
    for name, ptype, ndt, default in extras:
        if name in d:
            col = np.asarray(d[name].cpu().numpy(), dtype=ndt)
        else:
            col = np.full(n, default, dtype=ndt)
        table = table.append_column(name, pa.array(col, type=ptype))
    for name, ndt in (("soft_cps", np.int32), ("soft_mates", np.int32)):
        if name in d:
            arr = np.asarray(d[name].cpu().numpy(), dtype=ndt)
            table = table.append_column(name, _fixed_list(arr, pa.int32() if ndt is np.int32 else pa.int32(), 8))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)
    return n


def write_card(path: Path, n_rows: int, n_shards: int, tags: dict) -> None:
    tag_rows = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(tags.items())) or "| — | 0 |"
    path.write_text(
        f"""---
license: mit
task_categories:
- other
tags:
- chess
- soft-labels
- disagreement
- blunder
- stockfish
pretty_name: 100M overnight SWA policy mistakes
---

# {REPO}

Positions where overnight `eval_swa.pt`
(`outputs/sf19_ft/overnight_20260908`, SF 8000-node screen score 0.719)
**disagrees** with a teacher best move.

In-PV severity uses teacher MultiPV **STM** cps. Off-PV rows are not given
an invented drop; they are queued for Stockfish 19 analysis (`needs_sf=1`).
Holdout + flip hashes from the overnight union (20,961) are excluded.

**{n_rows:,}** rows in **{n_shards}** shards.

## Tags

| tag | rows |
|---|---:|
{tag_rows}

`tag` codes: 0 ok, 1 off_pv, 2 inaccuracy, 3 blunder, 4 conversion, 5 major, 6 disagree (no PV eval).

## Columns

Standard soft schema plus `origin`, `model_move_idx`, `tag`, `drop_cp`,
`model_in_pv`, and `soft_cps` / `soft_mates` when the teacher pack had them
(SF19 yes; Lichess mix / Syzygy are disagreement-only).

`origin`: 0 = SF19, 1 = Lichess mix, 2 = Syzygy.

Overnight holdout+flip hashes (20,961) are excluded from every source.
Legal greedy move (top-16 remask), same as play.
Analyzed shards may also include `teacher_move_idx` and deeper SF19 MultiPV.
""",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--out-dir", default="outputs/swa_mistakes")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-s", type=float, default=60.0)
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")
    token = _hf_token()
    if not token:
        raise SystemExit("HF_TOKEN missing")
    from huggingface_hub import HfApi, create_repo

    out = Path(args.out_dir)
    staging = out / "hf_staging"
    staging.mkdir(parents=True, exist_ok=True)
    state_path = out / "hf_upload.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"uploaded": {}, "n": 0, "tags": {}}
    api = HfApi(token=token)
    create_repo(args.repo, repo_type="dataset", exist_ok=True, token=token)

    def once() -> None:
        sources = [
            (out / "analyzed" / "inbox", "data"),
            (out / "inbox", "data/scan"),
        ]
        pending: list[tuple[Path, str, str]] = []
        for inbox, remote_dir in sources:
            if not inbox.exists():
                continue
            for sh in sorted(p for p in inbox.glob("shard_*") if (p / "soft_cache.pt").exists()):
                key = f"{remote_dir}/{sh.name}"
                if key not in state["uploaded"]:
                    pending.append((sh, remote_dir, key))
        log(f"pending={len(pending)} uploaded={state['n']:,}")
        for sh, remote_dir, key in pending:
            parquet = staging / f"{key.replace('/', '_')}.parquet"
            n = convert(sh, parquet)
            remote = f"{remote_dir}/{sh.name}.parquet"
            api.upload_file(
                path_or_fileobj=str(parquet),
                path_in_repo=remote,
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=f"add {remote} n={n:,}",
                token=token,
            )
            state["uploaded"][key] = {"n": n, "path": remote}
            state["n"] = int(sum(v["n"] for v in state["uploaded"].values()))
            tags = {}
            d = torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)
            if "tag" in d:
                for i, c in zip(*np.unique(d["tag"].numpy(), return_counts=True)):
                    tags[I_TO_TAG.get(int(i), str(int(i)))] = int(c)
            for k, v in tags.items():
                state.setdefault("tags", {})
                state["tags"][k] = int(state["tags"].get(k, 0)) + v
            state_path.write_text(json.dumps(state, indent=2))
            log(f"uploaded {remote} n={n:,} total={state['n']:,}")
            parquet.unlink(missing_ok=True)
        card = staging / "README.md"
        write_card(card, state["n"], len(state["uploaded"]), state.get("tags") or {})
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"card {state['n']:,} rows",
            token=token,
        )
        log(f"https://huggingface.co/datasets/{args.repo} rows={state['n']:,}")

    while True:
        once()
        if not args.watch:
            break
        time.sleep(max(15.0, args.poll_s))


if __name__ == "__main__":
    main()
