#!/usr/bin/env python3
"""Harvest >=26-piece best-first-move rows from Lichess evals.

Downloads one source parquet at a time, keeps the deepest / highest-knodes
PV1 per FEN, writes READY inbox shards. Does not snapshot the 42GB pack.

  MOVE_VOCAB_VERSION=compact python3 -u scripts/build_lichess_opening_bestline.py --go
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_lichess_evals_soft_cache import (  # noqa: E402
    accumulate_shard,
    write_inbox_shard,
)
from upload_exp201_hf import load_hf_token  # noqa: E402

SOURCE_REPO = "Lichess/chess-position-evaluations"
INBOX = ROOT / "outputs" / "lichess_opening_bestline" / "inbox"
MIN_PCS, MAX_PCS = 26, 32


def log(msg: str) -> None:
    print(msg, flush=True)


def list_source_parquets(token: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    files = [
        f
        for f in HfApi(token=token).list_repo_files(SOURCE_REPO, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    files.sort()
    return files


def download_parquet(name: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            SOURCE_REPO,
            name,
            repo_type="dataset",
            token=token,
        )
    )


def harvest(args: argparse.Namespace) -> None:
    token = load_hf_token()
    files = list_source_parquets(token)
    if args.max_shards > 0:
        files = files[: args.max_shards]
    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    target = args.target if args.target > 0 else 10**18
    log(
        f"opening harvest files={len(files)} pieces={MIN_PCS}-{MAX_PCS} "
        f"one_hot min_depth={args.min_depth} min_knodes={args.min_knodes} "
        f"flush_every={args.flush_every} target={'all' if args.target <= 0 else f'{args.target:,}'}"
    )

    acc: dict = {}
    written: set[str] = set()
    flushed = total_rows = total_kept = 0
    t0 = time.time()

    def _flush(chunk: dict) -> None:
        nonlocal flushed
        sh = write_inbox_shard(
            inbox,
            chunk,
            one_hot=True,
            tau=args.tau,
            meta={"min_pieces": MIN_PCS, "max_pieces": MAX_PCS, "phase": "opening"},
        )
        flushed += 1
        log(f"flush {sh.name} n={len(chunk):,} written={len(written) + len(chunk):,} inbox={inbox}")

    for i, name in enumerate(files):
        sealed = len(written) + len(acc)
        if sealed >= target:
            log(f"target reached unique={sealed:,} — stop before {name}")
            break
        log(f"[{i+1}/{len(files)}] download {name} acc={len(acc):,} written={len(written):,}")
        path = download_parquet(name, token)
        rows, kept = accumulate_shard(
            path,
            acc,
            min_depth=args.min_depth,
            min_knodes=args.min_knodes,
            target=target,
            batch_rows=args.batch_rows,
            min_pieces=MIN_PCS,
            max_pieces=MAX_PCS,
            one_hot=True,
            written=written,
            flush_every=int(args.flush_every),
            flush_cb=_flush,
        )
        total_rows += rows
        total_kept += kept
        if acc:
            _flush(acc)
            written.update(acc.keys())
            acc.clear()
        sealed = len(written)
        rate = sealed / max(time.time() - t0, 1e-6)
        log(
            f"  rows={rows:,} kept={kept:,} unique={sealed:,} "
            f"({rate:.0f} fen/s wall) {time.time()-t0:.0f}s"
        )
    if acc:
        _flush(acc)
        written.update(acc.keys())
        acc.clear()
    log(
        f"opening_done shards={flushed} unique={len(written):,} "
        f"rows_scanned={total_rows:,} {time.time()-t0:.1f}s inbox={inbox}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--inbox", default=str(INBOX))
    ap.add_argument("--min-depth", type=int, default=22)
    ap.add_argument("--min-knodes", type=int, default=5000)
    ap.add_argument("--tau", type=float, default=120.0)
    ap.add_argument("--batch-rows", type=int, default=250_000)
    ap.add_argument("--flush-every", type=int, default=200_000)
    ap.add_argument("--target", type=int, default=0, help="0 = no unique-FEN cap")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all 20 source files")
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")
    harvest(args)


if __name__ == "__main__":
    main()
