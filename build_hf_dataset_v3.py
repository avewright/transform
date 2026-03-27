#!/usr/bin/env python3
"""build_hf_dataset_v3.py - Streaming dedup build, one shard at a time.

Processes source parquet files individually, maintaining a disk-backed
DuckDB table of seen dedup-key hashes.  This allows building the full
870M+ row corpus without materialising the entire dataset at once.

Key differences from build_hf_dataset_v2.py:
  - Streams one remote file at a time (reads each URL once via httpfs)
  - Disk-backed hash table for dedup keys (~8 bytes/key in DuckDB)
  - Writes output parquet shards incrementally via pyarrow
  - Resume support: tracks processed files, skips on restart
  - Per-file progress with ETA + periodic heartbeat

Usage:
    python build_hf_dataset_v3.py --work-dir /path/to/workdir [--max-source-files N] [--skip-upload]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi

# -- constants ----------------------------------------------------------

DEFAULT_SOURCE_REPO = "avewright/chess-positions-lichess-sf"
DEFAULT_TARGET_REPO = "avewright/chess-positions-lichess-sf-v2-full-dedup-rowkey"
REMOTE_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
ROWS_PER_SHARD = 1_000_000

OUTPUT_COLUMNS = [
    "fen", "best_move", "eval_type", "eval_value",
    "wdl_win", "wdl_draw", "wdl_loss",
    "phase", "num_legal", "source", "game_id", "top_moves",
    "ply", "depth", "split", "source_file",
]


# -- helpers ------------------------------------------------------------

def _q(s: str) -> str:
    """SQL-quote a string literal."""
    return "'" + s.replace("'", "''") + "'"


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hf_token() -> str | None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def _eta(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    return f"{secs / 60:.1f}m" if secs < 3600 else f"{secs / 3600:.1f}h"


# -- source file listing -----------------------------------------------

def list_source_urls(repo: str, limit: int | None = None) -> list[str]:
    api = HfApi(token=_hf_token())
    paths = sorted(
        p for p in api.list_repo_files(repo, repo_type="dataset")
        if p.startswith("data/") and "-src" in p and p.endswith(".parquet")
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No source parquets in {repo}")
    return [REMOTE_BASE.format(repo=repo, path=p) for p in paths]


# -- DuckDB setup ------------------------------------------------------

def open_db(build_dir: Path) -> duckdb.DuckDBPyConnection:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "tmp").mkdir(exist_ok=True)
    (build_dir / "ext").mkdir(exist_ok=True)

    con = duckdb.connect(str(build_dir / "dedupe.duckdb"))
    con.execute(f"SET temp_directory = {_q(str((build_dir / 'tmp').resolve()))}")
    con.execute(f"SET extension_directory = {_q(str((build_dir / 'ext').resolve()))}")
    con.execute("SET preserve_insertion_order = false")
    try:
        con.execute("INSTALL httpfs; LOAD httpfs")
    except duckdb.Error:
        pass

    # dedup state - UBIGINT primary key for compact storage
    con.execute("CREATE TABLE IF NOT EXISTS seen_keys (dk UBIGINT PRIMARY KEY)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            url      VARCHAR PRIMARY KEY,
            rows_in  INTEGER,
            rows_new INTEGER,
            ts       TIMESTAMP DEFAULT current_timestamp
        )
    """)
    return con


# -- shard writer ------------------------------------------------------

class ShardWriter:
    """Incrementally write parquet shards for one split, with resume."""

    def __init__(self, root: Path, split: str, rows_per_shard: int = ROWS_PER_SHARD):
        self.dir = root / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.max_rows = rows_per_shard

        # resume: continue numbering after existing shards
        existing = sorted(self.dir.glob(f"{split}-*.parquet"))
        self._idx = len(existing)
        self.files: list[Path] = list(existing)
        self.total_rows = sum(pq.read_metadata(str(f)).num_rows for f in existing)

        self._writer: pq.ParquetWriter | None = None
        self._rows = 0

    def write(self, table: pa.Table) -> None:
        if len(table) == 0:
            return
        off = 0
        while off < len(table):
            if self._writer is None:
                self._open(table.schema)
            space = self.max_rows - self._rows
            n = min(len(table) - off, space)
            self._writer.write_table(table.slice(off, n))
            self._rows += n
            self.total_rows += n
            off += n
            if self._rows >= self.max_rows:
                self._close()

    def _open(self, schema: pa.Schema) -> None:
        p = self.dir / f"{self.split}-{self._idx:05d}.parquet"
        self._writer = pq.ParquetWriter(str(p), schema, compression="zstd")
        self._rows = 0
        self.files.append(p)

    def _close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None
            self._idx += 1
            self._rows = 0

    def close(self) -> None:
        self._close()


# -- per-file processing -----------------------------------------------

def process_file(
    con: duckdb.DuckDBPyConnection,
    url: str,
    test_thr: int,
    tw: ShardWriter,
    ew: ShardWriter,
) -> tuple[int, int]:
    """Process one remote parquet file.  Returns (rows_in, rows_new)."""
    uq = _q(url)
    sfq = _q(url.rsplit("/", 1)[-1])

    # Materialise one file into a temp table (reads remote URL once)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _b AS
        SELECT fen, best_move, eval_type, eval_value,
               wdl_win, wdl_draw, wdl_loss,
               phase, num_legal, source, game_id, top_moves,
               ply, depth,
               {sfq} AS source_file,
               hash(concat_ws('|', fen, best_move, eval_type,
                    cast(eval_value AS varchar)))::UBIGINT AS dk
        FROM read_parquet({uq})
    """)
    rows_in = con.execute("SELECT count(*) FROM _b").fetchone()[0]

    # Intra-file dedup (row_number) + inter-file dedup (anti-join seen_keys)
    arrow = con.execute(f"""
        WITH d AS (
            SELECT *, row_number() OVER (PARTITION BY dk) AS rn FROM _b
        )
        SELECT fen, best_move, eval_type, eval_value,
               wdl_win, wdl_draw, wdl_loss,
               phase, num_legal, source, game_id, top_moves,
               ply, depth, source_file,
               CASE WHEN (hash(fen) % 10000) < {test_thr}
                    THEN 'test' ELSE 'train' END AS split,
               dk
        FROM d
        WHERE rn = 1
          AND NOT EXISTS (SELECT 1 FROM seen_keys sk WHERE sk.dk = d.dk)
    """).fetch_arrow_table()

    rows_new = len(arrow)

    if rows_new > 0:
        # Bulk-insert new dedup keys
        con.register("_a", arrow)
        con.execute("INSERT OR IGNORE INTO seen_keys SELECT DISTINCT dk FROM _a")
        con.unregister("_a")

        # Write output (drop internal dk column)
        out = arrow.drop(["dk"])
        train_mask = pc.equal(out.column("split"), "train")
        test_mask = pc.equal(out.column("split"), "test")
        tw.write(out.filter(train_mask))
        ew.write(out.filter(test_mask))

    con.execute("DROP TABLE IF EXISTS _b")

    # Record this file as processed (for resume)
    con.execute(
        "INSERT OR IGNORE INTO processed_files (url, rows_in, rows_new) VALUES (?, ?, ?)",
        [url, rows_in, rows_new],
    )
    return rows_in, rows_new


# -- dataset card ------------------------------------------------------

def render_card(s: dict) -> str:
    return f"""---
pretty_name: Chess Positions Lichess SF V2 (Streaming Dedup)
license: mit
tags: [chess, stockfish, deduplicated]
size_categories: [100M<n<1B]
---

# {s["target_repo"].split("/")[-1]}

Streaming rebuild of `{s["source_repo"]}` with global exact dedup
on **(fen, best_move, eval_type, eval_value)**.

## Build Stats

| Metric | Value |
|---|---|
| Source files | {s["source_files"]:,} |
| Input rows | {s["input_rows"]:,} |
| Output rows | {s["output_rows"]:,} |
| Duplicates removed | {s["duplicates_removed"]:,} ({s["dup_rate"]:.2%}) |
| Unique dedup keys | {s["unique_keys"]:,} |
| Train | {s["train_rows"]:,} rows across {s["train_shards"]} shard(s) |
| Test | {s["test_rows"]:,} rows across {s["test_shards"]} shard(s) |

## Dedup Policy

- **Key**: `hash(fen || best_move || eval_type || eval_value)` as UBIGINT
- **Keep rule**: first seen (source files processed in lexicographic order)
- **Scope**: global across all source shards

## Split Policy

- Deterministic FEN-hash split applied *after* dedup
- Rule: `{s["split_policy"]["rule"]}`
- Test ratio: `{s["split_policy"]["test_ratio"]:.2%}`

## Schema

{chr(10).join(f"- `{col}`" for col in s["schema_columns"])}
"""


# -- upload ------------------------------------------------------------

def upload_dataset(
    repo_id: str,
    tw: ShardWriter,
    ew: ShardWriter,
    stats_path: Path,
    card_path: Path,
) -> None:
    token = _hf_token()
    if not token:
        raise RuntimeError("No HF_TOKEN found in .env or environment")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    ops: list[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)),
        CommitOperationAdd(path_in_repo="stats.json", path_or_fileobj=str(stats_path)),
    ]
    for w in (tw, ew):
        for i, f in enumerate(sorted(w.files)):
            ops.append(CommitOperationAdd(
                path_in_repo=f"data/{w.split}-{i:05d}.parquet",
                path_or_fileobj=str(f),
            ))

    api.create_commit(
        repo_id, repo_type="dataset", operations=ops,
        commit_message="Publish streaming-dedup dataset",
    )


# -- main --------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming dedup dataset build")
    ap.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    ap.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--test-ratio", type=float, default=0.01)
    ap.add_argument("--max-source-files", type=int, default=None)
    ap.add_argument("--rows-per-shard", type=int, default=ROWS_PER_SHARD)
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    work = args.work_dir
    bld = work / "build"
    exp = work / "export"
    sp = work / "stats.json"
    cp = work / "README.md"
    test_thr = int(args.test_ratio * 10_000)

    # Discover source files
    print(f"Listing source files from {args.source_repo} ...", flush=True)
    urls = list_source_urls(args.source_repo, args.max_source_files)
    print(f"Found {len(urls):,} source files\n", flush=True)

    # Connect DuckDB (creates tables if first run)
    con = open_db(bld)

    # Resume support: find already-processed files
    done = {r[0] for r in con.execute("SELECT url FROM processed_files").fetchall()}
    todo = [u for u in urls if u not in done]
    if len(todo) < len(urls):
        print(
            f"Resuming: {len(urls) - len(todo):,} already done, "
            f"{len(todo):,} remaining\n",
            flush=True,
        )

    # Shard writers (auto-continue numbering on resume)
    tw = ShardWriter(exp, "train", args.rows_per_shard)
    ew = ShardWriter(exp, "test", args.rows_per_shard)

    cum_in = cum_new = 0
    errors = []
    t0 = time.time()

    for i, url in enumerate(todo, 1):
        tf = time.time()
        try:
            r_in, r_new = process_file(con, url, test_thr, tw, ew)
        except Exception as exc:
            fname = url.rsplit("/", 1)[-1]
            print(f"  ERROR on {fname}: {exc}", file=sys.stderr, flush=True)
            errors.append((fname, str(exc)))
            continue

        cum_in += r_in
        cum_new += r_new
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (len(todo) - i) / rate if rate else 0
        dup_pct = (1 - r_new / max(1, r_in)) * 100
        done_n = len(urls) - len(todo) + i

        print(
            f"[{done_n}/{len(urls)}] {url.rsplit('/', 1)[-1]}  "
            f"{r_in:>8,} in | {r_new:>8,} new | {dup_pct:4.1f}% dup | "
            f"cum {cum_new:>12,} | {time.time() - tf:.1f}s | ETA {_eta(eta)}",
            flush=True,
        )

        # Heartbeat every 50 files
        if i % 50 == 0:
            sk = con.execute("SELECT count(*) FROM seen_keys").fetchone()[0]
            print(
                f"\n  === HEARTBEAT ===  files {done_n}/{len(urls)}  "
                f"rows {cum_new:,}/{cum_in:,}  "
                f"keys {sk:,}  train {tw.total_rows:,}  test {ew.total_rows:,}  "
                f"elapsed {_eta(elapsed)}\n",
                flush=True,
            )

    tw.close()
    ew.close()

    # Final stats
    sk = con.execute("SELECT count(*) FROM seen_keys").fetchone()[0]
    stats = {
        "created_at": _utcnow(),
        "source_repo": args.source_repo,
        "target_repo": args.target_repo,
        "source_files": len(urls),
        "input_rows": cum_in,
        "output_rows": cum_new,
        "duplicates_removed": cum_in - cum_new,
        "dup_rate": round((cum_in - cum_new) / max(1, cum_in), 4),
        "unique_keys": sk,
        "train_rows": tw.total_rows,
        "test_rows": ew.total_rows,
        "train_shards": len(tw.files),
        "test_shards": len(ew.files),
        "errors": len(errors),
        "schema_columns": OUTPUT_COLUMNS,
        "dedupe_policy": {
            "key_fields": ["fen", "best_move", "eval_type", "eval_value"],
            "hash_fn": "duckdb hash() -> UBIGINT",
            "keep": "first seen (source files in lexicographic order)",
        },
        "split_policy": {
            "rule": f"(hash(fen) % 10000) < {test_thr}",
            "test_ratio": args.test_ratio,
        },
    }

    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cp.write_text(render_card(stats), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(json.dumps({
        "input_rows": stats["input_rows"],
        "output_rows": stats["output_rows"],
        "duplicates_removed": stats["duplicates_removed"],
        "dup_rate": f"{stats['dup_rate']:.2%}",
        "train_rows": stats["train_rows"],
        "test_rows": stats["test_rows"],
        "train_shards": stats["train_shards"],
        "test_shards": stats["test_shards"],
        "errors": stats["errors"],
    }, indent=2))

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for fname, exc in errors:
            print(f"  {fname}: {exc}")

    if args.skip_upload:
        print("\nUpload skipped.")
    else:
        print(f"\nUploading to {args.target_repo} ...", flush=True)
        upload_dataset(args.target_repo, tw, ew, sp, cp)
        print(f"Done: https://huggingface.co/datasets/{args.target_repo}")

    con.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
