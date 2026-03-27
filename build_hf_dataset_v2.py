#!/usr/bin/env python3
"""Build a canonical deduped HF dataset from an existing shard-first dataset.

This script creates a new Hugging Face dataset repo from an existing one by:
1. Downloading the source parquet shards locally.
2. Globally deduplicating rows on (fen, best_move, eval_type, eval_value).
3. Reassigning train/test splits deterministically by FEN hash after dedupe.
4. Exporting a cleaner, documented schema as parquet shards.
5. Uploading the new dataset repo with a generated README dataset card.

The dedupe/export pass is streamed through DuckDB directly from parquet input
into parquet output to avoid materializing the full corpus in Python memory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi


DEFAULT_SOURCE_REPO = "avewright/chess-positions-lichess-sf"
DEFAULT_TARGET_REPO = "avewright/chess-positions-lichess-sf-v2-full-dedup-rowkey"
DEFAULT_WORK_DIR = Path("outputs") / "hf_dataset_v2"
DEFAULT_BUILD_DIR = DEFAULT_WORK_DIR / "build"
DEFAULT_EXPORT_DIR = DEFAULT_WORK_DIR / "export"
DEFAULT_STATS_PATH = DEFAULT_WORK_DIR / "stats.json"
DEFAULT_CARD_PATH = DEFAULT_WORK_DIR / "README.md"
REMOTE_BASE_TMPL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
KNOWN_SOURCE_SHARD_COUNTS = {
    "avewright/chess-positions-lichess-sf": {
        "00000": 193,
        "00001": 187,
        "00002": 187,
        "00003": 187,
        "00004": 187,
        "00005": 187,
        "00006": 192,
        "00007": 192,
        "00008": 195,
        "00009": 196,
        "00010": 196,
        "00011": 196,
        "00012": 196,
        "00013": 196,
        "00014": 196,
        "00015": 196,
        "00016": 196,
    }
}


EXPORTED_COLUMNS = [
    "fen",
    "best_move",
    "eval_type",
    "eval_value",
    "wdl_win",
    "wdl_draw",
    "wdl_loss",
    "phase",
    "num_legal",
    "source",
    "game_id",
    "top_moves",
    "ply",
    "depth",
    "split",
    "source_file",
    "position_key",
    "dedup_key",
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_hf_token() -> str | None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def discover_source_files(source_dir: Path, max_source_files: int | None = None) -> list[Path]:
    files = sorted((source_dir / "data").glob("*-src*.parquet"))
    if max_source_files is not None:
        files = files[:max_source_files]
    if not files:
        raise FileNotFoundError(
            f"No source parquet files found under {source_dir / 'data'}. "
            "Run without --skip-download or point --source-dir at an existing snapshot."
        )
    return files


def list_remote_source_urls(source_repo: str, max_source_files: int | None = None) -> list[str]:
    files: list[str] | None = None
    try:
        api = HfApi(token=load_hf_token())
        files = sorted(
            f for f in api.list_repo_files(source_repo, repo_type="dataset")
            if f.startswith("data/") and "-src" in f and f.endswith(".parquet")
        )
    except Exception:
        counts = KNOWN_SOURCE_SHARD_COUNTS.get(source_repo)
        if counts is None:
            raise
        files = []
        for src_idx, count in sorted(counts.items()):
            for shard_idx in range(count):
                files.append(f"data/train-src{src_idx}-{shard_idx:05d}.parquet")
            # The repo also contains per-source test shards; include them when reconstructing.
            for shard_idx in range(2):
                files.append(f"data/test-src{src_idx}-{shard_idx:05d}.parquet")
    if max_source_files is not None:
        files = files[:max_source_files]
    if not files:
        raise FileNotFoundError(f"No source parquet files found in dataset repo {source_repo}")
    return [REMOTE_BASE_TMPL.format(repo=source_repo, path=path) for path in files]


def build_source_relation_sql(source_files: list[str | Path], test_ratio: float) -> str:
    file_list_sql = ", ".join(
        _sql_quote(str(p.resolve()) if isinstance(p, Path) else p)
        for p in source_files
    )
    threshold = int(test_ratio * 10_000)
    return f"""
with src as (
    select
        fen,
        best_move,
        eval_type,
        eval_value,
        wdl_win,
        wdl_draw,
        wdl_loss,
        phase,
        num_legal,
        source,
        game_id,
        top_moves,
        ply,
        depth,
        regexp_extract(filename, '([^/\\\\]+)$', 1) as source_file
    from read_parquet([{file_list_sql}], filename=true)
),
ranked as (
    select
        fen,
        best_move,
        eval_type,
        eval_value,
        wdl_win,
        wdl_draw,
        wdl_loss,
        phase,
        num_legal,
        source,
        game_id,
        top_moves,
        ply,
        depth,
        case
            when (hash(fen) % 10000) < {threshold} then 'test'
            else 'train'
        end as split,
        source_file,
        fen as position_key,
        md5(concat_ws('|', fen, best_move, eval_type, cast(eval_value as varchar))) as dedup_key,
        row_number() over (
            partition by fen, best_move, eval_type, eval_value
            order by source_file
        ) as rn
    from src
)
select {", ".join(EXPORTED_COLUMNS)}
from ranked
where rn = 1
"""


def connect_duckdb(build_dir: Path) -> duckdb.DuckDBPyConnection:
    ensure_dir(build_dir)
    ext_dir = ensure_dir(build_dir / "duckdb_extensions")
    temp_dir = ensure_dir(build_dir / "duckdb_temp")
    db_path = build_dir / "dedupe.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET temp_directory = {_sql_quote(str(temp_dir.resolve()))}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET extension_directory = {_sql_quote(str(ext_dir.resolve()))}")
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
    except duckdb.Error:
        # Local-file-only runs do not need httpfs.
        pass
    return con


def export_split(
    con: duckdb.DuckDBPyConnection,
    base_query_sql: str,
    split_name: str,
    export_root: Path,
) -> list[Path]:
    split_dir = export_root / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    ensure_dir(split_dir)

    out_dir = split_dir / f"{split_name}.parquet"
    query = f"""
copy (
    select * from ({base_query_sql}) as deduped
    where split = {_sql_quote(split_name)}
) to {_sql_quote(str(out_dir.resolve()))}
(format parquet, compression zstd, per_thread_output true, row_group_size 100000)
"""
    print(f"\nExporting {split_name} split to {out_dir} ...", flush=True)
    t0 = time.time()
    con.execute(query)
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"DuckDB export produced no parquet files for split {split_name}")
    print(f"  Exported {len(files)} parquet file(s) in {time.time() - t0:.1f}s", flush=True)
    return files


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def sum_input_rows(source_files: list[str | Path]) -> int:
    total = 0
    for path in source_files:
        if isinstance(path, Path):
            total += parquet_row_count(path)
    return total


def relabeled_export_files(export_files: list[Path], split_name: str) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for idx, path in enumerate(sorted(export_files)):
        items.append((path, f"data/{split_name}-{idx:05d}.parquet"))
    return items


def compute_stats(
    con: duckdb.DuckDBPyConnection,
    export_files_by_split: dict[str, list[Path]],
    source_repo: str,
    target_repo: str,
    source_files: list[str | Path],
    test_ratio: float,
) -> dict:
    all_exported = export_files_by_split["train"] + export_files_by_split["test"]
    export_list_sql = ", ".join(_sql_quote(str(p.resolve())) for p in all_exported)

    total_rows, unique_positions = con.execute(
        f"""
        select count(*) as total_rows, count(distinct position_key) as unique_positions
        from read_parquet([{export_list_sql}])
        """
    ).fetchone()

    split_rows = {}
    for split_name, files in export_files_by_split.items():
        split_rows[split_name] = int(sum(parquet_row_count(p) for p in files))

    input_rows = sum_input_rows(source_files)

    stats = {
        "created_at": utc_now(),
        "source_repo": source_repo,
        "target_repo": target_repo,
        "dedupe_policy": {
            "name": "full-dedup-rowkey",
            "rowkey_fields": ["fen", "best_move", "eval_type", "eval_value"],
            "keep_policy": "first source_file in lexicographic order",
        },
        "split_policy": {
            "field": "fen",
            "bucket_modulus": 10000,
            "test_ratio": test_ratio,
            "rule": f"(hash(fen) % 10000) < {int(test_ratio * 10000)}",
        },
        "schema_columns": EXPORTED_COLUMNS,
        "source_files": len(source_files),
        "input_rows": int(input_rows) if input_rows else None,
        "output_rows": int(total_rows),
        "duplicate_rows_removed": int(input_rows - total_rows) if input_rows else None,
        "duplicate_rate_removed": ((input_rows - total_rows) / max(1, input_rows)) if input_rows else None,
        "unique_positions": int(unique_positions),
        "train_rows": split_rows["train"],
        "test_rows": split_rows["test"],
        "train_files": len(export_files_by_split["train"]),
        "test_files": len(export_files_by_split["test"]),
    }
    return stats


def render_dataset_card(stats: dict) -> str:
    duplicate_rate_pct = None if stats["duplicate_rate_removed"] is None else 100.0 * stats["duplicate_rate_removed"]
    dedupe_summary_lines = [
        f"- Source repo: `{stats['source_repo']}`",
        f"- Source parquet files: `{stats['source_files']:,}`",
    ]
    if stats["input_rows"] is not None:
        dedupe_summary_lines.extend([
            f"- Input rows: `{stats['input_rows']:,}`",
            f"- Output rows: `{stats['output_rows']:,}`",
            f"- Exact duplicate rows removed: `{stats['duplicate_rows_removed']:,}` (`{duplicate_rate_pct:.2f}%`)",
        ])
    else:
        dedupe_summary_lines.append(f"- Output rows: `{stats['output_rows']:,}`")
        dedupe_summary_lines.append("- Duplicate-removal counts omitted because the input was scanned remotely")
    return f"""---
pretty_name: Chess Positions Lichess SF V2 Full Dedup Rowkey
license: mit
task_categories:
- text-classification
language:
- en
tags:
- chess
- stockfish
- policy-value
- deduplicated
- parquet
size_categories:
- 100M<n<1B
---

# {stats["target_repo"].split("/")[-1]}

Canonical rebuild of `{stats["source_repo"]}` with a documented schema and global exact deduplication on:

- `fen`
- `best_move`
- `eval_type`
- `eval_value`

This variant is `v2-full-dedup-rowkey`.

## What Changed

- Global exact dedupe across all source shards
- Deterministic split assignment after dedupe
- Stable provenance column: `source_file`
- Explicit `position_key` and `dedup_key`
- Flat parquet shard layout under `data/train-*.parquet` and `data/test-*.parquet`

## Split Policy

Splits are assigned *after dedupe* using a deterministic FEN hash. This avoids exact duplicate leakage across train/test.

- Test ratio: `{stats["split_policy"]["test_ratio"]:.2%}`
- Rule: `{stats["split_policy"]["rule"]}`

## Dedup Policy

- Policy name: `{stats["dedupe_policy"]["name"]}`
- Row key: `{", ".join(stats["dedupe_policy"]["rowkey_fields"])}`
- Keep rule: `{stats["dedupe_policy"]["keep_policy"]}`

## Row Schema

{chr(10).join(f"- `{col}`" for col in stats["schema_columns"])}

## Summary Stats

{chr(10).join(dedupe_summary_lines)}
- Unique FEN positions after dedupe: `{stats["unique_positions"]:,}`
- Train rows: `{stats["train_rows"]:,}` across `{stats["train_files"]}` file(s)
- Test rows: `{stats["test_rows"]:,}` across `{stats["test_files"]}` file(s)

## Notes

- This dataset removes only **exact** duplicates on the documented row key.
- It does not attempt semantic or near-duplicate clustering.
- Multiple labels for the same FEN may still exist if `best_move` or evaluation fields differ.
"""


def upload_dataset(
    repo_id: str,
    export_files_by_split: dict[str, list[Path]],
    card_path: Path,
    stats_path: Path,
) -> None:
    token = load_hf_token()
    if not token:
        raise RuntimeError("No HF_TOKEN found in .env or environment.")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    operations: list[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)),
        CommitOperationAdd(path_in_repo="stats.json", path_or_fileobj=str(stats_path)),
    ]
    for split_name in ("train", "test"):
        operations.extend(
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
            for local_path, repo_path in relabeled_export_files(export_files_by_split[split_name], split_name)
        )

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish v2 full dedup rowkey dataset",
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and publish a deduped v2 HF chess dataset")
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, help="Source HF dataset repo")
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO, help="Target HF dataset repo")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Working directory")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_WORK_DIR / "source_snapshot", help="Local source snapshot directory")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="DuckDB working directory")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Output parquet directory")
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH, help="Output stats JSON path")
    parser.add_argument("--card-path", type=Path, default=DEFAULT_CARD_PATH, help="Output README path")
    parser.add_argument("--test-ratio", type=float, default=0.01, help="Deterministic test split ratio after dedupe")
    parser.add_argument("--skip-download", action="store_true", help="Use local parquet files under --source-dir instead of remote URLs")
    parser.add_argument("--skip-upload", action="store_true", help="Build locally but do not upload")
    parser.add_argument("--max-source-files", type=int, default=None, help="Limit input files for local dry runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.work_dir)
    ensure_dir(args.export_dir)

    if args.skip_download:
        source_files = discover_source_files(args.source_dir, max_source_files=args.max_source_files)
    else:
        print(f"Using remote parquet URLs from {args.source_repo} ...", flush=True)
        source_files = list_remote_source_urls(args.source_repo, max_source_files=args.max_source_files)
    print(f"Using {len(source_files):,} source parquet files", flush=True)

    con = connect_duckdb(args.build_dir)
    base_query_sql = build_source_relation_sql(source_files, args.test_ratio)

    export_files_by_split = {
        "train": export_split(con, base_query_sql, "train", args.export_dir),
        "test": export_split(con, base_query_sql, "test", args.export_dir),
    }

    stats = compute_stats(
        con,
        export_files_by_split,
        source_repo=args.source_repo,
        target_repo=args.target_repo,
        source_files=source_files,
        test_ratio=args.test_ratio,
    )
    args.stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.card_path.write_text(render_dataset_card(stats), encoding="utf-8")

    print("\nBuild summary:")
    print(json.dumps({
        "input_rows": stats["input_rows"],
        "output_rows": stats["output_rows"],
        "duplicate_rows_removed": stats["duplicate_rows_removed"],
        "train_rows": stats["train_rows"],
        "test_rows": stats["test_rows"],
        "stats_path": str(args.stats_path),
        "card_path": str(args.card_path),
    }, indent=2))

    if args.skip_upload:
        print("\nSkipping upload.")
        return

    print(f"\nUploading dataset to {args.target_repo} ...", flush=True)
    upload_dataset(args.target_repo, export_files_by_split, args.card_path, args.stats_path)
    print(f"Uploaded: https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise
