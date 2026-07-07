#!/usr/bin/env python3
"""Process Lichess source parquets one at a time with persistent logging.

This runner is designed for long unattended jobs on ephemeral machines:
  - downloads exactly one source parquet at a time to large storage
  - processes each parquet into its own resumable shard directory
  - uploads the aggregate dataset after each completed source parquet
  - deletes temporary source parquet files after successful processing/upload
  - records enough state on disk to resume after crashes without redoing work
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


SOURCE_REPO = "Lichess/chess-position-evaluations"
TARGET_REPO = "avewright/chess-positions-lichess-sf"
DEFAULT_WORK_ROOT = Path("/workspace/chess_hf_pipeline")
LEGACY_SOURCE0_SHARDS = Path("/root/transform/outputs/hf_dataset_shards/lichess_sf_formatted")
LEGACY_LICHESS_CACHE = Path("/root/transform/outputs/lichess_cache")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def state_path(work_root: Path) -> Path:
    return work_root / "orchestrator_state.json"


def events_path(work_root: Path) -> Path:
    return work_root / "orchestrator_events.jsonl"


def append_event(work_root: Path, event: str, **payload) -> None:
    events_path(work_root).parent.mkdir(parents=True, exist_ok=True)
    with events_path(work_root).open("a") as f:
        f.write(json.dumps({"ts": utc_now(), "event": event, **payload}, sort_keys=True) + "\n")


def save_state(work_root: Path, state: dict) -> None:
    state_path(work_root).parent.mkdir(parents=True, exist_ok=True)
    state_path(work_root).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def default_state(args: argparse.Namespace) -> dict:
    return {
        "source_repo": args.source_repo,
        "target_repo": args.target_repo,
        "work_root": str(args.work_root),
        "created_at": utc_now(),
        "source_files": [],
        "sources": {},
        "last_uploaded_sources": [],
        "last_upload_completed_at": None,
        "last_upload_row_counts": {},
        "legacy_cache_cleared": False,
    }


def load_state(args: argparse.Namespace) -> dict:
    path = state_path(args.work_root)
    if not path.exists():
        return default_state(args)
    state = json.loads(path.read_text())
    for key, expected in {
        "source_repo": args.source_repo,
        "target_repo": args.target_repo,
        "work_root": str(args.work_root),
    }.items():
        if state.get(key) != expected:
            raise RuntimeError(
                f"Existing orchestrator state at {path} does not match `{key}`: "
                f"{state.get(key)!r} != {expected!r}"
            )
    return state


def ensure_env(args: argparse.Namespace) -> None:
    args.work_root.mkdir(parents=True, exist_ok=True)
    hf_home = args.work_root / "hf_home"
    tmp_root = args.work_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    tempfile.tempdir = str(tmp_root)


def load_hf_token() -> str | None:
    env_path = Path("/root/transform/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def list_source_files(args: argparse.Namespace) -> list[str]:
    api = HfApi(token=load_hf_token())
    files = api.list_repo_files(args.source_repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))


def register_source(state: dict, filename: str) -> dict:
    src = state["sources"].setdefault(filename, {})
    src.setdefault("filename", filename)
    src.setdefault("status", "pending")
    src.setdefault("attempts", 0)
    src.setdefault("downloaded_path", None)
    src.setdefault("shard_dir", None)
    src.setdefault("rows", {})
    src.setdefault("last_error", None)
    src.setdefault("processed_at", None)
    src.setdefault("uploaded_at", None)
    src.setdefault("cleaned_download", False)
    return src


def bootstrap_legacy_source0(args: argparse.Namespace, state: dict) -> None:
    source0 = "data/train-00000-of-00017.parquet"
    register_source(state, source0)
    src = state["sources"][source0]
    dest_dir = args.work_root / "processed_sources" / "train-00000-of-00017"

    if src.get("status") in {"processed", "uploaded"} and dest_dir.exists():
        return
    if not LEGACY_SOURCE0_SHARDS.exists():
        return

    legacy_progress_path = LEGACY_SOURCE0_SHARDS / "progress.json"
    if not legacy_progress_path.exists():
        return

    legacy_progress = json.loads(legacy_progress_path.read_text())
    if not legacy_progress.get("upload_completed"):
        return

    append_event(args.work_root, "bootstrap_source0_started", source=source0, from_dir=str(LEGACY_SOURCE0_SHARDS))
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if not dest_dir.exists():
        shutil.copytree(LEGACY_SOURCE0_SHARDS, dest_dir)
    src["status"] = "uploaded"
    src["shard_dir"] = str(dest_dir)
    src["rows"] = {
        "valid_rows": legacy_progress.get("valid_rows", 0),
        "train_rows": legacy_progress.get("train_rows", 0),
        "test_rows": legacy_progress.get("test_rows", 0),
    }
    src["processed_at"] = legacy_progress.get("last_upload_at") or utc_now()
    src["uploaded_at"] = legacy_progress.get("last_upload_at") or utc_now()
    save_state(args.work_root, state)
    append_event(args.work_root, "bootstrap_source0_completed", source=source0, to_dir=str(dest_dir), rows=src["rows"])


def maybe_clear_legacy_cache(args: argparse.Namespace, state: dict) -> None:
    if not args.clear_legacy_cache:
        return
    if state.get("legacy_cache_cleared"):
        return
    if LEGACY_LICHESS_CACHE.exists():
        append_event(args.work_root, "legacy_cache_delete_started", path=str(LEGACY_LICHESS_CACHE))
        shutil.rmtree(LEGACY_LICHESS_CACHE)
        append_event(args.work_root, "legacy_cache_delete_completed", path=str(LEGACY_LICHESS_CACHE))
    state["legacy_cache_cleared"] = True
    save_state(args.work_root, state)


def download_source_parquet(args: argparse.Namespace, filename: str, state: dict) -> Path:
    src = register_source(state, filename)
    src["attempts"] = int(src.get("attempts", 0)) + 1
    src["status"] = "downloading"
    save_state(args.work_root, state)
    append_event(args.work_root, "download_started", source=filename)

    local_path = hf_hub_download(
        repo_id=args.source_repo,
        repo_type="dataset",
        filename=filename,
        token=load_hf_token(),
        cache_dir=str(args.work_root / "hf_cache"),
        local_dir=str(args.work_root / "downloads"),
    )

    local_path = Path(local_path)
    src["downloaded_path"] = str(local_path)
    src["status"] = "downloaded"
    save_state(args.work_root, state)
    append_event(
        args.work_root,
        "download_completed",
        source=filename,
        local_path=str(local_path),
        size_bytes=local_path.stat().st_size if local_path.exists() else None,
    )
    return local_path


def run_prepare_for_source(args: argparse.Namespace, filename: str, parquet_path: Path, state: dict) -> Path:
    src = register_source(state, filename)
    shard_dir = args.work_root / "processed_sources" / Path(filename).stem
    src["shard_dir"] = str(shard_dir)
    src["status"] = "processing"
    save_state(args.work_root, state)
    append_event(args.work_root, "process_started", source=filename, parquet_path=str(parquet_path), shard_dir=str(shard_dir))

    cmd = [
        sys.executable,
        str(Path("/root/transform/prepare_hf_dataset.py")),
        "--dry-run",
        "--parquet-path",
        str(parquet_path),
        "--shard-dir",
        str(shard_dir),
        "--repo-id",
        args.target_repo,
        "--workers",
        str(args.workers),
        "--rows-per-chunk",
        str(args.rows_per_chunk),
        "--rows-per-shard",
        str(args.rows_per_shard),
        "--min-depth",
        str(args.min_depth),
        "--test-ratio",
        str(args.test_ratio),
        "--max-shard-size",
        args.max_shard_size,
    ]
    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])

    subprocess.run(cmd, check=True)

    progress_file = shard_dir / "progress.json"
    progress = json.loads(progress_file.read_text())
    src["rows"] = {
        "valid_rows": progress.get("valid_rows", 0),
        "train_rows": progress.get("train_rows", 0),
        "test_rows": progress.get("test_rows", 0),
        "read_rows_completed": progress.get("read_rows_completed", 0),
        "skipped_rows": progress.get("skipped_rows", 0),
    }
    src["status"] = "processed"
    src["processed_at"] = utc_now()
    save_state(args.work_root, state)
    append_event(args.work_root, "process_completed", source=filename, shard_dir=str(shard_dir), rows=src["rows"])
    return shard_dir


def completed_source_files(state: dict) -> list[str]:
    result = []
    for filename in state.get("source_files", []):
        src = state["sources"].get(filename, {})
        if src.get("status") in {"processed", "uploaded"} and src.get("shard_dir"):
            result.append(filename)
    return result


def _source_index_from_filename(filename: str) -> str:
    """Extract source index like '00001' from 'data/train-00001-of-00017.parquet'."""
    stem = Path(filename).stem  # e.g. 'train-00001-of-00017'
    parts = stem.split("-")
    return parts[1] if len(parts) >= 2 else "unknown"


def source_progress_file(state: dict, filename: str) -> Path | None:
    shard_dir = state["sources"].get(filename, {}).get("shard_dir")
    if not shard_dir:
        return None
    progress_file = Path(shard_dir) / "progress.json"
    if not progress_file.exists():
        return None
    return progress_file


def hydrate_source_from_progress(state: dict, filename: str) -> bool:
    progress_file = source_progress_file(state, filename)
    if progress_file is None:
        return False
    progress = json.loads(progress_file.read_text())
    if int(progress.get("read_rows_completed", 0)) <= 0:
        return False

    src = register_source(state, filename)
    src["rows"] = {
        "valid_rows": progress.get("valid_rows", 0),
        "train_rows": progress.get("train_rows", 0),
        "test_rows": progress.get("test_rows", 0),
        "read_rows_completed": progress.get("read_rows_completed", 0),
        "skipped_rows": progress.get("skipped_rows", 0),
    }
    if src.get("status") not in {"uploaded", "processed"}:
        src["status"] = "processed"
    if src.get("processed_at") is None:
        src["processed_at"] = utc_now()
    return True


def upload_source_shards(args: argparse.Namespace, state: dict, filename: str) -> None:
    """Upload one completed source's parquet shards directly to HF via the Hub API.

    This avoids materializing the full dataset as Arrow locally.  Each source's
    shard files are committed directly as parquet objects in ``data/`` so the
    HF auto-loader discovers them by split prefix.
    """
    src = state["sources"][filename]
    shard_dir = Path(src["shard_dir"])
    src_idx = _source_index_from_filename(filename)

    train_files = sorted((shard_dir / "train").glob("*.parquet"))
    test_files = sorted((shard_dir / "test").glob("*.parquet"))

    if not train_files and not test_files:
        raise RuntimeError(f"No parquet shards found for {filename} under {shard_dir}")

    operations: list[CommitOperationAdd] = []
    for i, f in enumerate(train_files):
        operations.append(CommitOperationAdd(
            path_in_repo=f"data/train-src{src_idx}-{i:05d}.parquet",
            path_or_fileobj=str(f),
        ))
    for i, f in enumerate(test_files):
        operations.append(CommitOperationAdd(
            path_in_repo=f"data/test-src{src_idx}-{i:05d}.parquet",
            path_or_fileobj=str(f),
        ))

    row_counts = src.get("rows", {})
    append_event(
        args.work_root,
        "source_upload_started",
        source=filename,
        train_file_count=len(train_files),
        test_file_count=len(test_files),
        row_counts=row_counts,
    )

    print(f"\nUploading {filename} shards to {args.target_repo}:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Test files:  {len(test_files)}")
    print(f"  Total operations: {len(operations)}")

    token = load_hf_token()
    api = HfApi(token=token)
    api.create_repo(args.target_repo, repo_type="dataset", exist_ok=True, token=token)
    api.create_commit(
        repo_id=args.target_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"Add shards from {Path(filename).stem} "
            f"({row_counts.get('train_rows', '?')} train, "
            f"{row_counts.get('test_rows', '?')} test rows)"
        ),
        token=token,
    )

    uploaded_at = utc_now()
    src["status"] = "uploaded"
    src["uploaded_at"] = uploaded_at
    state["last_uploaded_sources"] = completed_source_files(state) + (
        [filename] if filename not in completed_source_files(state) else []
    )
    state["last_upload_completed_at"] = uploaded_at
    state["last_upload_row_counts"][filename] = row_counts
    save_state(args.work_root, state)
    append_event(
        args.work_root,
        "source_upload_completed",
        source=filename,
        train_file_count=len(train_files),
        test_file_count=len(test_files),
        row_counts=row_counts,
    )
    print(f"  Upload complete: https://huggingface.co/datasets/{args.target_repo}")


def cleanup_download(filename: str, state: dict, work_root: Path) -> None:
    src = state["sources"][filename]
    local_path = src.get("downloaded_path")
    if not local_path:
        return
    path = Path(local_path)
    if path.exists():
        append_event(work_root, "download_cleanup_started", source=filename, local_path=str(path))
        path.unlink()
        parent = path.parent
        while parent != work_root and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        append_event(work_root, "download_cleanup_completed", source=filename, local_path=str(path))
    src["cleaned_download"] = True
    src["downloaded_path"] = None


def process_one_source(args: argparse.Namespace, filename: str, state: dict) -> None:
    src = register_source(state, filename)
    if src.get("status") == "uploaded":
        return

    if hydrate_source_from_progress(state, filename):
        save_state(args.work_root, state)
        append_event(args.work_root, "source_resumed_from_progress", source=filename, shard_dir=src.get("shard_dir"))
        if args.process_only:
            append_event(args.work_root, "source_upload_skipped", source=filename, reason="--process-only")
        else:
            upload_source_shards(args, state, filename)
        cleanup_download(filename, state, args.work_root)
        save_state(args.work_root, state)
        return

    parquet_path = None
    if src.get("downloaded_path"):
        parquet_path = Path(src["downloaded_path"])
        if not parquet_path.exists():
            parquet_path = None

    if parquet_path is None:
        parquet_path = download_source_parquet(args, filename, state)

    run_prepare_for_source(args, filename, parquet_path, state)
    if args.process_only:
        append_event(args.work_root, "source_upload_skipped", source=filename, reason="--process-only")
    else:
        upload_source_shards(args, state, filename)
    cleanup_download(filename, state, args.work_root)
    save_state(args.work_root, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequentially process Lichess source parquets with resume + logging")
    parser.add_argument("--source-repo", default=SOURCE_REPO, help="HF source dataset repo")
    parser.add_argument("--target-repo", default=TARGET_REPO, help="HF target dataset repo")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT, help="Large-storage work directory")
    parser.add_argument("--workers", type=int, default=12, help="Workers for each parquet conversion")
    parser.add_argument("--rows-per-chunk", type=int, default=200_000, help="Rows per parquet batch")
    parser.add_argument("--rows-per-shard", type=int, default=250_000, help="Rows per local parquet shard")
    parser.add_argument("--min-depth", type=int, default=15, help="Minimum Stockfish depth")
    parser.add_argument("--test-ratio", type=float, default=0.01, help="Deterministic test split ratio")
    parser.add_argument("--max-shard-size", default="500MB", help="HF upload shard size")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug cap for each source parquet")
    parser.add_argument("--start-at", default="data/train-00001-of-00017.parquet", help="First source parquet to process")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after processing this many new source parquets")
    parser.add_argument("--clear-legacy-cache", action="store_true", help="Delete the old /root Lichess cache after bootstrapping")
    parser.add_argument("--process-only", action="store_true", help="Process and checkpoint sources locally but skip HF uploads")
    parser.add_argument("--max-retries-per-source", type=int, default=0, help="Retries per source parquet after failure (0 = retry forever)")
    parser.add_argument("--retry-backoff-seconds", type=int, default=60, help="Initial retry backoff after failure")
    parser.add_argument("--retry-backoff-max-seconds", type=int, default=1800, help="Maximum retry backoff")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_env(args)
    state = load_state(args)

    source_files = list_source_files(args)
    state["source_files"] = source_files
    save_state(args.work_root, state)

    bootstrap_legacy_source0(args, state)
    maybe_clear_legacy_cache(args, state)

    if args.start_at not in source_files:
        raise RuntimeError(f"Unknown --start-at source parquet: {args.start_at}")

    start_idx = source_files.index(args.start_at)
    processed_new = 0

    print(f"Work root:   {args.work_root}")
    print(f"State file:  {state_path(args.work_root)}")
    print(f"Event log:   {events_path(args.work_root)}")
    print(f"Source repo: {args.source_repo}")
    print(f"Target repo: {args.target_repo}")
    print(f"Source files: {len(source_files)} total")
    print(f"Starting at: {args.start_at}")

    for filename in source_files[start_idx:]:
        src = register_source(state, filename)
        if src.get("status") == "uploaded":
            print(f"\nSkipping already uploaded source: {filename}")
            continue

        print(f"\n=== Processing source parquet: {filename} ===")
        failure_count = 0
        while True:
            try:
                process_one_source(args, filename, state)
                break
            except Exception as exc:
                failure_count += 1
                src["status"] = "failed"
                src["last_error"] = repr(exc)
                save_state(args.work_root, state)
                append_event(
                    args.work_root,
                    "source_failed",
                    source=filename,
                    error=repr(exc),
                    failure_count=failure_count,
                )

                if args.max_retries_per_source > 0 and failure_count > args.max_retries_per_source:
                    raise

                backoff = min(
                    args.retry_backoff_seconds * (2 ** max(0, failure_count - 1)),
                    args.retry_backoff_max_seconds,
                )
                print(
                    f"Source {filename} failed on attempt {failure_count}: {exc!r}. "
                    f"Retrying in {backoff}s...",
                    flush=True,
                )
                append_event(
                    args.work_root,
                    "source_retry_scheduled",
                    source=filename,
                    failure_count=failure_count,
                    backoff_seconds=backoff,
                )
                time.sleep(backoff)

        processed_new += 1
        if args.stop_after is not None and processed_new >= args.stop_after:
            print(f"\nReached --stop-after={args.stop_after}. Stopping cleanly.")
            break

    print("\nAll requested source parquets completed.")


if __name__ == "__main__":
    main()
