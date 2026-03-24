"""Prepare and upload a bounded-memory chess dataset to Hugging Face.

Reads the Lichess Stockfish-evaluated parquet, converts it into the
well-formatted `avewright` schema, writes local parquet shards, and then
uploads those shards to Hugging Face without materializing the full dataset
in RAM.

Output schema (compatible with the repo's `avewright` loaders):
  - fen: str
  - best_move: str
  - eval_type: str           # "cp" or "mate"
  - eval_value: int
  - wdl_win: float
  - wdl_draw: float
  - wdl_loss: float
  - phase: str               # opening / middlegame / endgame
  - num_legal: int
  - source: str              # "lichess_stockfish"
  - game_id: str             # empty for this source
  - top_moves: str           # JSON list, single best move entry
  - ply: int
  - depth: int

The pipeline is intentionally shard-first:
  1. Stream the input parquet in bounded-size batches.
  2. Process each batch in parallel workers.
  3. Spill train/test parquet shards to disk.
  4. Upload the shard directory to Hugging Face.

This keeps memory usage stable on smaller RunPod instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_NAME = "lichess_stockfish"
DEFAULT_REPO_ID = "avewright/chess-positions-lichess-sf"
PARQUET_GLOB = (
    "outputs/lichess_cache/"
    "datasets--Lichess--chess-position-evaluations/"
    "snapshots/*/data/train-00000-of-00017.parquet"
)

OUTPUT_KEYS = [
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
]

OUTPUT_SCHEMA = pa.schema([
    ("fen", pa.string()),
    ("best_move", pa.string()),
    ("eval_type", pa.string()),
    ("eval_value", pa.int32()),
    ("wdl_win", pa.float32()),
    ("wdl_draw", pa.float32()),
    ("wdl_loss", pa.float32()),
    ("phase", pa.string()),
    ("num_legal", pa.int32()),
    ("source", pa.string()),
    ("game_id", pa.string()),
    ("top_moves", pa.string()),
    ("ply", pa.int32()),
    ("depth", pa.int32()),
])

WORKER_UCI_TO_IDX = None


def _empty_records() -> dict[str, list]:
    return {key: [] for key in OUTPUT_KEYS}


def _load_hf_token() -> str | None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def _manifest_path(shard_root: Path) -> Path:
    return shard_root / "progress.json"


def _events_path(shard_root: Path) -> Path:
    return shard_root / "events.jsonl"


def _default_progress(repo_id: str, parquet_path: str | Path, args: argparse.Namespace) -> dict:
    return {
        "repo_id": repo_id,
        "parquet_path": str(parquet_path),
        "min_depth": args.min_depth,
        "rows_per_chunk": args.rows_per_chunk,
        "rows_per_shard": args.rows_per_shard,
        "test_ratio": args.test_ratio,
        "max_rows": args.max_rows,
        "read_rows_completed": 0,
        "valid_rows": 0,
        "skipped_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "next_train_shard": 0,
        "next_test_shard": 0,
        "batches_completed": 0,
        "upload_completed": False,
        "last_upload_at": None,
    }


def _load_progress(shard_root: Path, repo_id: str, parquet_path: str | Path, args: argparse.Namespace) -> dict:
    path = _manifest_path(shard_root)
    if not path.exists():
        return _default_progress(repo_id, parquet_path, args)

    progress = json.loads(path.read_text())
    expected = {
        "repo_id": repo_id,
        "parquet_path": str(parquet_path),
        "min_depth": args.min_depth,
        "rows_per_chunk": args.rows_per_chunk,
        "rows_per_shard": args.rows_per_shard,
        "test_ratio": args.test_ratio,
        "max_rows": args.max_rows,
    }
    for key, expected_value in expected.items():
        existing_value = progress.get(key)
        if existing_value != expected_value:
            raise RuntimeError(
                f"Existing progress manifest at {path} does not match `{key}`: "
                f"{existing_value!r} != {expected_value!r}. "
                "Use a new --shard-dir or remove the old manifest."
            )
    return progress


def _save_progress(shard_root: Path, progress: dict) -> None:
    path = _manifest_path(shard_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")


def _append_event(shard_root: Path, event_type: str, **payload) -> None:
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event_type,
        **payload,
    }
    events_path = _events_path(shard_root)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def cp_to_wdl(cp: int | None, mate: int | None = None) -> tuple[float, float, float]:
    if mate is not None:
        return (1.0, 0.0, 0.0) if mate > 0 else (0.0, 0.0, 1.0)
    if cp is None:
        return (0.33, 0.34, 0.33)
    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


def fen_to_phase(fen: str) -> str:
    board_part = fen.split()[0]
    n_non_king = sum(1 for c in board_part if c.isalpha() and c.lower() != "k")
    if n_non_king >= 14:
        return "opening"
    if n_non_king >= 6:
        return "middlegame"
    return "endgame"


def _is_test_split(fen: str, test_ratio: float) -> bool:
    bucket_count = 10_000
    threshold = int(test_ratio * bucket_count)
    digest = hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % bucket_count
    return bucket < threshold


def _init_worker():
    global WORKER_UCI_TO_IDX
    if WORKER_UCI_TO_IDX is None:
        from move_vocab import UCI_TO_IDX
        WORKER_UCI_TO_IDX = UCI_TO_IDX


def _process_chunk(args):
    chunk_idx, fens, lines, depths, cps, mates, test_ratio = args

    records_by_split = {"train": _empty_records(), "test": _empty_records()}
    n_valid = 0
    n_skipped = 0
    t0 = time.time()

    for i, fen in enumerate(fens):
        try:
            line = lines[i]
            if not line:
                n_skipped += 1
                continue

            best_move = line.split()[0]
            if best_move not in WORKER_UCI_TO_IDX:
                n_skipped += 1
                continue

            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move)
            if move not in board.legal_moves:
                n_skipped += 1
                continue

            cp_val = cps[i]
            mate_val = mates[i]
            eval_type = "mate" if mate_val is not None else "cp"
            eval_value = mate_val if mate_val is not None else (cp_val if cp_val is not None else 0)
            wdl = cp_to_wdl(cp_val, mate_val)

            top_move = {"uci": best_move}
            if mate_val is not None:
                top_move["mate"] = mate_val
            else:
                top_move["cp"] = cp_val if cp_val is not None else 0

            split_name = "test" if _is_test_split(fen, test_ratio) else "train"
            target = records_by_split[split_name]
            target["fen"].append(fen)
            target["best_move"].append(best_move)
            target["eval_type"].append(eval_type)
            target["eval_value"].append(int(eval_value))
            target["wdl_win"].append(round(wdl[0], 6))
            target["wdl_draw"].append(round(wdl[1], 6))
            target["wdl_loss"].append(round(wdl[2], 6))
            target["phase"].append(fen_to_phase(fen))
            target["num_legal"].append(board.legal_moves.count())
            target["source"].append(SOURCE_NAME)
            target["game_id"].append("")
            target["top_moves"].append(json.dumps([top_move], separators=(",", ":")))
            target["ply"].append(board.ply())
            target["depth"].append(int(depths[i]))
            n_valid += 1
        except Exception:
            n_skipped += 1

    elapsed = time.time() - t0
    rate = n_valid / max(elapsed, 0.1)
    print(
        f"  Worker {chunk_idx}: {n_valid:,} valid, {n_skipped:,} skipped "
        f"({rate:.0f} pos/s, {elapsed:.0f}s)",
        flush=True,
    )
    return records_by_split, n_valid, n_skipped


def _record_count(records: dict[str, list]) -> int:
    if not records:
        return 0
    return len(records["fen"])


def _append_records(dst: dict[str, list], src: dict[str, list]) -> None:
    for key in OUTPUT_KEYS:
        dst[key].extend(src[key])


def _flush_records(records: dict[str, list], split_name: str, split_dir: Path, shard_idx: int, shard_root: Path) -> int:
    n_rows = _record_count(records)
    if n_rows == 0:
        return shard_idx

    split_dir.mkdir(parents=True, exist_ok=True)
    shard_path = split_dir / f"{split_name}-{shard_idx:05d}.parquet"
    table = pa.Table.from_pydict(records, schema=OUTPUT_SCHEMA)
    pq.write_table(table, shard_path, compression="zstd")
    size_mb = shard_path.stat().st_size / 1e6
    print(f"  Wrote {split_name} shard {shard_idx:05d}: {n_rows:,} rows ({size_mb:.1f} MB)")
    _append_event(
        shard_root,
        "shard_written",
        split=split_name,
        shard_index=shard_idx,
        rows=n_rows,
        path=str(shard_path),
        size_mb=round(size_mb, 2),
    )
    return shard_idx + 1


def _make_worker_chunks(batch_dict: dict[str, list], n_workers: int, batch_idx: int, test_ratio: float):
    total = len(batch_dict["fen"])
    if total == 0:
        return []

    chunk_size = math.ceil(total / max(1, n_workers))
    chunks = []
    for worker_idx in range(n_workers):
        start = worker_idx * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            break
        chunks.append((
            batch_idx * 10_000 + worker_idx,
            batch_dict["fen"][start:end],
            batch_dict["line"][start:end],
            batch_dict["depth"][start:end],
            batch_dict["cp"][start:end],
            batch_dict["mate"][start:end],
            test_ratio,
        ))
    return chunks


def process_parquet_to_shards(
    parquet_path: str | Path,
    shard_root: Path,
    *,
    max_rows: int | None,
    min_depth: int,
    n_workers: int,
    rows_per_chunk: int,
    rows_per_shard: int,
    test_ratio: float,
    progress: dict,
) -> dict[str, int]:
    """Convert parquet to local train/test parquet shards."""
    shard_root.mkdir(parents=True, exist_ok=True)
    train_dir = shard_root / "train"
    test_dir = shard_root / "test"

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)

    print(f"Reading {parquet_path}...")
    print(f"  Source rows available: {pf.metadata.num_rows:,}")
    print(f"  Processing rows:       {total_rows:,}")
    print(f"  Workers:               {n_workers}")
    print(f"  Rows per chunk:        {rows_per_chunk:,}")
    print(f"  Rows per shard:        {rows_per_shard:,}")
    print(f"  Test ratio:            {test_ratio:.2%}")

    buffers = {"train": _empty_records(), "test": _empty_records()}
    shard_counts = {
        "train": int(progress.get("next_train_shard", 0)),
        "test": int(progress.get("next_test_shard", 0)),
    }
    totals = {
        "read_rows": int(progress.get("read_rows_completed", 0)),
        "valid_rows": int(progress.get("valid_rows", 0)),
        "skipped_rows": int(progress.get("skipped_rows", 0)),
        "train_rows": int(progress.get("train_rows", 0)),
        "test_rows": int(progress.get("test_rows", 0)),
    }

    if totals["read_rows"] > total_rows:
        raise RuntimeError(
            f"Progress manifest says {totals['read_rows']:,} rows were already processed, "
            f"but this run only has {total_rows:,} rows available."
        )

    remaining = total_rows - totals["read_rows"]
    rows_to_skip = totals["read_rows"]
    batch_iter = pf.iter_batches(
        batch_size=rows_per_chunk,
        columns=["fen", "line", "depth", "cp", "mate"],
    )

    t0 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for batch_idx, batch in enumerate(batch_iter):
            if remaining <= 0:
                break

            raw_batch_size = len(batch)
            if rows_to_skip >= raw_batch_size:
                rows_to_skip -= raw_batch_size
                continue

            batch_dict = batch.to_pydict()
            batch_size = len(batch_dict["fen"])
            if batch_size == 0:
                continue

            if rows_to_skip > 0:
                for key in batch_dict:
                    batch_dict[key] = batch_dict[key][rows_to_skip:]
                batch_size = len(batch_dict["fen"])
                rows_to_skip = 0
                if batch_size == 0:
                    continue

            if batch_size > remaining:
                for key in batch_dict:
                    batch_dict[key] = batch_dict[key][:remaining]
                batch_size = remaining

            totals["read_rows"] += batch_size
            remaining -= batch_size

            # Cheap prefilter before we fan out to workers.
            keep_idx = []
            depths = batch_dict["depth"]
            lines = batch_dict["line"]
            for i in range(batch_size):
                depth = depths[i]
                line = lines[i]
                if depth is None or int(depth) < min_depth:
                    continue
                if not line:
                    continue
                keep_idx.append(i)

            filtered = {
                key: [batch_dict[key][i] for i in keep_idx]
                for key in batch_dict
            }

            print(
                f"\nBatch {batch_idx + 1}: {batch_size:,} rows read, "
                f"{len(filtered['fen']):,} after filter",
                flush=True,
            )

            worker_chunks = _make_worker_chunks(filtered, n_workers, batch_idx, test_ratio)
            if not worker_chunks:
                continue

            for records_by_split, n_valid, n_skipped in pool.imap_unordered(_process_chunk, worker_chunks):
                totals["valid_rows"] += n_valid
                totals["skipped_rows"] += n_skipped

                for split_name in ("train", "test"):
                    split_records = records_by_split[split_name]
                    n_rows = _record_count(split_records)
                    if n_rows == 0:
                        continue

                    _append_records(buffers[split_name], split_records)
                    totals[f"{split_name}_rows"] += n_rows

                    if _record_count(buffers[split_name]) >= rows_per_shard:
                        split_dir = train_dir if split_name == "train" else test_dir
                        shard_counts[split_name] = _flush_records(
                            buffers[split_name],
                            split_name,
                            split_dir,
                            shard_counts[split_name],
                            shard_root,
                        )
                        buffers[split_name] = _empty_records()

            progress["read_rows_completed"] = totals["read_rows"]
            progress["valid_rows"] = totals["valid_rows"]
            progress["skipped_rows"] = totals["skipped_rows"]
            progress["train_rows"] = totals["train_rows"]
            progress["test_rows"] = totals["test_rows"]
            progress["next_train_shard"] = shard_counts["train"]
            progress["next_test_shard"] = shard_counts["test"]
            progress["batches_completed"] = int(progress.get("batches_completed", 0)) + 1
            _save_progress(shard_root, progress)
            _append_event(
                shard_root,
                "batch_completed",
                batch_number=progress["batches_completed"],
                read_rows_completed=totals["read_rows"],
                valid_rows=totals["valid_rows"],
                skipped_rows=totals["skipped_rows"],
                train_rows=totals["train_rows"],
                test_rows=totals["test_rows"],
            )
            elapsed = time.time() - t0
            print(
                f"  Progress: {totals['valid_rows']:,} valid / {totals['read_rows']:,} read "
                f"({totals['valid_rows'] / max(elapsed, 0.1):.0f} pos/s overall)",
                flush=True,
            )

    for split_name in ("train", "test"):
        if _record_count(buffers[split_name]) > 0:
            split_dir = train_dir if split_name == "train" else test_dir
            shard_counts[split_name] = _flush_records(
                buffers[split_name],
                split_name,
                split_dir,
                shard_counts[split_name],
                shard_root,
            )

    elapsed = time.time() - t0
    progress["read_rows_completed"] = totals["read_rows"]
    progress["valid_rows"] = totals["valid_rows"]
    progress["skipped_rows"] = totals["skipped_rows"]
    progress["train_rows"] = totals["train_rows"]
    progress["test_rows"] = totals["test_rows"]
    progress["next_train_shard"] = shard_counts["train"]
    progress["next_test_shard"] = shard_counts["test"]
    _save_progress(shard_root, progress)
    print(
        f"\nProcessed {totals['valid_rows']:,} valid positions in {elapsed:.0f}s "
        f"({totals['valid_rows'] / max(elapsed, 0.1):.0f} pos/s)"
    )
    print(f"Skipped {totals['skipped_rows']:,}")
    print(f"Train rows: {totals['train_rows']:,}")
    print(f"Test rows:  {totals['test_rows']:,}")
    print(f"Shard root: {shard_root}")

    return totals


def upload_shards_to_hf(shard_root: Path, repo_id: str, token: str, max_shard_size: str) -> None:
    """Upload a local shard directory to Hugging Face."""
    from datasets import load_dataset

    train_files = sorted(str(p) for p in (shard_root / "train").glob("*.parquet"))
    test_files = sorted(str(p) for p in (shard_root / "test").glob("*.parquet"))

    if not train_files and not test_files:
        raise RuntimeError(f"No parquet shards found under {shard_root}")

    data_files = {}
    if train_files:
        data_files["train"] = train_files
    if test_files:
        data_files["test"] = test_files

    _append_event(
        shard_root,
        "upload_started",
        repo_id=repo_id,
        train_files=len(train_files),
        test_files=len(test_files),
        max_shard_size=max_shard_size,
    )
    print(f"\nLoading local shards from {shard_root}...")
    ds = load_dataset("parquet", data_files=data_files)
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds):,} rows")

    print(f"\nUploading to {repo_id}...")
    t0 = time.time()
    ds.push_to_hub(
        repo_id,
        token=token,
        max_shard_size=max_shard_size,
        commit_message=f"Upload shard-first Lichess Stockfish conversion ({SOURCE_NAME})",
    )
    print(f"  Uploaded in {time.time() - t0:.1f}s")
    print(f"  Dataset: https://huggingface.co/datasets/{repo_id}")
    _append_event(
        shard_root,
        "upload_completed",
        repo_id=repo_id,
        train_rows=len(ds["train"]) if "train" in ds else 0,
        test_rows=len(ds["test"]) if "test" in ds else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare chess dataset for Hugging Face without OOMs")
    parser.add_argument("--max-rows", type=int, default=None, help="Max source rows to process")
    parser.add_argument("--min-depth", type=int, default=15, help="Minimum Stockfish depth")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repo ID")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Parallel workers")
    parser.add_argument("--rows-per-chunk", type=int, default=200_000, help="Rows to read per parquet batch")
    parser.add_argument("--rows-per-shard", type=int, default=250_000, help="Rows to spill per local parquet shard")
    parser.add_argument("--test-ratio", type=float, default=0.01, help="Deterministic test split ratio")
    parser.add_argument("--max-shard-size", type=str, default="500MB", help="HF upload shard size")
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("outputs") / "hf_dataset_shards" / "lichess_sf_formatted",
        help="Local parquet shard directory",
    )
    parser.add_argument("--skip-process", action="store_true", help="Skip parquet conversion and upload existing local shards")
    parser.add_argument("--dry-run", action="store_true", help="Build local shards but skip upload")
    args = parser.parse_args()

    import glob as globmod

    parquet_files = globmod.glob(str(Path(__file__).resolve().parent / PARQUET_GLOB))
    if not parquet_files:
        print("ERROR: No parquet file found. Run the dataset download first.")
        sys.exit(1)

    parquet_path = parquet_files[0]

    progress = _load_progress(args.shard_dir, args.repo_id, parquet_path, args)
    print(f"Progress manifest: {_manifest_path(args.shard_dir)}")
    print(f"Event log:          {_events_path(args.shard_dir)}")
    print(f"Already completed:  {progress['read_rows_completed']:,} source rows")
    print(f"Existing shards:    train={progress['next_train_shard']}, test={progress['next_test_shard']}")

    if args.skip_process:
        totals = {
            "read_rows": progress["read_rows_completed"],
            "valid_rows": progress["valid_rows"],
            "skipped_rows": progress["skipped_rows"],
            "train_rows": progress["train_rows"],
            "test_rows": progress["test_rows"],
        }
        print("\nSkipping parquet conversion. Using existing local shards.")
    else:
        totals = process_parquet_to_shards(
            parquet_path,
            args.shard_dir,
            max_rows=args.max_rows,
            min_depth=args.min_depth,
            n_workers=max(1, args.workers),
            rows_per_chunk=max(1, args.rows_per_chunk),
            rows_per_shard=max(1, args.rows_per_shard),
            test_ratio=args.test_ratio,
            progress=progress,
        )

    if args.dry_run:
        print(f"\nDry run complete: {totals['valid_rows']:,} rows prepared. Skipping upload.")
        return

    hf_token = _load_hf_token()
    if not hf_token:
        print("ERROR: No HF_TOKEN found in .env or environment.")
        sys.exit(1)

    upload_shards_to_hf(args.shard_dir, args.repo_id, hf_token, args.max_shard_size)
    progress["upload_completed"] = True
    progress["last_upload_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_progress(args.shard_dir, progress)


if __name__ == "__main__":
    main()
