"""Shared IO: fingerprints, manifests, parquet shards, HF token."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def load_hf_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'").strip('"')
            break


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def json_read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _null_empty_lists(rows: list[dict], schema: pa.Schema) -> list[dict]:
    list_fields = [f.name for f in schema if pa.types.is_list(f.type) or pa.types.is_fixed_size_list(f.type)]
    if not list_fields:
        return rows
    cleaned = []
    for row in rows:
        item = dict(row)
        for name in list_fields:
            val = item.get(name)
            if val is not None and len(val) == 0:
                item[name] = None
        cleaned.append(item)
    return cleaned


def write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(_null_empty_lists(rows, schema), schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def read_parquets(paths: Iterable[Path]) -> pa.Table | None:
    files = [p for p in paths if p.exists()]
    if not files:
        return None
    return pq.read_table(files)


def shard_paths(folder: Path, prefix: str) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{prefix}-*.parquet"))


def next_shard(folder: Path, prefix: str) -> Path:
    existing = shard_paths(folder, prefix)
    n = 0
    if existing:
        n = int(existing[-1].stem.split("-")[-1]) + 1
    return folder / f"{prefix}-{n:05d}.parquet"


def annotation_id(source_name: str, source_path: str, source_row: int, annotation_type: str) -> str:
    raw = f"{source_name}|{source_path}|{source_row}|{annotation_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def membership_id(pool: str, split: str, annotation_id_: str) -> str:
    raw = f"{pool}|{split}|{annotation_id_}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def search_record_id(source_name: str, source_path: str, source_row: int) -> str:
    raw = f"{source_name}|{source_path}|{source_row}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
