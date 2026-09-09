"""Quality report over master parquet tables."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from chess_master.io_util import json_write, shard_paths
from chess_master.schema import UNKNOWN


def _read(folder: Path, prefix: str, columns=None):
    files = shard_paths(folder, prefix)
    if not files:
        return None
    return pq.read_table(files, columns=columns)


def _prefixes(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    names = {p.name.rsplit("-", 1)[0] for p in folder.glob("*.parquet")}
    return sorted(names)


def build_quality_report(out: Path) -> dict:
    pos_parts, ann_parts, mem_parts, q_parts = [], [], [], []
    for prefix in _prefixes(out / "positions"):
        t = _read(out / "positions", prefix)
        if t is not None:
            pos_parts.append(t)
    for prefix in _prefixes(out / "annotations"):
        t = _read(out / "annotations", prefix)
        if t is not None:
            ann_parts.append(t)
    for prefix in _prefixes(out / "membership"):
        t = _read(out / "membership", prefix)
        if t is not None:
            mem_parts.append(t)
    for prefix in _prefixes(out / "quarantine"):
        t = _read(out / "quarantine", prefix)
        if t is not None:
            q_parts.append(t)

    import pyarrow as pa

    positions = pa.concat_tables(pos_parts) if pos_parts else None
    annotations = pa.concat_tables(ann_parts) if ann_parts else None
    membership = pa.concat_tables(mem_parts) if mem_parts else None
    quarantine = pa.concat_tables(q_parts) if q_parts else None

    prefixes = {
        "positions": _prefixes(out / "positions"),
        "annotations": _prefixes(out / "annotations"),
        "membership": _prefixes(out / "membership"),
        "quarantine": _prefixes(out / "quarantine"),
    }
    prefix_counts = {}
    for kind, folder in (("positions", "positions"), ("annotations", "annotations"), ("membership", "membership")):
        prefix_counts[kind] = {}
        for prefix in prefixes[kind]:
            t = _read(out / folder, prefix, columns=["position_id"] if kind != "annotations" else ["annotation_id"])
            prefix_counts[kind][prefix] = 0 if t is None else t.num_rows

    report: dict = {
        "positions": 0 if positions is None else positions.num_rows,
        "unique_positions": 0 if positions is None else len(set(positions.column("position_id").to_pylist())),
        "annotations": 0 if annotations is None else annotations.num_rows,
        "membership": 0 if membership is None else membership.num_rows,
        "quarantine": 0 if quarantine is None else quarantine.num_rows,
        "by_prefix": prefix_counts,
        "missing_metadata": {},
        "conflicts": {},
        "rejected": {},
        "annotation_coverage": {},
        "source_counts": {},
        "depth": {},
        "split_overlaps": {},
    }
    if annotations is not None:
        src = Counter(annotations.column("source_name").to_pylist())
        types = Counter(annotations.column("annotation_type").to_pylist())
        report["source_counts"] = dict(src)
        report["annotation_coverage"] = {
            "by_type": dict(types),
            "value_eligible": int(pc.sum(annotations.column("value_eligible")).as_py() or 0),
            "policy_eligible": int(pc.sum(annotations.column("policy_eligible")).as_py() or 0),
            "depth_sentinels": int(pc.sum(pc.cast(annotations.column("depth_is_sentinel"), pa.int64())).as_py() or 0),
        }
        missing = {}
        for col, label in (
            ("engine_version", "engine_version_unknown"),
            ("network", "network_unknown"),
            ("game_id_raw", None),
        ):
            if col in annotations.column_names:
                vals = annotations.column(col).to_pylist()
                missing[col] = sum(1 for v in vals if v in (None, "", UNKNOWN))
        report["missing_metadata"] = missing
        depths = [d for d in annotations.column("depth").to_pylist() if d is not None]
        if depths:
            report["depth"] = {
                "min": min(depths),
                "max": max(depths),
                "n": len(depths),
                "sentinel_count": report["annotation_coverage"]["depth_sentinels"],
            }
        # Conflicting best moves on the same position.
        by_pos: dict[str, set[str]] = {}
        for pid, move, src_name in zip(
            annotations.column("position_id").to_pylist(),
            annotations.column("best_uci").to_pylist(),
            annotations.column("source_name").to_pylist(),
        ):
            if move:
                by_pos.setdefault(pid, set()).add(f"{src_name}:{move}")
        conflicts = {pid: sorted(v) for pid, v in by_pos.items() if len({x.split(":", 1)[1] for x in v}) > 1}
        report["conflicts"] = {
            "positions_with_disagreeing_best_moves": len(conflicts),
            "examples": dict(list(conflicts.items())[:8]),
        }
    if quarantine is not None:
        report["rejected"] = dict(Counter(quarantine.column("reason").to_pylist()))
    if membership is not None and positions is not None:
        train = set()
        evals = set()
        for pid, split, pool in zip(
            membership.column("position_id").to_pylist(),
            membership.column("split").to_pylist(),
            membership.column("pool_name").to_pylist(),
        ):
            if pool != "organized_chess_v1":
                continue
            if split == "train":
                train.add(pid)
            elif split == "eval":
                evals.add(pid)
        report["split_overlaps"] = {
            "organized_v1_train_eval_position_overlap": len(train & evals),
            "train": len(train),
            "eval": len(evals),
        }
    if positions is not None:
        report["phase"] = dict(Counter(positions.column("phase").to_pylist()))
        report["rule_state_available"] = int(sum(1 for v in positions.column("rule_state_available").to_pylist() if v))
    json_write(out / "quality_report.json", report)
    return report
