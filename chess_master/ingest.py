"""Resumable sample ingest and organized-mix migration."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch

from chess_master.io_util import (
    ROOT,
    json_read,
    json_write,
    load_hf_token,
    next_shard,
    shard_paths,
    write_parquet,
)

sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

from chess_master.contracts import legal_policy, qualify_source
from chess_master.rows import (
    from_lichess,
    from_mix_row,
    from_puzzle_packed,
    from_sf19,
    from_swa_scan,
    from_syzygy,
    membership_record,
)
from chess_master.schema import (
    ANNOTATIONS_SCHEMA,
    MEMBERSHIP_SCHEMA,
    POSITIONS_SCHEMA,
    QUARANTINE_SCHEMA,
    UNKNOWN,
)

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

MIX_DIR = ROOT / "outputs/organized_chess_v1"
SHARD_ROWS = 50_000
SAMPLE_SOURCES = ("sf19", "lichess", "puzzles", "syzygy", "swa_mistakes")


def _progress(out: Path) -> dict:
    return json_read(out / "progress.json", {"completed": [], "seen_positions": 0})


def _save_progress(out: Path, progress: dict) -> None:
    json_write(out / "progress.json", progress)


def _flush(out: Path, kind: str, rows: list[dict], schema, prefix: str) -> None:
    if not rows:
        return
    path = next_shard(out / kind, prefix)
    write_parquet(path, rows, schema)
    rows.clear()


class Writer:
    def __init__(self, out: Path, prefix: str):
        self.out = out
        self.prefix = prefix
        self.positions: list[dict] = []
        self.annotations: list[dict] = []
        self.membership: list[dict] = []
        self.quarantine: list[dict] = []
        self.seen_pos: set[str] = set()
        self.counts = {"positions": 0, "annotations": 0, "membership": 0, "quarantine": 0}

    def load_seen(self) -> None:
        for p in shard_paths(self.out / "positions", self.prefix):
            t = pq.read_table(p, columns=["position_id"])
            self.seen_pos.update(t.column("position_id").to_pylist())
        self.counts["positions"] = len(self.seen_pos)
        self.counts["annotations"] = sum(
            pq.read_table(p, columns=["annotation_id"]).num_rows
            for p in shard_paths(self.out / "annotations", self.prefix)
        )
        self.counts["membership"] = sum(
            pq.read_table(p, columns=["membership_id"]).num_rows
            for p in shard_paths(self.out / "membership", self.prefix)
        )

    def add_position(self, pos: dict) -> None:
        pid = pos["position_id"]
        if pid in self.seen_pos:
            return
        self.seen_pos.add(pid)
        self.positions.append(pos)
        self.counts["positions"] += 1
        if len(self.positions) >= SHARD_ROWS:
            _flush(self.out, "positions", self.positions, POSITIONS_SCHEMA, self.prefix)

    def add_annotation(self, ann: dict) -> None:
        self.annotations.append(ann)
        self.counts["annotations"] += 1
        if len(self.annotations) >= SHARD_ROWS:
            _flush(self.out, "annotations", self.annotations, ANNOTATIONS_SCHEMA, self.prefix)

    def add_membership(self, mem: dict) -> None:
        self.membership.append(mem)
        self.counts["membership"] += 1
        if len(self.membership) >= SHARD_ROWS:
            _flush(self.out, "membership", self.membership, MEMBERSHIP_SCHEMA, self.prefix)

    def reject(self, source_name: str, source_path: str, source_row: int, reason: str,
               position_id: str | None = None, detail: str | None = None) -> None:
        self.quarantine.append({
            "source_name": source_name,
            "source_path": source_path,
            "source_row": int(source_row),
            "reason": reason,
            "position_id": position_id,
            "detail": detail,
        })
        self.counts["quarantine"] += 1
        if len(self.quarantine) >= SHARD_ROWS:
            _flush(self.out, "quarantine", self.quarantine, QUARANTINE_SCHEMA, self.prefix)

    def close(self) -> None:
        _flush(self.out, "positions", self.positions, POSITIONS_SCHEMA, self.prefix)
        _flush(self.out, "annotations", self.annotations, ANNOTATIONS_SCHEMA, self.prefix)
        _flush(self.out, "membership", self.membership, MEMBERSHIP_SCHEMA, self.prefix)
        _flush(self.out, "quarantine", self.quarantine, QUARANTINE_SCHEMA, self.prefix)


def _accept(pos: dict, ann: dict, fields: dict, source_name: str) -> str | None:
    if pos["legality_status"] == "invalid":
        return "invalid_position"
    q = qualify_source(source_name, fields)
    if q:
        return q
    return None


def _fields_from_encoded(d: dict, i: int, extra: dict | None = None) -> dict:
    extra = extra or {}
    return {
        "board_array": d["board_array"][i],
        "turn": d["turn"][i],
        "castling": d["castling"][i],
        "ep_square": d["ep_square"][i],
        "move_idx": d["move_idx"][i],
        "soft_indices": d["soft_indices"][i],
        "soft_probs": d["soft_probs"][i],
        "label_depth": d["label_depth"][i] if "label_depth" in d else extra.get("label_depth"),
        "depth": d["label_depth"][i] if "label_depth" in d else extra.get("label_depth"),
        "split": extra.get("split", d["split"][i] if "split" in d else 0),
        "policy_mask": extra.get("policy_mask", d["policy_mask"][i] if "policy_mask" in d else 1),
        "nodes_budget": extra.get("nodes_budget", d["nodes_budget"][i] if "nodes_budget" in d else None),
        "nodes_requested": extra.get("nodes_budget", d["nodes_budget"][i] if "nodes_budget" in d else None),
        "nodes_achieved": extra.get("nodes", d["nodes"][i] if "nodes" in d else None),
        "wdl": extra.get("wdl", d["wdl"][i] if "wdl" in d else None),
        "tb_wdl": extra.get("tb_wdl", extra.get("wdl")),
        "tb_dtz": extra.get("dtz", d["dtz"][i] if "dtz" in d else None),
        "dtz": extra.get("dtz", d["dtz"][i] if "dtz" in d else None),
    }


def _hf_download(repo: str, name: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    load_hf_token()
    return Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))


def _pack_table(table) -> dict:
    names = set(table.column_names)
    out = {}
    n = table.num_rows
    for k in names:
        col = table[k]
        if k in ("board_array", "soft_indices", "soft_probs", "wdl", "soft_cps", "soft_mates"):
            out[k] = np.asarray(col.to_pylist())
        else:
            try:
                out[k] = np.asarray(col.to_numpy(zero_copy_only=False))
            except Exception:
                out[k] = np.asarray(col.to_pylist(), dtype=object)
    out["_n"] = n
    return out


def ingest_source_sample(out: Path, name: str, n: int, pins: dict) -> dict:
    writer = Writer(out, f"sample-{name}")
    rejected: dict[str, int] = {}
    accepted = 0

    def bump(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    if name == "sf19":
        rev = pins["revision"]
        path = _hf_download(pins["repo"], "data/shard_000001.parquet", rev)
        pf = pq.ParquetFile(path)
        offset = 0
        for batch in pf.iter_batches(batch_size=1024):
            if accepted >= n:
                break
            d = _pack_table(batch)
            raw = batch.to_pydict()
            for i in range(d["_n"]):
                if accepted >= n:
                    break
                extra = {k: raw[k][i] for k in raw}
                pos, ann = from_sf19(d, i, source_path="data/shard_000001.parquet", revision=rev, extra=extra)
                reason = _accept(pos, ann, _fields_from_encoded(d, i, extra), "sf19")
                src_row = offset + i
                if reason:
                    writer.reject(name, "data/shard_000001.parquet", src_row, reason, pos.get("position_id"))
                    bump(reason)
                    continue
                writer.add_position(pos)
                writer.add_annotation(ann)
                writer.add_membership(membership_record(
                    pos, ann, pool="sample", split="inspect", game_raw=extra.get("game_id"),
                    source_name="sf19", included=False, exposure="sample_only",
                ))
                accepted += 1
            offset += d["_n"]

    elif name == "lichess":
        rev = pins["revision"]
        path = _hf_download(pins["repo"], "data-00000.parquet", rev)
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=max(n * 8, 256)))
        d = _pack_table(batch)
        for i in range(min(d["_n"], n * 8)):
            if accepted >= n:
                break
            pos, ann = from_lichess(d, i, source_path="data-00000.parquet", revision=rev)
            reason = _accept(pos, ann, _fields_from_encoded(d, i), "lichess")
            if reason:
                writer.reject(name, "data-00000.parquet", i, reason, pos.get("position_id"))
                bump(reason)
                continue
            writer.add_position(pos)
            writer.add_annotation(ann)
            writer.add_membership(membership_record(
                pos, ann, pool="sample", split="inspect", game_raw=None,
                source_name="lichess", included=False, exposure="sample_only",
            ))
            accepted += 1

    elif name == "syzygy":
        rev = pins["revision"]
        local = ROOT / "outputs/hf_soft/syzygy_soft.pt"
        d = torch.load(local, map_location="cpu", weights_only=False)
        source_path = str(local.relative_to(ROOT))
        for i in range(int(d["turn"].shape[0])):
            if accepted >= n:
                break
            extra = {}
            if "wdl" in d:
                extra["tb_wdl"] = int(d["wdl"][i])
            if "dtz" in d:
                extra["dtz"] = int(d["dtz"][i])
            pos, ann = from_syzygy(d, i, source_path=source_path, revision=rev, extra=extra)
            reason = _accept(pos, ann, _fields_from_encoded(d, i, extra), "syzygy")
            if reason:
                writer.reject(name, source_path, i, reason, pos.get("position_id"))
                bump(reason)
                continue
            writer.add_position(pos)
            writer.add_annotation(ann)
            writer.add_membership(membership_record(
                pos, ann, pool="sample", split="inspect", game_raw=None,
                source_name="syzygy", included=False, exposure="sample_only",
            ))
            accepted += 1

    elif name == "puzzles":
        from build_organized_chess_mix import pack_puzzle

        rev = pins["revision"]
        path = _hf_download(pins["repo"], "data/train-00000-of-00003.parquet", rev)
        pf = pq.ParquetFile(path)
        offset = 0
        for batch in pf.iter_batches(batch_size=512):
            if accepted >= n:
                break
            raw = batch.to_pydict()
            for j in range(batch.num_rows):
                if accepted >= n:
                    break
                puzzle = {k: raw[k][j] for k in raw}
                packed, meta = pack_puzzle(puzzle)
                src_row = offset + j
                if packed is None:
                    writer.reject(name, "data/train-00000-of-00003.parquet", src_row, "puzzle_unusable")
                    bump("puzzle_unusable")
                    continue
                d = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in packed.items()}
                pos, ann = from_puzzle_packed(
                    d, 0, meta, source_path="data/train-00000-of-00003.parquet",
                    revision=rev, source_row=src_row,
                )
                fields = _fields_from_encoded(d, 0)
                ok, why = legal_policy(
                    fields["board_array"], fields["turn"], fields["castling"], fields["ep_square"],
                    fields["move_idx"], fields["soft_indices"], fields["soft_probs"],
                )
                if not ok:
                    writer.reject(name, "data/train-00000-of-00003.parquet", src_row, why, pos.get("position_id"))
                    bump(why)
                    continue
                writer.add_position(pos)
                writer.add_annotation(ann)
                writer.add_membership(membership_record(
                    pos, ann, pool="sample", split="inspect", game_raw=meta.get("GameId"),
                    source_name="puzzles", included=False, exposure="sample_only",
                ))
                accepted += 1
            offset += batch.num_rows

    elif name == "swa_mistakes":
        rev = pins["revision"]
        path = _hf_download(pins["repo"], "data/scan/shard_000000.parquet", rev)
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=max(n * 4, 256)))
        d = _pack_table(batch)
        pairs = 0
        for i in range(d["_n"]):
            if pairs >= n:
                break
            items = from_swa_scan(d, i, source_path="data/scan/shard_000000.parquet", revision=rev)
            pos = items[0][0]
            teacher = items[0][1]
            reason = _accept(pos, teacher, _fields_from_encoded(d, i), "sf19")
            # Scan rows are not required to pass SF19 budget gates; keep legal policy only.
            ok, why = legal_policy(
                d["board_array"][i], d["turn"][i], d["castling"][i], d["ep_square"][i],
                d["move_idx"][i], d["soft_indices"][i], d["soft_probs"][i],
            )
            if not ok:
                writer.reject(name, "data/scan/shard_000000.parquet", i, why, pos.get("position_id"))
                bump(why)
                continue
            writer.add_position(pos)
            for _, ann in items:
                writer.add_annotation(ann)
                writer.add_membership(membership_record(
                    pos, ann, pool="sample", split="inspect", game_raw=None,
                    source_name="swa_mistakes", included=False, exposure="sample_only",
                ))
            pairs += 1
        accepted = pairs
    else:
        raise KeyError(name)

    writer.close()
    return {"accepted": accepted, "rejected": rejected, **writer.counts}


def ingest_samples(out: Path, inventory: dict, *, per_source: int = 64) -> dict:
    progress = _progress(out)
    key = "sample"
    report = {"per_source": per_source, "sources": {}}
    if key in progress.get("completed", []) and (out / "samples" / "cross_source.json").exists():
        return json_read(out / "samples" / "cross_source.json")

    hf = inventory.get("hf", {})
    pins = {
        "sf19": hf.get("sf19", {}),
        "lichess": hf.get("lichess_hf", {}),
        "puzzles": hf.get("puzzles", {}),
        "syzygy": hf.get("syzygy_hf", {}),
        "swa_mistakes": hf.get("swa_mistakes", {}),
    }
    for name in SAMPLE_SOURCES:
        print(f"sample {name}", flush=True)
        report["sources"][name] = ingest_source_sample(out, name, per_source, pins[name])

    sample = _cross_source_sample(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    json_write(out / "samples" / "cross_source.json", sample)
    json_write(out / "sample_report.json", report)
    progress.setdefault("completed", [])
    if key not in progress["completed"]:
        progress["completed"].append(key)
    _save_progress(out, progress)
    return report


def _table_rows(folder: Path, prefix: str, limit: int) -> list[dict]:
    rows = []
    for p in shard_paths(folder, prefix):
        t = pq.read_table(p)
        rows.extend(t.to_pylist())
        if len(rows) >= limit:
            break
    return rows[:limit]


def _cross_source_sample(out: Path) -> dict:
    by_source = {}
    for name in SAMPLE_SOURCES:
        anns = _table_rows(out / "annotations", f"sample-{name}", 4)
        poss = {r["position_id"]: r for r in _table_rows(out / "positions", f"sample-{name}", 64)}
        slim = []
        for ann in anns:
            pos = poss.get(ann["position_id"], {})
            slim.append({
                "source_name": ann.get("source_name"),
                "annotation_type": ann.get("annotation_type"),
                "position_id": ann.get("position_id"),
                "fen_4": pos.get("fen_4"),
                "phase": pos.get("phase"),
                "non_king_count": pos.get("non_king_count"),
                "in_check": pos.get("in_check"),
                "legality_status": pos.get("legality_status"),
                "best_uci": ann.get("best_uci"),
                "policy_uci": ann.get("policy_uci"),
                "policy_probs": ann.get("policy_probs"),
                "policy_transform": ann.get("policy_transform"),
                "value_perspective": ann.get("value_perspective"),
                "value_eligible": int(ann.get("value_eligible") or 0),
                "tb_wdl": ann.get("tb_wdl"),
                "tb_dtz": ann.get("tb_dtz"),
                "mate_is_dtz_proxy": int(ann.get("mate_is_dtz_proxy") or 0),
                "puzzle_id": ann.get("puzzle_id"),
                "puzzle_moves": ann.get("puzzle_moves"),
                "model_tag": ann.get("model_tag"),
                "regret_status": ann.get("regret_status"),
                "depth": ann.get("depth"),
                "depth_is_sentinel": ann.get("depth_is_sentinel"),
            })
        by_source[name] = slim
    return {
        "note": "Small accepted rows from every source. Positions and annotations are linked, not flattened.",
        "by_source": by_source,
    }


def _load_mix_pt(name: str, split: str) -> dict:
    path = MIX_DIR / f"{name}_{split}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_provenance(name: str) -> dict[tuple[str, int], dict]:
    path = MIX_DIR / f"{name}_provenance.jsonl"
    out: dict[tuple[str, int], dict] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            out[(rec["split"], int(rec["row"]))] = rec
    return out


def ingest_organized_mix(out: Path, inventory: dict) -> dict:
    progress = _progress(out)
    key = "organized_chess_v1"
    if key in progress.get("completed", []):
        return json_read(out / "mix_ingest_report.json", {"status": "already_complete"})

    mix_pins = inventory.get("local", {}).get("organized_chess_v1", {})
    revs = mix_pins.get("source_revisions") or {}
    writer = Writer(out, "mix")
    writer.load_seen()
    report = {"sources": {}, "status": "building"}
    completed = set(progress.get("completed", []))

    for name in ("sf19", "lichess", "puzzles", "syzygy"):
        src_key = f"mix:{name}"
        if src_key in completed:
            print(f"resume skip mix {name}", flush=True)
            continue
        print(f"migrate {name}", flush=True)
        prov = _load_provenance(name)
        rejected: dict[str, int] = {}
        kept = {"train": 0, "eval": 0}
        rev = revs.get(name) or UNKNOWN
        for split in ("train", "eval"):
            d = _load_mix_pt(name, split)
            n = int(d["turn"].shape[0])
            for i in range(n):
                rec = prov.get((split, i), {})
                extra = {}
                if name == "puzzles":
                    extra.update(rec.get("puzzle") or {})
                    extra["input_row"] = rec.get("input_row", i)
                if name == "syzygy":
                    extra["tb_wdl"] = rec.get("tb_wdl")
                    extra["dtz"] = rec.get("dtz")
                if name == "sf19":
                    extra["nodes_budget"] = rec.get("nodes_budget")
                    extra["nodes"] = rec.get("nodes_achieved")
                    extra["label_depth"] = rec.get("label_depth")
                    extra["game_id"] = rec.get("game_id")
                    extra["split"] = 0 if split == "train" else 0
                source_path = rec.get("input") or f"{name}_{split}.pt"
                source_row = int(rec.get("input_row", i))
                pos, ann = from_mix_row(
                    d, i, name, source_path=str(source_path), revision=rev, extra=extra,
                )
                # Mix rows already passed organized-mix gates. Keep them; flag, don't drop.
                fields = _fields_from_encoded(d, i, extra)
                reason = qualify_source(name, fields)
                if reason:
                    ann["quality_status"] = "accepted_from_frozen_mix"
                    ann["reject_reason"] = f"would_fail_current_gate:{reason}"
                    rejected[reason] = rejected.get(reason, 0) + 1
                writer.add_position(pos)
                writer.add_annotation(ann)
                writer.add_membership(membership_record(
                    pos, ann, pool="organized_chess_v1", split=split,
                    game_raw=rec.get("game_id") or (rec.get("puzzle") or {}).get("GameId"),
                    source_name=name, mix_row=i, included=True,
                    exposure="pool_member" if split == "train" else "holdout",
                ))
                kept[split] += 1
                if (i + 1) % 50000 == 0:
                    print(f"  {name} {split} {i + 1}/{n}", flush=True)
            del d
        report["sources"][name] = {"kept": kept, "gate_flags": rejected, "revision": rev}
        completed.add(src_key)
        progress["completed"] = sorted(completed)
        _save_progress(out, progress)
        writer.close()
        writer = Writer(out, "mix")
        writer.load_seen()
        print(f"  {name} train={kept['train']} eval={kept['eval']}", flush=True)

    writer.close()
    writer.load_seen()
    report["status"] = "complete"
    report["counts"] = writer.counts
    json_write(out / "mix_ingest_report.json", report)
    completed.add(key)
    progress["completed"] = sorted(completed)
    _save_progress(out, progress)
    return report
