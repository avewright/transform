"""Export a recipe to trainer-compatible soft_cache.pt / deep_cache.pt."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from chess_master.io_util import json_write, shard_paths
from chess_master.recipes import load_recipe, requested_counts
from chess_master.schema import SOURCE_IDS


def _cat_tables(folder: Path, prefix: str, columns=None):
    files = shard_paths(folder, prefix)
    if not files:
        return None
    return pq.read_table(files, columns=columns)


def _by_id(table, key: str) -> dict[str, dict]:
    if table is None:
        return {}
    return {row[key]: row for row in table.to_pylist()}


def _encode_export_row(pos: dict, ann: dict, source_name: str) -> dict:
    ba = np.asarray(pos["board_array"], dtype=np.int8)
    si = np.asarray(ann["trainer_soft_indices"], dtype=np.int64)
    sp = np.asarray(ann["trainer_soft_probs"], dtype=np.float32)
    wdl = ann.get("value_wdl")
    if wdl is None or int(ann.get("value_valid") or 0) != 1:
        wdl = [0.0, 1.0, 0.0]
        valid = 0
    else:
        valid = 1
    ep = int(pos["ep_square"])
    if ep <= 0:
        ep = -1
    return {
        "board_array": torch.from_numpy(ba.copy()),
        "turn": torch.tensor(int(pos["turn"]), dtype=torch.int8),
        "castling": torch.tensor(int(pos["castling"]), dtype=torch.int8),
        "ep_square": torch.tensor(ep, dtype=torch.int8),
        "move_idx": torch.tensor(int(si[0] if ann.get("best_uci") is None else _best_idx(ann)), dtype=torch.int64),
        "cp": torch.tensor(int(ann["original_cp"] or 0), dtype=torch.int32),
        "mate": torch.tensor(int(ann["original_mate"] or 0), dtype=torch.int32),
        "soft_indices": torch.from_numpy(si.copy()),
        "soft_probs": torch.from_numpy(sp.copy()),
        "label_depth": torch.tensor(int(ann["depth"] or 0), dtype=torch.int16),
        "phase": torch.tensor(int(pos["phase"]), dtype=torch.int8),
        "source": torch.tensor(SOURCE_IDS[source_name], dtype=torch.int8),
        "wdl": torch.tensor(wdl, dtype=torch.float32),
        "value_valid": torch.tensor(valid, dtype=torch.int8),
    }


def _best_idx(ann: dict) -> int:
    idxs = ann.get("policy_indices") or []
    if idxs:
        return int(idxs[0])
    si = ann.get("trainer_soft_indices") or [-1]
    return int(si[0])


def _stack(rows: list[dict]) -> dict:
    from data_loader import ep_square_to_file

    keys = list(rows[0].keys())
    out = {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}
    out["ep_file"] = ep_square_to_file(out["ep_square"]).to(torch.int8)
    return out


def export_recipe(master: Path, recipe_name: str, dest: Path, *, force: bool = False) -> dict:
    recipe = load_recipe(recipe_name)
    dest.mkdir(parents=True, exist_ok=True)
    man_path = dest / "manifest.json"
    if man_path.exists() and not force:
        import json
        prev = json.loads(man_path.read_text())
        if prev.get("status") == "complete":
            raise SystemExit(f"Refusing to overwrite complete export at {dest}; pass --force")

    pool = recipe.get("membership_pool")
    if pool != "organized_chess_v1":
        raise NotImplementedError("v1 exporter reproduces the frozen membership pool only")

    mem = _cat_tables(master / "membership", "mix")
    pos = _by_id(_cat_tables(master / "positions", "mix"), "position_id")
    ann = _by_id(_cat_tables(master / "annotations", "mix"), "annotation_id")
    if mem is None:
        raise SystemExit("no mix membership tables; run ingest-mix first")

    requested = requested_counts(recipe)
    buckets: dict[str, list[dict]] = {k: [] for k in recipe["sources"]}
    eval_buckets: dict[str, list[dict]] = {k: [] for k in recipe["sources"]}
    skipped = {"missing_join": 0, "wrong_pool": 0}

    for rec in mem.to_pylist():
        if rec.get("pool_name") != pool:
            skipped["wrong_pool"] += 1
            continue
        a = ann.get(rec["annotation_id"])
        p = pos.get(rec["position_id"])
        if a is None or p is None:
            skipped["missing_join"] += 1
            continue
        src = a["source_name"]
        if src not in buckets:
            continue
        row = _encode_export_row(p, a, src)
        dest_list = eval_buckets[src] if rec.get("split") == "eval" else buckets[src]
        dest_list.append((int(rec.get("mix_row") or 0), row))

    actual = {}
    source_train = {}
    source_eval = {}
    for src in recipe["sources"]:
        buckets[src].sort(key=lambda x: x[0])
        eval_buckets[src].sort(key=lambda x: x[0])
        train_rows = [r for _, r in buckets[src]]
        eval_rows = [r for _, r in eval_buckets[src]]
        if len(train_rows) != requested[src]:
            # Exact membership replay: report shortfall, do not relax quality to fill.
            pass
        source_train[src] = _stack(train_rows)
        source_eval[src] = _stack(eval_rows)
        torch.save(source_train[src], dest / f"{src}_train.pt")
        torch.save(source_eval[src], dest / f"{src}_eval.pt")
        actual[src] = len(train_rows)

    soft_names = ("sf19", "lichess", "puzzles")
    keys = [k for k in source_train["sf19"] if all(k in source_train[n] for n in soft_names)]
    soft = {k: torch.cat([source_train[n][k] for n in soft_names]) for k in keys}
    torch.save(soft, dest / "soft_cache.pt")
    torch.save(source_train["syzygy"], dest / "deep_cache.pt")

    total = sum(actual.values())
    report = {
        "status": "complete",
        "recipe": recipe["name"],
        "dataset_version": recipe["dataset_version"],
        "seed": recipe.get("seed"),
        "requested": requested,
        "actual_counts": actual,
        "actual_proportions": {k: (actual[k] / total if total else 0) for k in actual},
        "eval_counts": {k: int(source_eval[k]["turn"].shape[0]) for k in recipe["sources"]},
        "unique_train_positions": total,
        "overlap_policy": recipe.get("overlap_policy"),
        "quality_relaxation": "never",
        "fallbacks": [],
        "exclusions": recipe.get("split_exclusions"),
        "skipped": skipped,
        "trainer": recipe.get("trainer"),
        "note": "Membership replay of organized_chess_v1. Quotas were applied upstream; this export does not relax them.",
    }
    json_write(dest / "manifest.json", report)
    return report
