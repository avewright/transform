#!/usr/bin/env python3
"""Build one frozen overnight-base + 15% verified-correction mix.

Does not start training. Does not touch incumbent weights.
Lichess replay (overnight bonus 20%) is the replaced share.
Puzzles are not in this mix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, list_repo_files

from autoresearch_8gb.pipeline import (
    attach_static_targets,
    concat_soft_tables,
    hflip_cache_slice,
    make_val_membership,
)
from build_hf_elo_mix import position_hashes
from build_organized_chess_mix import (
    DUMMY_WDL,
    DTYPES,
    fingerprint,
    json_write,
    legal_row,
    load_hf_token,
    policy_ok,
)
from data_loader import compute_wdl, ep_square_to_file
from harvest_swa_mistakes import I_TO_TAG, TAG_TO_I

CORR_REPO = "avewright/chess-soft-100m-swa-mistakes"
CORR_REV = "69b75e12ab2867c8b9a63d385065dfe646cf86e1"
CORR_FILE = "data/corrections/verified_246k.parquet"
CORR_REPORT = "data/corrections/mix_v2_report.json"
SF19_REPO = "avewright/chess-soft-sf19"
SF19_REV = "68cef6c9ba62c62f904f25305f1a7489dab825e0"
SYZ_REPO = "avewright/chess-soft-syzygy"
SYZ_REV = "3889985c482837aa3082e182d8f123e005e2c0d4"
INCUMBENT_REPO = "avewright/chess-transformer-100m-overnight_20260908"
INCUMBENT_REV = "2314e519cf17157ad677421b2ac68eba0bce58fb"
INCUMBENT_SHA = "ea3255e60ff981800c11cdb960e7b0a45575f9d83a4984a99d8e54619fa3e7f4"
SOURCE_SYZYGY = 2
SOURCE_MISTAKE = 3
SOURCE_SF19 = 4
SUBSTANTIAL = frozenset({
    TAG_TO_I["inaccuracy"], TAG_TO_I["blunder"],
    TAG_TO_I["conversion"], TAG_TO_I["major"],
})
EXCLUDE_TAGS = frozenset({
    TAG_TO_I["ok"], TAG_TO_I["off_pv"], TAG_TO_I["disagree"],
})
SOFT_KEYS = (
    "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
    "soft_indices", "soft_probs", "label_depth", "phase", "source", "wdl",
    "value_valid", "ep_file",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def packed_fast(table) -> dict:
    out = {}
    n = table.num_rows
    names = set(table.column_names)
    for k, dt in DTYPES.items():
        if k not in names:
            continue
        col = table[k].combine_chunks()
        if k in ("board_array", "soft_indices", "soft_probs"):
            width = 64 if k == "board_array" else 8
            vals = np.asarray(col.values.to_numpy(zero_copy_only=False))
            arr = np.ascontiguousarray(vals.reshape(n, width))
            out[k] = torch.from_numpy(np.array(arr, copy=True)).to(dt)
        else:
            out[k] = torch.as_tensor(np.asarray(col.to_numpy()), dtype=dt)
    extra_map = {
        "tag": torch.int8, "drop_cp": torch.int32, "needs_sf": torch.int8,
        "model_in_pv": torch.int8, "origin": torch.int8, "split": torch.int8,
        "nodes": torch.int32, "nodes_budget": torch.int32,
        "model_move_idx": torch.int64, "teacher_move_idx": torch.int64,
        "policy_mask": torch.int8,
    }
    for k, dt in extra_map.items():
        if k in names:
            out[k] = torch.as_tensor(np.asarray(table[k].to_numpy()), dtype=dt)
    if "soft_cps" in names:
        col = table["soft_cps"].combine_chunks()
        vals = np.asarray(col.values.to_numpy(zero_copy_only=False)).reshape(n, 8)
        out["soft_cps"] = torch.from_numpy(np.array(vals, copy=True)).to(torch.int32)
    if "wdl" in names:
        col = table["wdl"].combine_chunks()
        if hasattr(col, "values"):
            vals = np.asarray(col.values.to_numpy(zero_copy_only=False)).reshape(n, 3)
            out["wdl"] = torch.from_numpy(np.array(vals, copy=True)).to(torch.float32)
    if "ep_square" in out:
        out["ep_square"] = torch.where(out["ep_square"] <= 0, -1, out["ep_square"]).to(torch.int8)
    return out


def take_rows(data: dict, idx: np.ndarray) -> dict:
    out = {}
    n = None
    for k, v in data.items():
        if torch.is_tensor(v) and v.ndim and n is None:
            n = int(v.shape[0])
        if torch.is_tensor(v) and n is not None and int(v.shape[0]) == n:
            out[k] = v[idx].contiguous()
    return out


def canonical_hashes(d: dict) -> np.ndarray:
    h = position_hashes(d).astype(np.uint64)
    idx = torch.where(d["castling"] == 0)[0]
    if len(idx):
        h[idx.numpy()] = np.minimum(h[idx.numpy()], position_hashes(hflip_cache_slice(d, idx)).astype(np.uint64))
    return h


def flip_blocked(data: dict, val_mask: np.ndarray) -> np.ndarray:
    hs = position_hashes(data).astype(np.uint64)
    blocked = np.unique(hs[val_mask])
    cast = data["castling"].cpu().numpy()
    flip_src = np.flatnonzero(val_mask & (cast == 0))
    if flip_src.size:
        extra = position_hashes(
            hflip_cache_slice(data, torch.from_numpy(flip_src.astype(np.int64)))
        ).astype(np.uint64)
        blocked = np.unique(np.concatenate([blocked, extra]))
    return blocked


def hex_keys_to_u64(keys: list[str]) -> np.ndarray:
    return np.asarray([int(k, 16) for k in keys], dtype=np.uint64)


def ensure_train_fields(data: dict, *, source: int, value_valid: int) -> dict:
    n = int(data["turn"].shape[0])
    data["source"] = torch.full((n,), source, dtype=torch.int8)
    if "wdl" not in data or data["wdl"].ndim != 2:
        data["wdl"] = compute_wdl(data["cp"], data["mate"]) if value_valid else DUMMY_WDL.expand(n, 3).clone()
    if "value_valid" not in data:
        data["value_valid"] = torch.full((n,), value_valid, dtype=torch.int8)
    if value_valid == 0:
        data["wdl"] = DUMMY_WDL.expand(n, 3).clone()
        data["value_valid"] = torch.zeros(n, dtype=torch.int8)
    if "ep_file" not in data:
        data["ep_file"] = ep_square_to_file(data["ep_square"]).to(torch.int8)
    if "label_depth" not in data:
        data["label_depth"] = torch.zeros(n, dtype=torch.int16)
    if "phase" not in data:
        data["phase"] = torch.ones(n, dtype=torch.int8)
    attach_static_targets(data)
    return data


def audit_corrections(raw: dict, official: dict, incumbent_sha: str) -> dict:
    n = int(raw["turn"].shape[0])
    tags = raw["tag"].numpy() if "tag" in raw else np.full(n, TAG_TO_I["disagree"], dtype=np.int8)
    tag_hist = {I_TO_TAG.get(int(k), str(int(k))): int((tags == k).sum()) for k in np.unique(tags)}
    hs = position_hashes(raw).astype(np.uint64)
    n_unique = int(np.unique(hs).size)
    needs = raw["needs_sf"].numpy() if "needs_sf" in raw else None
    in_pv = raw["model_in_pv"].numpy() if "model_in_pv" in raw else None
    drop = raw["drop_cp"].numpy() if "drop_cp" in raw else None
    origin = raw["origin"].numpy() if "origin" in raw else None
    model_mv = raw["model_move_idx"].numpy() if "model_move_idx" in raw else None
    teacher_mv = raw["move_idx"].numpy()

    perspective = {"method": "compare_cp_to_soft_cps0_by_turn", "rows_compared": 0}
    if "soft_cps" in raw:
        sc0 = raw["soft_cps"][:, 0].numpy()
        cp = raw["cp"].numpy()
        turn = raw["turn"].numpy()
        ok = sc0 != 0
        if ok.any():
            same = (cp[ok] == sc0[ok]).mean()
            flipped_black = (cp[ok & (turn == 1)] == -sc0[ok & (turn == 1)]).mean() if (ok & (turn == 1)).any() else None
            perspective.update({
                "rows_compared": int(ok.sum()),
                "cp_equals_soft_cps0": float(same),
                "black_cp_equals_neg_soft_cps0": None if flipped_black is None else float(flipped_black),
                "interpretation": (
                    "STM" if same > 0.9 else
                    "likely_white_absolute" if flipped_black is not None and flipped_black > 0.9 else
                    "mixed_or_unknown"
                ),
            })

    budget = {}
    for k in ("nodes", "nodes_budget", "label_depth"):
        if k in raw:
            v = raw[k].numpy()
            budget[k] = {
                "min": int(v.min()), "p50": int(np.median(v)),
                "mean": float(v.mean()), "max": int(v.max()),
                "zero_frac": float((v == 0).mean()),
            }

    unresolved = {
        "off_pv_tag": int((tags == TAG_TO_I["off_pv"]).sum()),
        "disagree_tag": int((tags == TAG_TO_I["disagree"]).sum()),
        "ok_tag": int((tags == TAG_TO_I["ok"]).sum()),
        "needs_sf": None if needs is None else int((needs != 0).sum()),
        "not_in_pv": None if in_pv is None else int((in_pv == 0).sum()),
    }

    keep = np.isin(tags, np.fromiter(SUBSTANTIAL, dtype=np.int8))
    keep &= ~np.isin(tags, np.fromiter(EXCLUDE_TAGS, dtype=np.int8))
    if needs is not None:
        keep &= needs == 0
    if model_mv is not None:
        keep &= model_mv != teacher_mv
        keep &= model_mv >= 0
    # majors may have drop_cp unset; require a drop for cp-tagged rows
    if drop is not None:
        cp_tags = np.isin(tags, np.array([
            TAG_TO_I["inaccuracy"], TAG_TO_I["blunder"],
        ], dtype=np.int8))
        keep &= (~cp_tags) | (drop >= 75)

    rejected = {
        "not_substantial_or_harmless": int((~np.isin(tags, np.fromiter(SUBSTANTIAL, dtype=np.int8))).sum()),
        "needs_sf": 0 if needs is None else int(((needs != 0) & np.isin(tags, np.fromiter(SUBSTANTIAL, dtype=np.int8))).sum()),
        "model_equals_teacher": 0 if model_mv is None else int((model_mv == teacher_mv).sum()),
    }

    report = {
        "n_in": n,
        "unique_positions": n_unique,
        "internal_dups": n - n_unique,
        "official_pack": {
            "n": official.get("n"),
            "unique": official.get("unique"),
            "tags": official.get("tags"),
            "padded_with_ok": official.get("padded_with_ok"),
        },
        "tag_hist": tag_hist,
        "origin_hist": None if origin is None else {
            str(int(k)): int((origin == k).sum()) for k in np.unique(origin)
        },
        "checkpoint_provenance": {
            "claimed": "overnight eval_swa.pt / outputs/sf19_ft/overnight_20260908",
            "file_hash_embedded_in_pack": False,
            "local_incumbent_sha256": incumbent_sha,
            "expected_incumbent_sha256": INCUMBENT_SHA,
            "matches_locked_incumbent": incumbent_sha == INCUMBENT_SHA,
            "note": "Pack card names the overnight SWA; rows do not store the weight hash.",
        },
        "score_perspective": perspective,
        "search_budget": budget,
        "unresolved_off_pv": unresolved,
        "severity": {
            "substantial_tags": {k: tag_hist.get(k, 0) for k in ("inaccuracy", "blunder", "conversion", "major")},
            "drop_cp": None if drop is None else {
                "min": int(drop.min()), "p50": int(np.median(drop)),
                "mean": float(drop.mean()), "max": int(drop.max()),
            },
        },
        "filter": {
            "include": ["inaccuracy", "blunder", "conversion", "major"],
            "exclude": ["ok", "off_pv", "disagree", "needs_sf!=0", "model_move==teacher"],
            "n_keep_before_legal": int(keep.sum()),
            "rejected": rejected,
        },
        "verified_enough": bool(
            int(keep.sum()) >= 50_000
            and (needs is None or int((needs[keep] != 0).sum()) == 0)
            and tag_hist.get("ok", 0) + tag_hist.get("off_pv", 0) < n  # pack is not only unresolved
            and incumbent_sha == INCUMBENT_SHA
        ),
    }
    report["keep_mask_n"] = int(keep.sum())
    return report, keep


def download(repo: str, name: str, *, repo_type: str, revision: str) -> Path:
    return Path(hf_hub_download(repo, name, repo_type=repo_type, revision=revision))


def _dedup_keep_first(data: dict) -> dict:
    hs = position_hashes(data).astype(np.uint64)
    _, first = np.unique(hs, return_index=True)
    first.sort()
    return take_rows(data, first)


def load_tagged_mistakes() -> tuple[dict, dict]:
    """Load analyzed shards first, then scan shards. Prefer analyzed labels on dups.

    verified_246k.parquet is a stripped soft export (no tag/needs_sf). Do not audit from it.
    """
    files = list_repo_files(CORR_REPO, repo_type="dataset", revision=CORR_REV)
    analyzed = sorted(f for f in files if f.startswith("data/shard_") and f.endswith(".parquet"))
    scans = sorted(f for f in files if f.startswith("data/scan/shard_") and f.endswith(".parquet"))
    meta = {"analyzed_files": analyzed, "scan_files": scans, "n_analyzed": 0, "n_scan": 0}

    def _load(paths: list[str]) -> dict | None:
        chunks = []
        for fn in paths:
            d = packed_fast(pq.read_table(download(CORR_REPO, fn, repo_type="dataset", revision=CORR_REV)))
            log(f"  mistakes {fn} n={int(d['turn'].shape[0]):,}")
            chunks.append(d)
        if not chunks:
            return None
        return concat_soft_tables(chunks)

    log("load analyzed mistake shards (preferred annotations)")
    a = _load(analyzed)
    log("load scan mistake shards")
    s = _load(scans)
    if a is not None:
        meta["n_analyzed"] = int(a["turn"].shape[0])
        a = _dedup_keep_first(a)
    if s is not None:
        meta["n_scan"] = int(s["turn"].shape[0])
        s = _dedup_keep_first(s)
    if a is None and s is None:
        raise SystemExit("no tagged mistake shards")
    if a is None:
        return s, meta
    if s is None:
        return a, meta
    a_h = position_hashes(a).astype(np.uint64)
    s_h = position_hashes(s).astype(np.uint64)
    extra = take_rows(s, np.flatnonzero(~np.isin(s_h, a_h)))
    merged = concat_soft_tables([a, extra]) if int(extra["turn"].shape[0]) else a
    meta["n_scan_only"] = int(extra["turn"].shape[0])
    meta["prefer"] = "analyzed_over_scan_on_duplicate_positions"
    return merged, meta


def load_sf19_eval_keys(path: Path) -> np.ndarray:
    man = json.loads(path.read_text())
    keys = man.get("keys_hex") or []
    return hex_keys_to_u64(keys)


def recover_sf19(out: Path, blocked: np.ndarray, occupied: np.ndarray) -> tuple[dict, dict, dict]:
    files = [
        f for f in list_repo_files(SF19_REPO, repo_type="dataset", revision=SF19_REV)
        if f.endswith(".parquet") and (f.startswith("data/") or f.startswith("overnight_20260908/"))
    ]
    files.sort()
    train_chunks, eval_chunks = [], []
    stats = {"files": 0, "rows": 0, "split1": 0, "quality": 0, "blocked": 0, "occupied": 0}
    eval_man = download(SF19_REPO, "eval_manifest.json", repo_type="dataset", revision=SF19_REV)
    eval_keys = load_sf19_eval_keys(eval_man)
    blocked = np.unique(np.concatenate([blocked, eval_keys])) if blocked.size else eval_keys
    for fn in files:
        local = download(SF19_REPO, fn, repo_type="dataset", revision=SF19_REV)
        table = pq.read_table(local)
        d = packed_fast(table)
        n = int(d["turn"].shape[0])
        stats["files"] += 1
        stats["rows"] += n
        split = d["split"].numpy() if "split" in d else np.zeros(n, dtype=np.int8)
        # shard_000000 is the upstream eval split
        if fn.endswith("data/shard_000000.parquet"):
            split = np.ones(n, dtype=np.int8)
        hs = position_hashes(d).astype(np.uint64)
        quality = np.ones(n, dtype=bool)
        if "policy_mask" in d:
            quality &= d["policy_mask"].numpy() == 1
        if "nodes_budget" in d:
            quality &= d["nodes_budget"].numpy() >= 100_000
        if "label_depth" in d:
            quality &= d["label_depth"].numpy() >= 12
        is_eval = split != 0
        stats["split1"] += int(is_eval.sum())
        stats["quality"] += int((~quality).sum())
        ev_idx = np.flatnonzero(is_eval)
        if ev_idx.size:
            eval_chunks.append(take_rows(d, ev_idx))
        tr = (~is_eval) & quality
        vs_b = np.isin(hs, blocked)
        vs_o = np.isin(hs, occupied)
        stats["blocked"] += int((tr & vs_b).sum())
        stats["occupied"] += int((tr & vs_o & ~vs_b).sum())
        tr &= ~vs_b & ~vs_o
        tr_idx = np.flatnonzero(tr)
        if tr_idx.size:
            train_chunks.append(take_rows(d, tr_idx))
        log(f"  sf19 {fn} n={n:,} train_keep={int(tr_idx.size):,}")
        del d, table
    train = concat_soft_tables(train_chunks)
    ev = concat_soft_tables(eval_chunks) if eval_chunks else None
    # drop internal dups, keep first
    hs = position_hashes(train).astype(np.uint64)
    _, first = np.unique(hs, return_index=True)
    first.sort()
    train = take_rows(train, first)
    stats["train_unique"] = int(first.size)
    return train, ev, {"stats": stats, "eval_manifest": str(eval_man), "n_eval_keys": int(eval_keys.size)}


def recover_syzygy(out: Path, blocked: np.ndarray, occupied: np.ndarray) -> tuple[dict, dict, dict]:
    files = [
        f for f in list_repo_files(SYZ_REPO, repo_type="dataset", revision=SYZ_REV)
        if f.endswith(".parquet")
    ]
    chunks = []
    for fn in sorted(files):
        local = download(SYZ_REPO, fn, repo_type="dataset", revision=SYZ_REV)
        d = packed_fast(pq.read_table(local))
        chunks.append(d)
        log(f"  syzygy {fn} n={int(d['turn'].shape[0]):,}")
    data = concat_soft_tables(chunks)
    man = make_val_membership(data, n_hold=2000, seed=202, source="deep")
    hs = position_hashes(data).astype(np.uint64)
    val_h = np.asarray(man["hashes"], dtype=np.uint64)
    blocked_syz = np.asarray(man["blocked_hashes"], dtype=np.uint64)
    blocked_all = np.unique(np.concatenate([blocked, blocked_syz])) if blocked.size else blocked_syz
    val_mask = np.isin(hs, val_h)
    keep = ~np.isin(hs, blocked_all) & ~np.isin(hs, occupied)
    # unique
    keep_idx = np.flatnonzero(keep)
    hs_k = hs[keep_idx]
    _, first = np.unique(hs_k, return_index=True)
    train = take_rows(data, keep_idx[np.sort(first)])
    ev = take_rows(data, np.flatnonzero(val_mask)[:2000])
    return train, ev, man


def select_corrections(raw: dict, keep: np.ndarray, blocked: np.ndarray) -> tuple[dict, dict]:
    idx = np.flatnonzero(keep)
    hs = position_hashes(raw).astype(np.uint64)
    vs_block = np.isin(hs[idx], blocked)
    idx = idx[~vs_block]
    if "drop_cp" in raw:
        idx = idx[torch.argsort(raw["drop_cp"][idx], descending=True).numpy()]
    # unique, first (highest drop after sort)
    seen = set()
    uniq = []
    n_legal_fail = 0
    n_policy_fail = 0
    for i in idx:
        h = int(hs[i])
        if h in seen:
            continue
        if not policy_ok(raw, int(i)):
            n_policy_fail += 1
            continue
        if not legal_row(raw, int(i)):
            n_legal_fail += 1
            continue
        seen.add(h)
        uniq.append(int(i))
    taken = np.asarray(uniq, dtype=np.int64)
    out = take_rows(raw, taken)
    tags = out["tag"].numpy() if "tag" in out else None
    drop = out["drop_cp"].numpy() if "drop_cp" in out else None
    report = {
        "n_kept": int(taken.size),
        "unique": int(taken.size),
        "blocked": int(vs_block.sum()),
        "policy_fail": n_policy_fail,
        "legal_fail": n_legal_fail,
        "tags": None if tags is None else {
            I_TO_TAG.get(int(k), str(int(k))): int((tags == k).sum()) for k in np.unique(tags)
        },
        "drop_cp": None if drop is None else {
            "min": int(drop.min()), "p50": int(np.median(drop)), "max": int(drop.max()),
        },
        "repeat_exposure_if_8000x64x0.15": (8000 * 64 * 0.15) / max(int(taken.size), 1),
    }
    return out, report


def recover_resume(dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = download(
        INCUMBENT_REPO, "training_resume.pt", repo_type="model", revision=INCUMBENT_REV,
    ).resolve()
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    import shutil
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    ckpt = torch.load(dest, map_location="cpu", weights_only=False)
    info = {
        "path": str(dest),
        "sha256": sha256_file(dest),
        "eval_only": bool(ckpt.get("eval_only")),
        "has_optimizer": "optimizer_state_dict" in ckpt,
        "has_model": "model_state_dict" in ckpt,
        "n_keys": len(ckpt) if isinstance(ckpt, dict) else None,
        "note": "Recovered for a later live continuation only. Do not load this optimizer into SWA weights.",
        "do_not_use_this_arm": True,
    }
    del ckpt
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/overnight_corr15")
    args = ap.parse_args()
    load_hf_token()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    inc = ROOT / "outputs/sf19_ft/overnight_20260908/eval_swa.pt"
    if not inc.exists():
        raise SystemExit(f"missing incumbent SWA {inc}")
    inc_sha = sha256_file(inc)
    if inc_sha != INCUMBENT_SHA:
        raise SystemExit(f"incumbent sha mismatch {inc_sha}")

    resume_path = ROOT / "outputs/sf19_ft/overnight_20260908/training_resume.pt"
    log("recover training_resume.pt (not used for this arm)")
    resume_info = recover_resume(resume_path)
    json_write(out / "training_resume_info.json", resume_info)
    log(f"  resume sha={resume_info['sha256'][:12]} eval_only={resume_info['eval_only']} opt={resume_info['has_optimizer']}")
    if resume_info["sha256"] == inc_sha:
        log("WARN training_resume is the same file as eval_swa; live optimizer was not recovered")

    log("download + audit tagged mistake shards (not the stripped 246k export)")
    official = {}
    try:
        official = json.loads(
            download(CORR_REPO, CORR_REPORT, repo_type="dataset", revision=CORR_REV).read_text()
        ).get("corrections", {})
    except Exception:
        official = {}
    raw, shard_meta = load_tagged_mistakes()
    audit, keep = audit_corrections(raw, official, inc_sha)
    audit["file"] = {
        "repo": CORR_REPO, "revision": CORR_REV,
        "stripped_export": CORR_FILE,
        "stripped_export_note": "no tag/needs_sf/drop_cp; not used as the audited source",
        "shards": shard_meta,
    }
    json_write(out / "corrections_audit.json", audit)
    log(f"  n={audit['n_in']:,} unique={audit['unique_positions']:,} keep={audit['keep_mask_n']:,} verified={audit['verified_enough']}")
    if not audit["verified_enough"]:
        raise SystemExit("correction pack failed audit; not treating as verified")

    log("select frozen corrections (holdouts applied after pool recovery)")
    corr, corr_rep = select_corrections(raw, keep, np.zeros(0, dtype=np.uint64))
    del raw
    occupied = position_hashes(corr).astype(np.uint64)
    blocked = np.zeros(0, dtype=np.uint64)
    log(f"  corrections kept={int(corr['turn'].shape[0]):,} unique={int(np.unique(occupied).size):,}")

    log("recover SF19 overnight pool")
    sf19, sf19_eval, sf19_rep = recover_sf19(out, blocked, occupied)
    blocked = flip_blocked(sf19_eval, np.ones(int(sf19_eval["turn"].shape[0]), dtype=bool)) if sf19_eval else blocked
    hs = position_hashes(sf19).astype(np.uint64)
    keep_sf = ~np.isin(hs, blocked) & ~np.isin(hs, occupied)
    sf19 = take_rows(sf19, np.flatnonzero(keep_sf))
    log(f"  sf19 train={int(sf19['turn'].shape[0]):,} eval={0 if sf19_eval is None else int(sf19_eval['turn'].shape[0]):,}")

    log("recover Syzygy pool")
    syz, syz_eval, syz_man = recover_syzygy(out, blocked, occupied)
    blocked = np.unique(np.concatenate([
        blocked,
        np.asarray(syz_man["blocked_hashes"], dtype=np.uint64),
    ]))
    log(f"  syzygy train={int(syz['turn'].shape[0]):,} eval={int(syz_eval['turn'].shape[0]):,}")

    corr_hs = position_hashes(corr).astype(np.uint64)
    corr_keep = ~np.isin(corr_hs, blocked)
    n_drop = int((~corr_keep).sum())
    if n_drop:
        corr = take_rows(corr, np.flatnonzero(corr_keep))
        corr_rep["blocked_after_pool"] = n_drop
        corr_rep["n_kept"] = int(corr["turn"].shape[0])
        corr_rep["unique"] = int(corr["turn"].shape[0])
        corr_rep["repeat_exposure_if_8000x64x0.15"] = (8000 * 64 * 0.15) / max(int(corr["turn"].shape[0]), 1)
        log(f"  dropped {n_drop} corrections that hit recovered holdouts")
    corr = ensure_train_fields(corr, source=SOURCE_MISTAKE, value_valid=0)
    # SF19 values stay on; corrections use teacher policy only unless wdl is clean
    if "wdl" in corr:
        w = corr["wdl"].float().reshape(-1, 3)
        valid = torch.isfinite(w).all(1) & (w >= 0).all(1) & ((w.sum(1) - 1).abs() <= 0.001)
        # origin 0 = SF19 may have real WDL; others masked
        if "origin" in corr:
            valid = valid & (corr["origin"] == 0)
        corr["value_valid"] = valid.to(torch.int8)
        dummy = DUMMY_WDL.expand(int(corr["turn"].shape[0]), 3).clone()
        corr["wdl"] = torch.where(valid.unsqueeze(1), corr["wdl"], dummy)

    sf19 = ensure_train_fields(sf19, source=SOURCE_SF19, value_valid=1)
    syz = ensure_train_fields(syz, source=SOURCE_SYZYGY, value_valid=0)
    if sf19_eval is not None:
        sf19_eval = ensure_train_fields(sf19_eval, source=SOURCE_SF19, value_valid=1)
    syz_eval = ensure_train_fields(syz_eval, source=SOURCE_SYZYGY, value_valid=0)

    torch.save(sf19, out / "soft_cache.pt")
    torch.save(syz, out / "deep_cache.pt")
    torch.save(corr, out / "bonus_cache.pt")
    if sf19_eval is not None:
        torch.save(sf19_eval, out / "eval_sf19.pt")
    torch.save(syz_eval, out / "eval_syzygy.pt")

    block_man = {
        "method": "overnight_corr15_union",
        "n_blocked": int(blocked.size),
        "blocked_hashes": [int(x) for x in blocked.tolist()],
        "includes_flips": True,
        "sources": {
            "sf19_eval_manifest": "saved_split_v1 shard_000000 + computed flips",
            "syzygy": syz_man.get("method"),
            "lichess_replay_holdout": "not recovered as a list on this pod",
        },
        "overnight_union_20961": "not recovered as an exact list; used recoverable SF19+syzygy holdouts and flips",
        "note": "Correction set is frozen. Do not resample during the train arm.",
    }
    json_write(out / "blocked_manifest.json", block_man)

    n_sf = int(sf19["turn"].shape[0])
    n_syz = int(syz["turn"].shape[0])
    n_c = int(corr["turn"].shape[0])
    drawn_8k = 8000 * 64
    report = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "do_not_overwrite": True,
        "question": (
            "From overnight SWA, fresh optimizer, original LRs: does a 15% verified "
            "correction bonus beat the cached incumbent on the 2050/2200 screen?"
        ),
        "base": "overnight pools (SF19 soft + Syzygy deep). Not the puzzle-heavy master mix.",
        "replaced_share": {
            "overnight_mix": {"sf19": 0.75, "lichess_replay": 0.20, "syzygy": 0.05},
            "this_arm": {"sf19": 0.80, "corrections": 0.15, "syzygy": 0.05, "lichess_replay": 0.0, "puzzles": 0.0},
            "replaces": "Lichess replay bonus slot (20% -> 0%). The leftover 5% goes to SF19 (75% -> 80%).",
        },
        "incumbent": {"path": str(inc), "sha256": inc_sha},
        "training_resume": resume_info,
        "corrections": {
            "repo": CORR_REPO,
            "revision": CORR_REV,
            "file": CORR_FILE,
            **corr_rep,
            "audit_file": "corrections_audit.json",
        },
        "pools": {
            "sf19_train": n_sf,
            "syzygy_train": n_syz,
            "corrections": n_c,
            "sf19_recover": sf19_rep,
            "syzygy_holdout": {k: syz_man[k] for k in ("method", "n_hold", "n_blocked", "leakage") if k in syz_man},
        },
        "sampling_8k": {
            "soft_draws": int(drawn_8k * 0.80),
            "correction_draws": int(drawn_8k * 0.15),
            "deep_draws": int(drawn_8k * 0.05),
            "unique_corrections": n_c,
            "correction_repeat_exposure": (drawn_8k * 0.15) / max(n_c, 1),
            "sf19_repeat_exposure": (drawn_8k * 0.80) / max(n_sf, 1),
        },
        "trainer": {
            "resume": str(inc),
            "resume_kind": "weights_only_fresh_optimizer",
            "soft_cache": "soft_cache.pt",
            "deep_cache": "deep_cache.pt",
            "bonus_cache": "bonus_cache.pt",
            "deep_mix_frac": 0.05,
            "bonus_mix_frac": 0.15,
            "muon_lr": 0.0007,
            "adam_lr": 1e-5,
            "batch_size": 64,
            "max_steps": 8000,
        },
        "artifacts": {p.name: fingerprint(p) for p in out.glob("*.pt")},
    }
    json_write(out / "FROZEN.json", report)
    json_write(out / "manifest.json", report)
    log("FROZEN " + json.dumps({
        "sf19": n_sf, "syzygy": n_syz, "corrections": n_c,
        "repeat_corr": report["sampling_8k"]["correction_repeat_exposure"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
