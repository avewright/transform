#!/usr/bin/env python3
"""Freeze organized_chess_v1 and build one SF19→correction swap variant.

The 45% SF19-family share stays 45% of the 1M pool:
  baseline  450k ordinary SF19
  variant   300k ordinary SF19 + 150k verified substantial corrections

Lichess, puzzles, Syzygy, and all eval shards are unchanged. No new positions
are generated. Does not start training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from build_organized_chess_mix import (
    DUMMY_WDL,
    DTYPES,
    SOURCE_SF19,
    canonical_hashes,
    fingerprint,
    json_write,
    legal_row,
    load_hf_token,
    policy_ok,
    squeeze_scalars,
)
from data_loader import compute_wdl, ep_square_to_file
from harvest_swa_mistakes import I_TO_TAG, TAG_TO_I

CORR_REPO = "avewright/chess-soft-100m-swa-mistakes"
CORR_REV = "69b75e12ab2867c8b9a63d385065dfe646cf86e1"
CORR_FILE = "data/corrections/verified_246k.parquet"
CORR_REPORT = "data/corrections/mix_v2_report.json"
INCUMBENT_REPO = "avewright/chess-transformer-100m-overnight_20260908"
INCUMBENT_REV = "2314e519cf17157ad677421b2ac68eba0bce58fb"
INCUMBENT_DIR = ROOT / "outputs/sf19_ft/overnight_20260908"
SOURCE_MISTAKE = 3
SUBSTANTIAL = frozenset({
    TAG_TO_I["inaccuracy"], TAG_TO_I["blunder"],
    TAG_TO_I["conversion"], TAG_TO_I["major"],
})
SOFT_KEYS = (
    "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
    "soft_indices", "soft_probs", "label_depth", "phase", "source", "wdl",
    "value_valid",
)


def load_pt(path: Path) -> dict:
    return squeeze_scalars(torch.load(path, map_location="cpu", weights_only=False))


def blocked_set(baseline: Path) -> np.ndarray:
    man = json.loads((baseline / "blocked_manifest.json").read_text())
    return np.asarray(man["blocked_hashes"], dtype=np.uint64)


def freeze_baseline(baseline: Path) -> dict:
    existing = baseline / "FROZEN.json"
    man = json.loads((baseline / "manifest.json").read_text())
    if existing.exists() and man.get("status") == "frozen":
        return json.loads(existing.read_text())
    if man.get("status") not in ("complete", "frozen"):
        raise SystemExit(f"{baseline} is not a complete mix")
    artifacts = {p.name: fingerprint(p) for p in sorted(baseline.glob("*.pt"))}
    frozen = {
        "role": "baseline",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": man.get("seed"),
        "actual_counts": man.get("actual_counts"),
        "artifacts": artifacts,
        "blocked_manifest": "blocked_manifest.json",
        "do_not_overwrite": True,
        "variant": "outputs/organized_chess_v1_corr15",
        "note": "Frozen 1M organized mix. Scale only after the correction variant compare.",
    }
    json_write(existing, frozen)
    man["status"] = "frozen"
    man["frozen"] = {k: frozen[k] for k in ("role", "frozen_at", "do_not_overwrite", "variant", "note")}
    json_write(baseline / "manifest.json", man)
    return frozen


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
    if "tag" in names:
        out["tag"] = torch.as_tensor(np.asarray(table["tag"].to_numpy()), dtype=torch.int8)
    if "drop_cp" in names:
        out["drop_cp"] = torch.as_tensor(np.asarray(table["drop_cp"].to_numpy()), dtype=torch.int32)
    out["source"] = torch.full((n,), SOURCE_MISTAKE, dtype=torch.int8)
    out["ep_square"] = torch.where(out["ep_square"] <= 0, -1, out["ep_square"]).to(torch.int8)
    return squeeze_scalars(out)


def wdl_valid_mask(wdl: torch.Tensor) -> torch.Tensor:
    w = wdl.float().reshape(-1, 3)
    return (
        torch.isfinite(w).all(1)
        & (w >= 0).all(1)
        & ((w.sum(1) - 1).abs() <= 0.001)
    )


def select_corrections(data: dict, *, n_take: int, blocked: np.ndarray, occupied: np.ndarray) -> tuple[dict, dict]:
    n = int(data["turn"].shape[0])
    keep = np.ones(n, dtype=bool)
    report = {"n_in": n, "rejected": {}}
    if "tag" in data:
        sub = np.isin(data["tag"].numpy(), np.fromiter(SUBSTANTIAL, dtype=np.int8))
        report["rejected"]["not_substantial"] = int((~sub).sum())
        keep &= sub
    hs = canonical_hashes(data)
    vs_block = np.isin(hs, blocked)
    vs_base = np.isin(hs, occupied)
    report["rejected"]["blocked"] = int(vs_block.sum())
    report["rejected"]["overlap_baseline"] = int((vs_base & ~vs_block).sum())
    keep &= ~vs_block & ~vs_base
    idx = np.flatnonzero(keep)
    if "drop_cp" in data:
        idx = idx[torch.argsort(data["drop_cp"][idx], descending=True).numpy()]
    taken = []
    n_bad = 0
    for i in idx:
        if policy_ok(data, int(i)) and legal_row(data, int(i)):
            taken.append(int(i))
            if len(taken) >= n_take:
                break
        else:
            n_bad += 1
    report["rejected"]["invalid_position_or_targets"] = n_bad
    if len(taken) < n_take:
        raise RuntimeError(f"only {len(taken)} qualifying corrections, need {n_take}")
    taken_i = np.asarray(taken, dtype=np.int64)
    report.update(n_kept=int(taken_i.size), unique=int(np.unique(hs[taken_i]).size))
    if "tag" in data:
        tags = data["tag"].numpy()[taken_i]
        report["tags"] = {I_TO_TAG.get(int(k), str(int(k))): int((tags == k).sum()) for k in np.unique(tags)}
    if "drop_cp" in data:
        drop = data["drop_cp"].numpy()[taken_i]
        report["drop_cp"] = {"min": int(drop.min()), "p50": int(np.median(drop)), "max": int(drop.max())}
    out = {k: v[taken_i].contiguous() for k, v in data.items() if torch.is_tensor(v)}
    out["source"] = torch.full((n_take,), SOURCE_MISTAKE, dtype=torch.int8)
    if "wdl" not in out or out["wdl"].ndim != 2:
        out["wdl"] = compute_wdl(out["cp"], out["mate"])
    valid = wdl_valid_mask(out["wdl"]).to(torch.int8)
    out["value_valid"] = valid
    dummy = DUMMY_WDL.expand(n_take, 3).clone()
    out["wdl"] = torch.where(valid.bool().unsqueeze(1), out["wdl"], dummy)
    if "ep_file" not in out:
        out["ep_file"] = ep_square_to_file(out["ep_square"]).to(torch.int8)
    return squeeze_scalars(out), report


def replace_sf19(sf19: dict, corrections: dict, n_keep_sf19: int, seed: int) -> dict:
    n = int(sf19["turn"].shape[0])
    rng = np.random.RandomState(seed)
    keep = rng.permutation(n)[:n_keep_sf19]
    kept = {k: v[keep].contiguous() for k, v in sf19.items() if torch.is_tensor(v) and v.shape[0] == n}
    keys = [k for k in SOFT_KEYS if k in kept and k in corrections]
    return squeeze_scalars({k: torch.cat([kept[k], corrections[k]], dim=0) for k in keys})


def ensure_value_mask(data: dict, valid: int) -> dict:
    n = int(data["turn"].shape[0])
    if "value_valid" not in data:
        data["value_valid"] = torch.full((n,), valid, dtype=torch.int8)
    if "wdl" not in data:
        data["wdl"] = DUMMY_WDL.expand(n, 3).clone()
    if "ep_file" not in data:
        data["ep_file"] = ep_square_to_file(data["ep_square"]).to(torch.int8)
    return data


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_incumbent() -> dict:
    load_hf_token()
    INCUMBENT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Path(hf_hub_download(
        INCUMBENT_REPO, "eval_swa.pt", repo_type="model", revision=INCUMBENT_REV,
    ))
    cfg = Path(hf_hub_download(
        INCUMBENT_REPO, "model_config.json", repo_type="model", revision=INCUMBENT_REV,
    ))
    link_or_copy(ckpt, INCUMBENT_DIR / "eval_swa.pt")
    link_or_copy(cfg, INCUMBENT_DIR / "model_config.json")
    return {
        "repo": INCUMBENT_REPO,
        "revision": INCUMBENT_REV,
        "path": str((INCUMBENT_DIR / "eval_swa.pt").relative_to(ROOT)),
        "sha256": fingerprint(INCUMBENT_DIR / "eval_swa.pt"),
    }


def occupied_hashes(baseline: Path) -> np.ndarray:
    chunks = []
    for name in ("sf19", "lichess", "puzzles", "syzygy"):
        for split in ("train", "eval"):
            d = load_pt(baseline / f"{name}_{split}.pt")
            chunks.append(canonical_hashes(d))
            del d
    return np.unique(np.concatenate(chunks))


def build(args) -> dict:
    load_hf_token()
    baseline = Path(args.baseline)
    if not baseline.is_absolute():
        baseline = ROOT / baseline
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    n_corr = int(args.correction_rows)
    if out.exists() and (out / "manifest.json").exists() and not args.force:
        prev = json.loads((out / "manifest.json").read_text())
        if prev.get("status") == "complete":
            raise SystemExit(f"Refusing to overwrite complete variant at {out}; pass --force")
    frozen = freeze_baseline(baseline)
    out.mkdir(parents=True, exist_ok=True)
    incumbent = stage_incumbent()

    corr_path = Path(hf_hub_download(
        CORR_REPO, CORR_FILE, repo_type="dataset", revision=CORR_REV,
    ))
    official = {}
    try:
        rep_path = Path(hf_hub_download(
            CORR_REPO, CORR_REPORT, repo_type="dataset", revision=CORR_REV,
        ))
        official = json.loads(rep_path.read_text()).get("corrections", {})
    except Exception:
        official = {}
    print("loading corrections", corr_path, flush=True)
    raw = packed_fast(pq.read_table(corr_path))
    print(f"corrections in={int(raw['turn'].shape[0])}", flush=True)
    blocked = blocked_set(baseline)
    occupied = occupied_hashes(baseline)
    corr, corr_rep = select_corrections(raw, n_take=n_corr, blocked=blocked, occupied=occupied)
    del raw

    sf19 = ensure_value_mask(load_pt(baseline / "sf19_train.pt"), 1)
    n_keep = int(sf19["turn"].shape[0]) - n_corr
    if n_keep <= 0:
        raise SystemExit("correction_rows >= SF19 train rows")
    mixed_sf = replace_sf19(sf19, corr, n_keep, args.seed)
    del sf19
    torch.save(mixed_sf, out / "sf19_family_train.pt")
    torch.save(corr, out / "corrections_train.pt")

    lich = ensure_value_mask(load_pt(baseline / "lichess_train.pt"), 0)
    puz = ensure_value_mask(load_pt(baseline / "puzzles_train.pt"), 0)
    keys = [k for k in SOFT_KEYS if k in mixed_sf and k in lich and k in puz]
    soft = squeeze_scalars({k: torch.cat([mixed_sf[k], lich[k], puz[k]], dim=0) for k in keys})
    if "ep_file" not in soft:
        soft["ep_file"] = ep_square_to_file(soft["ep_square"]).to(torch.int8)
    torch.save(soft, out / "soft_cache.pt")
    link_or_copy(baseline / "deep_cache.pt", out / "deep_cache.pt")
    link_or_copy(baseline / "blocked_manifest.json", out / "blocked_manifest.json")
    for name in ("sf19", "lichess", "puzzles", "syzygy"):
        link_or_copy(baseline / f"{name}_eval.pt", out / f"{name}_eval.pt")

    n_soft = int(soft["turn"].shape[0])
    n_sf = int((soft["source"] == SOURCE_SF19).sum())
    n_m = int((soft["source"] == SOURCE_MISTAKE).sum())
    n_lich = int(lich["turn"].shape[0])
    n_puz = int(puz["turn"].shape[0])
    n_syz = int(load_pt(out / "deep_cache.pt")["turn"].shape[0])
    report = {
        "status": "complete",
        "baseline": str(baseline.relative_to(ROOT)),
        "baseline_frozen": frozen,
        "variant": "sf19_bucket_swap_15pct_of_pool",
        "hypothesis": (
            "Same 1M pool and 45% SF19-family share. Replace 150k ordinary SF19 "
            "rows with verified substantial SWA corrections. Other buckets fixed."
        ),
        "seed": args.seed,
        "incumbent": incumbent,
        "corrections": {
            "repo": CORR_REPO,
            "revision": CORR_REV,
            "file": CORR_FILE,
            "sha256": fingerprint(corr_path),
            "official_pack": {
                "n": official.get("n"),
                "unique": official.get("unique"),
                "tags": official.get("tags"),
                "padded_with_ok": official.get("padded_with_ok", False),
            },
            **corr_rep,
            "value_valid": int(corr["value_valid"].sum()),
            "padded_with_ok": False,
        },
        "actual_counts": {
            "sf19": n_sf,
            "corrections": n_m,
            "lichess": n_lich,
            "puzzles": n_puz,
            "syzygy": n_syz,
        },
        "drawn_if_deep_mix_0.05": {
            "sf19": 0.95 * n_sf / n_soft,
            "corrections": 0.95 * n_m / n_soft,
            "lichess": 0.95 * n_lich / n_soft,
            "puzzles": 0.95 * n_puz / n_soft,
            "syzygy": 0.05,
        },
        "shared_eval": "baseline eval shards; not claimed unseen by old checkpoints",
        "trainer": {
            "soft_cache": "soft_cache.pt",
            "deep_cache": "deep_cache.pt",
            "deep_mix_frac": 0.05,
            "bonus_mix_frac": 0,
            "note": "Corrections live inside soft_cache. Do not also set bonus_mix_frac.",
        },
        "artifacts": {
            p.name: fingerprint(p)
            for p in (out / "soft_cache.pt", out / "corrections_train.pt", out / "sf19_family_train.pt")
        },
    }
    json_write(out / "manifest.json", report)
    print("VARIANT", out, json.dumps(report["actual_counts"]), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="outputs/organized_chess_v1")
    p.add_argument("--output", default="outputs/organized_chess_v1_corr15")
    p.add_argument("--correction-rows", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--force", action="store_true")
    build(p.parse_args())
