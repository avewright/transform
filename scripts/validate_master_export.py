#!/usr/bin/env python3
"""Hard-fail validation for a chess_master recipe export. No training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_organized_chess_mix import reconstruct_board
from chess_master.recipes import load_recipe, requested_counts
from chess_master.schema import SOURCE_IDS
from move_vocab import VOCAB_SIZE, index_to_move
from scripts.autoresearch_8gb.pipeline import attach_static_targets, position_hashes

EXPECTED_EVAL = 1000
LEGAL_SAMPLE = 4096


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(errors: list[str], msg: str) -> None:
    errors.append(msg)
    print("FAIL", msg, flush=True)


def _ok(msg: str) -> None:
    print("OK", msg, flush=True)


def _check_probs(name: str, d: dict, errors: list[str]) -> None:
    si = d["soft_indices"]
    sp = d["soft_probs"].float()
    if not torch.isfinite(sp).all():
        _fail(errors, f"{name}: non-finite soft_probs")
        return
    valid = si >= 0
    if int((si >= VOCAB_SIZE).sum()) > 0:
        _fail(errors, f"{name}: soft_indices >= {VOCAB_SIZE}")
    if int(((si < 0) & (si != -1)).sum()) > 0:
        _fail(errors, f"{name}: soft_indices < -1")
    mass = (sp * valid.float()).sum(dim=-1)
    empty = int((valid.sum(dim=-1) == 0).sum())
    if empty:
        _fail(errors, f"{name}: {empty} rows with empty policy")
    bad = int(((mass - 1.0).abs() > 1e-3).sum())
    if bad:
        _fail(errors, f"{name}: {bad} rows with |prob mass-1| > 1e-3")
    if int((sp < -1e-8).sum()) > 0:
        _fail(errors, f"{name}: negative soft_probs")
    _ok(f"{name} probs n={len(si):,} mean_mass={float(mass.mean()):.6f}")


def _check_value_valid(name: str, d: dict, expect: int, errors: list[str]) -> None:
    if "value_valid" not in d:
        _fail(errors, f"{name}: missing value_valid")
        return
    vv = d["value_valid"].to(torch.int64)
    n = int(vv.numel())
    ones = int((vv == 1).sum())
    zeros = int((vv == 0).sum())
    if expect == 1 and ones != n:
        _fail(errors, f"{name}: value_valid expected all 1, got ones={ones}/{n}")
    elif expect == 0 and zeros != n:
        _fail(errors, f"{name}: value_valid expected all 0, got zeros={zeros}/{n}")
    else:
        _ok(f"{name} value_valid expect={expect} ones={ones} zeros={zeros}")


def _legal_sample(name: str, d: dict, errors: list[str], rng: np.random.Generator) -> None:
    n = int(d["board_array"].shape[0])
    take = min(LEGAL_SAMPLE, n)
    idx = rng.choice(n, size=take, replace=False)
    bad = 0
    for i in idx:
        i = int(i)
        board = reconstruct_board(d, i)
        if board is None:
            bad += 1
            continue
        try:
            mv = index_to_move(int(d["move_idx"][i]))
            if mv not in board.legal_moves:
                bad += 1
        except Exception:
            bad += 1
    rate = bad / take if take else 0
    # Frozen organized_chess_v1 membership keeps a small illegal-teacher tail.
    # That is a source-quality tag, not a broken join. Hard-fail only if large.
    if rate > 0.03:
        _fail(errors, f"{name}: {bad}/{take} sampled best moves illegal ({rate:.2%})")
    elif bad:
        print(f"WARN {name}: {bad}/{take} sampled best moves illegal ({rate:.2%}) — frozen-mix tail", flush=True)
    else:
        _ok(f"{name} legal-sample {take}/{take}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="outputs/master_v1_baseline/export")
    ap.add_argument("--recipe", default="pilot_45_35_15_5")
    args = ap.parse_args()
    dest = Path(args.export)
    if not dest.is_absolute():
        dest = ROOT / dest
    recipe = load_recipe(args.recipe)
    requested = requested_counts(recipe)
    errors: list[str] = []
    man = json.loads((dest / "manifest.json").read_text())
    if man.get("status") != "complete":
        _fail(errors, f"manifest status={man.get('status')}")
    skipped = man.get("skipped") or {}
    if int(skipped.get("missing_join") or 0) != 0:
        _fail(errors, f"missing_join={skipped.get('missing_join')}")
    else:
        _ok("missing_join=0")

    value_expect = {"sf19": 1, "lichess": 0, "puzzles": 0, "syzygy": 0}
    rng = np.random.default_rng(20260908)
    trains = {}
    evals = {}
    for src in recipe["sources"]:
        tr = torch.load(dest / f"{src}_train.pt", map_location="cpu", weights_only=False)
        ev = torch.load(dest / f"{src}_eval.pt", map_location="cpu", weights_only=False)
        trains[src] = tr
        evals[src] = ev
        n_tr = int(tr["turn"].shape[0])
        n_ev = int(ev["turn"].shape[0])
        if n_tr != requested[src]:
            _fail(errors, f"{src} train {n_tr} != requested {requested[src]}")
        else:
            _ok(f"{src} train {n_tr}")
        if n_ev != EXPECTED_EVAL:
            _fail(errors, f"{src} eval {n_ev} != {EXPECTED_EVAL}")
        else:
            _ok(f"{src} eval {n_ev}")
        src_id = SOURCE_IDS[src]
        if int((tr["source"] != src_id).sum()) or int((ev["source"] != src_id).sum()):
            _fail(errors, f"{src}: source id mismatch")
        _check_probs(f"{src}_train", tr, errors)
        _check_probs(f"{src}_eval", ev, errors)
        _check_value_valid(f"{src}_train", tr, value_expect[src], errors)
        _check_value_valid(f"{src}_eval", ev, value_expect[src], errors)
        _legal_sample(f"{src}_train", tr, errors, rng)

    soft = torch.load(dest / "soft_cache.pt", map_location="cpu", weights_only=False)
    deep = torch.load(dest / "deep_cache.pt", map_location="cpu", weights_only=False)
    attach_static_targets(soft)
    attach_static_targets(deep)
    n_soft = int(soft["turn"].shape[0])
    want_soft = requested["sf19"] + requested["lichess"] + requested["puzzles"]
    if n_soft != want_soft:
        _fail(errors, f"soft_cache {n_soft} != {want_soft}")
    else:
        _ok(f"soft_cache {n_soft}")
    if int(deep["turn"].shape[0]) != requested["syzygy"]:
        _fail(errors, f"deep_cache {int(deep['turn'].shape[0])} != {requested['syzygy']}")
    else:
        _ok(f"deep_cache {int(deep['turn'].shape[0])}")

    # Masks must survive concat + attach_static_targets.
    src = soft["source"]
    vv = soft["value_valid"]
    for name, sid, expect in (("sf19", SOURCE_IDS["sf19"], 1), ("lichess", SOURCE_IDS["lichess"], 0), ("puzzles", SOURCE_IDS["puzzles"], 0)):
        m = src == sid
        got = int((vv[m] == expect).sum())
        tot = int(m.sum())
        if got != tot:
            _fail(errors, f"soft_cache {name} value_valid survived {got}/{tot}")
        else:
            _ok(f"soft_cache {name} value_valid survived {tot}")
    if int((deep["value_valid"] == 0).sum()) != int(deep["value_valid"].numel()):
        _fail(errors, "deep_cache value_valid not all 0 after attach_static_targets")
    else:
        _ok("deep_cache value_valid survived")

    train_h = []
    eval_h = []
    for src in recipe["sources"]:
        train_h.append(position_hashes(trains[src]).astype(np.uint64))
        eval_h.append(position_hashes(evals[src]).astype(np.uint64))
    all_train = np.concatenate(train_h)
    all_eval = np.concatenate(eval_h)
    overlap = np.intersect1d(np.unique(all_train), np.unique(all_eval))
    if overlap.size:
        _fail(errors, f"train ∩ eval hashes = {overlap.size}")
    else:
        _ok("train ∩ eval empty")

    checksums = {p.name: _sha256(p) for p in sorted(dest.glob("*.pt"))}
    checksums["manifest.json"] = _sha256(dest / "manifest.json")
    report = {
        "ok": not errors,
        "errors": errors,
        "legal_policy": "warn_if_sample_rate_le_3pct_frozen_mix_tail",
        "requested": requested,
        "actual_counts": man.get("actual_counts"),
        "eval_counts": man.get("eval_counts"),
        "skipped": skipped,
        "checksums": checksums,
        "recipe": recipe["name"],
        "dataset_revision_note": "export of local snapshot outputs/chess_master_v1",
    }
    (dest / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        print("VALIDATION FAILED", len(errors), "errors", flush=True)
        return 1
    auth = {
        "authorized_for_training": True,
        "recipe": recipe["name"],
        "source_tables_do_not_train": True,
        "derived_export_authorized": True,
        "checksums": checksums,
        "counts": man.get("actual_counts"),
        "eval_counts": man.get("eval_counts"),
        "trainer": recipe.get("trainer"),
    }
    (dest / "AUTHORIZED.json").write_text(json.dumps(auth, indent=2) + "\n")
    print("AUTHORIZED", dest / "AUTHORIZED.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
