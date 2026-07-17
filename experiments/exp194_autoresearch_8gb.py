#!/usr/bin/env python3
"""exp194: 8GB Elo Autoresearch Lab.

Trains ~25M architecture/data/optimizer variants under a fixed budget, then
promotes champions using policy Elo games only (elo_eval_latest). Soft metrics
are diagnostic — never used for crowning.

Spec: docs/superpowers/specs/2026-07-15-elo-autoresearch-8gb-design.md

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp194_autoresearch_8gb.py --go --smoke
  MOVE_VOCAB_VERSION=compact python experiments/exp194_autoresearch_8gb.py --go
  MOVE_VOCAB_VERSION=compact python experiments/exp194_autoresearch_8gb.py --go --max-trials 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_8gb.elo_trial import run_elo_trial
from autoresearch_8gb.pareto import should_promote_champion, update_pareto
from autoresearch_8gb.train_trial import find_cache, resolve_trial_config, train_trial

SPACE_PATH = ROOT / "scripts" / "autoresearch_8gb" / "search_space.json"
OUT_ROOT = ROOT / "outputs" / "autoresearch_8gb"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_space() -> dict:
    return json.loads(SPACE_PATH.read_text(encoding="utf-8"))


def load_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_journal(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def completed_ids(journal: list[dict], *, min_steps: int = 0) -> set[str]:
    """Ids with a successful Elo-backed result and enough train steps."""
    done = set()
    for row in journal:
        if row.get("status") != "done" or not row.get("id"):
            continue
        if row.get("elo_estimate") is None:
            continue
        if int(row.get("steps") or 0) < min_steps:
            continue
        done.add(row["id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--train-minutes", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--only", type=str, nargs="*", default=None, help="Run only these trial ids")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle trial order after baseline")
    ap.add_argument("--skip-elo", action="store_true", help="Train only (debug)")
    ap.add_argument("--output-dir", type=str, default=str(OUT_ROOT))
    ap.add_argument("--soft-cache", type=str, default=None)
    ap.add_argument("--deep-cache", type=str, default=None)
    ap.add_argument(
        "--min-steps-done", type=int, default=2000,
        help="Ignore prior 'done' trials with fewer train steps (force re-run).",
    )
    ap.add_argument("--force", action="store_true", help="Re-run all selected trials")
    args = ap.parse_args()

    if not args.go:
        print("DRY RUN. Pass --go to start autoresearch.")
        print(f"Search space: {SPACE_PATH}")
        return

    space = load_space()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    journal_path = out_root / "trials.jsonl"
    journal = load_journal(journal_path)
    done = set() if args.force else completed_ids(journal, min_steps=args.min_steps_done)

    soft = Path(args.soft_cache) if args.soft_cache else find_cache(space["soft_cache_candidates"])
    deep = Path(args.deep_cache) if args.deep_cache else find_cache(space["deep_cache_candidates"])
    train_minutes = args.train_minutes if args.train_minutes is not None else (
        2.0 if args.smoke else float(space.get("default_train_minutes", 45))
    )
    max_steps = args.max_steps if args.max_steps is not None else (
        15 if args.smoke else int(space.get("default_max_steps", 3000))
    )
    elo_noise = float(space.get("elo_noise", 100))
    speed_frac = float(space.get("speed_promote_frac", 0.20))

    # Resolve all trials
    resolved = [resolve_trial_config(t, space) for t in space["trials"]]
    # Baseline first
    baseline_id = space.get("baseline_id", resolved[0]["id"])
    ordered = sorted(resolved, key=lambda t: (0 if t["id"] == baseline_id else 1, t["id"]))
    if args.only:
        only = set(args.only)
        ordered = [t for t in ordered if t["id"] in only]
    if args.shuffle and len(ordered) > 1:
        head = [t for t in ordered if t["id"] == baseline_id]
        tail = [t for t in ordered if t["id"] != baseline_id]
        random.shuffle(tail)
        ordered = head + tail

    champion = None
    champ_path = out_root / "champion.json"
    if champ_path.exists():
        champion = json.loads(champ_path.read_text(encoding="utf-8"))

    print(f"[{utcnow()}] autoresearch start out={out_root}")
    print(f"  soft={soft} deep={deep}")
    print(f"  train_minutes={train_minutes} max_steps={max_steps} smoke={args.smoke}")
    print(f"  already_done={sorted(done)}")

    if soft is None and not args.smoke:
        print("ERROR: no soft cache found. Pass --soft-cache PATH or place cache under outputs/")
        sys.exit(1)

    n_run = 0
    for trial in ordered:
        if args.max_trials is not None and n_run >= args.max_trials:
            break
        tid = trial["id"]
        if tid in done:
            print(f"skip done {tid}")
            continue

        trial_dir = out_root / "trials" / tid
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== TRIAL {tid}: {trial.get('desc','')} ===")

        row: dict = {
            "id": tid,
            "desc": trial.get("desc", ""),
            "started_at": utcnow(),
            "config": trial,
            "status": "running",
        }
        append_journal(journal_path, {**row, "event": "start"})

        train_result = train_trial(
            trial,
            trial_dir,
            soft_cache=soft,
            deep_cache=deep,
            max_steps=max_steps,
            max_minutes=train_minutes,
            smoke=args.smoke,
        )
        row.update(train_result)
        row["pos_s"] = float(train_result.get("pos_s") or 0.0)

        if train_result.get("status") in ("oom", "failed"):
            row["status"] = train_result["status"]
            row["finished_at"] = utcnow()
            append_journal(journal_path, row)
            write_json(trial_dir / "result.json", row)
            # Shrink batch hint for future (logged only)
            print(f"  FAIL {tid}: {train_result.get('error')}")
            n_run += 1
            continue

        elo_result = None
        if args.skip_elo:
            row["status"] = "trained_no_elo"
            row["elo_estimate"] = None
        else:
            ckpt = train_result.get("ckpt_path")
            mcfg = train_result.get("model_config_path")
            print(f"  elo eval {ckpt}")
            elo_result = run_elo_trial(
                ckpt,
                f"ar8gb_{tid}",
                model_config=mcfg,
                smoke=args.smoke,
            )
            row["elo_raw"] = {
                "rc": elo_result.get("rc"),
                "json_path": elo_result.get("json_path"),
                "estimate": elo_result.get("estimate"),
            }
            row["elo_estimate"] = elo_result.get("elo")
            if elo_result.get("rc") != 0 and elo_result.get("elo") is None:
                row["status"] = "elo_failed"
                row["error"] = elo_result.get("stderr_tail") or elo_result.get("stdout_tail")
            else:
                row["status"] = "done"

        row["finished_at"] = utcnow()
        append_journal(journal_path, row)
        write_json(trial_dir / "result.json", row)

        # Reload journal for pareto
        journal = load_journal(journal_path)
        # Collapse to latest row per id with status done
        latest: dict[str, dict] = {}
        for r in journal:
            if r.get("id") and r.get("status") == "done":
                latest[r["id"]] = r
        front = update_pareto(list(latest.values()))
        write_json(out_root / "pareto.json", {
            "updated_at": utcnow(),
            "front": [
                {"id": t["id"], "elo": t.get("elo_estimate"), "pos_s": t.get("pos_s")}
                for t in front
            ],
        })

        if row.get("status") == "done" and should_promote_champion(
            row, champion, elo_noise=elo_noise, speed_promote_frac=speed_frac,
        ):
            champion = {
                "id": tid,
                "elo_estimate": row.get("elo_estimate"),
                "pos_s": row.get("pos_s"),
                "ckpt_path": row.get("ckpt_path"),
                "model_config_path": row.get("model_config_path"),
                "config": trial,
                "promoted_at": utcnow(),
                "n_params": row.get("n_params"),
            }
            write_json(champ_path, champion)
            # Copy weights
            if row.get("ckpt_path") and Path(row["ckpt_path"]).exists():
                shutil.copy2(row["ckpt_path"], out_root / "champion.pt")
            print(f"  NEW CHAMPION {tid} elo={row.get('elo_estimate')} pos_s={row.get('pos_s'):.1f}")
        else:
            print(f"  done {tid} elo={row.get('elo_estimate')} pos_s={row.get('pos_s'):.1f} status={row.get('status')}")

        n_run += 1

    print(f"\n[{utcnow()}] autoresearch finished trials_this_run={n_run}")
    if champ_path.exists():
        print(f"champion: {champ_path.read_text(encoding='utf-8')[:500]}")


if __name__ == "__main__":
    main()
