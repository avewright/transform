#!/usr/bin/env python3
"""Freeze the exp270 75/20/5 pretrain mix. No puzzles, no correction, no live ingest.

Writes outputs/exp270_mix_v1/{soft_cache.pt,deep_cache.pt,manifest.json,FROZEN.json}.
SF19 is the limiting source (~2.01M audited today). Harvested expand50m shards
are NOT pulled in here — remix after that pool is audited.

Usage:
  python scripts/build_exp270_mix.py --freeze-only   # recipe + holdout inventory
  python scripts/build_exp270_mix.py --go            # download + pack (RunPod / fat disk)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_organized_chess_mix import (  # noqa: E402
    SOURCE_LICHESS,
    SOURCE_SF19,
    SOURCE_SYZYGY,
    collect_blocked,
    json_write,
    load_hf_token,
)
import build_organized_chess_mix as organized  # noqa: E402

RECIPE_PATH = ROOT / "chess_master" / "recipes" / "exp270_75_20_5.json"
OUT_DEFAULT = ROOT / "outputs" / "exp270_mix_v1"
SF19_REPO = "avewright/chess-soft-sf19"
LICHESS_REPO = "avewright/chess-soft-multipv-lichess"
SYZYGY_REPO = "avewright/chess-soft-syzygy"


def load_recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def freeze_only(out: Path) -> dict:
    recipe = load_recipe()
    out.mkdir(parents=True, exist_ok=True)
    blocked, block_files = collect_blocked(out)
    doc = {
        "dataset_version": recipe["dataset_version"],
        "recipe": recipe,
        "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
        "status": "recipe_frozen",
        "output": str(out),
        "existing_block_manifests": block_files,
        "existing_blocked_count": len(blocked),
        "sf19_pool_note": (
            "Audited HF SF19 is ~2.01M after holdout skip. "
            "50M is the harvest target under outputs/sf19_soft/expand50m/. "
            "Do not live-ingest those shards into this first capacity run."
        ),
        "value_supervision": recipe["value_eligible"],
        "incumbent_untouched": True,
    }
    json_write(out / "dataset_manifest.json", doc)
    print(json.dumps({k: doc[k] for k in (
        "dataset_version", "status", "existing_blocked_count", "sf19_pool_note"
    )}, indent=2), flush=True)
    print(f"wrote {out / 'dataset_manifest.json'}", flush=True)
    return doc


def build_mix(out: Path, *, force: bool, sf19_n: int, eval_rows: int, seed: int) -> dict:
    recipe = load_recipe()
    organized.SOURCES = {
        "sf19": (SF19_REPO, recipe["weights"]["sf19"], SOURCE_SF19),
        "lichess": (LICHESS_REPO, recipe["weights"]["lichess"], SOURCE_LICHESS),
        "syzygy": (SYZYGY_REPO, recipe["weights"]["syzygy"], SOURCE_SYZYGY),
    }
    organized.LIMITATIONS = list(recipe.get("limitations") or organized.LIMITATIONS)
    # SF19 is the scarce audited teacher. Size the mix so 75% == available SF19.
    rows = int(round(sf19_n / recipe["weights"]["sf19"]))
    args = SimpleNamespace(
        output=str(out),
        rows=rows,
        eval_rows=eval_rows,
        seed=seed,
        force=force,
    )
    print(
        f"exp270 mix rows={rows:,} (sf19≈{int(rows*0.75):,} lichess≈{int(rows*0.20):,} "
        f"syzygy≈{int(rows*0.05):,})",
        flush=True,
    )
    man = organized.build(args)
    json_write(out / "dataset_manifest.json", {
        "dataset_version": recipe["dataset_version"],
        "recipe": recipe,
        "status": "frozen",
        "mix_manifest": man,
        "incumbent_untouched": True,
    })
    (out / "FROZEN.json").write_text(
        json.dumps({"frozen_at": __import__("datetime").datetime.now().isoformat(), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return man


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--go", action="store_true")
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--output", default=str(OUT_DEFAULT))
    p.add_argument("--sf19-n", type=int, default=2_010_006, help="Audited SF19 rows available (HF pack)")
    p.add_argument("--eval-rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    load_hf_token()
    if args.go:
        build_mix(out, force=args.force, sf19_n=args.sf19_n, eval_rows=args.eval_rows, seed=args.seed)
        return
    freeze_only(out)
    if not args.freeze_only:
        print("recipe frozen. Pass --go on the pod to download and pack.", flush=True)


if __name__ == "__main__":
    main()
