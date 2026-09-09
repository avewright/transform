"""CLI for inventory, sample ingest, mix migration, export, and reporting."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from chess_master.export import export_recipe
from chess_master.ingest import ingest_organized_mix, ingest_samples
from chess_master.inventory import build_inventory
from chess_master.io_util import ROOT, json_write
from chess_master.report import build_quality_report
from chess_master.schema import DATASET_VERSION


def default_out() -> Path:
    return ROOT / "outputs" / DATASET_VERSION


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(default_out()))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inventory")
    s = sub.add_parser("sample")
    s.add_argument("--per-source", type=int, default=64)
    sub.add_parser("ingest-mix")
    e = sub.add_parser("export")
    e.add_argument("--recipe", default="pilot_45_35_15_5")
    e.add_argument("--dest", default="outputs/chess_master_v1/exports/pilot_45_35_15_5")
    e.add_argument("--force", action="store_true")
    sub.add_parser("report")
    b = sub.add_parser("build")
    b.add_argument("--per-source", type=int, default=64)
    b.add_argument("--skip-mix", action="store_true", help="Stop after sample review artifacts")
    b.add_argument("--export", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "inventory":
        report = build_inventory(out)
        print("inventory", out / "inventory.json", flush=True)
        return
    if args.cmd == "sample":
        inv = build_inventory(out)
        report = ingest_samples(out, inv, per_source=args.per_source)
        print("sample", report, flush=True)
        return
    if args.cmd == "ingest-mix":
        inv = build_inventory(out)
        report = ingest_organized_mix(out, inv)
        print("mix", report.get("status"), report.get("counts"), flush=True)
        return
    if args.cmd == "export":
        dest = Path(args.dest)
        if not dest.is_absolute():
            dest = ROOT / dest
        report = export_recipe(out, args.recipe, dest, force=args.force)
        print("export", report["actual_counts"], flush=True)
        return
    if args.cmd == "report":
        report = build_quality_report(out)
        print("report", out / "quality_report.json", flush=True)
        print({k: report.get(k) for k in ("positions", "annotations", "source_counts", "conflicts", "rejected")}, flush=True)
        return
    if args.cmd == "build":
        inv = build_inventory(out)
        sample = ingest_samples(out, inv, per_source=args.per_source)
        json_write(out / "input_manifest.json", {
            "dataset_version": DATASET_VERSION,
            "inventory": str((out / "inventory.json").relative_to(ROOT)),
            "sample": sample,
            "do_not_upload": True,
            "do_not_train": True,
        })
        print("sample complete; review outputs/chess_master_v1/samples/cross_source.json", flush=True)
        if args.skip_mix:
            return
        mix = ingest_organized_mix(out, inv)
        print("mix", mix.get("status"), flush=True)
        report = build_quality_report(out)
        print("quality", {k: report.get(k) for k in ("positions", "annotations", "source_counts")}, flush=True)
        if args.export:
            dest = out / "exports" / "pilot_45_35_15_5"
            export_recipe(out, "pilot_45_35_15_5", dest)
        return
    raise SystemExit(args.cmd)


if __name__ == "__main__":
    main()
