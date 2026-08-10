#!/usr/bin/env python3
"""train → pure-policy Elo → promote cycle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.common import CHAMPION_DIR, DEFAULT_SEED_CKPT, ROOT, ensure_best_aliases, load_protocol
from harness.promote import promote_from_elo_json
from harness import train_soft_ft
from harness import elo as elo_mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Max-Elo loop: soft FT → policy Elo → promote")
    ap.add_argument("--name", required=True, help="Run name (outputs/runs/<name>)")
    ap.add_argument("--init", default=None, help="Init ckpt (default: champion or FT3h seed)")
    ap.add_argument("--soft-cache", required=True)
    ap.add_argument("--deep-soft-cache", default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--soft-frac", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=0.38)
    ap.add_argument("--deep-mix-frac", type=float, default=0.42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-elo", action="store_true")
    ap.add_argument("--skip-promote", action="store_true")
    ap.add_argument("--games-per-opening-per-color", type=int, default=None)
    ap.add_argument("--elos", type=int, nargs="+", default=None)
    ap.add_argument("--force-promote", action="store_true")
    args = ap.parse_args(argv)

    out = ROOT / "outputs" / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    init = args.init
    if init is None:
        champ = CHAMPION_DIR / "champion.pt"
        init = str(champ if champ.exists() else DEFAULT_SEED_CKPT)

    # 1) Train
    if not args.skip_train:
        train_argv = [
            "--init",
            init,
            "--soft-cache",
            args.soft_cache,
            "--out",
            str(out),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--soft-frac",
            str(args.soft_frac),
            "--soft-alpha",
            str(args.soft_alpha),
            "--deep-mix-frac",
            str(args.deep_mix_frac),
        ]
        if args.deep_soft_cache:
            train_argv.extend(["--deep-soft-cache", args.deep_soft_cache])
        if args.smoke:
            train_argv.append("--smoke")
        rc = train_soft_ft.main(train_argv)
        if rc != 0:
            return rc
    ensure_best_aliases(out)
    best = out / "best.pt"
    if not best.exists():
        latest = out / "latest.pt"
        if latest.exists():
            best = latest
        else:
            print(f"ERROR: no checkpoint in {out}", flush=True)
            return 1

    # 2) Pure policy Elo
    elo_json = None
    if not args.skip_elo:
        elo_argv = [
            "--ckpt",
            str(best),
            "--mode",
            "policy",
            "--out-prefix",
            f"runs_{args.name}_policy",
            "--no-book",
            "--no-syzygy",
        ]
        if args.games_per_opening_per_color is not None:
            elo_argv.extend(
                ["--games-per-opening-per-color", str(args.games_per_opening_per_color)]
            )
        if args.elos is not None:
            elo_argv.extend(["--elos", *[str(e) for e in args.elos]])
        if args.smoke:
            elo_argv.extend(["--games-per-opening-per-color", "1", "--elos", "1450", "1600"])
        rc = elo_mod.main(elo_argv)
        if rc != 0:
            return rc
        elo_json = ROOT / "outputs" / f"elo_eval_runs_{args.name}_policy.json"
    else:
        elo_json = ROOT / "outputs" / f"elo_eval_runs_{args.name}_policy.json"

    # 3) Promote
    result = {"skipped_promote": True}
    if not args.skip_promote:
        if elo_json is None or not elo_json.exists():
            print(f"ERROR: missing elo json {elo_json}", flush=True)
            return 1
        result = promote_from_elo_json(
            elo_json,
            ckpt=best,
            force=args.force_promote,
            dry_run=False,
        )
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "name": args.name,
        "out": str(out),
        "best": str(best),
        "elo_json": str(elo_json) if elo_json else None,
        "promote": result,
        "champion": str(CHAMPION_DIR / "CHAMPION.json"),
    }
    (out / "loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
