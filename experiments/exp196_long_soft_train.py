#!/usr/bin/env python3
"""exp196: Serious local train on the 300k soft mix.

Hypothesis: Longer training on the merged openings+walk MultiPV cache will
break the 1320 Elo floor that short autoresearch budgets could not.

Recipe: wider_shallower (~same params as baseline, faster) + soft T=4 + SWA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_8gb.elo_trial import run_elo_trial
from autoresearch_8gb.train_trial import resolve_trial_config, train_trial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--soft-cache", default="outputs/autoresearch_8gb/soft_cache_200k.pt")
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--train-minutes", type=float, default=180.0)
    ap.add_argument("--skip-elo", action="store_true")
    ap.add_argument("--output-dir", default="outputs/exp196_long_soft_train")
    args = ap.parse_args()
    if not args.go:
        print("Pass --go")
        return

    space = json.loads((ROOT / "scripts/autoresearch_8gb/search_space.json").read_text())
    # Start from faster iso-param arch, then add Elo-friendly train knobs.
    raw = {
        "id": "exp196_wider_soft_swa",
        "desc": "wider_shallower + soft T=4 + SWA long train",
        "inherits": "wider_shallower",
        "model_overrides": {"dropout": 0.0},
        "train_overrides": {
            "soft_frac": 1.0,
            "soft_alpha": 0.6,
            "soft_temp": 4.0,
            "soft_temp_weight": 0.45,
            "use_swa": True,
            "swa_start_frac": 0.75,
            "batch_size": 96,
            "accum_steps": 2,
            "hflip_p": 0.5,
            "value_weight": 0.15,
        },
    }
    trial = resolve_trial_config(raw, space)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    soft = Path(args.soft_cache)
    if not soft.exists():
        raise SystemExit(f"missing soft cache: {soft}")

    max_steps = 30 if args.smoke else args.max_steps
    max_minutes = 3.0 if args.smoke else args.train_minutes
    print(f"exp196 soft={soft} steps={max_steps} minutes={max_minutes}")
    print(f"  trial={trial['id']} {trial['desc']}")

    result = train_trial(
        trial,
        out,
        soft_cache=soft,
        deep_cache=None,
        max_steps=max_steps,
        max_minutes=max_minutes,
        smoke=args.smoke,
    )
    (out / "train_result.json").write_text(json.dumps(result, indent=2, default=str))
    print("train_result", {k: result.get(k) for k in ("status", "steps", "pos_s", "ckpt_path", "n_params")})

    if args.skip_elo or result.get("status") in ("oom", "failed"):
        return
    ckpt = result.get("ckpt_path")
    mcfg = result.get("model_config_path")
    elo = run_elo_trial(ckpt, "exp196_long", model_config=mcfg, smoke=args.smoke)
    summary = {
        "train": {k: result.get(k) for k in ("status", "steps", "pos_s", "ckpt_path", "n_params")},
        "elo": elo.get("elo"),
        "elo_estimate": elo.get("estimate"),
        "soft_cache": str(soft),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("summary", summary)


if __name__ == "__main__":
    main()
