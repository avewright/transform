#!/usr/bin/env python3
"""exp197: HF soft mix long train — break the 1320 Elo floor.

Hypothesis: Local short budgets + shallow MultiPV pinned Elo at the SF
UCI_LimitStrength floor. Training `wider_shallower` and `meta_shaw_elo`
for a matched long budget on avewright HF soft packs
(chess-soft-multipv-lichess + chess-soft-syzygy + local ballast) will
produce a real Elo signal and a promotable champion.

Usage:
  # 1) build mix (once)
  python scripts/build_hf_elo_mix.py --go
  # 2) train + Elo
  python experiments/exp197_hf_soft_long_train.py --go
  python experiments/exp197_hf_soft_long_train.py --go --only wider_shallower
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

DEFAULT_SOFT = ROOT / "outputs/hf_soft_mix/soft_cache.pt"
DEFAULT_DEEP = ROOT / "outputs/hf_soft_mix/deep_soft.pt"

TRIALS = [
    {
        "id": "exp197_wider_hf",
        "desc": "wider_shallower + HF soft mix + soft T=4 + SWA",
        "inherits": "wider_shallower",
        "model_overrides": {"dropout": 0.0},
        "train_overrides": {
            "soft_frac": 1.0,
            "soft_alpha": 0.55,
            "soft_temp": 4.0,
            "soft_temp_weight": 0.4,
            "deep_mix_frac": 0.4,
            "use_swa": True,
            "swa_start_frac": 0.75,
            "batch_size": 96,
            "accum_steps": 2,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
        },
    },
    {
        "id": "exp197_meta_shaw_hf",
        "desc": "meta_shaw_elo + HF soft mix (architecture challenger)",
        "inherits": "meta_shaw_elo",
        "model_overrides": {"dropout": 0.0},
        "train_overrides": {
            "soft_frac": 1.0,
            "soft_alpha": 0.55,
            "soft_temp": 4.0,
            "soft_temp_weight": 0.4,
            "deep_mix_frac": 0.4,
            "use_swa": True,
            "swa_start_frac": 0.75,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
        },
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--soft-cache", default=str(DEFAULT_SOFT))
    ap.add_argument("--deep-cache", default=str(DEFAULT_DEEP))
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--train-minutes", type=float, default=180.0)
    ap.add_argument("--skip-elo", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--output-dir", default="outputs/exp197_hf_soft_long_train")
    args = ap.parse_args()
    if not args.go:
        print("Pass --go")
        return

    soft = Path(args.soft_cache)
    deep = Path(args.deep_cache) if args.deep_cache else None
    if not soft.exists():
        raise SystemExit(
            f"missing soft cache: {soft}\n"
            "Run: python scripts/build_hf_elo_mix.py --go"
        )
    if deep is not None and not deep.exists():
        print(f"warn: deep cache missing ({deep}); training without deep_mix", flush=True)
        deep = None

    space = json.loads((ROOT / "scripts/autoresearch_8gb/search_space.json").read_text())
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    summaries = []
    for raw in TRIALS:
        tid = raw["id"]
        if only and not any(o in tid or o in (raw.get("inherits") or "") for o in only):
            continue

        trial = resolve_trial_config(raw, space)
        out = out_root / trial["id"]
        out.mkdir(parents=True, exist_ok=True)
        max_steps = 30 if args.smoke else args.max_steps
        max_minutes = 3.0 if args.smoke else args.train_minutes
        print(f"\n=== {trial['id']} soft={soft} deep={deep} steps={max_steps} min={max_minutes} ===", flush=True)

        result = train_trial(
            trial,
            out,
            soft_cache=soft,
            deep_cache=deep,
            max_steps=max_steps,
            max_minutes=max_minutes,
            smoke=args.smoke,
        )
        (out / "train_result.json").write_text(json.dumps(result, indent=2, default=str))
        print("train_result", {k: result.get(k) for k in ("status", "steps", "pos_s", "ckpt_path", "n_params")}, flush=True)

        elo = None
        if not args.skip_elo and result.get("status") not in ("oom", "failed"):
            ckpt = result.get("ckpt_path")
            mcfg = result.get("model_config_path")
            elo = run_elo_trial(ckpt, trial["id"], model_config=mcfg, smoke=args.smoke)

        summary = {
            "id": trial["id"],
            "desc": trial["desc"],
            "train": {k: result.get(k) for k in ("status", "steps", "pos_s", "ckpt_path", "n_params")},
            "elo": (elo or {}).get("elo"),
            "elo_estimate": (elo or {}).get("estimate"),
            "soft_cache": str(soft),
            "deep_cache": str(deep) if deep else None,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        summaries.append(summary)
        print("summary", summary, flush=True)

    (out_root / "summaries.json").write_text(json.dumps(summaries, indent=2, default=str))
    print("\n=== ALL DONE ===", flush=True)
    for s in summaries:
        print(f"  {s['id']}: elo_est={s.get('elo_estimate')} steps={s['train'].get('steps')}", flush=True)


if __name__ == "__main__":
    main()
