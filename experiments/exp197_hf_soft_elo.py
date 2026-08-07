#!/usr/bin/env python3
"""exp197: Elo-gated train on HF soft mix (lichess MultiPV + Syzygy).

Hypothesis: Local 1320 floor was a shallow/short-budget artifact. Training the
two best lab arches on deep HF soft (≥d12) + Syzygy/deep mix for a matched long
budget will produce a real Elo lift.

Recipes (matched budget):
  1. wider_shallower + soft T=4 + SWA  (speed Pareto control)
  2. meta_shaw_elo                     (geometry bet)

Usage:
  python experiments/exp197_hf_soft_elo.py --go --trial wider_shallower
  python experiments/exp197_hf_soft_elo.py --go --only wider_shallower meta_shaw_elo
  python experiments/exp197_hf_soft_elo.py --go --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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

# Prefer the richer mix if present (deep+syzygy+harvest), else elo mix.
_PREF_MIX = ROOT / "outputs" / "hf_soft_mix"
_FALLBACK_MIX = ROOT / "outputs" / "hf_elo_mix"
if (_PREF_MIX / "soft_cache.pt").exists():
    MIX_DIR = _PREF_MIX
    SOFT = MIX_DIR / "soft_cache.pt"
    DEEP = MIX_DIR / "deep_soft.pt"
    if not DEEP.exists():
        DEEP = MIX_DIR / "deep_cache.pt"
else:
    MIX_DIR = _FALLBACK_MIX
    SOFT = MIX_DIR / "soft_cache.pt"
    DEEP = MIX_DIR / "deep_cache.pt"

TRIALS = {
    "wider_shallower": {
        "id": "exp197_wider_hf",
        "desc": "wider_shallower + HF soft + deep mix + softT4 + SWA",
        "inherits": "wider_shallower",
        "model_overrides": {"dropout": 0.0},
        "train_overrides": {
            "soft_frac": 1.0,
            "soft_alpha": 0.55,
            "soft_temp": 4.0,
            "soft_temp_weight": 0.4,
            "deep_mix_frac": 0.25,
            "use_swa": True,
            "swa_start_frac": 0.75,
            "batch_size": 96,
            "accum_steps": 2,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
        },
    },
    "meta_shaw_elo": {
        "id": "exp197_meta_shaw_hf",
        "desc": "meta_shaw_elo on HF soft + deep mix",
        "inherits": "meta_shaw_elo",
        "model_overrides": {},
        "train_overrides": {
            "deep_mix_frac": 0.25,
            "value_weight": 0.15,
            "min_depth": 12,
        },
    },
}


def ensure_mix(smoke: bool) -> None:
    if SOFT.exists() and (DEEP.exists() or smoke):
        print(f"mix ready soft={SOFT} deep={DEEP}")
        return
    builder = ROOT / "scripts" / "build_hf_soft_mix.py"
    if not builder.exists():
        builder = ROOT / "scripts" / "build_hf_elo_mix.py"
    cmd = [sys.executable, str(builder), "--go", "--output-dir", str(MIX_DIR)]
    if smoke:
        cmd.append("--smoke")
    print("building HF mix:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def run_one(
    key: str,
    *,
    soft: Path,
    deep_path: Path | None,
    out: Path,
    space: dict,
    max_steps: int,
    max_minutes: float,
    smoke: bool,
    skip_elo: bool,
    flat: bool,
) -> dict:
    raw = TRIALS[key]
    trial = resolve_trial_config(raw, space)
    # Flat mode (--trial): write directly into --output-dir for waiter scripts.
    # Nested mode (--only multi): out/trial_id/
    trial_out = out if flat else (out / trial["id"])
    trial_out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {trial['id']} soft={soft} deep={deep_path} steps={max_steps} out={trial_out} ===", flush=True)

    result = train_trial(
        trial,
        trial_out,
        soft_cache=soft,
        deep_cache=deep_path,
        max_steps=max_steps,
        max_minutes=max_minutes,
        smoke=smoke,
    )
    (trial_out / "train_result.json").write_text(json.dumps(result, indent=2, default=str))
    print("train_result", {k: result.get(k) for k in ("status", "steps", "pos_s", "ckpt_path", "n_params")})

    elo = None
    if not skip_elo and result.get("status") not in ("oom", "failed"):
        elo = run_elo_trial(
            result.get("ckpt_path"),
            trial["id"],
            model_config=result.get("model_config_path"),
            smoke=smoke,
        )
    summary = {
        "trial": trial["id"],
        "inherits": key,
        "train": {k: result.get(k) for k in ("status", "steps", "pos_s", "ckpt_path", "n_params")},
        "elo": None if elo is None else elo.get("elo"),
        "elo_estimate": None if elo is None else elo.get("estimate"),
        "soft_cache": str(soft),
        "deep_cache": str(deep_path) if deep_path else None,
    }
    (trial_out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # Also write at out root in flat mode (waiter looks here).
    if flat:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("summary", summary, flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trial", type=str, default=None, help="Single recipe; writes flat into --output-dir")
    ap.add_argument("--only", nargs="*", default=None, help="One or more recipes (nested dirs)")
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--train-minutes", type=float, default=240.0)
    ap.add_argument("--skip-elo", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--soft-cache", default=str(SOFT))
    ap.add_argument("--deep-cache", default=str(DEEP))
    args = ap.parse_args()
    if not args.go:
        print("Pass --go")
        return

    if not args.skip_build:
        ensure_mix(args.smoke)

    soft = Path(args.soft_cache)
    deep = Path(args.deep_cache)
    if not soft.exists():
        raise SystemExit(f"missing soft cache: {soft}")
    deep_path = deep if deep.exists() else None

    space = json.loads((ROOT / "scripts/autoresearch_8gb/search_space.json").read_text())
    max_steps = 30 if args.smoke else args.max_steps
    max_minutes = 3.0 if args.smoke else args.train_minutes

    if args.trial:
        if args.trial not in TRIALS:
            raise SystemExit(f"unknown trial {args.trial}; choose from {list(TRIALS)}")
        out = Path(args.output_dir or f"outputs/exp197_{args.trial}")
        out.mkdir(parents=True, exist_ok=True)
        run_one(
            args.trial,
            soft=soft,
            deep_path=deep_path,
            out=out,
            space=space,
            max_steps=max_steps,
            max_minutes=max_minutes,
            smoke=args.smoke,
            skip_elo=args.skip_elo,
            flat=True,
        )
        return

    keys = args.only if args.only is not None else list(TRIALS.keys())
    out_root = Path(args.output_dir or "outputs/exp197_hf_soft_elo")
    out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for key in keys:
        if key not in TRIALS:
            raise SystemExit(f"unknown trial {key}; choose from {list(TRIALS)}")
        summaries.append(
            run_one(
                key,
                soft=soft,
                deep_path=deep_path,
                out=out_root,
                space=space,
                max_steps=max_steps,
                max_minutes=max_minutes,
                smoke=args.smoke,
                skip_elo=args.skip_elo,
                flat=False,
            )
        )
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print("\nall summaries", summaries)


if __name__ == "__main__":
    main()
