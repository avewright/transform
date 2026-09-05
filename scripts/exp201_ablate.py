#!/usr/bin/env python3
"""Matched ablations for the corrected exp201 100M baseline.

Do not run these on the live training GPU. Each arm uses the same starting
weights, data manifest, example budget, and eval protocol. Existing
checkpoints have no optimizer state — every arm is a weights-only warm start
with the same LR-transition rule.

Arms (run one at a time, after the corrected baseline exists):
  1. continue_lr     muon=0.001  adam=1.5e-5  (current)
  2. continue_lr_2x  muon=0.002  adam=3.0e-5  + 200-step warmup, then cosine to floor
  3. deep_mix_0.2    vs 0.4, after the LR winner is chosen
  4. dropout         only if profiling shows a real benefit; not with the above

Selection: held-out quality + search-free match score, not train loss alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "exp201_ablations"

ARMS = {
    "continue_lr": {
        "muon_lr": 0.001,
        "adam_lr": 1.5e-5,
        "warmup": 0,
        "min_lr_frac": 1.0,
        "deep_mix_frac": 0.4,
    },
    "continue_lr_2x": {
        "muon_lr": 0.002,
        "adam_lr": 3.0e-5,
        "warmup": 200,
        "min_lr_frac": 0.5,
        "deep_mix_frac": 0.4,
    },
    "deep_mix_0.2": {
        "muon_lr": 0.001,
        "adam_lr": 1.5e-5,
        "warmup": 0,
        "min_lr_frac": 1.0,
        "deep_mix_frac": 0.2,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write or args.list or not args.arm:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "arms.json").write_text(json.dumps(ARMS, indent=2), encoding="utf-8")
        print(json.dumps(ARMS, indent=2))
        return
    raise SystemExit(
        f"arm {args.arm} = {ARMS[args.arm]}\n"
        "Not launching: refuse to start a second GPU job beside the live run."
    )


if __name__ == "__main__":
    main()
