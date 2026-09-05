#!/usr/bin/env python3
"""Search-free strength eval for exp201.

Engine rule: game state → fixed-compute net → legal-masked policy argmax.
No MCTS, alpha-beta, child-board eval, book, tablebase, or external engine
during play. Frozen checkpoint. Paired colors. Ply-cap terminations are
reported separately from genuine chess draws.

This is a protocol runner. It does not claim FIDE / Lichess / UCI_Elo
equivalence. Internal match Elo is labeled as such.

Usage:
  python scripts/eval_exp201_searchfree.py --protocol outputs/exp201_recurrent_64/searchfree_protocol.json
  python scripts/eval_exp201_searchfree.py --write-protocol
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "exp201_recurrent_64"

SCREEN = {
    "name": "exp201_searchfree_screen",
    "mode": "policy_argmax",
    "search": False,
    "book": False,
    "syzygy": False,
    "movetime": 0.05,
    "ply_cap": 160,
    "games_per_opening_per_color": 1,
    "paired_colors": True,
    "threads": 1,
    "hash": 16,
    "rating_pool": "Stockfish UCI_Elo (NOT FIDE / Lichess / internal tournament Elo)",
    "time_control": "movetime=50ms, 1 thread, opponent UCI_LimitStrength",
    "elos": [1320, 1450, 1600],
    "openings": [
        [],
        ["e2e4", "e7e5"],
        ["d2d4", "d7d5"],
        ["e2e4", "c7c5"],
    ],
    "report_separately": ["ply_cap_termination", "repetition_draw", "stalemate", "fifty_move", "insufficient"],
    "notes": (
        "Screening suite only. Too few games for a 3000-Elo claim. "
        "Promote only after the larger match protocol with CIs."
    ),
}

PROMOTE = {
    **SCREEN,
    "name": "exp201_searchfree_promote",
    "games_per_opening_per_color": 4,
    "elos": [1320, 1450, 1600, 1750, 1900, 2050, 2200],
    "openings": SCREEN["openings"] + [
        ["d2d4", "g8f6"],
        ["e2e4", "e7e6"],
        ["c2c4", "e7e5"],
        ["g1f3", "d7d5"],
        ["e2e4", "g8f6"],
        ["d2d4", "f7f5"],
    ],
    "notes": (
        "Promotion suite. Still UCI_Elo, not FIDE. Require overlapping CIs "
        "and a freeze of opponent version/settings. Compare previous live "
        "checkpoints pairwise (same openings, swapped colors)."
    ),
}


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def write_protocol() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "searchfree_protocol.json"
    path.write_text(
        json.dumps({"screen": SCREEN, "promote": PROMOTE}, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-protocol", action="store_true")
    ap.add_argument("--protocol", default=str(OUT / "searchfree_protocol.json"))
    ap.add_argument("--run", action="store_true", help="Invoke elo_eval_latest with the screen protocol")
    ap.add_argument("--ckpt", default=str(OUT / "latest.pt"))
    args = ap.parse_args()
    path = write_protocol() if args.write_protocol or not Path(args.protocol).exists() else Path(args.protocol)
    print(f"protocol {path}", flush=True)
    if not args.run:
        print("pass --run to play games (CPU; do not do this on the training GPU)", flush=True)
        return
    from autoresearch_8gb.elo_trial import run_elo_trial
    proto = json.loads(path.read_text())["screen"]
    result = run_elo_trial(
        args.ckpt,
        "exp201_searchfree_screen",
        model_config=OUT / "model_config.json",
        movetime=float(proto["movetime"]),
        games_per_opening_per_color=int(proto["games_per_opening_per_color"]),
        elos=list(proto["elos"]),
        stop_after_bracket=True,
        smoke=False,
    )
    (OUT / "searchfree_screen_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("elo", "estimate", "rc", "json_path")}, indent=2))


if __name__ == "__main__":
    main()
