#!/usr/bin/env python3
"""Rank sf19 checkpoint comparison artifacts (H2H + optional SF ladders)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = ("init", "step1500", "swa")
H2H_FILES = {
    ("init", "step1500"): "h2h_init_vs_1500.json",
    ("init", "swa"): "h2h_init_vs_swa.json",
    ("step1500", "swa"): "h2h_1500_vs_swa.json",
}
SCREEN_ELO = {
    "init": "elo_eval_sf19_screen_init_n8000.json",
    "step1500": "elo_eval_sf19_screen_1500_n8000.json",
    "swa": "elo_eval_sf19_screen_swa_n8000.json",
}
LARGE_ELO = {
    "init": "elo_eval_sf19_large_init_n8000.json",
    "step1500": "elo_eval_sf19_large_1500_n8000.json",
    "swa": "elo_eval_sf19_large_swa_n8000.json",
}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _h2h_row(d: Path, a: str, b: str) -> dict | None:
    raw = _load(d / H2H_FILES[(a, b)])
    if raw is None:
        return None
    n = int(raw["n_games"])
    b_pts = float(raw["b_points"])
    a_pts = float(raw["a_points"])
    return {
        "a": a,
        "b": b,
        "n": n,
        "a_points": a_pts,
        "b_points": b_pts,
        "b_score": b_pts / n if n else None,
        "b_match_elo_vs_a": raw.get("b_match_elo_vs_a"),
        "terminations": raw.get("terminations"),
    }


def _elo_row(root: Path, name: str, mapping: dict[str, str]) -> dict | None:
    raw = _load(root / "outputs" / mapping[name])
    if raw is None:
        return None
    est = raw.get("estimate") or {}
    summaries = raw.get("summaries") or []
    return {
        "name": name,
        "estimated_elo": est.get("estimated_elo"),
        "lower_bound": est.get("lower_bound"),
        "upper_bound": est.get("upper_bound"),
        "note": est.get("note"),
        "protocol": {
            "book": (raw.get("protocol") or {}).get("book"),
            "syzygy": (raw.get("protocol") or {}).get("syzygy"),
            "nodes": (raw.get("protocol") or {}).get("nodes"),
            "openings": (raw.get("protocol") or {}).get("openings"),
        },
        "levels": [
            {
                "sf_elo": s.get("sf_elo"),
                "score": s.get("score"),
                "wins": s.get("wins"),
                "draws": s.get("draws"),
                "losses": s.get("losses"),
                "n": s.get("n") or s.get("games"),
            }
            for s in summaries
        ],
    }


def _tourney_points(pairs: list[dict]) -> dict[str, float]:
    pts = {n: 0.0 for n in NAMES}
    played = {n: 0 for n in NAMES}
    for p in pairs:
        pts[p["a"]] += p["a_points"]
        pts[p["b"]] += p["b_points"]
        played[p["a"]] += p["n"]
        played[p["b"]] += p["n"]
    return {
        n: {
            "points": pts[n],
            "games": played[n],
            "score": (pts[n] / played[n]) if played[n] else None,
        }
        for n in NAMES
    }


def _rank_key(name: str, tourney: dict, screen: dict, large: dict) -> tuple:
    # H2H is deterministic greedy policy and is the screen. SF ladders confirm.
    # Prefer completed large Elo, then H2H, then 1-game screen Elo.
    lg = (large.get(name) or {}).get("estimated_elo")
    sc = (screen.get(name) or {}).get("estimated_elo")
    h2h = (tourney.get(name) or {}).get("score")
    return (
        lg is not None,
        float(lg) if lg is not None else -1e9,
        h2h is not None,
        float(h2h) if h2h is not None else -1e9,
        sc is not None,
        float(sc) if sc is not None else -1e9,
    )


def build(compare_dir: Path, include_large: bool) -> dict:
    root = compare_dir.parents[2] if compare_dir.name == "compare1" else Path("/root/transform")
    if not (root / "outputs").exists():
        root = Path("/root/transform")
    pairs = []
    for a, b in H2H_FILES:
        row = _h2h_row(compare_dir, a, b)
        if row:
            pairs.append(row)
    tourney = _tourney_points(pairs)
    screen = {n: _elo_row(root, n, SCREEN_ELO) for n in NAMES}
    large = {n: _elo_row(root, n, LARGE_ELO) for n in NAMES} if include_large else {}
    order = sorted(NAMES, key=lambda n: _rank_key(n, tourney, screen, large), reverse=True)
    beats_init = None
    if "init" in order and order[0] != "init":
        winner = order[0]
        h2h = next((p for p in pairs if set((p["a"], p["b"])) == {"init", winner}), None)
        if h2h:
            if h2h["b"] == winner:
                beats_init = h2h["b_points"] > h2h["a_points"]
            else:
                beats_init = h2h["a_points"] > h2h["b_points"]
        w_src = large.get(winner) or screen.get(winner) or {}
        i_src = large.get("init") or screen.get("init") or {}
        w_elo = w_src.get("estimated_elo")
        i_elo = i_src.get("estimated_elo")
        if w_elo is not None and i_elo is not None:
            elo_beats = w_elo > i_elo
            beats_init = (True if beats_init is None else beats_init) and elo_beats
    return {
        "h2h": pairs,
        "tourney": tourney,
        "screen_sf": {k: v for k, v in screen.items() if v},
        "large_sf": {k: v for k, v in large.items() if v},
        "rank": order,
        "provisional_winner": order[0] if order else None,
        "beats_init": beats_init,
        "note": (
            "step1500 stays the provisional FT candidate until a larger "
            "no-book / no-Syzygy / fixed-node ladder beats init. "
            "Do not promote latest.pt (step 2000)."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--write", default="")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--names-only", action="store_true")
    ap.add_argument("--include-large", action="store_true")
    args = ap.parse_args()
    payload = build(Path(args.dir), include_large=args.include_large)
    if args.names_only:
        names = payload["rank"][: args.top] if args.top else payload["rank"]
        print("\n".join(names))
        return
    text = json.dumps(payload, indent=2)
    print(text)
    if args.write:
        Path(args.write).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
