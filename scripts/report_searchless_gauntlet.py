#!/usr/bin/env python3
"""Paired-opening report for the searchless 2050/2200 gauntlet.

Does not treat the 32-game 1750/1900 screen as an Elo result.
Overnight SWA stays incumbent unless a challenger's paired-opening
bootstrap CI for (challenger - incumbent) is entirely above 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.elo import estimate_elo  # noqa: E402

PREFIX = {
    "incumbent": "elo_eval_searchless_incumbent_n8000_r4.json",
    "control": "elo_eval_searchless_control_n8000_r4.json",
    "astra": "elo_eval_searchless_astra_n8000_r4.json",
    "refreshed": "elo_eval_searchless_refreshed_n8000_r4.json",
}
REFERENCE = ROOT / "outputs/elo_eval_overnight_swa_2050_2200_n8000.json"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize(raw: dict) -> dict:
    games = raw.get("games") or []
    by_opp: dict[int, list[float]] = defaultdict(list)
    wdl: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for g in games:
        elo = int(g["sf_elo"])
        s = float(g["score"])
        by_opp[elo].append(s)
        if s == 1.0:
            wdl[elo][0] += 1
        elif s == 0.5:
            wdl[elo][1] += 1
        else:
            wdl[elo][2] += 1
    levels = []
    for elo in sorted(by_opp):
        sc = by_opp[elo]
        w, d, l = wdl[elo]
        levels.append({
            "sf_elo": elo,
            "games": len(sc),
            "score": float(np.mean(sc)),
            "w": w,
            "d": d,
            "l": l,
        })
    est = estimate_elo(levels)
    return {
        "checkpoint": raw.get("checkpoint"),
        "n_games": len(games),
        "overall": float(np.mean([g["score"] for g in games])) if games else None,
        "levels": levels,
        "estimate": est,
        "protocol": raw.get("protocol") or raw.get("config"),
    }


def _index(games: list[dict]) -> dict[tuple, float]:
    out = {}
    for g in games:
        key = (int(g["sf_elo"]), g.get("opening_name") or g.get("opening"), g["model_color"], int(g.get("repeat_idx", 0)))
        out[key] = float(g["score"])
    return out


def paired_opening_bootstrap(
    games_a: list[dict],
    games_b: list[dict],
    *,
    n_boot: int = 5000,
    seed: int = 20260908,
) -> dict | None:
    """Resample openings. Keep both colors, repeats, and opponents inside each opening."""
    ia, ib = _index(games_a), _index(games_b)
    common = sorted(set(ia) & set(ib))
    if not common:
        return None
    by_open: dict[str, list[tuple]] = defaultdict(list)
    for key in common:
        by_open[str(key[1])].append(key)
    openings = sorted(by_open)
    obs_diffs = [ia[k] - ib[k] for k in common]
    observed = float(np.mean(obs_diffs))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(openings, size=len(openings), replace=True)
        diffs = []
        for name in sample:
            for key in by_open[name]:
                diffs.append(ia[key] - ib[key])
        boots[i] = float(np.mean(diffs))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n_paired": len(common),
        "n_openings": len(openings),
        "observed_diff": observed,
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "a_better": bool(lo > 0.0),
        "b_better": bool(hi < 0.0),
    }


def decide(rows: dict[str, dict], diffs: dict[str, dict]) -> dict:
    incumbent = "incumbent"
    winner = incumbent
    reason = "incumbent retained; no challenger CI entirely above 0"
    for name in ("control", "astra", "refreshed"):
        d = diffs.get(f"{name}_minus_{incumbent}")
        if d and d.get("a_better"):
            winner = name
            reason = f"{name} paired-opening CI entirely above 0 vs incumbent"
            break
    return {"winner": winner, "reason": reason, "incumbent": incumbent}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/searchless_gauntlet"))
    ap.add_argument("--write", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root_out = ROOT / "outputs"

    raws = {}
    rows = {}
    for name, fn in PREFIX.items():
        raw = _load(root_out / fn)
        raws[name] = raw
        rows[name] = _summarize(raw) if raw else {"missing": True, "path": str(root_out / fn)}

    diffs = {}
    names = [n for n, r in rows.items() if not r.get("missing")]
    pairs = []
    for a in names:
        for b in names:
            if a != b and (a, b) in {
                ("control", "incumbent"),
                ("astra", "incumbent"),
                ("astra", "control"),
                ("refreshed", "incumbent"),
                ("refreshed", "astra"),
                ("refreshed", "control"),
            }:
                pairs.append((a, b))
    for a, b in pairs:
        if raws.get(a) and raws.get(b):
            diffs[f"{a}_minus_{b}"] = paired_opening_bootstrap(raws[a]["games"], raws[b]["games"])

    required = ("incumbent", "control", "astra")
    decision = decide(rows, diffs) if all(not rows[n].get("missing") for n in required) else {
        "winner": "incumbent",
        "reason": "gauntlet incomplete; keep incumbent",
        "incumbent": "incumbent",
    }

    ref = _load(REFERENCE)
    report = {
        "note": "32-game 1750/1900 screen is not an Elo result. Reference ~2150 is overnight SWA vs 2050/2200.",
        "reference_overnight_swa_2050_2200": _summarize(ref) if ref else None,
        "arms": rows,
        "paired_opening_diffs": diffs,
        "decision": decision,
    }
    text = json.dumps(report, indent=2) + "\n"
    dest = Path(args.write) if args.write else out / "report.json"
    dest.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
