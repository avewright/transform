"""Pareto front + Elo champion promotion for autoresearch."""
from __future__ import annotations

from typing import Any


def elo_of(trial: dict[str, Any]) -> float | None:
    e = trial.get("elo_estimate")
    if e is None:
        return None
    if isinstance(e, dict):
        for k in ("elo", "estimate", "point_estimate"):
            if k in e and e[k] is not None:
                return float(e[k])
        return None
    return float(e)


def pos_s_of(trial: dict[str, Any]) -> float:
    return float(trial.get("pos_s") or 0.0)


def update_pareto(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated on maximize(elo), maximize(pos_s). Failed trials skipped."""
    ok = [t for t in trials if t.get("status") == "done" and elo_of(t) is not None]
    front: list[dict[str, Any]] = []
    for t in ok:
        e, s = elo_of(t), pos_s_of(t)
        dominated = False
        for u in ok:
            if u is t:
                continue
            eu, su = elo_of(u), pos_s_of(u)
            if eu is None:
                continue
            if (eu >= e and su >= s) and (eu > e or su > s):
                dominated = True
                break
        if not dominated:
            front.append(t)
    front.sort(key=lambda t: (-(elo_of(t) or 0.0), -pos_s_of(t)))
    return front


def should_promote_champion(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    *,
    elo_noise: float = 100.0,
    speed_promote_frac: float = 0.20,
) -> bool:
    """Elo-only promotion with noise band; speed can break Elo ties."""
    ce = elo_of(candidate)
    if ce is None or candidate.get("status") != "done":
        return False
    if champion is None:
        return True
    he = elo_of(champion)
    if he is None:
        return True
    if ce >= he + elo_noise:
        return True
    if abs(ce - he) < elo_noise:
        cs, hs = pos_s_of(candidate), pos_s_of(champion)
        if hs <= 0:
            return cs > 0
        return cs >= hs * (1.0 + speed_promote_frac)
    return False
