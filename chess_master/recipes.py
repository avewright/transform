"""Recipe loading and overlap-aware mix planning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chess_master.io_util import ROOT

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"


def load_recipe(name: str | Path) -> dict[str, Any]:
    path = Path(name)
    if not path.exists():
        path = RECIPES_DIR / name
    if not path.exists() and not str(name).endswith(".json"):
        path = RECIPES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(name)
    recipe = json.loads(path.read_text(encoding="utf-8"))
    recipe["_path"] = str(path)
    if recipe.get("quality_relaxation", "never") != "never":
        raise ValueError("quality_relaxation must be 'never'")
    weights = recipe.get("weights") or {}
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {sum(weights.values())}")
    return recipe


def requested_counts(recipe: dict) -> dict[str, int]:
    n = int(recipe["output_size"])
    weights = recipe["weights"]
    out = {k: int(round(n * float(v))) for k, v in weights.items()}
    drift = n - sum(out.values())
    if drift and "lichess" in out:
        out["lichess"] += drift
    return out


def overlap_assignment(eligible_sources: list[str], policy: str, priority: list[str]) -> str:
    """A puzzle-endgame with SF19 is one row, not two weighted draws."""
    if not eligible_sources:
        raise ValueError("no eligible sources")
    if policy != "exclusive_source_bucket":
        raise ValueError(f"unsupported overlap policy {policy}")
    for name in priority:
        if name in eligible_sources:
            return name
    return eligible_sources[0]
