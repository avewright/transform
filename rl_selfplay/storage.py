"""Save and load self-play position shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_positions(
    path: Path,
    positions: list[dict],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"positions": positions, "meta": meta}, tmp)
    tmp.replace(path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"n_positions": len(positions), **meta}, f, indent=2)


def load_positions(path: Path) -> tuple[list[dict], dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["positions"], data.get("meta", {})
