"""Save and load self-play position shards / cumulative datasets."""

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


def append_dataset(
    dataset_dir: Path,
    positions: list[dict],
    meta: dict,
) -> Path:
    """Append positions to a cumulative dataset under dataset_dir.

    Layout:
      dataset_dir/manifest.json
      dataset_dir/shards/shard_XXXXXX.pt
    """
    dataset_dir = Path(dataset_dir)
    shard_dir = dataset_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"n_positions": 0, "n_mcts": 0, "n_sf": 0, "shards": []}

    shard_idx = len(manifest["shards"])
    shard_name = f"shard_{shard_idx:06d}.pt"
    shard_path = shard_dir / shard_name

    n_mcts = sum(1 for p in positions if p.get("source", "mcts") != "sf")
    n_sf = sum(1 for p in positions if p.get("source") == "sf")
    shard_meta = {
        **meta,
        "shard_idx": shard_idx,
        "n_positions": len(positions),
        "n_mcts": n_mcts,
        "n_sf": n_sf,
    }
    save_positions(shard_path, positions, shard_meta)

    manifest["shards"].append({
        "path": f"shards/{shard_name}",
        "n_positions": len(positions),
        "n_mcts": n_mcts,
        "n_sf": n_sf,
        "iteration": meta.get("iteration"),
    })
    manifest["n_positions"] += len(positions)
    manifest["n_mcts"] += n_mcts
    manifest["n_sf"] += n_sf
    manifest["last_meta"] = {
        k: v for k, v in meta.items() if k != "config"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return shard_path
