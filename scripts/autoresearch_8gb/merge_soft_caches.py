"""Concatenate compatible soft_cache.pt files (same tensor keys / shapes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)


def load(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in CORE if k not in d]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="soft_cache.pt paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    chunks = [load(Path(p)) for p in args.inputs]
    n0 = [int(c["board_array"].shape[0]) for c in chunks]
    print("sizes", list(zip(args.inputs, n0)), flush=True)

    out: dict = {}
    for k in CORE:
        out[k] = torch.cat([c[k] for c in chunks], dim=0)
    # optional extras if present in all
    for k in ("label_depth", "phase", "source", "n_pieces", "wdl", "dtz"):
        if all(k in c for c in chunks):
            out[k] = torch.cat([c[k] for c in chunks], dim=0)

    if args.max_rows is not None and out["board_array"].shape[0] > args.max_rows:
        idx = torch.randperm(out["board_array"].shape[0])[: args.max_rows]
        out = {k: v[idx].contiguous() for k, v in out.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    report = {"n": int(out["board_array"].shape[0]), "sources": list(zip(args.inputs, n0))}
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path} n={report['n']:,}", flush=True)


if __name__ == "__main__":
    main()
