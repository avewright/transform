#!/usr/bin/env python3
"""Merge soft_cache.pt files (dedupe by board+turn+castling+ep; later inputs win).

Keeps core tensors always. Optional tensors (phase, label_depth, …) are kept
only when present in every input that contributed a surviving row's source —
practically: intersection of keys across all inputs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)


def row_key(data, i):
    return (
        bytes(data["board_array"][i].tolist()),
        int(data["turn"][i]),
        int(data["castling"][i]),
        int(data["ep_square"][i]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    chunks = []
    last = {}
    key_sets = []
    for path in args.inputs:
        data = torch.load(path, map_location="cpu", weights_only=False)
        n = int(data["board_array"].shape[0])
        print(f"load {path}: {n:,} keys={sorted(data.keys())}", flush=True)
        for k in CORE:
            if k not in data:
                raise KeyError(f"{path} missing core key {k}")
        key_sets.append(set(data.keys()))
        ci = len(chunks)
        chunks.append(data)
        for i in range(n):
            last[row_key(data, i)] = (ci, i)

    final = list(last.values())
    print(f"unique={len(final):,}", flush=True)

    opt = set.intersection(*key_sets) - set(CORE)
    keep = list(CORE) + sorted(opt)
    if opt:
        print(f"optional kept: {sorted(opt)}", flush=True)

    out = {k: torch.stack([chunks[ci][k][ii] for ci, ii in final]) for k in keep}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(outp) + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, outp)
    print(f"saved {out['board_array'].shape[0]:,} → {outp}", flush=True)


if __name__ == "__main__":
    main()
