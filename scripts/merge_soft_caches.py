#!/usr/bin/env python3
"""Merge multiple soft_cache.pt files (dedupe by board+turn+castling+ep; later wins)."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import torch

KEYS = ("board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate", "soft_indices", "soft_probs")

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
    for path in args.inputs:
        data = torch.load(path, map_location="cpu", weights_only=False)
        n = int(data["board_array"].shape[0])
        print(f"load {path}: {n:,}", flush=True)
        ci = len(chunks)
        chunks.append(data)
        for i in range(n):
            last[row_key(data, i)] = (ci, i)
    final = list(last.values())
    print(f"unique={len(final):,}", flush=True)
    out = {k: torch.stack([chunks[ci][k][ii] for ci, ii in final]) for k in KEYS}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(outp) + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, outp)
    print(f"saved {out['board_array'].shape[0]:,} → {outp}", flush=True)

if __name__ == "__main__":
    main()
