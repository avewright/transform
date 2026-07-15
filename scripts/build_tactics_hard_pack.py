#!/usr/bin/env python3
"""Build high-rated puzzle hard pack where FT3h misses the solution."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_inference import load_checkpoint  # noqa: E402
from data_loader import board_array_to_fused, ep_square_to_file  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/exp193_puzzle_soft/soft_cache.pt")
    ap.add_argument("--checkpoint", default="outputs/exp191_soft_ft3h_edge_end/best.pt")
    ap.add_argument("--out-dir", default="outputs/policy_tactics_hard")
    ap.add_argument("--min-rating", type=int, default=2200)
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(args.src, map_location="cpu", weights_only=False)
    n0 = int(data["board_array"].shape[0])
    rating = data["puzzle_rating"]
    mask = rating >= args.min_rating
    idx0 = mask.nonzero(as_tuple=False).squeeze(1)
    print(f"src={args.src} n={n0:,} rating>={args.min_rating} → {idx0.numel():,}", flush=True)

    # Subsample before scoring if huge
    g = torch.Generator().manual_seed(args.seed)
    if idx0.numel() > args.max_n * 3:
        perm = torch.randperm(idx0.numel(), generator=g)[: args.max_n * 3]
        idx0 = idx0[perm]
        print(f"pre-subsample to {idx0.numel():,} for scoring", flush=True)

    model = load_checkpoint(args.checkpoint, device)
    model.eval()

    keep_local = []
    for start in range(0, idx0.numel(), args.batch_size):
        end = min(start + args.batch_size, idx0.numel())
        sel = idx0[start:end]
        ba = data["board_array"][sel]
        board_input = {
            "fused_ids": board_array_to_fused(ba).to(device),
            "turn": data["turn"][sel].long().to(device),
            "castling": data["castling"][sel].long().to(device),
            "ep_file": ep_square_to_file(data["ep_square"][sel]).long().to(device),
        }
        hard = data["move_idx"][sel].long()
        pred = model(board_input)["policy_logits"].float().argmax(dim=-1).cpu()
        hit = (pred != hard).nonzero(as_tuple=False).squeeze(1)
        keep_local.append(sel[hit])
        if (start // args.batch_size) % 10 == 0:
            so_far = sum(int(t.numel()) for t in keep_local)
            print(f"  scored {end:,}/{idx0.numel():,} keep={so_far:,}", flush=True)

    idx = torch.cat(keep_local) if keep_local else torch.empty(0, dtype=torch.long)
    print(f"disagreements={idx.numel():,}", flush=True)
    if idx.numel() < 500:
        raise SystemExit(f"too few rows ({idx.numel()})")

    # Dedup
    ba = data["board_array"][idx]
    turn = data["turn"][idx]
    castling = data["castling"][idx]
    ep = data["ep_square"][idx]
    seen: set[bytes] = set()
    uniq = []
    for i in range(idx.numel()):
        key = (
            ba[i].numpy().tobytes()
            + bytes([int(turn[i]) & 0xFF, int(castling[i]) & 0xFF])
            + int(ep[i]).to_bytes(2, "little", signed=True)
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(i)
    idx = idx[torch.tensor(uniq, dtype=torch.long)]
    print(f"deduped={idx.numel():,}", flush=True)

    if idx.numel() > args.max_n:
        perm = torch.randperm(idx.numel(), generator=g)[: args.max_n]
        idx = idx[perm]
        print(f"capped to {args.max_n:,}", flush=True)

    keys = [
        "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
        "soft_indices", "soft_probs", "label_depth", "phase",
    ]
    out_data = {k: data[k][idx].contiguous() for k in keys if k in data}
    k = int(idx.numel())
    sk = out_data["soft_indices"].shape[1]
    soft_i = torch.full((k, sk), -1, dtype=torch.int64)
    soft_p = torch.zeros((k, sk), dtype=torch.float32)
    soft_i[:, 0] = out_data["move_idx"].long()
    soft_p[:, 0] = 1.0
    out_data["soft_indices"] = soft_i
    out_data["soft_probs"] = soft_p
    out_data["source"] = torch.ones(k, dtype=torch.int8)
    if "puzzle_rating" in data:
        out_data["puzzle_rating"] = data["puzzle_rating"][idx].contiguous()

    out_pt = out / "soft_cache.pt"
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(out_data, tmp)
    os.replace(tmp, out_pt)

    ratings = out_data.get("puzzle_rating")
    report = {
        "src": args.src,
        "checkpoint": args.checkpoint,
        "min_rating": args.min_rating,
        "n": k,
        "rating_mean": float(ratings.float().mean()) if ratings is not None else None,
        "rating_min": int(ratings.min()) if ratings is not None else None,
        "mode": "puzzle_hard_model_miss",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_pt} n={k:,}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
