#!/usr/bin/env python3
"""Build blunder-hard pack from policy_loss soft cache.

Existing soft_cache.pt has no per-row blunder tags (harvest dropped them).
We reconstruct the corrective set by scoring FT3h on each row and keeping
positions where the model's greedy move ≠ SF best (move_idx). Hard target
remains SF best — train with soft_alpha=0.
"""
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
    ap.add_argument("--src", default="outputs/policy_loss_soft/soft_cache.pt")
    ap.add_argument("--checkpoint", default="outputs/exp191_soft_ft3h_edge_end/best.pt")
    ap.add_argument("--out-dir", default="outputs/policy_blunder_hard")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--mode",
        choices=("disagree", "not_top1"),
        default="disagree",
        help="disagree: model argmax != SF best; not_top1: same (alias)",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(args.src, map_location="cpu", weights_only=False)
    n = int(data["board_array"].shape[0])
    print(f"src={args.src} n={n:,} ckpt={args.checkpoint}", flush=True)

    model = load_checkpoint(args.checkpoint, device)
    model.eval()

    keep = torch.zeros(n, dtype=torch.bool)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        ba = data["board_array"][start:end]
        turn = data["turn"][start:end].long()
        castling = data["castling"][start:end].long()
        ep = data["ep_square"][start:end]
        hard = data["move_idx"][start:end].long()
        board_input = {
            "fused_ids": board_array_to_fused(ba).to(device),
            "turn": turn.to(device),
            "castling": castling.to(device),
            "ep_file": ep_square_to_file(ep).long().to(device),
        }
        logits = model(board_input)["policy_logits"].float()
        pred = logits.argmax(dim=-1).cpu()
        keep[start:end] = pred != hard
        if (start // args.batch_size) % 5 == 0:
            print(
                f"  scored {end:,}/{n:,} keep_so_far={int(keep[:end].sum()):,}",
                flush=True,
            )

    idx = keep.nonzero(as_tuple=False).squeeze(1)
    k = int(idx.numel())
    print(f"kept {k:,}/{n:,} ({100.0 * k / max(n, 1):.1f}%) model≠SF-best", flush=True)
    if k < 100:
        raise SystemExit(f"too few rows kept ({k}); abort")

    # Deduplicate by board fingerprint (no repeated FENs in the pack)
    ba = data["board_array"][idx]
    turn = data["turn"][idx]
    castling = data["castling"][idx]
    ep = data["ep_square"][idx]
    seen: set[bytes] = set()
    uniq_pos = []
    for i in range(k):
        key = (
            ba[i].numpy().tobytes()
            + bytes([int(turn[i]) & 0xFF, int(castling[i]) & 0xFF])
            + int(ep[i]).to_bytes(2, "little", signed=True)
        )
        if key in seen:
            continue
        seen.add(key)
        uniq_pos.append(i)
    uniq = torch.tensor(uniq_pos, dtype=torch.long)
    idx = idx[uniq]
    k = int(idx.numel())
    print(f"deduped unique boards={k:,}", flush=True)

    out_data = {key: val[idx].contiguous() for key, val in data.items()}
    # Hard CE uses move_idx; collapse soft to one-hot on best for cleanliness
    sk = out_data["soft_indices"].shape[1]
    soft_i = torch.full((k, sk), -1, dtype=torch.int64)
    soft_p = torch.zeros((k, sk), dtype=torch.float32)
    soft_i[:, 0] = out_data["move_idx"].long()
    soft_p[:, 0] = 1.0
    out_data["soft_indices"] = soft_i
    out_data["soft_probs"] = soft_p

    out_pt = out / "soft_cache.pt"
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(out_data, tmp)
    os.replace(tmp, out_pt)

    report = {
        "src": args.src,
        "checkpoint": args.checkpoint,
        "src_n": n,
        "kept_n": k,
        "keep_frac": k / max(n, 1),
        "mode": "model_disagrees_sf_best",
        "deduped": True,
        "note": "hard target=move_idx (SF best); train with soft_alpha=0; deep-max-epochs=1",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "DONE").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_pt} n={k:,}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
