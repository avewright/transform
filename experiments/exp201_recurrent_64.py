#!/usr/bin/env python3
"""exp201: ~100M squares-only recurrent transformer (compact vocab).

Architecture (see chess_squares64.py):
  - 64×64 attention only (no ctx tokens in the sequence)
  - Fused piece-color embeds (WQ ≠ BQ)
  - Turn / castling / EP as FiLM transforms on the square stream
  - Trunk: prefix 4 + bank 7×3 unrolls + suffix 4  (29 effective depth)
  - After backward: average_recurrent_grads() before optimizer.step()

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp201_recurrent_64.py
  MOVE_VOCAB_VERSION=compact python experiments/exp201_recurrent_64.py --smoke
  MOVE_VOCAB_VERSION=compact python experiments/exp201_recurrent_64.py --smoke --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_squares64 import (
    DEFAULT_100M_SQUARES64_CONFIG,
    average_recurrent_grads,
    build_squares64,
    count_parameters,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp201_recurrent_64"


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(
            f"Expected compact vocab 1968, got {VOCAB_SIZE}. "
            "Export MOVE_VOCAB_VERSION=compact."
        )


def smoke(device: torch.device) -> dict:
    cfg = DEFAULT_100M_SQUARES64_CONFIG
    model = build_squares64(cfg).to(device)
    n = count_parameters(model)
    print(
        f"params={n:,} ({n/1e6:.1f}M)  unique_layers={cfg.unique_layers}  "
        f"effective_depth={cfg.effective_depth}  "
        f"bank={cfg.recurrent_layers}×{cfg.recurrent_unrolls}"
    )

    B = 4
    board_input = {
        "fused_ids": torch.randint(0, 13, (B, 64), device=device),
        "turn": torch.randint(0, 2, (B,), device=device),
        "castling": torch.randint(0, 16, (B,), device=device),
        "ep_file": torch.randint(0, 9, (B,), device=device),
    }
    # Distinct WQ / BQ slots
    board_input["fused_ids"][:, 3] = 5
    board_input["fused_ids"][:, 59] = 11

    model.train()
    out = model(board_input)
    assert out["square_hidden"].shape == (B, 64, cfg.hidden_dim), out["square_hidden"].shape
    assert out["policy_logits"].shape[-1] == VOCAB_SIZE

    loss = out["policy_logits"].float().pow(2).mean() + out["value_logits"].float().pow(2).mean()
    loss.backward()

    # Recurrent grads should be non-None; averaging must be safe.
    bank_grads_before = [
        p.grad.detach().abs().mean().item()
        for p in model.recurrent_parameters() if p.grad is not None
    ]
    average_recurrent_grads(model)
    bank_grads_after = [
        p.grad.detach().abs().mean().item()
        for p in model.recurrent_parameters() if p.grad is not None
    ]
    ratio = (
        sum(bank_grads_after) / max(sum(bank_grads_before), 1e-12)
        if bank_grads_before else float("nan")
    )

    enc = model.encoder
    wq = enc.piece_color_embed.weight[5].detach()
    bq = enc.piece_color_embed.weight[11].detach()
    cos = torch.nn.functional.cosine_similarity(wq, bq, dim=0).item()

    summary = {
        "params": n,
        "params_m": round(n / 1e6, 2),
        "vocab_size": VOCAB_SIZE,
        "config": cfg.to_dict(),
        "unique_layers": cfg.unique_layers,
        "effective_depth": cfg.effective_depth,
        "policy_shape": list(out["policy_logits"].shape),
        "square_hidden_shape": list(out["square_hidden"].shape),
        "wq_bq_cosine": round(cos, 4),
        "recurrent_grad_scale_after_avg": round(ratio, 4),
        "smoke_loss": float(loss.detach().cpu()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "smoke.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    print(
        "Train step pattern:\n"
        "  loss.backward()\n"
        "  average_recurrent_grads(model)  # /3 on bank grads\n"
        "  optimizer.step()"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_100M_SQUARES64_CONFIG
    print("DEFAULT_100M_SQUARES64_CONFIG")
    print(
        f"  {cfg.hidden_dim}d / {cfg.num_heads}H | "
        f"prefix={cfg.prefix_layers} bank={cfg.recurrent_layers}×{cfg.recurrent_unrolls} "
        f"suffix={cfg.suffix_layers} | effective={cfg.effective_depth} unique={cfg.unique_layers}"
    )
    print("  attention=64×64  side-info=FiLM  embeds=fused piece×color  vocab=compact")

    if not args.smoke:
        n = count_parameters(build_squares64(cfg))
        print(f"  params≈{n:,} ({n/1e6:.1f}M) — pass --smoke for forward/backward check")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    smoke(device)


if __name__ == "__main__":
    main()
