#!/usr/bin/env python3
"""exp200: ~1B linear-attention chess transformer (fused piece-color embeds).

Architecture
------------
- MOVE_VOCAB_VERSION=compact (1968) — required
- FusedBoardEncoder: one embedding per {empty, WP..WK, BP..BK} — no separate
  color channel, no STM flip (absolute white/black queens stay distinct)
- Linear (kernelized) attention trunk: 20L / 2048d / 16H / SwiGLU×4 (~1.01B)
- Spatial policy + WDL value aux (training signal only)
- Grad checkpoint ON (A100+; will OOM on 24GB at full width)
- NO search: no MCTS, no lookahead, no latent search heads at train or play.
  Elo is pure argmax(legal-masked policy).

This is a scaffold. Architecture alone will not hit 2500 Elo.

What you still need for ~2500 Elo @ 0 sims (do not skip)
-------------------------------------------------------
Current floor in-repo is ~1700–1850 *policy* Elo. 2500 with no search means
the net must be the entire engine — data + capacity + distillation.

1. Soft MultiPV teacher at scale (primary lever)
   - ≥5–20M+ shallow MultiPV rows (depth 4–10, MultiPV≥8)
   - Deep mix (depth≥14–20) + Syzygy EG pack
   - Soft CE + hard ballast (soft_frac≈0.7–0.85); Elo-gate every N steps
   - Teacher stronger than target (SF depth that plays >>2500)

2. Distill MultiPV mass, not just best-move CE
   - Soft targets over legal moves beat one-hot SF; this is how a searchless
     net absorbs "search" without running it at inference

3. Value is aux only
   - WDL helps representation; it does NOT pick moves
   - use_search_* heads stay OFF; no MCTS in eval

4. Elo protocol for pure policy
   - Promote only on verified SF-limited gauntlet (32–48 games/level)
   - Bracket must reach 2200–2600 or you cannot claim 2500
   - Always legal-mask; argmax only

5. Train systems
   - NorMuon/Muon on ≥2D trunk + AdamW aux (embeds/heads/norms)
   - Long budget over soft cache; SWA last 20–30% if Elo wobbles
   - H-flip aug on soft batches

6. Hardware
   - bf16 + grad ckpt on A100-class; download ckpts before terminate

Honesty: searchless 2500 is much harder than 2500-with-MCTS. If soft data
stays at ~1–2M rows, expect a wall well below 2500.

Usage
-----
  MOVE_VOCAB_VERSION=compact python experiments/exp200_1b_linear_fused.py
  MOVE_VOCAB_VERSION=compact python experiments/exp200_1b_linear_fused.py --smoke
  MOVE_VOCAB_VERSION=compact python experiments/exp200_1b_linear_fused.py --smoke --device cpu
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

from chess_transformer_factory import (
    DEFAULT_1B_LINEAR_CONFIG,
    ChessTransformerConfig,
    build_model,
    count_parameters,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp200_1b_linear_fused"


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(
            f"Expected compact vocab 1968, got {VOCAB_SIZE}. "
            "Export MOVE_VOCAB_VERSION=compact before import/run."
        )


def smoke(config: ChessTransformerConfig, device: torch.device) -> dict:
    assert not config.use_search_policy_head and not config.use_search_value_head
    model = build_model(config).to(device)
    n = count_parameters(model)
    print(f"params={n:,} ({n/1e9:.3f}B) device={device}")
    print(
        f"encoder={config.encoder_type} linear_attn={config.use_linear_attention} "
        f"stm={config.normalize_stm} value_bins={config.n_value_classes} "
        f"search=OFF"
    )

    B, S = 2, 64
    board_input = {
        "fused_ids": torch.zeros(B, S, dtype=torch.long, device=device),
        "turn": torch.zeros(B, dtype=torch.long, device=device),
        "castling": torch.zeros(B, dtype=torch.long, device=device),
        "ep_file": torch.zeros(B, dtype=torch.long, device=device),
    }
    board_input["fused_ids"][0, 3] = 5    # white queen
    board_input["fused_ids"][0, 59] = 11  # black queen

    model.train()
    out = model(board_input)
    loss = out["policy_logits"].float().pow(2).mean() + out["value_logits"].float().pow(2).mean()
    loss.backward()

    enc = model.encoder
    assert hasattr(enc, "piece_color_embed"), "expected fused piece-color table"
    assert not hasattr(enc, "color_proj"), "no separate color encoder"
    wq = enc.piece_color_embed.weight[5].detach()
    bq = enc.piece_color_embed.weight[11].detach()
    cos = torch.nn.functional.cosine_similarity(wq, bq, dim=0).item()

    summary = {
        "params": n,
        "params_b": round(n / 1e9, 3),
        "vocab_size": VOCAB_SIZE,
        "search": False,
        "config": config.to_dict(),
        "policy_shape": list(out["policy_logits"].shape),
        "value_shape": list(out["value_logits"].shape),
        "wq_bq_cosine": round(cos, 4),
        "smoke_loss": float(loss.detach().cpu()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "smoke.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'smoke.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="Build model + one forward/backward")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_1B_LINEAR_CONFIG
    overrides = {}
    if args.layers is not None:
        overrides["num_layers"] = args.layers
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.no_grad_checkpoint:
        overrides["gradient_checkpointing"] = False
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)

    print("DEFAULT_1B_LINEAR_CONFIG (pure policy, no search)")
    print(
        f"  {cfg.num_layers}L / {cfg.hidden_dim}d / {cfg.num_heads}H  "
        f"encoder={cfg.encoder_type} linear={cfg.use_linear_attention}"
    )
    print(
        "  Elo path: soft MultiPV@scale → legal-mask argmax → Elo-gated promote "
        "(no MCTS / no lookahead)"
    )

    if not args.smoke:
        print(
            "Pass --smoke to instantiate. Full train: reuse exp191 soft/hard "
            "recipe once MultiPV caches are large enough."
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    smoke(cfg, device)


if __name__ == "__main__":
    main()
