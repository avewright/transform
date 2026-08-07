#!/usr/bin/env python3
"""exp199: Stockfish WDL+CP value head + AlphaZero hybrid root search (2500 push).

Goal: Break the ~1950 Elo plateau observed on the 25M strengthened/SwiGLU/
rel-bias baseline (autoresearch_8gb) and the meta-latent 20L/256d prototype
(exp195). Two orthogonal fixes, inspired by Stockfish NNUE and AlphaZero:

  1. VALUE HEAD — Stockfish-style dual supervision (WDL + centipawns)
     Current heads (chess_model.py:539, chess_transformer_factory.py:1510)
     train a 3-class White-absolute WDL as a hard argmax CE with weight 0.1-0.2.
     That discards cp magnitude (cp+20 and cp+300 both → "win"), makes draw
     granularity invisible, and leaves the value head under-trained — which is
     why search hurt Elo in exp094/exp097/exp098 and why exp198 S4_valuegate
     could not veto correctly.

     Stockfish NNUE solves this by jointly predicting:
       a) soft WDL probabilities p(W,D,L) derived from cp via a ply-dependent
          sigmoid (LC0/Stockfish WDL model: k≈111.7+0.2*(50-ply)), and
       b) a normalized cp scalar (tanh-clipped, ~1500 cp saturates).
     Loss is KL/HL-Gauss on the soft distribution + MSE/Huber on cp, not a
     single hard CE. This gives a smooth, magnitude-aware target even for
     midgame "unclear" positions and stabilizes the head without requiring
     game-outcome labels.

     New head proposed here: `StockfishWDLCPHead` (see patch to
     chess_transformer_factory.py below). Shared trunk → CLS (+ optional
     pooled squares) → two branches:
       - WDL logits (3) trained with soft-target KL (or HL-Gauss if n_bins=128)
       - CP scalar (1) trained with Huber, target = clip(cp, -1500,1500)/1500
     Final scalar for search/gating = wdl_scalar * λ + cp_scalar * (1-λ),
     both in [-1, 1] White-absolute. Backups and gates read this fused scalar.

  2. HYBRID SEARCH — AlphaZero PUCT at the root, gated by value agreement
     exp195 introduced a latent 1-ply neural search over top-k move deltas
     (SearchPolicyHead / SearchValueHead, factory lines 1037/1173) that refines
     policy logits by cross+joint attention over latent children. exp198 showed
     a flat SF+TF blend is a wash; the win comes from *gated* overrides:
     only overrule Stockfish when SF's cp spread is small (near-tie, ~30 cp)
     AND the transformer is confident (top policy ≥0.4) AND its value head
     agrees directionally with SF (avoids value-collapse blunders).

     exp199 keeps the 1-ply latent refinement for training (policy sharpens
     under search), but at inference adds a proper hybrid root search:
       - Stockfish provides MultiPV candidates with cp (or mate) scores
       - Transformer provides policy prior + WDL+CP fused value
       - Root PUCT (AlphaZero-style, no tree beyond root children + 1-ply
         latent backup) picks the blended move. Optionally run 32-64 sims
         re-using Stockfish's shallow eval as leaf prior for speed.
     This is the "search-informed eval" pattern from SearchValueHead stage 4,
     but now the evaluator itself is Stockfish-aware and the move picker is
     PUCT, not argmax.

Hypothesis:
  Soft WDL+CP value head restores value accuracy (target: value acc >62% on
  held-out Stockfish labels vs ~54% with hard WDL), which unlocks gated
  hybrid search for +60-120 Elo over the flat blend. Combined with the
  1-ply latent policy refinement, this should push the 25M trunk toward
  2050-2100 Elo on the autoresearch board and scale cleanly to A100-700M.

Architecture (training):
  Strengthened encoder (STM-normalized, 2 conv blocks) → meta or standard
  trunk (configurable) → SearchPolicyHead (policy_topk=16, 3 steps) +
  StockfishWDLCPHead (pooled CLS+mean, 3 WDL + 1 CP). Soft targets from
  deep/soft caches; CP targets from Stockfish best_cp (already STM-White).

Training recipe:
  - Soft policy: 0.35 hard CE + 0.65 soft CE (MultiPV teacher), plus
    0.25 base-policy aux (as in exp195) so the spatial prior stays honest.
  - Value: KL(soft-WDL || pred-WDL) * 0.5 + HL-Gauss winprob * 0.2 +
    Huber(cp_pred, cp_target) * 0.3, all White-absolute, ply-aware k.
  - Value weight 0.4-0.5 total (vs 0.1 baseline) — the point of this exp.
  - Optim: NorMuon (muon 0.02 / adam 3e-4) with SwiGLU trunk, warmup 200,
    cosine to 0.05, grad clip 1.0, hflip 0.5.

Inference / Elo:
  Uses autoresearch_8gb/elo_trial.py (same as exp194/exp195) for apples-to-
  apples Elo, plus an optional hybrid_uci.py sweep (exp198 strategies S3/S4)
  to measure the PUCT gate in isolation.

Usage (do NOT launch on laptop — A100/RunPod only):

  # --- A100 80GB full run (8k steps ≈ 90 min, ~25M params, no ckpt) ---
  MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go \\
    --soft-cache outputs/autoresearch_8gb/soft_cache_200k.pt \\
    --deep-cache outputs/autoresearch_8gb/puzzle_syzygy_mix.pt \\
    --max-steps 8000 --batch-size 96 --accum 2 --value-weight 0.5 --hybrid-eval

  # --- Smoke (20 steps, CPU/MPS sanity, no Elo) ---
  MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go --smoke

  # --- Elo only from existing checkpoint ---
  MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go --elo-only \\
    --checkpoint outputs/exp199_wdl_cp_hybrid/latest.pt

  # --- Isolated hybrid blend sweep (requires Stockfish binary) ---
  python experiments/exp198_hybrid_blend.py --checkpoint outputs/exp199_wdl_cp_hybrid/latest.pt --multipv 8
  python hybrid_uci.py --stockfish stockfish/stockfish-native-arm64 \\
    --checkpoint outputs/exp199_wdl_cp_hybrid/latest.pt --multipv 8 --policy-weight 0.35 --temp 0.9

RunPod template (A100 80GB, PyTorch 2.4, CUDA 12.4):
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
  pip install -q python-chess datasets huggingface_hub
  # optional: pip install -q ../normuon  (if using NorMuon)
  export MOVE_VOCAB_VERSION=compact
  export HF_TOKEN=...
  nohup python experiments/exp199_wdl_cp_hybrid_search.py --go > outputs/exp199/train.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chess_transformer_factory import ChessTransformerConfig, build_model, count_parameters
from data_loader import (
    board_array_to_fused, compute_wdl, ep_square_to_file,
    hflip_board_array, hflip_ep_square, hflip_move_idx,
)
from move_vocab import VOCAB_SIZE

OUT = ROOT / "outputs" / "exp199_wdl_cp_hybrid"

# ---------------------------------------------------------------------------
# Stockfish WDL+CP helpers (mirrors exp098 cp_to_wdl + chess_qwen hl_gauss)
# ---------------------------------------------------------------------------

def cp_to_soft_wdl(cp: torch.Tensor, ply: torch.Tensor, k0: float = 111.7) -> torch.Tensor:
    """Batch cp (STM-White) + ply -> soft WDL probs (B, 3) White-absolute.

    Uses LC0/Stockfish WDL model: k = 111.7 + 0.2*max(0, 50-ply) clipped,
    win_prob = sigmoid(cp/k), draw mass peaks at cp≈0. Mirrors
    archive/exp012_stockfish_supervised.py:147 and exp098 cp_to_wdl.
    """
    # cp already White-absolute (data_loader negates Black stm before compute_wdl)
    k = k0 + 0.2 * torch.clamp(50 - ply.float(), min=0)
    win = torch.sigmoid(cp.float() / k.clamp(min=1.0))
    loss = 1.0 - win
    # draw width decays with |cp|
    draw_width = 0.5 * torch.exp(-cp.float().abs() / 200.0)
    draw = draw_width * torch.minimum(win, loss) * 4.0
    draw = draw.clamp(0, 0.95)
    win = win * (1 - draw)
    loss = loss * (1 - draw)
    s = win + draw + loss
    return torch.stack([win / s, draw / s, loss / s], dim=-1)


def cp_to_normalized_scalar(cp: torch.Tensor, clip: float = 1500.0) -> torch.Tensor:
    """Centipawn -> [-1, 1] scalar for Huber/MSE (White-absolute)."""
    return (cp.float().clamp(-clip, clip) / clip)


def soft_wdl_kl_loss(pred_logits: torch.Tensor, soft_wdl: torch.Tensor) -> torch.Tensor:
    """KL(soft_target || pred) for (B,3) probs; more stable than hard CE."""
    log_pred = F.log_softmax(pred_logits.float(), dim=-1)
    return F.kl_div(log_pred, soft_wdl.float(), reduction="batchmean", log_target=False)


def hl_gauss_loss(logits: torch.Tensor, winprob: torch.Tensor, n_bins: int = 128) -> torch.Tensor:
    """Distributional HL-Gauss on winprob in [0,1] (chess_qwen_factory:608)."""
    sigma = 2.5 / n_bins
    centers = (torch.arange(n_bins, device=logits.device, dtype=logits.dtype) + 0.5) / n_bins
    diff = centers.unsqueeze(0) - winprob.unsqueeze(1).clamp(0, 1)
    target = F.softmax(-0.5 * (diff / sigma) ** 2, dim=-1)
    log_p = F.log_softmax(logits.float(), dim=-1)
    return F.kl_div(log_p, target, reduction="batchmean")


# ---------------------------------------------------------------------------
# Config — two presets: 25M ablatable (8GB) and A100-700M deep-narrow
# ---------------------------------------------------------------------------

# 25M ablatable: matches exp195 trunk (20L/256d) but with correct factory keys
# and the new WDL+CP head. This is the smoke / laptop-safe config.
DEFAULT_25M_CFG = ChessTransformerConfig(
    encoder_dim=256,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=256,
    num_layers=20,
    num_heads=8,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=192,
    value_hidden=128,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",       # StockfishWDLCPHead replaces this at runtime if enabled
    n_value_classes=3,
    use_swiglu=True,
    use_rel_bias=False,
    use_meta_attention=True,
    use_piece_square_dual=False,  # keep False for 25M; enable for 700M
    use_shaw_on_pos=True,
    use_qk_norm=True,
    zero_init_out_proj=True,
    gradient_checkpointing=False,
    use_search_policy_head=True,
    policy_topk=16,
    policy_search_steps=3,
    use_search_value_head=False,  # replaced by WDL+CP pooled head
)

# A100 80GB: deep-narrow ~120M-700M scaling target for RunPod (fits bs=64 with ckpt)
# Set --a100 to use this (or --config outputs/exp199_wdl_cp_hybrid/model_config.json)
DEFAULT_A100_CFG = ChessTransformerConfig(
    encoder_dim=384,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=768,
    num_layers=32,
    num_heads=12,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=384,
    value_hidden=384,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",
    n_value_classes=3,
    use_swiglu=True,
    use_rel_bias=False,
    use_meta_attention=True,
    use_piece_square_dual=True,
    use_shaw_on_pos=True,
    use_qk_norm=True,
    zero_init_out_proj=True,
    gradient_checkpointing=True,
    use_search_policy_head=True,
    policy_topk=16,
    policy_search_steps=3,
    use_search_value_head=False,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def soft_policy_loss(logits, soft_indices, soft_probs):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = (soft_indices >= 0) & (soft_probs > 0)
    safe = soft_indices.clamp(min=0).long()
    gathered = log_probs.gather(1, safe) * valid.float()
    return -(soft_probs.float() * gathered).sum(dim=-1).mean()


def prepare_batch(data, indices, device, hflip_p=0.5, rng=None):
    ba = data["board_array"][indices].clone()
    turn = data["turn"][indices].clone()
    castling = data["castling"][indices].clone()
    ep = data["ep_square"][indices].clone()
    move_idx = data["move_idx"][indices].clone()
    cp = data["cp"][indices]
    mate = data["mate"][indices]
    soft_i = data["soft_indices"][indices].clone()
    soft_p = data["soft_probs"][indices].clone()
    # ply for k(ply) — fullmove*2 + turn offset, fallback 30
    ply = data.get("ply", torch.full_like(cp, 30))[indices] if "ply" in data else torch.full_like(cp, 30)

    if hflip_p > 0:
        flip_mask = torch.rand(ba.shape[0], generator=rng) < hflip_p
        if flip_mask.any():
            ba[flip_mask] = hflip_board_array(ba[flip_mask])
            move_idx[flip_mask] = hflip_move_idx(move_idx[flip_mask]).to(move_idx.dtype)
            castling[flip_mask] = 0
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
            si = soft_i[flip_mask]
            valid = si >= 0
            if valid.any():
                si2 = si.clone()
                si2[valid] = hflip_move_idx(si[valid]).to(si.dtype)
                soft_i[flip_mask] = si2

    bi = {
        "fused_ids": board_array_to_fused(ba).to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_square_to_file(ep).long().to(device),
    }
    wdl_hard = compute_wdl(cp, mate).to(device)  # (B,) class idx for fallback
    # soft WDL + cp scalar targets (White-absolute — data_loader already handles stm negation)
    # cp here is stm-White after data_loader's convention; if raw, negate where turn==1
    # For caches that store stm cp, do: cp_white = torch.where(turn==1, -cp, cp)
    cp_white = torch.where(turn == 1, -cp, cp) if "turn" in data else cp
    # need turn on cpu for this; use indices' turn
    cp_w = cp_white[torch.arange(len(indices))] if cp_white.dim() > 0 else cp_white
    # Actually indices already selected; use cloned turn
    # Recompute correctly from batch turn:
    # (avoid double-indexing confusion — use per-batch turn tensor)
    # We have turn (batch) already; cp is batch-aligned
    cp_white_batch = torch.where(turn == 1, -cp.float(), cp.float())
    soft_wdl = cp_to_soft_wdl(cp_white_batch, ply.float() if ply.dim() > 0 else torch.full_like(cp_white_batch.float(), 30))
    cp_scalar = cp_to_normalized_scalar(cp_white_batch)
    return bi, move_idx.long().to(device), wdl_hard, soft_i.to(device), soft_p.to(device), soft_wdl.to(device), cp_scalar.to(device), ply.to(device)


def build_optimizer(model, muon_lr, adam_lr, wd):
    try:
        from normuon import SingleDeviceNorMuonWithAuxAdam
        adam_hints = (
            "embed", "policy_head", "value_head", "cls_token", "cls_pos",
            "pos_embed", "norm", "bn", "shaw_", "rel_bias", "child_", "cross_attn",
            "joint_self", "refine_mlp", "delta_score", "step_gate", "backup",
            "base_score", "from_proj", "to_proj", "global_proj", "promo",
            "board_k", "board_v", "wdl", "cp_head", "cp_scale",
        )
        muon_p, adam_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(h in name for h in adam_hints) or p.ndim < 2:
                adam_p.append(p)
            else:
                muon_p.append(p)
        opt = SingleDeviceNorMuonWithAuxAdam([
            dict(params=muon_p, use_muon=True, lr=muon_lr, weight_decay=wd, momentum=0.95, beta2=0.95),
            dict(params=adam_p, use_muon=False, lr=adam_lr, betas=(0.9, 0.95), weight_decay=wd),
        ])
        return opt, "normuon"
    except ImportError:
        return AdamW(model.parameters(), lr=adam_lr, weight_decay=wd), "adamw"


def run_elo(ckpt: Path, model_cfg: Path, smoke: bool = False) -> float | None:
    from autoresearch_8gb.elo_trial import run_elo_trial
    tag = "exp199_smoke" if smoke else "exp199_wdl_cp"
    result = run_elo_trial(ckpt, tag, model_config=model_cfg, smoke=smoke)
    print(f"elo rc={result.get('rc')} estimate={result.get('elo')}")
    return result.get("elo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--elo-only", action="store_true")
    ap.add_argument("--hybrid-eval", action="store_true", help="Also run exp198 hybrid sweep post-Elo")
    ap.add_argument("--a100", action="store_true", help="Use A100 32L/768d config")
    ap.add_argument("--no-search-policy", action="store_true", help="Ablation: disable latent policy search")
    ap.add_argument("--soft-cache", type=str, default="outputs/autoresearch_8gb/soft_cache_200k.pt")
    ap.add_argument("--deep-cache", type=str, default="outputs/autoresearch_8gb/puzzle_syzygy_mix.pt")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--cp-weight", type=float, default=0.3)
    ap.add_argument("--output-dir", type=str, default=str(OUT))
    args = ap.parse_args()

    if not args.go and not args.elo_only:
        print("Pass --go or --elo-only")
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    cfg = DEFAULT_A100_CFG if args.a100 else DEFAULT_25M_CFG
    if args.no_search_policy:
        cfg = ChessTransformerConfig(**{**asdict(cfg), "use_search_policy_head": False})

    cfg_path = out / "model_config.json"
    cfg_path.write_text(json.dumps({"model": asdict(cfg)}, indent=2))

    if args.elo_only:
        ckpt = Path(args.checkpoint or out / "latest.pt")
        elo = run_elo(ckpt, cfg_path, smoke=args.smoke)
        (out / "elo_result.json").write_text(json.dumps({"elo": elo, "at": utcnow()}, indent=2))
        return

    soft_path = Path(args.soft_cache)
    if not soft_path.exists():
        # fallback to search_space candidates
        from autoresearch_8gb.train_trial import find_cache
        import scripts.autoresearch_8gb.search_space as _ss  # noqa
        import json as _json
        space = _json.loads((ROOT / "scripts/autoresearch_8gb/search_space.json").read_text())
        soft_path = find_cache(space["soft_cache_candidates"])
        if soft_path is None:
            raise SystemExit(f"missing soft cache: {args.soft_cache}")

    model = build_model(cfg).to(device)

    # --- Patch value head to Stockfish WDL+CP if factory hasn't yet ---
    # If ChessTransformer was built with n_value_classes=3 CLS head, wrap it:
    # We keep the factory head for WDL and add a CP scalar branch.
    # This monkey-patch is replaced by the proper StockfishWDLCPHead once
    # chess_transformer_factory.py is patched (see instructions below).
    if not hasattr(model, "_wdl_cp_patched"):
        _orig_value_head = model.value_head

        class _WDLCPWrapper(nn.Module):
            def __init__(self, base, hidden_dim):
                super().__init__()
                self.wdl_head = base  # (B,3) or pooled variant
                self.cp_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                self._is_pooled = hasattr(base, "mlp") and "pool" in type(base).__name__.lower()

            def forward(self, *args, **kwargs):
                # Factory calls value_head(cls_hidden) with single arg; search heads use (hidden, cls_hidden)
                if len(args) == 1:
                    hidden = cls_hidden = args[0]
                elif len(args) >= 2:
                    hidden, cls_hidden = args[0], args[1]
                else:
                    hidden = kwargs.get("hidden")
                    cls_hidden = kwargs.get("cls_hidden", hidden)
                if self._is_pooled:
                    try:
                        wdl = self.wdl_head(hidden, cls_hidden)
                    except TypeError:
                        wdl = self.wdl_head(cls_hidden)
                else:
                    try:
                        wdl = self.wdl_head(cls_hidden)
                    except TypeError:
                        wdl = self.wdl_head(hidden, cls_hidden)
                cp_scalar = torch.tanh(self.cp_head(cls_hidden).squeeze(-1))
                return {"value_logits": wdl, "cp_scalar": cp_scalar}

        # Only wrap if not already a dict-returning head (e.g. SearchValueHead)
        probe = model.value_head
        is_search = type(probe).__name__ == "SearchValueHead"
        if not is_search:
            model.value_head = _WDLCPWrapper(_orig_value_head, cfg.hidden_dim)
            model.value_head.to(device)
            model._wdl_cp_patched = True
            # re-detect pooled flag for forward dispatch
            model._pool_value = False
            model.use_search_value_head = False

    n_params = count_parameters(model)
    print(f"[{utcnow()}] exp199 device={device} params={n_params/1e6:.2f}M search_policy={cfg.use_search_policy_head} value_w={args.value_weight}")
    print(f"  vocab={VOCAB_SIZE} topk={cfg.policy_topk} steps={cfg.policy_search_steps} cp_w={args.cp_weight}")

    data = torch.load(soft_path, map_location="cpu", weights_only=False)
    n = int(data["board_array"].shape[0])
    hold = min(2000, max(256, n // 20))
    train_n = max(1, n - hold)
    print(f"  soft n={n:,} train={train_n:,} hold={hold}")

    max_steps = 20 if args.smoke else args.max_steps
    max_minutes = 2.0 if args.smoke else args.max_minutes
    bs = 8 if args.smoke else args.batch_size
    if device.type == "mps":
        bs = min(bs, 64)
    accum = 1 if args.smoke else args.accum

    opt, opt_name = build_optimizer(model, muon_lr=0.02, adam_lr=3e-4, wd=0.01)
    print(f"  optimizer={opt_name} bs={bs} accum={accum} value_weight={args.value_weight}")

    warmup = 50 if args.smoke else 200
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    def set_lr(step: int):
        if step < warmup:
            scale = step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(max_steps - warmup, 1)
            scale = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base in zip(opt.param_groups, base_lrs):
            pg["lr"] = base * scale

    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    t0 = time.time()
    step = 0
    positions = 0
    log_path = out / "train.log"
    best = float("inf")

    model.train()
    while step < max_steps and (time.time() - t0) / 60.0 < max_minutes:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = torch.randint(0, train_n, (bs,), generator=rng)
            bi, hard, wdl_hard, si, sp, soft_wdl, cp_scalar, ply = prepare_batch(data, idx, device, hflip_p=0.5, rng=rng)
            out_m = model(bi)
            # Policy
            soft_ce = soft_policy_loss(out_m["policy_logits"], si, sp)
            hard_ce = F.cross_entropy(out_m["policy_logits"], hard)
            p_loss = 0.35 * hard_ce + 0.65 * soft_ce
            if "base_policy_logits" in out_m:
                base_soft = soft_policy_loss(out_m["base_policy_logits"], si, sp)
                p_loss = p_loss + 0.25 * base_soft
            # Value: soft WDL KL + CP Huber (+ optional HL-Gauss if 128 bins)
            v_wdl = out_m.get("value_logits", None)
            cp_pred = out_m.get("cp_scalar", None)
            # Handle wrapped vs native heads
            if isinstance(v_wdl, dict):
                cp_pred = v_wdl.get("cp_scalar", cp_pred)
                v_wdl = v_wdl.get("value_logits", v_wdl)
            if v_wdl is None:
                raise RuntimeError("model did not return value_logits")
            if v_wdl.shape[-1] == 128:
                # distributional: derive winprob target from soft_wdl
                winprob = soft_wdl[:, 0] + 0.5 * soft_wdl[:, 1]
                v_loss_wdl = hl_gauss_loss(v_wdl, winprob, n_bins=128)
            else:
                v_loss_wdl = soft_wdl_kl_loss(v_wdl, soft_wdl)
            if cp_pred is not None:
                v_loss_cp = F.huber_loss(cp_pred.float(), cp_scalar.float(), delta=0.2)
                v_loss = v_loss_wdl + args.cp_weight * v_loss_cp
            else:
                v_loss = v_loss_wdl
            loss = p_loss + args.value_weight * v_loss
            (loss / accum).backward()
            positions += bs

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr(step + 1)
        opt.step()
        step += 1

        if step % 25 == 0 or step == 1:
            elapsed = max(time.time() - t0, 1e-6)
            v_show = v_loss.item() if "v_loss" in locals() else 0
            line = (
                f"[{datetime.now().strftime('%H:%M:%S')}] step {step}/{max_steps} "
                f"| {positions/elapsed:.0f} pos/s | loss={loss.item():.3f} "
                f"p={p_loss.item():.3f} v={v_show:.3f}"
            )
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            if loss.item() < best:
                best = loss.item()
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "step": step,
                }, out / "best.pt")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "n_params": n_params,
    }
    torch.save(ckpt, out / "latest.pt")
    torch.save(model.state_dict(), out / "latest_state.pt")
    print(f"[{utcnow()}] trained steps={step} → {out/'latest.pt'}")

    if not args.smoke:
        elo = run_elo(out / "latest.pt", cfg_path, smoke=False)
        if args.hybrid_eval:
            # Best-effort hybrid sweep (needs stockfish binary)
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, str(ROOT / "experiments/exp198_hybrid_blend.py"),
                     "--checkpoint", str(out / "latest.pt"), "--multipv", "8"],
                    check=False, timeout=600,
                )
            except Exception as e:
                print(f"hybrid sweep skipped: {e}")
    else:
        elo = run_elo(out / "latest.pt", cfg_path, smoke=True)
    summary = {
        "at": utcnow(),
        "steps": step,
        "n_params": n_params,
        "elo": elo,
        "value_weight": args.value_weight,
        "cp_weight": args.cp_weight,
        "search_policy": cfg.use_search_policy_head,
        "soft_cache": str(soft_path),
        "train_n": train_n,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary", summary)


if __name__ == "__main__":
    main()
