#!/usr/bin/env python3
"""exp195: Meta-attention trunk + latent neural search (no MCTS).

Hypothesis: A refine-transformer over latent children (from/to square features)
beats one-shot spatial policy on Elo, using the same MultiPV soft data.

Architecture:
  Strengthened encoder → meta-factored layers → LatentSearchPolicyHead
  (spatial prior → top-K latent children → cross/joint attn refine → π)
  + WDL value head; backed-up child value as aux.

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp195_meta_latent_search.py --go --smoke
  MOVE_VOCAB_VERSION=compact python experiments/exp195_meta_latent_search.py --go
  MOVE_VOCAB_VERSION=compact python experiments/exp195_meta_latent_search.py --go --elo-only \\
      --checkpoint outputs/exp195_meta_latent_search/latest.pt
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

OUT = ROOT / "outputs" / "exp195_meta_latent_search"

# ~25M meta + latent search — fits M5 Pro 24GB unified.
DEFAULT_CFG = ChessTransformerConfig(
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
    value_head_type="cls",
    n_value_classes=3,
    use_swiglu=True,
    use_rel_bias=False,
    use_meta_attention=True,
    use_piece_square_dual=True,
    use_shaw_on_pos=False,
    use_qk_norm=True,
    zero_init_out_proj=True,
    use_latent_search=True,
    latent_topk=16,
    latent_search_steps=3,
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
    wdl = compute_wdl(cp, mate).to(device)
    return bi, move_idx.long().to(device), wdl, soft_i.to(device), soft_p.to(device)


def wdl_to_scalar(wdl: torch.Tensor) -> torch.Tensor:
    """White-absolute WDL one-hot/class → scalar in [-1, 1]."""
    if wdl.dtype in (torch.long, torch.int64, torch.int32):
        # class index 0=W, 1=D, 2=L
        mapped = torch.tensor([1.0, 0.0, -1.0], device=wdl.device)
        return mapped[wdl]
    # soft probs
    return wdl[:, 0] - wdl[:, 2]


def build_optimizer(model, muon_lr, adam_lr, wd):
    try:
        from normuon import SingleDeviceNorMuonWithAuxAdam
        adam_hints = (
            "embed", "policy_head", "value_head", "cls_token", "cls_pos",
            "pos_embed", "norm", "bn", "shaw_", "rel_bias", "child_", "cross_attn",
            "joint_self", "refine_mlp", "delta_score", "step_gate", "backup",
            "base_score", "from_proj", "to_proj", "global_proj", "promo",
            "board_k", "board_v",
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
    tag = "exp195_smoke" if smoke else "exp195_latent"
    result = run_elo_trial(ckpt, tag, model_config=model_cfg, smoke=smoke)
    print(f"elo rc={result.get('rc')} estimate={result.get('elo')}")
    return result.get("elo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--elo-only", action="store_true")
    ap.add_argument("--no-latent", action="store_true", help="Ablation: spatial head only")
    ap.add_argument("--soft-cache", type=str, default="outputs/autoresearch_8gb/soft_cache_200k.pt")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--max-minutes", type=float, default=120.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--output-dir", type=str, default=str(OUT))
    args = ap.parse_args()

    if not args.go and not args.elo_only:
        print("Pass --go or --elo-only")
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    cfg = DEFAULT_CFG
    if args.no_latent:
        cfg = ChessTransformerConfig(**{**asdict(cfg), "use_latent_search": False})

    cfg_path = out / "model_config.json"
    cfg_path.write_text(json.dumps({"model": asdict(cfg)}, indent=2))

    if args.elo_only:
        ckpt = Path(args.checkpoint or out / "latest.pt")
        elo = run_elo(ckpt, cfg_path, smoke=args.smoke)
        (out / "elo_result.json").write_text(json.dumps({"elo": elo, "at": utcnow()}, indent=2))
        return

    soft_path = Path(args.soft_cache)
    if not soft_path.exists():
        raise SystemExit(f"missing soft cache: {soft_path}")

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[{utcnow()}] exp195 device={device} params={n_params/1e6:.2f}M latent={cfg.use_latent_search}")
    print(f"  vocab={VOCAB_SIZE} topk={cfg.latent_topk} steps={cfg.latent_search_steps}")

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
    print(f"  optimizer={opt_name} bs={bs} accum={accum}")

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
    best_soft = float("inf")

    model.train()
    while step < max_steps and (time.time() - t0) / 60.0 < max_minutes:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = torch.randint(0, train_n, (bs,), generator=rng)
            bi, hard, wdl, si, sp = prepare_batch(data, idx, device, hflip_p=0.5, rng=rng)
            out_m = model(bi)
            soft_ce = soft_policy_loss(out_m["policy_logits"], si, sp)
            hard_ce = F.cross_entropy(out_m["policy_logits"], hard)
            p_loss = 0.35 * hard_ce + 0.65 * soft_ce
            if "base_policy_logits" in out_m:
                base_soft = soft_policy_loss(out_m["base_policy_logits"], si, sp)
                p_loss = p_loss + 0.25 * base_soft
            v_loss = F.cross_entropy(out_m["value_logits"], wdl)
            loss = p_loss + 0.2 * v_loss
            if "backed_up_value" in out_m:
                target = wdl_to_scalar(wdl)
                loss = loss + 0.15 * F.mse_loss(out_m["backed_up_value"], target)
            (loss / accum).backward()
            positions += bs

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr(step + 1)
        opt.step()
        step += 1

        if step % 25 == 0 or step == 1:
            elapsed = max(time.time() - t0, 1e-6)
            line = (
                f"[{datetime.now().strftime('%H:%M:%S')}] step {step}/{max_steps} "
                f"| {positions/elapsed:.0f} pos/s | loss={loss.item():.3f} "
                f"p={p_loss.item():.3f} v={v_loss.item():.3f}"
            )
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            if loss.item() < best_soft:
                best_soft = loss.item()
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
    # elo_eval / build_model path expects raw or state_dict; also write plain state
    torch.save(model.state_dict(), out / "latest_state.pt")
    # Prefer full ckpt with config for load_eval_model(model_config=...)
    print(f"[{utcnow()}] trained steps={step} → {out/'latest.pt'}")

    if not args.smoke:
        elo = run_elo(out / "latest.pt", cfg_path, smoke=False)
    else:
        elo = run_elo(out / "latest.pt", cfg_path, smoke=True)
    summary = {
        "at": utcnow(),
        "steps": step,
        "n_params": n_params,
        "elo": elo,
        "latent": cfg.use_latent_search,
        "soft_cache": str(soft_path),
        "train_n": train_n,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary", summary)


if __name__ == "__main__":
    main()
