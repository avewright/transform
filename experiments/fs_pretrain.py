#!/usr/bin/env python3
"""Pre-train a small (linear-attn, meta, or vanilla) model from scratch on the
Stockfish fishtest self-play cache.

Combines:
  - policy: CE on the move Stockfish played (strong self-play games)
  - value : regression toward Stockfish cp -> win prob, with the game result as
            the primary Q target (AZ-style terminal), blending SF cp as auxiliary.

This replaces slow random self-play with real strong-engine games for the
from-scratch small model (Max Elo/FLOP with limited compute).

Usage:
  python experiments/fs_pretrain.py \
     --ckpt outputs/linattn_rl/init.pt --data outputs/fishtest_cache.pt \
     --steps 3000 --batch 64 --out outputs/linattn_pretrain
"""
from __future__ import annotations
import os, sys, argparse, time, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from chess_transformer_factory import build_model, ChessTransformerConfig, count_parameters


def load_cache(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    def ef(ep):
        r = torch.zeros_like(ep); m = ep >= 0; r[m] = (ep[m] % 8) + 1; return r
    return {
        "ba": c["board_array"], "turn": c["turn"], "castling": c["castling"],
        "ep": ef(c["ep_square"]), "mi": c["move_idx"], "cp": c["cp"],
        "q": c["result_q"],
    }


def cp_to_target(cp):
    """Map SF centipawn (from mover's perspective sign) to a win-prob target."""
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/linattn_rl/init.pt")
    ap.add_argument("--data", default="outputs/fishtest_cache.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-w", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/fs_pretrain")
    ap.add_argument("--max-rows", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("mps"); random.seed(args.seed)
    d = load_cache(args.data)
    N = d["ba"].shape[0]
    N = min(N, args.max_rows)
    print(f"train rows: {N}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config"); cfg = ChessTransformerConfig(**cfg) if not isinstance(cfg, ChessTransformerConfig) else cfg
    model = build_model(cfg)
    state = ck.get("model_state_dict", ck)
    try:
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    except Exception as e:
        print("state load warn:", str(e)[:80])
    model = model.to(dev); model.train()
    print(f"model params {count_parameters(model)/1e6:.1f}M")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_val = cfg.n_value_classes if hasattr(cfg, "n_value_classes") else 3
    t0 = time.time()
    def batch():
        idx = torch.randint(0, N, (args.batch,))
        inp = {"fused_ids": d["ba"][idx].to(dev).long(), "turn": d["turn"][idx].to(dev).long(),
               "castling": d["castling"][idx].to(dev).long(), "ep_file": d["ep"][idx].to(dev).long()}
        targets = d["mi"][idx].to(dev).long()
        # value targets: game result Q is binary; CE on 3-class WDL
        q = d["q"][idx].float()
        # convert Q in [-1,1] to 3-class win/draw/loss probs
        w_win = torch.clamp((1 + q) / 2, 0, 1)
        w_loss = torch.clamp((1 - q) / 2, 0, 1)
        w_draw = 1 - w_win - w_loss
        v_target = torch.stack([w_win, w_draw, w_loss], dim=-1).to(dev)
        return inp, targets, v_target

    for s in range(args.steps):
        inp, tgt, v_t = batch()
        opt.zero_grad()
        out = model(inp)
        p_loss = F.cross_entropy(out["policy_logits"], tgt)
        vl = out["value_logits"]
        if vl.shape[-1] == 3:
            v_loss = F.kl_div(F.log_softmax(vl, dim=-1), v_t, reduction="batchmean")
        else:
            win_pct = v_t[:, 0] + 0.5 * v_t[:, 1]
            v_loss = F.mse_loss(torch.sigmoid(vl.squeeze(-1)), win_pct)
        loss = p_loss + args.value_w * v_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"step {s+1}: loss={loss.item():.4f} p={p_loss.item():.3f} v={v_loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                "step": args.steps, "vocab_version": "compact"}, f"{args.out}/latest.pt")
    print("saved", args.out)


if __name__ == "__main__":
    main()
