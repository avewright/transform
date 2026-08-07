#!/usr/bin/env python3
"""Train the 437M VALUE HEAD on Stockfish fishtest game results (via LoRA).

The 437M value head correlates only 0.33 with SF eval — too weak to beat
Stockfish via search. This trains the value head harder on real game outcomes
(+ SF evals) so that search with this value can actually win.

Freezes everything except the value head + a small LoRA on attention, keeping
steps fast on MPS. Reports value-vs-SF correlation before/after.

Usage:
  python experiments/value_train.py --steps 2000
"""
from __future__ import annotations
import os, sys, argparse, time, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from chess_transformer_factory import build_model, ChessTransformerConfig
sys.path.insert(0, str(ROOT / "experiments"))
from lora_tac import LoRALinear, apply_lora, freeze_all
sys.path.insert(0, str(ROOT / "experiments"))
from fs_pretrain import load_cache
import chess, json
from chess_features import batch_boards_to_fused_token_ids


def value_corr(model, rows, dev):
    preds, sfs = [], []
    for d in rows:
        b = chess.Board(d["fen"])
        inp = batch_boards_to_fused_token_ids([b], dev)
        with torch.no_grad():
            r = model(inp)
        w = r["value_logits"][0].float()
        if w.shape[-1] == 3:
            wp = F.softmax(w, dim=-1)[0].item()
        else:
            wp = torch.sigmoid(w.mean()).item()
        preds.append(wp)
        mv = d.get("move_values")
        if mv:
            sfs.append(mv[0]["cp"])
    if len(preds) != len(sfs):
        return None, len(preds)
    import numpy as np
    rp = np.array(preds); rs = np.array(sfs)
    dp = rp - rp.mean(); ds = rs - rs.mean()
    r = float((dp * ds).sum() / (((dp ** 2).sum()) ** 0.5 * ((ds ** 2).sum()) ** 0.5 + 1e-9))
    return r, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/hf_437m/best_model.pt")
    ap.add_argument("--data", default="outputs/fishtest_cache.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-w", type=float, default=0.5)
    ap.add_argument("--vhead-lr", type=float, default=3e-4)  # for real value-head params
    ap.add_argument("--out", default="outputs/value_train")
    ap.add_argument("--max-rows", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("mps")
    d = load_cache(args.data)
    N = min(d["ba"].shape[0], args.max_rows)
    print(f"train rows: {N}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config"); cfg = ChessTransformerConfig(**cfg) if not isinstance(cfg, ChessTransformerConfig) else cfg
    model = build_model(cfg); model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()})

    # move to dev first so everything (incl. buffers) is on MPS
    model = model.to(dev); model.eval()
    test_rows = [json.loads(l) for l in open(ROOT / "data" / "sf_test_500.jsonl")][:250]
    r0, n0 = value_corr(model, test_rows, dev)
    print(f"[before] value-SF corr: {r0:.3f} n={n0}")

    # freeze trunk; train only the value/policy heads
    trainable = []
    for name, p in model.named_parameters():
        if "value_head" in name or "policy_head" in name:
            p.requires_grad_(True); trainable.append(p)
        else:
            p.requires_grad_(False)

    model = model.to(dev); model.train()
    opt = AdamW(trainable, lr=args.vhead_lr, weight_decay=0.01)
    t0 = time.time()
    for s in range(args.steps):
        idx = torch.randint(0, N, (args.batch,))
        inp = {"fused_ids": d["ba"][idx].to(dev).long(), "turn": d["turn"][idx].to(dev).long(),
               "castling": d["castling"][idx].to(dev).long(), "ep_file": d["ep"][idx].to(dev).long()}
        q = d["q"][idx].float()
        ww = torch.clamp((1 + q) / 2, 0, 1); wl = torch.clamp((1 - q) / 2, 0, 1)
        v_t = torch.stack([ww, 1 - ww - wl, wl], dim=-1).to(dev)
        opt.zero_grad()
        out = model(inp)
        vl = out["value_logits"]
        v_loss = F.kl_div(F.log_softmax(vl, dim=-1), v_t, reduction="batchmean")
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"step {s+1}: v_loss={v_loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    rd, nd = value_corr(model, test_rows, dev)
    print(f"[after]  value-SF corr: {rd:.3f} n={nd}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                "step": args.steps, "vocab_version": "compact"}, f"{args.out}/latest.pt")
    print("saved", args.out)


if __name__ == "__main__":
    main()
