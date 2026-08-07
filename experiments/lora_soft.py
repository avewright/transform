#!/usr/bin/env python3
"""LoRA fine-tune the 437M on SOFT MultiPV Stockfish targets (not hard CE).

This corrects the failed hard-1-hot experiment: the model needs ranked move
quality (soft multiPV), which hard puzzle answers destroyed. Uses
`soft_cache_200k.pt` genuine multiPV distributions, trained via LoRA on MPS.

Validation: held-out SF tactical probe (avg cp cost of picked vs best).
Usage:
  python experiments/lora_soft.py --steps 3000
"""
from __future__ import annotations
import os, sys, argparse, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from chess_transformer_factory import build_model, ChessTransformerConfig
from move_vocab import legal_move_mask, IDX_TO_UCI, VOCAB_SIZE
from chess_features import batch_boards_to_fused_token_ids
import chess

TEST = ROOT / "data" / "sf_test_500.jsonl"
sys.path.insert(0, str(ROOT / "experiments"))
from lora_tac import LoRALinear, apply_lora, freeze_all  # reuse


def load_soft(path, max_rows):
    c = torch.load(path, map_location="cpu", weights_only=False)
    def ef(ep):
        r = torch.zeros_like(ep); m = ep >= 0; r[m] = (ep[m] % 8) + 1; return r
    return {
        "ba": c["board_array"][:max_rows], "turn": c["turn"][:max_rows],
        "castling": c["castling"][:max_rows], "ep": ef(c["ep_square"][:max_rows]),
        "move_idx": c["move_idx"][:max_rows], "si": c["soft_indices"][:max_rows],
        "sp": c["soft_probs"][:max_rows],
    }


def soft_targets(move_idx, si, sp, batch_idx, n):
    # build (B, VOCAB) target distribution
    t = torch.zeros(len(batch_idx), n, dtype=torch.float32)
    for r, i in enumerate(batch_idx):
        t[r, move_idx[i]] += sp[i].sum()  # ensure top-1 included via hard? keep pure soft
        for idx, p in zip(si[i].tolist(), sp[i].tolist()):
            if p > 0:
                t[r, idx] += p
    # normalize per-row
    s = t.sum(dim=-1, keepdim=True)
    t = torch.where(s > 0, t / s.clamp_min(1e-6), t)
    return t  # (B, VOCAB) on cpu; caller moves to device


def probe(model, rows, dev):
    agree = cost_sum = ncost = within50 = 0
    for d in rows:
        b = chess.Board(d["fen"])
        inp = batch_boards_to_fused_token_ids([b], dev)
        with torch.no_grad():
            r = model(inp)
        lg = r["policy_logits"][0].float()
        mask = legal_move_mask(b).to(dev); lg[~mask] = float("-inf")
        picked = IDX_TO_UCI[lg.argmax().item()]
        cp_by = {m["uci"]: m["cp"] for m in d.get("move_values", [])}
        if d["best_uci"] not in cp_by or picked not in cp_by:
            continue
        cost = cp_by[d["best_uci"]] - cp_by[picked]
        agree += (picked == d["best_uci"]); cost_sum += cost; ncost += 1
        within50 += (cost <= 50)
    return agree/max(1,ncost), cost_sum/max(1,ncost), within50/max(1,ncost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ckpt", default="outputs/hf_437m/best_model.pt")
    ap.add_argument("--soft-cache", default="outputs/hf_soft_mix/soft_cache.pt",
                    help="Prefer deep HF mix (path-to-2500); not shallow 200k harvest")
    ap.add_argument("--max-train", type=int, default=500000)
    ap.add_argument("--out", default="outputs/lora_soft_hfmix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("mps"); random.seed(args.seed)
    d = load_soft(args.soft_cache, args.max_train)
    N = d["ba"].shape[0]
    print(f"soft cache rows: {N}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config"); cfg = ChessTransformerConfig(**cfg) if not isinstance(cfg, ChessTransformerConfig) else cfg
    model = build_model(cfg); model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()})
    freeze_all(model); apply_lora(model, rank=args.rank, alpha=32.0)
    model = model.to(dev); model.train()
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    test_rows = [json.loads(l) for l in open(TEST) if l.strip()][:500]
    def report(tag):
        model.eval(); a, c, w = probe(model, test_rows, dev)
        print(f"[{tag}] test agree={a:.0%} avgcost={c:+.0f}cp within50={w:.0%}", flush=True)
        model.train()
    report("before")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    for s in range(args.steps):
        idx = torch.randint(0, N, (args.batch,))
        inp = {"fused_ids": d["ba"][idx].to(dev).long(), "turn": d["turn"][idx].to(dev).long(),
               "castling": d["castling"][idx].to(dev).long(), "ep_file": d["ep"][idx].to(dev).long()}
        tg = soft_targets(d["move_idx"], d["si"], d["sp"], idx.tolist(), VOCAB_SIZE).to(dev)
        opt.zero_grad()
        out = model(inp)
        lp = F.log_softmax(out["policy_logits"], dim=-1)
        loss = -(tg * lp).sum(dim=-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"step {s+1}: loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    report("after")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    st = {k: v.clone() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save({"lora_state": st, "config": cfg}, f"{args.out}/lora.pt")
    print("saved", args.out)


if __name__ == "__main__":
    main()
