#!/usr/bin/env python3
"""Train the model on the 60k Lichess puzzle cache (tactical), validate on the
SF tactical test positions. Fast base (31M) to iterate cheaply and confirm the
large-puzzle-tactical lever before spending on the 437M.

Usage:
  python experiments/tac_cache_finetune.py --steps 2000 --ckpt outputs/autoresearch_8gb/trials/wider_shallower/latest.pt
"""
from __future__ import annotations
import os, sys, argparse, json, random, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import chess, torch
import torch.nn.functional as F
from torch.optim import AdamW
from chess_inference import load_checkpoint
from move_vocab import legal_move_mask, IDX_TO_UCI, UCI_TO_IDX, VOCAB_SIZE
from chess_features import batch_boards_to_fused_token_ids

TEST = ROOT / "data" / "sf_test_500.jsonl"


def load_puzzle_cache(path, max_rows=None):
    c = torch.load(path, map_location="cpu", weights_only=False)
    ba = c["board_array"][:max_rows]
    move_idx = c["move_idx"][:max_rows]
    turn = c["turn"][:max_rows]
    castling = c["castling"][:max_rows]
    ep = c["ep_square"][:max_rows]
    return ba, move_idx, turn, castling, ep


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
        if d["best_uci"] not in cp_by:
            continue
        cost = cp_by[d["best_uci"]] - cp_by.get(picked, -10_000)
        if cp_by.get(picked) is None:
            continue
        agree += (picked == d["best_uci"])
        cost_sum += cost; ncost += 1
        within50 += (cost <= 50)
    return agree/max(1,ncost), cost_sum/max(1,ncost), within50/max(1,ncost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ckpt", default="outputs/autoresearch_8gb/trials/wider_shallower/latest.pt")
    ap.add_argument("--cache", default="outputs/exp193_tactical/soft_cache.pt")
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--out", default="outputs/tac_cache_ft")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("mps")
    random.seed(args.seed)
    ba, mi, turn, castling, ep = load_puzzle_cache(args.cache, args.max_train)
    # ep_square (-1 or 0-63) -> ep_file encoding
    def ep_to_file(ep):
        r = torch.zeros_like(ep)
        mask = ep >= 0
        r[mask] = (ep[mask] % 8) + 1
        return r
    n = ba.shape[0]
    print(f"puzzle cache rows: {n}")
    test_rows = [json.loads(l) for l in open(TEST) if l.strip()][:500]

    model = load_checkpoint(args.ckpt, dev); model.train()
    def report(tag):
        model.eval()
        a, c, w = probe(model, test_rows, dev)
        print(f"[{tag}] test  agree={a:.0%} avgcost={c:+.0f}cp within50={w:.0%}", flush=True)
        model.train()
    report("before")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    for s in range(args.steps):
        idx = torch.randint(0, n, (args.batch,))
        inp = {
            "fused_ids": ba[idx].to(dev).long(),
            "turn": turn[idx].to(dev).long(),
            "castling": castling[idx].to(dev).long(),
            "ep_file": ep_to_file(ep[idx]).to(dev).long(),
        }
        targets = mi[idx].to(dev).long()
        opt.zero_grad()
        out = model(inp)
        loss = F.cross_entropy(out["policy_logits"], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s+1) % 500 == 0:
            print(f"step {s+1}: loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    report("after")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": None}, f"{args.out}/latest.pt")
    print("saved", args.out)


if __name__ == "__main__":
    main()
