#!/usr/bin/env python3
"""Quick lever-probe: does a SHORT tactical finetune of the 437M reduce the
tactical blunder rate? Trains on tactical positions (SF best-move CE), probes
on a held-out split. Bounded (~few hundred steps). If held-out tactical cost
drops, "train on tactics" is the lever; if not, we need search or more data.

Usage:
  python experiments/tac_finetune_probe.py --steps 300
"""
from __future__ import annotations
import os, sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import chess, torch
import torch.nn.functional as F
from torch.optim import AdamW
from chess_inference import load_checkpoint
from chess_features import batch_boards_to_fused_token_ids
from move_vocab import legal_move_mask, IDX_TO_UCI
import argparse

TEST = ROOT / "data" / "sf_test_500.jsonl"


def load_positions(path):
    out = []
    for i, line in enumerate(open(path)):
        d = json.loads(line); out.append(d)
        if i + 1 >= 500: break
    return out


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
        if d["best_uci"] not in cp_by: continue
        bc = cp_by[d["best_uci"]]
        pc = cp_by.get(picked)
        if pc is None: continue
        cost = bc - pc
        agree += (picked == d["best_uci"])
        cost_sum += cost; ncost += 1
        within50 += (cost <= 50)
    return agree/ncost, cost_sum/ncost, within50/ncost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="outputs/hf_437m/best_model.pt")
    ap.add_argument("--out", default="outputs/tac_ft_probe")
    ap.add_argument("--data", default=str(TEST))
    ap.add_argument("--max-rows", type=int, default=500)
    args = ap.parse_args()

    dev = torch.device("mps")
    random.seed(args.seed)
    all_d = load_positions(args.data)
    all_d = all_d[: args.max_rows]
    # keep positions with SF best-move labels AND a tactic (2nd best much worse)
    tactic = [d for d in all_d if d.get("move_values") and d["best_uci"] in
              {m["uci"] for m in d["move_values"]}]
    random.shuffle(tactic)
    split = int(len(tactic)*0.7)
    train_rows, eval_rows = tactic[:split], tactic[split:]
    print(f"tactical positions: train={len(train_rows)} eval={len(eval_rows)} (total={len(tactic)})")

    model = load_checkpoint(args.ckpt, dev); model.train()
    base = {k: v.clone() for k, v in model.state_dict().items()}

    def report(tag):
        model.eval()
        a, c, w = probe(model, eval_rows, dev)
        print(f"[{tag}] eval  agree={a:.0%} avgcost={c:+.0f}cp within50={w:.0%}", flush=True)
        model.train()

    report("before")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # only few steps
    steps = args.steps
    for s in range(steps):
        batch = random.sample(train_rows, min(args.batch, len(train_rows)))
        boards = [chess.Board(d["fen"]) for d in batch]
        inp = batch_boards_to_fused_token_ids(boards, dev)
        targets = torch.tensor([move_to_idx(d["best_uci"], b) for d, b in zip(batch, boards)],
                               device=dev, dtype=torch.long)
        opt.zero_grad()
        out = model(inp)
        loss = F.cross_entropy(out["policy_logits"], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s+1) % 100 == 0:
            print(f"step {s+1}: loss={loss.item():.3f}", flush=True)
    report("after")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, f"{args.out}/tac_ft.pt")
    print("saved", args.out)


def move_to_idx(uci, board):
    from move_vocab import move_to_index
    return move_to_index(chess.Move.from_uci(uci))


if __name__ == "__main__":
    main()
