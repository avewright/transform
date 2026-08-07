#!/usr/bin/env python3
"""Probe model policy quality vs SF move-values on the 500-pos test set.

Cheap (no games): measures (a) top-1 agreement with SF's best move, and
(b) the cost-in-centipawns of the model's chosen move vs SF best. High cp cost
= tactical weakness = where search or tactics-training must help.

This is the quick "find the lever" probe.
"""
from __future__ import annotations
import os, sys, json, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import chess
import torch.nn.functional as F
from chess_inference import load_checkpoint
from chess_features import batch_boards_to_fused_token_ids
from move_vocab import legal_move_mask, IDX_TO_UCI, move_to_index

TEST = ROOT / "data" / "sf_test_500.jsonl"


def main():
    dev = "mps"
    model = load_checkpoint("outputs/hf_437m/best_model.pt", dev)
    model.eval()
    checkmates, rows = [], []
    for i, line in enumerate(open(TEST)):
        d = json.loads(line)
        fen, best, mv = d["fen"], d["best_uci"], d.get("move_values")
        b = chess.Board(fen)
        inp = batch_boards_to_fused_token_ids([b], dev)
        with torch.no_grad():
            r = model(inp)
        lg = r["policy_logits"][0].float()
        mask = legal_move_mask(b).to(dev); lg[~mask] = float("-inf")
        picked = IDX_TO_UCI[lg.argmax().item()]
        # cp cost of picked move vs best
        cp_by_uci = {m["uci"]: m["cp"] for m in mv} if mv else {}
        best_cp = cp_by_uci.get(best, 0)
        picked_cp = cp_by_uci.get(picked, None)
        cost = (best_cp - picked_cp) if picked_cp is not None else None
        rows.append({
            "fen": fen, "phase": d.get("phase"), "best": best, "picked": picked,
            "agree": picked == best, "best_cp": best_cp, "picked_cp": picked_cp,
            "cost": cost, "mate_avail": any(m.get("type") == "mate" for m in mv) if mv else True,
        })
        if i + 1 >= 500:
            break
    # aggregate
    n = len(rows)
    agree = sum(r["agree"] for r in rows)
    costs = [r["cost"] for r in rows if r["cost"] is not None]
    avg_cost = sum(costs) / len(costs)
    # match-to-best (within 50cp)
    match50 = sum(1 for r in rows if r["cost"] is not None and r["cost"] <= 50)
    big_err = sum(1 for r in rows if r["cost"] is not None and r["cost"] >= 150)
    by_phase = {}
    blunders = sorted(rows, key=lambda r: -(r["cost"] or 0))[:8]
    print(f"positions evaluated: {n}")
    print(f"top-1 agree with SF best: {agree}/{n} = {agree/n:.2%}")
    print(f"avg cp cost of picked vs best: {avg_cost:+.0f} cp")
    print(f"within 50cp of best: {match50}/{n} = {match50/n:.2%}")
    print(f"big errors (>=150cp off): {big_err}/{n}")
    print("worst picks (highest cost):")
    for r in blunders:
        if r["cost"] is not None and r["cost"] > 0:
            print(f"  {r['phase']}: SF={r['best']}({r['best_cp']}) model={r['picked']}({r['picked_cp']}) cost={r['cost']}")
    # write results
    json.dump(rows, open(ROOT / "outputs" / "tac_probe_437m.json", "w"), indent=1)
    print("saved outputs/tac_probe_437m.json")


if __name__ == "__main__":
    main()
