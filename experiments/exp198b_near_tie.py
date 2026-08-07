#!/usr/bin/env python3
"""exp198b: Does the transformer break near-ties better than SF dice-rolls?

Hypothesis: On positions where Stockfish's shallow eval shows a genuine near-tie
(top-2 |Δcp| tiny), the transformer's *structural* attention read may pick the
better move than SF's noisy cp noise -- the only regime a post-hoc override could
earn Elo. We evaluate on a suite of middlegame/endgame positions with richer
structure (vs the opening-heavy set that already agreed).

Score = deep-SF cp of each engine's pick. If transformer picks >= SF shallow top
on near-ties, the hybrid has a place to live.
"""
from __future__ import annotations

import os, sys, math, chess, chess.engine
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION","compact")
import torch, torch.nn.functional as F
from hybrid_uci import HybridEngine
from chess_features import batch_boards_to_fused_token_ids
from move_vocab import IDX_TO_UCI, legal_move_mask

# Middlegame/tactical-ish positions where NNUE value can be less certain.
POSITIONS = {
    "mg_equal": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PB1NP1/PP2BPP1/R1BQ1RK1 w - - 1 10",
    "mg_double_pawn": "r1bq1rk1/pp2bppp/2n1p3/2pp4/2PP4/2PB1NP1/PP3PP1/R1BQR1K1 b - - 0 12",
    "end_knight": "r4rk1/1ppbqppp/p1np1n2/8/2B1P3/2N2N2/PPP2PPP/R1BQR1K1 w - - 6 13",
    "end_bad_bishop": "2r3k1/pp2rppp/2p2n2/3q4/3P4/1B4P1/PPPQ2BP/R4R1K w - - 1 20",
    "closed_mg": "rnbq1rk1/pp3ppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 8",
    "hanging_pieces": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 6 6",
    "opposite_castle": "r1bq1rk1/ppp1bppp/2np1n2/8/2BpP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 10",
    "isolated_d5": "rnbq1rk1/pp2bppp/4pn2/2pp4/3P4/2PB1NP1/PP2BPP1/R1BQ1RK1 b - - 0 9",
}


@torch.no_grad()
def tf_policy(eng, board):
    inp = batch_boards_to_fused_token_ids([board], eng.dev)
    r = eng.model(inp)
    logits = r["policy_logits"][0].float()
    mask = legal_move_mask(board).to(eng.dev); logits[~mask] = float("-inf")
    probs = F.softmax(logits / eng.temp, dim=-1)
    top = probs.topk(12).indices
    return {IDX_TO_UCI[i]: probs[i].item() for i in top.tolist()}


def main():
    eng = HybridEngine(sf_binary="stockfish/stockfish-native-arm64",
                       checkpoint="outputs/hf_437m/best_model.pt",
                       multipv=8, policy_weight=0.6, sf_time=1.5)
    deep = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-native-arm64")
    deep.configure({"Threads": 8, "Hash": 512})

    print(f"{'pos':<18} {'gap':>5} {'sf_top':>8} {'tf_top':>8} {'deep_sf':>7} {'deep_tf':>7}  note")
    sf_win=tf_win=near=0
    for name, fen in POSITIONS.items():
        b = chess.Board(fen)
        root = eng._get_root_analysis(b)
        if not root:
            continue
        cps = [m["cp"] for m in root]
        sf_best = max(root, key=lambda m:m["cp"])
        gap = max(cps)-min(cps)  # near-tie if small
        pol = tf_policy(eng, b)
        tf_best = max(root, key=lambda m: pol.get(m["uci"],0.0))
        # deep ref
        di = deep.analyse(b, chess.engine.Limit(time=7), multipv=12)
        dc = {pv["pv"][0].uci(): pv.get("score").white().score(mate_score=32000)
              for pv in di if pv.get("pv")}
        ds = dc.get(sf_best["uci"],0); dt = dc.get(tf_best["uci"],0)
        near_flag = gap < 20
        note = "NEAR-TIE" if near_flag else ""
        if near_flag: near+=1
        if ds>=0: sf_win+=1
        if dt>=ds: tf_win+=1
        print(f"{name:<18} {gap:>5} {sf_best['uci']:>8} {tf_best['uci']:>8} {ds:>7} {dt:>7}  {note}")
    print(f"\nSF deep>=0: {sf_win}/{len(POSITIONS)} | TF>=SF: {tf_win}/{len(POSITIONS)} | near-ties: {near}")
    deep.quit(); eng.engine.quit()

if __name__=="__main__":
    main()
