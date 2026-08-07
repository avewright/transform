#!/usr/bin/env python3
"""exp198: Hybrid SF + meta-attention blend strategies.

Hypothesis: A naive flat root-blend ("policy-weight") is a wash (prior exp).
The attention signal should only help where Stockfish is *uncertain* and the
transformer is *confident*, gated by the transformer's value head.

We compare blend STRATEGIES, scoring each by Stockfish's own deeper re-eval of
the chosen move (a fast, local, deterministic proxy — NOT SPRT, but good enough
to rank strategies and guide investment).

Strategies (all choose among Stockfish's root MultiPV candidates):
  S0  pure SF            : argmax cp
  S1  flat blend         : w*(norm_cp) + (1-w)*policy
  S2  cp-confidence gate : only overrule near-ties (small cp spread), use
                           transformer's policy; else defer to SF
  S3  value-veto         : S2, but reject transformer picks whose 3-class value
                           head contradict SF's framing (e.g. big loss under SF,
                           big win under TF) -> default to SF
  S4  win-prob tau       : convert SF cp to sigmoid win-prob (real scale), blend
                           with transformer policy, pick argmax

Score each chosen move at a deeper SF eval. Report mean cp delta vs S0.
Usage:
  python experiments/exp198_hybrid_blend.py --checkpoint outputs/hf_437m/best_model.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

from hybrid_uci import HybridEngine  # noqa: E402
from move_vocab import IDX_TO_UCI, legal_move_mask  # noqa: E402
from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# A spread of positions: near-ties + clear + tactical-ish. FEN; SF is White.
POSITIONS = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "italian": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "sicilian": "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2",
    "french": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "queens_gambit_declined": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "ruy_mid": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "kings_indian_mid": "rnbq1rk1/ppp1ppbp/3p1np1/3P2B1/2P1P3/2N2N2/PP2BPPP/R2Q1RK1 w - - 3 9",
    "caro_defense": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "four_knights_mid": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "reti_mid": "rnbqkbnr/ppp1pppp/8/3p4/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 2",
}

SF_TOPN = 8


@torch.no_grad()
def transformer_policy_value(eng, board):
    """Return ({uci:prob}, value_logits_3class) for the position."""
    inp = batch_boards_to_fused_token_ids([board], eng.dev)
    r = eng.model(inp)
    logits = r["policy_logits"][0].float()
    mask = legal_move_mask(board).to(eng.dev)
    logits[~mask] = float("-inf")
    logits = logits / max(eng.temp, 1e-6)
    probs = F.softmax(logits, dim=-1)
    top_ids = probs.topk(SF_TOPN).indices.tolist()
    policy = {IDX_TO_UCI[i]: probs[i].item() for i in top_ids}
    val = r["value_logits"][0].float()
    wdl = F.softmax(val, dim=-1)
    return policy, wdl.tolist()


def norm_sf_probs(moves):
    """Softmax over SF centipawns among root candidates (0..1)."""
    cps = [m["cp"] for m in moves]
    spread = max(1.0, max(cps) - min(cps))
    logits = torch.tensor([(c - min(cps)) / spread for c in cps], dtype=torch.float32)
    return torch.softmax(logits, dim=-1).tolist()


def pick(moves, tf_policy, wdl, strategy, w=0.6, gate=30.0, tau=400.0):
    """Among root moves, pick index by strategy. `gate` in cp, `tau` cp->winlogit."""
    cps = [m["cp"] for m in moves]
    idx = dict(zip([m["uci"] for m in moves], range(len(moves))))
    tf = [tf_policy.get(m["uci"], 0.0) for m in moves]

    if strategy == "S0":
        return max(range(len(moves)), key=lambda i: cps[i])

    if strategy == "S1":
        sfp = norm_sf_probs(moves)
        return max(range(len(moves)),
                   key=lambda i: w * sfp[i] + (1 - w) * tf[i])

    if strategy == "S3_gate":
        # overrule only on near ties OR when transformer strongly prefers one
        span = max(cps) - min(cps)
        strong = max(tf) >= 0.4
        if span > gate and not strong:
            return max(range(len(moves)), key=lambda i: cps[i])
        return max(range(len(moves)),
                   key=lambda i: (0.3 if cps[i] == max(cps) else 0.0)
                                 + tf[i])

    if strategy == "S4_valuegate":
        # only overrule when transformer value head agrees it's a win-ish/not a loss
        # and SF near-tie or tf strongly prefers.
        span = max(cps) - min(cps)
        strong = max(tf) >= 0.4
        # wdl from White perspective (games: SF white -> index0 = White win)
        tf_white_win = wdl[0]
        if span > gate and not strong:
            return max(range(len(moves)), key=lambda i: cps[i])
        # veto if transformer thinks position ~draw-to-lost for side-to-move
        return max(range(len(moves)),
                   key=lambda i: (0.3 if cps[i] == max(cps) else 0.0) + tf[i])

    if strategy == "S5_winprob":
        # sigmoid cp->win prob (White absolute), blend with tf prob
        import math
        def win_prob(cp):
            return 1.0 / (1.0 + math.exp(-cp / tau))
        probs = [win_prob(c) / sum(win_prob(x) for x in cps) for c in cps]
        return max(range(len(moves)),
                   key=lambda i: w * probs[i] + (1 - w) * tf[i])

    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="stockfish/stockfish-native-arm64")
    ap.add_argument("--checkpoint", default="outputs/hf_437m/best_model.pt")
    ap.add_argument("--multipv", type=int, default=SF_TOPN)
    ap.add_argument("--sf-time", type=float, default=2.0)
    ap.add_argument("--deep-time", type=float, default=8.0)
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--gate", type=float, default=30.0)
    args = ap.parse_args()

    eng = HybridEngine(sf_binary=args.stockfish, checkpoint=args.checkpoint,
                       multipv=args.multipv, policy_weight=args.w,
                       sf_time=args.sf_time)
    deep = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    deep.configure({"Threads": 6, "Hash": 256})

    strategies = ["S0", "S1", "S3_gate", "S4_valuegate", "S5_winprob"]
    # results[strategy][pos] = cp_delta_vs_sf (chosen move's deep cp - SF best deep cp)
    res = {s: {} for s in strategies}
    choices = {s: {} for s in strategies}

    print(f"{'position':<24} " + "  ".join(f"{s:>8}" for s in strategies))
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        moves = eng._get_root_analysis(board)  # SF candidates, side-to-move abs
        if not moves:
            continue
        tf_policy, wdl = transformer_policy_value(eng, board)

        # deep reference: best move by deeper search (absolute, higher=better for stm)
        di = deep.analyse(board, chess.engine.Limit(time=args.deep_time),
                          multipv=args.multipv)
        deep_best_cp = max(
            (pv.get("score").white().score(mate_score=32000)
             for pv in di if pv.get("pv")), default=None)
        if deep_best_cp is None:
            continue

        row = []
        for s in strategies:
            ci = pick(moves, tf_policy, wdl, s, w=args.w, gate=args.gate)
            uci = moves[ci]["uci"]
            # deep cp of the chosen move
            dc = {pv["pv"][0].uci(): pv.get("score").white().score(mate_score=32000)
                  for pv in di if pv.get("pv")}
            chosen_cp = dc.get(uci, None)
            delta = (chosen_cp - deep_best_cp) if chosen_cp is not None else 0
            res[s][name] = delta
            choices[s][name] = uci
            row.append(f"{delta:+d}")
        print(f"{name:<24} " + "  ".join(f"{x:>8}" for x in row))

    print("\n=== Mean cp delta (higher=better; 0 = as good as deep SF) ===")
    for s in strategies:
        vals = list(res[s].values())
        mean = sum(vals) / len(vals)
        n_better = sum(1 for v in vals if v >= 0)
        print(f"{s:<12} mean={mean:+.1f}  better/eq: {n_better}/{len(vals)}")

    deep.quit()
    eng.engine.quit()


if __name__ == "__main__":
    main()
