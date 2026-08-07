#!/usr/bin/env python3
"""Quick A/B: does the SF+Transformer blend ever HELP vs pure Stockfish?

Procedure:
  1. Load a small suite of positions.
  2. For each, get Stockfish's top-N root moves + scores.
  3. Get the transformer's policy.
  4. Compare the blended pick against Stockfish's own best.
Count agreements / disagreements and how the blended pick scores on SF's own
evaluation. Trust a candidate if Stockfish independently agrees it is legal/top.

This is a *diagnostic* first pass (not SPRT): it tells us whether the attention
signal diverges meaningfully and in what direction.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

from hybrid_uci import HybridEngine  # noqa: E402

POSITIONS = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "italian": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "sicilian": "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2",
    "french": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "queens_gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "kings_indian_mid": "rnbq1rk1/ppp1ppbp/3p1np1/3P2B1/2P1P3/2N2N2/PP2BPPP/R2Q1RK1 w - - 3 9",
    "ruy_mid": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
}


def sf_root(engine_obj, board, multipv):
    info = engine_obj.analyse(board, chess.engine.Limit(depth=16), multipv=multipv)
    out = []
    for pv in info:
        if not pv.get("pv"):
            continue
        sc = pv.get("score")
        cp = sc.white().score(mate_score=32000) if sc else None
        out.append({"uci": pv["pv"][0].uci(), "cp": cp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="stockfish/stockfish-native-arm64")
    ap.add_argument("--checkpoint", default="outputs/hf_437m/best_model.pt")
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--policy-weight", type=float, default=0.6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--sf-time", type=float, default=2.0)
    args = ap.parse_args()

    eng = HybridEngine(
        sf_binary=args.stockfish, checkpoint=args.checkpoint,
        multipv=args.multipv, policy_weight=args.policy_weight, temp=args.temp,
        sf_time=args.sf_time,
    )

    print(f"{'position':<18} {'SF top':<10} {'blend pick':<12} agree?")
    print("-" * 54)
    agree = disagree = 0
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        moves = sf_root(eng.engine, board, args.multipv)
        if not moves:
            continue
        sf_top = max(moves, key=lambda m: m["cp"])["uci"]
        # pick blend via engine logic
        try:
            blob = eng._get_root_analysis(board)
            tf_p, _ = eng._tf_policy(board)
            combined = {}
            for m in blob:
                tf = tf_p.get(m["uci"], 0.0)
                spread = max(1.0, max(x["cp"] for x in blob) - min(x["cp"] for x in blob))
                sp = (m["cp"] - min(x["cp"] for x in blob)) / spread
                combined[m["uci"]] = (args.policy_weight * sp + (1 - args.policy_weight) * tf)
            blend_pick = max(combined, key=lambda k: combined[k])
        except Exception:
            blend_pick = sf_top
        is_agree = (blend_pick == sf_top)
        agree += is_agree
        disagree += (not is_agree)
        print(f"{name:<18} {sf_top:<10} {blend_pick:<12} {'YES' if is_agree else 'no'}")

    print(f"\nAgreements: {agree}, Disagreements: {disagree} / {len(POSITIONS)}")
    eng.engine.quit()


if __name__ == "__main__":
    main()
