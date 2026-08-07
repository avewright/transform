#!/usr/bin/env python3
"""Measure move-divergence-from-Stockfish (isolated: SF first, then models)."""
from __future__ import annotations
import os, sys, gc, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import chess, chess.engine
import torch.nn.functional as F
from chess_inference import load_checkpoint
from chess_features import batch_boards_to_fused_token_ids
from move_vocab import legal_move_mask, IDX_TO_UCI

FENS = [
 "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PB1NP1/PP2BPP1/R1BQ1RK1 w - - 1 10",
 "rnbq1rk1/ppp2ppp/4pn2/3p4/3P1B2/2N1PN2/PP3PPP/R2Q1RK1 b - - 0 10",
 "r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 6 6",
 "rnbq1rk1/ppp1ppbp/3p1np1/3P2B1/2P1P3/2N2N2/PP2BPPP/R2Q1RK1 w - - 3 9",
 "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
 "r1bq1rk1/ppp1bppp/2np1n2/8/2BpP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 10",
]
MODELS = [
    ("437M(meta)", "outputs/hf_437m/best_model.pt"),
    ("wider_shallower", "outputs/autoresearch_8gb/trials/wider_shallower/latest.pt"),
]


def main():
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-native-arm64")
    engine.configure({"Threads": 4, "Hash": 64})
    # 1) SF best move per FEN first (no torch/model in memory yet)
    sf_best = {}
    for fen in FENS:
        try:
            info = engine.analyse(chess.Board(fen), chess.engine.Limit(time=2.0), multipv=8)
            best = None; bc = None
            for pv in info:
                if not pv.get("pv"):
                    continue
                sc = pv.get("score").white().score(mate_score=32000)
                if bc is None or sc > bc:
                    bc = sc; best = pv["pv"][0].uci()
            sf_best[fen] = best
        except Exception as e:
            print("sf err", e); sf_best[fen] = None
    engine.quit()
    print(f"SF best moves computed for {sum(1 for v in sf_best.values() if v)}/{len(FENS)} positions")

    # 2) Now load models and compute argmax divergence
    device = "mps"
    for tag, path in MODELS:
        m = load_checkpoint(path, device); m.eval()
        div = tot = 0
        for fen in FENS:
            sb = sf_best.get(fen)
            if sb is None:
                continue
            b = chess.Board(fen)
            inp = batch_boards_to_fused_token_ids([b], device)
            with torch.no_grad():
                r = m(inp)
            lg = r["policy_logits"][0].float()
            mask = legal_move_mask(b).to(device); lg[~mask] = float("-inf")
            mv = IDX_TO_UCI[lg.argmax().item()]
            tot += 1; div += (mv != sb)
        print(f"{tag:<18} divergence-from-SF: {div}/{tot}")
        del m; gc.collect(); torch.mps.empty_cache()


if __name__ == "__main__":
    main()
