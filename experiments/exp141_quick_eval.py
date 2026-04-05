"""exp141: Quick greedy ELO eval for checkpoints during training.

Runs a fast 8-game greedy eval (no MCTS) against SF at specified ELO.
Designed to run alongside training by using minimal GPU (<200ms per move).

Usage:
  python experiments/exp141_quick_eval.py --checkpoint outputs/exp140_25m_10m/best_model.pt \
    --config outputs/exp140_25m_10m/config.json --sf-elo 1320
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model, ChessTransformerConfig
from move_vocab import legal_move_mask, IDX_TO_UCI

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
]


def resolve_stockfish():
    candidates = [
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ]
    for c in candidates:
        if c.exists():
            return c
    binary = shutil.which("stockfish")
    if binary:
        return Path(binary)
    raise FileNotFoundError("Stockfish not found")


def elo_diff(s):
    if s <= 0: return -400
    if s >= 1: return 400
    return -400 * math.log10(1/s - 1)


def wilson_ci(s, n, z=1.96):
    p = s/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / d
    return max(0, c-m), min(1, c+m)


@torch.no_grad()
def get_greedy_move(model, board):
    board_input = batch_boards_to_fused_token_ids([board], DEVICE)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        result = model(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(DEVICE)
    logits[~mask] = float("-inf")
    idx = logits.argmax().item()
    return chess.Move.from_uci(IDX_TO_UCI[idx])


def play_game(model, sf_engine, sf_elo, model_color, opening, ply_cap=160):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move = get_greedy_move(model, board)
        else:
            move = sf_engine.play(board, chess.engine.Limit(time=0.05)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == model_color else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="Model config JSON")
    ap.add_argument("--sf-elo", type=int, default=1320)
    ap.add_argument("--games", type=int, default=16)
    args = ap.parse_args()

    # Load model
    if args.config:
        cfg = ChessTransformerConfig.from_json(args.config)
    else:
        cfg = None
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")
    if "step" in ckpt:
        print(f"Checkpoint: step={ckpt['step']}, acc={ckpt.get('best_acc', '?')}")

    # Setup SF
    sf_path = resolve_stockfish()
    sf = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo, "Threads": 1, "Hash": 32})

    # Run games
    total_score = 0.0
    results = []
    print(f"\nGreedy eval: {args.games} games vs SF{args.sf_elo}")
    print("-" * 50)

    for gi in range(args.games):
        color = chess.WHITE if gi % 2 == 0 else chess.BLACK
        opening = OPENINGS[gi % len(OPENINGS)]
        score = play_game(model, sf, args.sf_elo, color, opening)
        total_score += score
        r = "W" if score == 1.0 else ("D" if score == 0.5 else "L")
        results.append(r)

        avg = total_score / (gi + 1)
        lo, hi = wilson_ci(total_score, gi + 1)
        elo = args.sf_elo + elo_diff(avg)
        c = "white" if color == chess.WHITE else "black"
        print(f"  G{gi+1:2d}/{args.games}: {r} ({c}) | "
              f"score={avg:.3f} [{lo:.3f},{hi:.3f}] ~{elo:.0f} ELO")

    sf.quit()

    # Summary
    n = args.games
    avg = total_score / n
    lo, hi = wilson_ci(total_score, n)
    elo = args.sf_elo + elo_diff(avg)
    w = results.count("W")
    d = results.count("D")
    l = results.count("L")
    print(f"\nFINAL: {w}W-{d}D-{l}L | score={avg:.3f} [{lo:.3f},{hi:.3f}] | ~{elo:.0f} ELO")


if __name__ == "__main__":
    main()
