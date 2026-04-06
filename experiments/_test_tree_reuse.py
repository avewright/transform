"""Quick validation that tree reuse works across moves.

Plays 1 game with tree reuse ON vs OFF and compares NN eval counts.
Tree reuse should save ~10-30% of NN evals by reusing explored subtrees.

Usage:
  python experiments/_test_tree_reuse.py
  python experiments/_test_tree_reuse.py --checkpoint outputs/exp149_scratch_204m/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import torch

from chess_transformer_factory import build_model, ChessTransformerConfig
from uci_engine import MCTSSearch, SyzygyProbe
from opening_book import get_book_move

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CKPT = ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt"

FIXED_OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
SIMS = 50  # Low sims for quick test


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    cfg = ChessTransformerConfig(**ckpt["config"]) if "config" in ckpt else None
    model = build_model(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model


def play_game(model, use_tree_reuse, max_moves=30):
    """Play a short game, return total NN evals used."""
    syzygy = SyzygyProbe()
    mcts = MCTSSearch(
        model, DEVICE, syzygy,
        c_puct=2.5, batch_size=8,
        root_noise_alpha=0.03, root_noise_frac=0.0,  # No noise for reproducibility
        use_fp16=True, use_transpositions=True,
    )
    mcts.new_game()

    board = chess.Board()
    for uci in FIXED_OPENING:
        board.push(chess.Move.from_uci(uci))

    total_nn_evals = 0
    moves_played = 0

    for i in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        move, info = mcts.search(board, max_sims=SIMS)
        nn = info.get("nn_evals", 0)
        total_nn_evals += nn

        if use_tree_reuse:
            # Advance tree past our move
            mcts.advance_tree(move)
        else:
            # Reset tree (no reuse)
            mcts.root = None

        board.push(move)
        moves_played += 1

        # Simulate opponent (play the highest-policy move)
        if board.is_game_over(claim_draw=True):
            break
        # Use model's own policy for opponent (deterministic)
        opp_move, opp_info = mcts.search(board, max_sims=SIMS)
        opp_nn = opp_info.get("nn_evals", 0)
        total_nn_evals += opp_nn

        if use_tree_reuse:
            mcts.advance_tree(opp_move)
        else:
            mcts.root = None

        board.push(opp_move)
        moves_played += 1

    return total_nn_evals, moves_played


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = ap.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    print(f"Device: {DEVICE}")
    print(f"Sims per move: {SIMS}")
    print(f"Opening: {' '.join(FIXED_OPENING)}")
    print()

    # Run WITHOUT tree reuse
    print("Running WITHOUT tree reuse...")
    nn_no_reuse, moves_no = play_game(model, use_tree_reuse=False)
    print(f"  Moves: {moves_no}, NN evals: {nn_no_reuse}")

    # Run WITH tree reuse
    print("Running WITH tree reuse...")
    nn_reuse, moves_reuse = play_game(model, use_tree_reuse=True)
    print(f"  Moves: {moves_reuse}, NN evals: {nn_reuse}")

    print()
    if moves_reuse > 0 and moves_no > 0:
        per_move_no = nn_no_reuse / moves_no
        per_move_reuse = nn_reuse / moves_reuse
        savings = (per_move_no - per_move_reuse) / per_move_no * 100
        print(f"Evals/move WITHOUT reuse: {per_move_no:.1f}")
        print(f"Evals/move WITH reuse:    {per_move_reuse:.1f}")
        if savings > 0:
            print(f"Tree reuse SAVES {savings:.1f}% of NN evals per move ✓")
        else:
            print(f"Tree reuse uses {-savings:.1f}% MORE evals (unexpected!) ✗")
    else:
        print("Could not compare (0 moves played)")


if __name__ == "__main__":
    main()
