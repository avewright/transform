"""exp110_search: Value reranking at inference time.

Tests whether using the value head to rerank top-K policy moves improves ELO.

For each position:
  1. Get top-K moves from policy head
  2. For each candidate, push the move and evaluate the resulting position
  3. Use opponent's value prediction (flipped) + policy confidence to score
  4. Pick the move with the best combined score

This is a zero-cost inference improvement that requires no additional training.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import IDX_TO_UCI, VOCAB_SIZE, index_to_move, legal_move_mask
from play import ChessTransformer200M

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_stockfish_path() -> Path:
    configured = os.environ.get("STOCKFISH_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    binary = shutil.which("stockfish")
    if binary:
        candidates.append(Path(binary))
    candidates.extend([
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Stockfish not found")


SF_PATH = resolve_stockfish_path()


def load_model(checkpoint_path: str, device: torch.device):
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return model


@torch.no_grad()
def get_policy_move(model, board: chess.Board, device: torch.device, temperature=0.0):
    """Standard greedy policy move."""
    board_input = batch_boards_to_fused_token_ids([board], device)
    result = model(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    move_idx = logits.argmax().item()
    move = index_to_move(move_idx)
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, mask.sum().item()))
    top_moves = [(IDX_TO_UCI[i], f"{p*100:.1f}%") for i, p in zip(topk.indices.tolist(), topk.values.tolist())]
    wdl = F.softmax(result["value_logits"][0].float(), dim=-1).tolist()
    return move, {"top_moves": top_moves, "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]}}


@torch.no_grad()
def get_value_rerank_move(
    model, board: chess.Board, device: torch.device,
    top_k: int = 8, policy_weight: float = 0.3,
    temperature: float = 0.0,
):
    """Value reranking: score top-K policy moves by pushing each and evaluating.

    Score = (1 - policy_weight) * value_score + policy_weight * policy_prob
    where value_score = model's predicted win probability from the resulting position
    (from the current player's perspective).
    """
    board_input = batch_boards_to_fused_token_ids([board], device)
    result = model(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    n_legal = mask.sum().item()
    k = min(top_k, n_legal)
    topk = torch.topk(probs, k)

    if k <= 1:
        # Only one legal move
        move_idx = topk.indices[0].item()
        move = index_to_move(move_idx)
        wdl = F.softmax(result["value_logits"][0].float(), dim=-1).tolist()
        return move, {"top_moves": [(IDX_TO_UCI[move_idx], "100.0%")],
                      "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]}, "search": "trivial"}

    # Evaluate each candidate by pushing the move and running the model
    candidate_boards = []
    candidate_moves = []
    candidate_probs = []

    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move in board.legal_moves:
            new_board = board.copy()
            new_board.push(move)
            candidate_boards.append(new_board)
            candidate_moves.append(move)
            candidate_probs.append(prob)

    if not candidate_boards:
        move_idx = topk.indices[0].item()
        return index_to_move(move_idx), {"search": "fallback"}

    # Batch evaluate all candidate positions
    board_inputs = batch_boards_to_fused_token_ids(candidate_boards, device)
    results = model(board_inputs)
    value_logits = results["value_logits"].float()  # (K, 3) White-absolute: [P(W wins), P(draw), P(W loses)]
    value_probs = F.softmax(value_logits, dim=-1)

    # WDL is White-absolute, NOT side-to-move relative.
    # After pushing our move, the child boards are from the opponent's turn,
    # but the WDL still reports White-absolute probabilities.
    # We want the value from the perspective of the parent mover (us).
    # If we are White: our_win = child_wdl[0] (P(W wins))
    # If we are Black: our_win = child_wdl[2] (P(W loses) = P(B wins))
    # parent_turn is the turn BEFORE we pushed (i.e. our color)
    parent_turn = candidate_boards[0].turn  # opponent's turn = not our turn
    # Actually parent_turn here is opponent's turn since we pushed. We need the original mover.
    # The original board.turn is our color. After push, candidate_boards have opponent's turn.
    # So if candidate_boards[0].turn == BLACK, we were WHITE.
    if candidate_boards[0].turn == chess.BLACK:
        # We are White: our win prob = P(W wins) = idx 0
        our_win_prob = value_probs[:, 0].tolist()
        our_draw_prob = value_probs[:, 1].tolist()
    else:
        # We are Black: our win prob = P(W loses) = idx 2
        our_win_prob = value_probs[:, 2].tolist()
        our_draw_prob = value_probs[:, 1].tolist()

    # Combined score: value_score = win + 0.5 * draw (expected score)
    value_scores = [w + 0.5 * d for w, d in zip(our_win_prob, our_draw_prob)]

    # Normalize policy probs
    total_prob = sum(candidate_probs)
    norm_probs = [p / total_prob for p in candidate_probs]

    # Combined score
    combined = [
        (1.0 - policy_weight) * vs + policy_weight * pp
        for vs, pp in zip(value_scores, norm_probs)
    ]

    best_idx = max(range(len(combined)), key=lambda i: combined[i])
    best_move = candidate_moves[best_idx]

    # Build info
    top_moves = []
    for i in sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:5]:
        m = candidate_moves[i]
        top_moves.append((m.uci(), f"v={value_scores[i]:.3f} p={norm_probs[i]:.3f} c={combined[i]:.3f}"))

    wdl = F.softmax(result["value_logits"][0].float(), dim=-1).tolist()
    return best_move, {
        "top_moves": top_moves,
        "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]},
        "search": "value_rerank",
        "k": k,
        "best_value_score": value_scores[best_idx],
    }


# ── Game playing ──
def play_game(engine, model, move_fn, sf_elo, model_color, opening, movetime, ply_cap):
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move, _ = move_fn(model, board, DEVICE)
        else:
            move = engine.play(board, chess.engine.Limit(time=movetime)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == model_color:
        score = 1.0
    else:
        score = 0.0

    return {
        "sf_elo": sf_elo,
        "model_color": "white" if model_color == chess.WHITE else "black",
        "opening": opening,
        "result": board.result(claim_draw=True),
        "score": score,
        "plies": len(board.move_stack),
        "termination": outcome.termination.name if outcome else "PLY_CAP",
    }


DEFAULT_OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
]


def run_eval(model, move_fn, sf_elo, games_per_opening_per_color=1, movetime=0.05, ply_cap=160):
    """Run games against limited-strength Stockfish."""
    engine = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})

    results = []
    try:
        for opening in DEFAULT_OPENINGS:
            for color in [chess.WHITE, chess.BLACK]:
                for _ in range(games_per_opening_per_color):
                    result = play_game(engine, model, move_fn, sf_elo, color, opening, movetime, ply_cap)
                    results.append(result)
                    print(f"  {result['model_color']} vs SF{sf_elo}: {result['result']} "
                          f"({result['plies']} plies, {result['termination']})", flush=True)
    finally:
        engine.quit()

    total = len(results)
    score = sum(r["score"] for r in results) / total if total else 0
    w = sum(1 for r in results if r["score"] == 1.0)
    d = sum(1 for r in results if r["score"] == 0.5)
    l = sum(1 for r in results if r["score"] == 0.0)
    return {"sf_elo": sf_elo, "games": total, "score": score, "w": w, "d": d, "l": l}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, default="outputs/hf_checkpoint/best_model.pt")
    parser.add_argument("--top-k", type=int, default=8, help="Number of policy candidates to rerank")
    parser.add_argument("--policy-weight", type=float, default=0.3, help="Weight for policy prob in combined score")
    parser.add_argument("--elos", type=int, nargs="+", default=[1320, 1600, 1750, 1900])
    parser.add_argument("--games-per", type=int, default=1, help="Games per opening per color")
    parser.add_argument("--compare-greedy", action="store_true", help="Also run greedy baseline for comparison")
    args = parser.parse_args()

    model = load_model(args.checkpoint, DEVICE)
    out_dir = ROOT / "outputs" / "exp110_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Value reranking move function
    def value_rerank_fn(model, board, device, temperature=0.0):
        return get_value_rerank_move(model, board, device, top_k=args.top_k, policy_weight=args.policy_weight)

    print(f"\n=== Value Reranking (top-{args.top_k}, policy_weight={args.policy_weight}) ===")
    rerank_results = []
    for elo in args.elos:
        print(f"\nvs SF {elo}:")
        summary = run_eval(model, value_rerank_fn, elo, args.games_per)
        rerank_results.append(summary)
        print(f"  Score: {summary['score']:.3f} (W{summary['w']}/D{summary['d']}/L{summary['l']})")

    if args.compare_greedy:
        print(f"\n=== Greedy Baseline ===")
        greedy_results = []
        for elo in args.elos:
            print(f"\nvs SF {elo}:")
            summary = run_eval(model, get_policy_move, elo, args.games_per)
            greedy_results.append(summary)
            print(f"  Score: {summary['score']:.3f} (W{summary['w']}/D{summary['d']}/L{summary['l']})")
    else:
        greedy_results = None

    # Save results
    result = {
        "checkpoint": args.checkpoint,
        "top_k": args.top_k,
        "policy_weight": args.policy_weight,
        "value_rerank": rerank_results,
        "greedy": greedy_results,
    }
    out_path = out_dir / "search_comparison.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
