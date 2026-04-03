"""exp094: 1-ply search-enhanced evaluation.

Hypothesis: Using the model's value head for 1-ply lookahead during gameplay
should give 100-200+ ELO for free — no additional training needed.

For each position, instead of just taking the policy head's top move:
  1. Get top-K moves from the policy head (K=8)
  2. For each candidate move, push it on the board and evaluate with the value head
  3. Pick the move that maximizes expected value (win_prob - loss_prob)

This is a simple "alpha-beta depth 1" using the neural value function.
It's the cheapest possible search and should catch tactical blunders
where the policy head is confident but the value head disagrees.

Usage:
    python experiments/exp094_search_eval.py \
        --checkpoint outputs/exp090_full_legal_temp05_continue_ckpt/checkpoints/latest.pt \
        --output-tag exp094_search_d1 \
        --search-depth 1 --top-k 8

    # Can also do depth-2 (much slower but potentially stronger):
    python experiments/exp094_search_eval.py \
        --checkpoint outputs/exp093_ema_curriculum_d8/best_model.pt \
        --output-tag exp094_search_d2 \
        --search-depth 2 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import IDX_TO_UCI, UCI_TO_IDX, VOCAB_SIZE, index_to_move, legal_move_mask

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
DEFAULT_TEST_ELOS = [1320, 1450, 1600, 1750, 1900]


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
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Stockfish not found")


SF = resolve_stockfish_path()
LOG: Path
JSON_OUT: Path


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_model(checkpoint_path: str | Path, device: torch.device):
    from play import ChessTransformer200M
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Value-based search
# ---------------------------------------------------------------------------

@torch.no_grad()
def board_value(model, board: chess.Board, device: torch.device) -> float:
    """Get the value estimate for a board position.

    Returns expected score from the side-to-move's perspective: win_prob - loss_prob.
    Ranges from -1 (certain loss) to +1 (certain win).
    """
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    wdl = F.softmax(result["value_logits"][0].float(), dim=-1)
    # WDL is White-absolute: [P(W wins), P(draw), P(W loses)]
    # Convert to side-to-move perspective
    white_value = (wdl[0] - wdl[2]).item()
    return white_value if board.turn == chess.WHITE else -white_value


@torch.no_grad()
def search_move_depth1(
    model, board: chess.Board, device: torch.device, top_k: int = 8,
) -> tuple[chess.Move, dict]:
    """1-ply search: evaluate top-K policy moves with value head, pick best."""
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    k = min(top_k, int(mask.sum().item()))
    topk = torch.topk(probs, k)

    best_value = float("-inf")
    best_move = None
    candidates = []

    for idx, policy_prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue

        board.push(move)
        # Value is from opponent's perspective after our move, so negate
        child_value = -board_value(model, board, device)
        board.pop()

        candidates.append({
            "uci": move.uci(),
            "policy_prob": f"{policy_prob*100:.1f}%",
            "value": round(child_value, 4),
        })

        if child_value > best_value:
            best_value = child_value
            best_move = move

    if best_move is None:
        best_move = next(iter(board.legal_moves))

    # Get WDL for the current position
    wdl_logits = result["value_logits"][0].float()
    wdl = F.softmax(wdl_logits, dim=-1).tolist()

    return best_move, {
        "search_depth": 1,
        "candidates_evaluated": len(candidates),
        "top_candidates": candidates[:5],
        "wdl": {"win": wdl[0], "draw": wdl[1], "loss": wdl[2]},
        "chosen_value": round(best_value, 4),
    }


@torch.no_grad()
def search_move_depth2(
    model, board: chess.Board, device: torch.device, top_k: int = 5,
) -> tuple[chess.Move, dict]:
    """2-ply search: for each top-K move, evaluate opponent's best reply."""
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    k = min(top_k, int(mask.sum().item()))
    topk = torch.topk(probs, k)

    best_value = float("-inf")
    best_move = None
    candidates = []

    for idx, policy_prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue

        board.push(move)

        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                # We just moved, so if there's a winner it's us (checkmate)
                child_value = 1.0 if outcome.winner != board.turn else -1.0
            else:
                child_value = 0.0  # draw
        else:
            # Opponent's best reply (minimax)
            opp_inp = batch_boards_to_fused_token_ids([board], device)
            opp_result = model(opp_inp)
            opp_logits = opp_result["policy_logits"][0].float()
            opp_mask = legal_move_mask(board).to(device)
            opp_logits[~opp_mask] = float("-inf")
            opp_probs = F.softmax(opp_logits, dim=-1)
            opp_k = min(top_k, int(opp_mask.sum().item()))
            opp_topk = torch.topk(opp_probs, opp_k)

            worst_for_us = float("inf")
            for opp_idx in opp_topk.indices.tolist():
                opp_move = index_to_move(opp_idx)
                if opp_move not in board.legal_moves:
                    continue
                board.push(opp_move)
                # Now it's our turn again — get value from our perspective
                leaf_value = board_value(model, board, device)
                board.pop()
                worst_for_us = min(worst_for_us, leaf_value)

            child_value = worst_for_us if worst_for_us != float("inf") else 0.0

        board.pop()

        candidates.append({
            "uci": move.uci(),
            "policy_prob": f"{policy_prob*100:.1f}%",
            "value": round(child_value, 4),
        })

        if child_value > best_value:
            best_value = child_value
            best_move = move

    if best_move is None:
        best_move = next(iter(board.legal_moves))

    wdl_logits = result["value_logits"][0].float()
    wdl = F.softmax(wdl_logits, dim=-1).tolist()

    return best_move, {
        "search_depth": 2,
        "candidates_evaluated": len(candidates),
        "top_candidates": candidates[:5],
        "wdl": {"win": wdl[0], "draw": wdl[1], "loss": wdl[2]},
        "chosen_value": round(best_value, 4),
    }


@torch.no_grad()
def greedy_move(model, board, device, temperature=0.0):
    """Standard greedy policy move (no search), for baseline comparison."""
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    move_idx = logits.argmax().item()
    move = index_to_move(move_idx)
    wdl_logits = result["value_logits"][0].float()
    wdl = F.softmax(wdl_logits, dim=-1).tolist()
    return move, {"wdl": {"win": wdl[0], "draw": wdl[1], "loss": wdl[2]}}


# ---------------------------------------------------------------------------
# Game play and elo eval (adapted from elo_eval_latest.py)
# ---------------------------------------------------------------------------

def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_one(engine, model, move_fn, sf_elo, model_color, opening, movetime, ply_cap):
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


def summarize_results(sf_elo, results):
    games = len(results)
    total_score = sum(r["score"] for r in results)
    score = total_score / games if games else 0.0
    ci_low, ci_high = wilson_interval(total_score, games)
    return {
        "sf_elo": sf_elo,
        "games": games,
        "score": score,
        "score_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "w": sum(1 for r in results if r["score"] == 1.0),
        "d": sum(1 for r in results if r["score"] == 0.5),
        "l": sum(1 for r in results if r["score"] == 0.0),
        "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0.0,
    }


def estimate_elo(summaries):
    ordered = sorted(summaries, key=lambda s: s["sf_elo"])
    above = [s for s in ordered if s["score"] >= 0.5]
    below = [s for s in ordered if s["score"] < 0.5]
    lower_bound = max((s["sf_elo"] for s in above), default=None)
    upper_bound = min((s["sf_elo"] for s in below), default=None)

    if lower_bound is None:
        return {"estimated_elo": ordered[0]["sf_elo"], "lower_bound": None, "upper_bound": ordered[0]["sf_elo"]}
    if upper_bound is None:
        return {"estimated_elo": ordered[-1]["sf_elo"], "lower_bound": ordered[-1]["sf_elo"], "upper_bound": None}

    ls = next(s for s in ordered if s["sf_elo"] == lower_bound)
    us = next(s for s in ordered if s["sf_elo"] == upper_bound)
    if ls["score"] == us["score"]:
        est = lower_bound
    else:
        frac = (0.5 - ls["score"]) / (us["score"] - ls["score"])
        est = round(lower_bound + frac * (upper_bound - lower_bound))
    return {"estimated_elo": est, "lower_bound": lower_bound, "upper_bound": upper_bound}


def run_eval(model, move_fn, search_tag: str, test_elos, openings, movetime=0.05, ply_cap=160):
    summaries = []
    all_games = []

    for sf_elo in test_elos:
        engine = chess.engine.SimpleEngine.popen_uci(str(SF))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})
        results = []
        try:
            for opening in openings:
                for color in [chess.WHITE, chess.BLACK]:
                    result = play_one(engine, model, move_fn, sf_elo, color, opening, movetime, ply_cap)
                    results.append(result)
                    log(f"game {json.dumps({k: result[k] for k in ['sf_elo', 'model_color', 'result', 'score', 'plies', 'termination']})}")
        finally:
            engine.quit()

        summary = summarize_results(sf_elo, results)
        summaries.append(summary)
        all_games.extend(results)
        log(f"level_summary {json.dumps(summary)}")

        # Early stop: if we found both bounds, done
        above = [s for s in summaries if s["score"] >= 0.5]
        below = [s for s in summaries if s["score"] < 0.5]
        if above and below:
            log("bracketing complete, stopping early")
            break

    elo_est = estimate_elo(summaries)
    log(f"elo_estimate search={search_tag} {json.dumps(elo_est)}")
    return {"search_mode": search_tag, "elo": elo_est, "summaries": summaries, "games": all_games}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="exp094: Search-enhanced ELO evaluation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-tag", type=str, default="exp094_search")
    p.add_argument("--search-depth", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--test-elos", type=int, nargs="+", default=DEFAULT_TEST_ELOS)
    return p.parse_args()


def main():
    global LOG, JSON_OUT
    args = parse_args()

    output_dir = ROOT / "outputs"
    LOG = output_dir / f"elo_eval_{args.output_tag}.log"
    JSON_OUT = output_dir / f"elo_eval_{args.output_tag}.json"

    model = load_model(args.checkpoint, DEVICE)

    # Set up the move function based on search depth
    if args.search_depth == 0:
        move_fn = greedy_move
        search_tag = "greedy_d0"
    elif args.search_depth == 1:
        def move_fn(model, board, device, temperature=0.0):
            return search_move_depth1(model, board, device, top_k=args.top_k)
        search_tag = f"value_search_d1_k{args.top_k}"
    else:
        def move_fn(model, board, device, temperature=0.0):
            return search_move_depth2(model, board, device, top_k=args.top_k)
        search_tag = f"value_search_d2_k{args.top_k}"

    log(f"exp094: search-enhanced elo eval")
    log(f"checkpoint={args.checkpoint}")
    log(f"search_depth={args.search_depth} top_k={args.top_k} tag={search_tag}")
    log(f"device={DEVICE}")

    # Run baseline (greedy) first for comparison
    log("=" * 60)
    log("BASELINE: greedy (no search)")
    log("=" * 60)
    baseline_results = run_eval(model, greedy_move, "greedy_d0", args.test_elos, DEFAULT_OPENINGS)

    # Run search-enhanced
    if args.search_depth > 0:
        log("=" * 60)
        log(f"SEARCH: depth={args.search_depth} top_k={args.top_k}")
        log("=" * 60)
        search_results = run_eval(model, move_fn, search_tag, args.test_elos, DEFAULT_OPENINGS)
    else:
        search_results = None

    # Save combined results
    output = {
        "checkpoint": str(args.checkpoint),
        "search_depth": args.search_depth,
        "top_k": args.top_k,
        "baseline": baseline_results,
        "search": search_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    log(f"results saved to {JSON_OUT}")

    # Summary
    baseline_elo = baseline_results["elo"]["estimated_elo"]
    log(f"\nBASELINE ELO: {baseline_elo}")
    if search_results:
        search_elo = search_results["elo"]["estimated_elo"]
        delta = (search_elo or 0) - (baseline_elo or 0)
        log(f"SEARCH ELO:   {search_elo} (delta={delta:+d})")

    log("done")


if __name__ == "__main__":
    main()
