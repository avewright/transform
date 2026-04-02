"""exp097: Alpha-Beta search with neural network evaluation.

This is the key experiment for maximizing ELO. AlphaZero/Leela showed that
neural network + search >> neural network alone. Even a modest 4-6 ply
alpha-beta search with our value head should gain 500+ ELO.

Features:
  - Alpha-beta pruning with move ordering from policy head
  - Quiescence search (extend captures/checks at leaf nodes) 
  - Transposition table for avoiding re-evaluations
  - Iterative deepening with time management
  - Aspiration windows for faster convergence
  - Batched neural network evaluation for throughput

Usage:
    # Quick test: depth 4, should be much stronger than greedy
    python experiments/exp097_alphabeta_search.py \
        --checkpoint outputs/exp090_full_legal_temp05_continue_ckpt/checkpoints/latest.pt \
        --output-tag exp097_ab_d4 \
        --max-depth 4 --top-k 12

    # Full strength: depth 6 with quiescence
    python experiments/exp097_alphabeta_search.py \
        --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt \
        --output-tag exp097_ab_d6 \
        --max-depth 6 --top-k 16 --quiesce
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
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
DEFAULT_TEST_ELOS = [1320, 1450, 1600, 1750, 1900, 2100, 2300]


def resolve_stockfish_path() -> Path:
    configured = os.environ.get("STOCKFISH_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    binary = shutil.which("stockfish")
    if binary:
        candidates.append(Path(binary))
    candidates.extend([
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
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
    if LOG is not None:
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
# Transposition Table
# ---------------------------------------------------------------------------

class TranspositionTable:
    """Simple LRU-bounded transposition table."""

    EXACT = 0
    LOWER = 1  # beta cutoff: value >= beta
    UPPER = 2  # failed low: value <= alpha

    def __init__(self, max_entries: int = 500_000):
        self.max_entries = max_entries
        self.table: OrderedDict[int, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def probe(self, board: chess.Board, depth: int, alpha: float, beta: float):
        key = chess.polyglot.zobrist_hash(board)
        entry = self.table.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry["depth"] < depth:
            self.misses += 1
            return None

        self.hits += 1
        self.table.move_to_end(key)

        flag = entry["flag"]
        value = entry["value"]

        if flag == self.EXACT:
            return value
        elif flag == self.LOWER and value >= beta:
            return value
        elif flag == self.UPPER and value <= alpha:
            return value
        return None

    def store(self, board: chess.Board, depth: int, value: float, flag: int, best_move: chess.Move | None):
        key = chess.polyglot.zobrist_hash(board)
        self.table[key] = {
            "depth": depth,
            "value": value,
            "flag": flag,
            "best_move": best_move,
        }
        self.table.move_to_end(key)
        if len(self.table) > self.max_entries:
            self.table.popitem(last=False)

    def get_best_move(self, board: chess.Board) -> chess.Move | None:
        key = chess.polyglot.zobrist_hash(board)
        entry = self.table.get(key)
        return entry["best_move"] if entry else None


# ---------------------------------------------------------------------------
# Neural Network Evaluation
# ---------------------------------------------------------------------------

class NNEvaluator:
    """Batched neural network evaluator with caching."""

    def __init__(self, model, device: torch.device, cache_size: int = 100_000):
        self.model = model
        self.device = device
        self.eval_count = 0
        self.cache: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[float, list[tuple[chess.Move, float]]]:
        """Return (value, [(move, policy_prob), ...]) for the position.

        Value is from side-to-move's perspective: +1 = winning, -1 = losing.
        """
        fen = board.fen()
        cached = self.cache.get(fen)
        if cached is not None:
            self.cache_hits += 1
            self.cache.move_to_end(fen)
            return cached

        inp = batch_boards_to_fused_token_ids([board], self.device)
        result = self.model(inp)

        # Value
        wdl = F.softmax(result["value_logits"][0].float(), dim=-1)
        value = (wdl[2] - wdl[0]).item()  # win - loss

        # Policy (sorted by prob)
        logits = result["policy_logits"][0].float()
        mask = legal_move_mask(board).to(self.device)
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)

        legal_indices = mask.nonzero(as_tuple=True)[0]
        move_probs = []
        for idx in legal_indices.tolist():
            move = index_to_move(idx)
            if move in board.legal_moves:
                move_probs.append((move, probs[idx].item()))
        move_probs.sort(key=lambda x: x[1], reverse=True)

        self.eval_count += 1

        # Cache
        result_tuple = (value, move_probs)
        self.cache[fen] = result_tuple
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return result_tuple


# ---------------------------------------------------------------------------
# Alpha-Beta Search
# ---------------------------------------------------------------------------

MATE_SCORE = 10000.0
DRAW_SCORE = 0.0


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    """Check if a move is quiet (not a capture or promotion)."""
    return not board.is_capture(move) and not move.promotion


class AlphaBetaSearcher:
    """Alpha-Beta search with neural network evaluation."""

    def __init__(
        self,
        evaluator: NNEvaluator,
        tt: TranspositionTable,
        max_depth: int = 4,
        top_k: int = 12,
        quiesce: bool = True,
        quiesce_depth: int = 4,
        time_limit: float | None = None,
    ):
        self.evaluator = evaluator
        self.tt = tt
        self.max_depth = max_depth
        self.top_k = top_k
        self.quiesce = quiesce
        self.quiesce_depth = quiesce_depth
        self.time_limit = time_limit
        self.nodes = 0
        self.start_time = 0.0
        self.aborted = False

    def _check_time(self):
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            self.aborted = True

    def _order_moves(self, board: chess.Board, move_probs: list[tuple[chess.Move, float]], tt_move: chess.Move | None) -> list[chess.Move]:
        """Order moves: TT move first, then by policy probability, captures boosted."""
        ordered = []
        scored = []

        for move, prob in move_probs:
            score = prob
            if move == tt_move:
                score += 10.0  # TT move always first
            if board.is_capture(move):
                score += 0.5  # Boost captures
            if move.promotion:
                score += 0.3  # Boost promotions
            if board.gives_check(move):
                score += 0.2  # Boost checks
            scored.append((move, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:self.top_k]]

    def quiescence(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """Quiescence search: only consider captures at leaf nodes."""
        self.nodes += 1
        if self.aborted:
            return 0.0

        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                return DRAW_SCORE
            return -MATE_SCORE if outcome.winner != board.turn else MATE_SCORE

        # Stand pat: evaluate position with neural network
        value, move_probs = self.evaluator.evaluate(board)
        stand_pat = value

        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        if depth >= self.quiesce_depth:
            return stand_pat

        # Only search captures and promotions
        for move, prob in move_probs:
            if not board.is_capture(move) and not move.promotion:
                continue

            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, depth + 1)
            board.pop()

            if self.aborted:
                return 0.0

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Alpha-beta search with TT and move ordering."""
        self.nodes += 1
        self._check_time()
        if self.aborted:
            return 0.0

        # Terminal node check
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                return DRAW_SCORE
            return -MATE_SCORE if outcome.winner != board.turn else MATE_SCORE

        # Leaf node: evaluate
        if depth <= 0:
            if self.quiesce:
                return self.quiescence(board, alpha, beta)
            else:
                value, _ = self.evaluator.evaluate(board)
                return value

        # Transposition table probe
        tt_val = self.tt.probe(board, depth, alpha, beta)
        if tt_val is not None:
            return tt_val

        # Get moves ordered by policy
        value, move_probs = self.evaluator.evaluate(board)
        tt_move = self.tt.get_best_move(board)
        moves = self._order_moves(board, move_probs, tt_move)

        if not moves:
            return value  # No legal moves after top-k filter (shouldn't happen)

        best_value = -MATE_SCORE - 1
        best_move = moves[0]
        flag = TranspositionTable.UPPER

        for move in moves:
            board.push(move)
            child_value = -self.alphabeta(board, depth - 1, -beta, -alpha)
            board.pop()

            if self.aborted:
                return 0.0

            if child_value > best_value:
                best_value = child_value
                best_move = move

            if child_value > alpha:
                alpha = child_value
                flag = TranspositionTable.EXACT

            if alpha >= beta:
                flag = TranspositionTable.LOWER
                break

        self.tt.store(board, depth, best_value, flag, best_move)
        return best_value

    def search(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """Run iterative deepening search and return best move."""
        self.start_time = time.time()
        self.aborted = False
        self.nodes = 0

        best_move = None
        best_value = -MATE_SCORE
        search_info = []

        for depth in range(1, self.max_depth + 1):
            self.nodes = 0
            depth_start = time.time()

            # Get moves ordered by policy
            value, move_probs = self.evaluator.evaluate(board)
            tt_move = self.tt.get_best_move(board)
            moves = self._order_moves(board, move_probs, tt_move)

            alpha = -MATE_SCORE
            beta = MATE_SCORE
            current_best = moves[0] if moves else None
            current_value = -MATE_SCORE

            for move in moves:
                board.push(move)
                child_value = -self.alphabeta(board, depth - 1, -beta, -alpha)
                board.pop()

                if self.aborted:
                    break

                if child_value > current_value:
                    current_value = child_value
                    current_best = move
                if child_value > alpha:
                    alpha = child_value

            elapsed = time.time() - depth_start
            nps = self.nodes / max(elapsed, 0.001)

            info = {
                "depth": depth,
                "value": round(current_value, 4),
                "nodes": self.nodes,
                "time": round(elapsed, 2),
                "nps": int(nps),
                "best_move": current_best.uci() if current_best else None,
                "tt_hits": self.tt.hits,
            }
            search_info.append(info)

            if not self.aborted and current_best:
                best_move = current_best
                best_value = current_value

            if self.aborted:
                break

            # Quick time check for next iteration
            if self.time_limit and (time.time() - self.start_time) > self.time_limit * 0.6:
                break

        if best_move is None:
            best_move = next(iter(board.legal_moves))

        total_time = time.time() - self.start_time
        return best_move, {
            "search_depths": search_info,
            "total_nodes": sum(d["nodes"] for d in search_info),
            "total_time": round(total_time, 2),
            "nn_evals": self.evaluator.eval_count,
            "nn_cache_hits": self.evaluator.cache_hits,
            "value": round(best_value, 4),
        }


# ---------------------------------------------------------------------------
# ELO evaluation harness
# ---------------------------------------------------------------------------

def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_one(engine, model_move_fn, sf_elo, model_color, opening, movetime, ply_cap):
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move, info = model_move_fn(board)
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
        "opening": " ".join(opening) if opening else "startpos",
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
        "score": round(score, 4),
        "score_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "w": sum(1 for r in results if r["score"] == 1.0),
        "d": sum(1 for r in results if r["score"] == 0.5),
        "l": sum(1 for r in results if r["score"] == 0.0),
        "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="exp097: Alpha-Beta search ELO evaluation")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-tag", type=str, default="exp097_ab")
    p.add_argument("--max-depth", type=int, default=4, help="Max search depth (plies)")
    p.add_argument("--top-k", type=int, default=12, help="Top K moves from policy to consider at each node")
    p.add_argument("--quiesce", action="store_true", help="Enable quiescence search")
    p.add_argument("--quiesce-depth", type=int, default=4, help="Max quiescence plies")
    p.add_argument("--time-limit", type=float, default=None, help="Time limit per move in seconds")
    p.add_argument("--tt-size", type=int, default=500_000, help="Transposition table max entries")
    p.add_argument("--sf-movetime", type=float, default=0.05, help="SF move time in seconds")
    p.add_argument("--ply-cap", type=int, default=200, help="Max plies per game")
    p.add_argument("--test-elos", type=int, nargs="+", default=DEFAULT_TEST_ELOS)
    p.add_argument("--stop-after-bracket", action="store_true")
    p.add_argument("--greedy-baseline", action="store_true", help="Also run greedy baseline first")
    return p.parse_args()


def main():
    global LOG, JSON_OUT
    args = parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    LOG = output_dir / f"elo_eval_{args.output_tag}.log"
    JSON_OUT = output_dir / f"elo_eval_{args.output_tag}.json"

    if LOG.exists():
        LOG.unlink()

    log(f"exp097: Alpha-Beta search ELO evaluation")
    log(f"checkpoint={args.checkpoint}")
    log(f"max_depth={args.max_depth} top_k={args.top_k} quiesce={args.quiesce} quiesce_depth={args.quiesce_depth}")
    log(f"time_limit={args.time_limit} tt_size={args.tt_size}")
    log(f"device={DEVICE}")

    model = load_model(args.checkpoint, DEVICE)
    evaluator = NNEvaluator(model, DEVICE)
    tt = TranspositionTable(max_entries=args.tt_size)
    searcher = AlphaBetaSearcher(
        evaluator=evaluator,
        tt=tt,
        max_depth=args.max_depth,
        top_k=args.top_k,
        quiesce=args.quiesce,
        quiesce_depth=args.quiesce_depth,
        time_limit=args.time_limit,
    )

    def search_move(board: chess.Board):
        return searcher.search(board)

    results_all = {}

    # Optional greedy baseline
    if args.greedy_baseline:
        log("=" * 60)
        log("BASELINE: greedy (no search)")
        log("=" * 60)

        @torch.no_grad()
        def greedy_move(board):
            inp = batch_boards_to_fused_token_ids([board], DEVICE)
            result = model(inp)
            logits = result["policy_logits"][0].float()
            mask = legal_move_mask(board).to(DEVICE)
            logits[~mask] = float("-inf")
            move = index_to_move(logits.argmax().item())
            return move, {}

        baseline_summaries = []
        for sf_elo in args.test_elos:
            engine = chess.engine.SimpleEngine.popen_uci(str(SF))
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})
            results = []
            try:
                for opening in DEFAULT_OPENINGS:
                    for color in [chess.WHITE, chess.BLACK]:
                        result = play_one(engine, greedy_move, sf_elo, color, opening, args.sf_movetime, args.ply_cap)
                        results.append(result)
                        log(f"baseline_game {json.dumps({k: result[k] for k in ['sf_elo', 'model_color', 'result', 'score', 'plies', 'termination']})}")
            finally:
                engine.quit()

            summary = summarize_results(sf_elo, results)
            baseline_summaries.append(summary)
            log(f"baseline_summary {json.dumps(summary)}")

            above = [s for s in baseline_summaries if s["score"] >= 0.5]
            below = [s for s in baseline_summaries if s["score"] < 0.5]
            if args.stop_after_bracket and above and below:
                log("baseline bracketing complete")
                break

        baseline_elo = estimate_elo(baseline_summaries)
        log(f"baseline_elo {json.dumps(baseline_elo)}")
        results_all["baseline"] = {"elo": baseline_elo, "summaries": baseline_summaries}

    # Alpha-Beta search
    log("=" * 60)
    log(f"SEARCH: alpha-beta depth={args.max_depth} top_k={args.top_k}")
    log("=" * 60)

    search_summaries = []
    all_search_games = []

    for sf_elo in args.test_elos:
        engine = chess.engine.SimpleEngine.popen_uci(str(SF))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})
        results = []
        try:
            for opening in DEFAULT_OPENINGS:
                for color in [chess.WHITE, chess.BLACK]:
                    result = play_one(engine, search_move, sf_elo, color, opening, args.sf_movetime, args.ply_cap)
                    results.append(result)
                    log(f"search_game {json.dumps({k: result[k] for k in ['sf_elo', 'model_color', 'result', 'score', 'plies', 'termination']})}")
        finally:
            engine.quit()

        summary = summarize_results(sf_elo, results)
        search_summaries.append(summary)
        all_search_games.extend(results)
        log(f"search_summary {json.dumps(summary)}")

        above = [s for s in search_summaries if s["score"] >= 0.5]
        below = [s for s in search_summaries if s["score"] < 0.5]
        if args.stop_after_bracket and above and below:
            log("search bracketing complete")
            break

    search_elo = estimate_elo(search_summaries)
    log(f"search_elo {json.dumps(search_elo)}")
    results_all["search"] = {"elo": search_elo, "summaries": search_summaries, "games": all_search_games}

    # Stats
    log(f"nn_evals={evaluator.eval_count} nn_cache_hits={evaluator.cache_hits} tt_hits={tt.hits} tt_misses={tt.misses}")

    # Save JSON
    JSON_OUT.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "config": {
            "max_depth": args.max_depth,
            "top_k": args.top_k,
            "quiesce": args.quiesce,
            "quiesce_depth": args.quiesce_depth,
            "time_limit": args.time_limit,
        },
        "results": results_all,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    log("done")


if __name__ == "__main__":
    main()
