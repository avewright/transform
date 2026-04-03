"""exp104: Policy-guided alpha-beta search.

Source: stockfish_md/improvements.md §6 (Neural Network Move Ordering)
        stockfish_md/architecture.md (Alpha-beta, LMR, killer moves, TT)
        alphazero/possible_improvements.md §11 (Batched inference)

Hypothesis: Our policy head (43% top-1, 76% top-3 accuracy) is strong enough
to serve as an excellent move ordering heuristic for alpha-beta search. Stockfish's
strength comes largely from search + move ordering, not just evaluation.

Key insight: exp094 showed that MCTS with our value head DESTROYS ELO (-344 pts).
But alpha-beta is different from MCTS:
  - Alpha-beta with good move ordering prunes most of the tree
  - The value head only needs to be ORDINAL (rank positions), not cardinal
  - Even a noisy value head helps when you search 3-4 ply deep and prune well
  - Policy ordering means we search the best move first → massive beta cutoffs

Architecture:
  1. Root: get policy logits, sort legal moves by probability
  2. Expand top-K moves (K=8 at root, K=5 at depth>0)
  3. Alpha-beta with:
     - Policy-ordered move generation (best moves first)
     - Transposition table (Zobrist hashing via python-chess)
     - Null move pruning (skip a move to get a fast bound)
     - Late move reductions (search unpromising moves at reduced depth)
  4. Leaf evaluation: value head WDL → scalar

This is a SEARCH experiment, not a training experiment. It modifies eval-time
behavior only. Can be tested immediately on the best checkpoint.

Usage:
    # Quick demo game
    python experiments/exp104_policy_guided_search.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode demo

    # ELO evaluation
    python experiments/exp104_policy_guided_search.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode elo --depth 3

    # Depth sweep
    python experiments/exp104_policy_guided_search.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode sweep
"""

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.polyglot
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, index_to_move, legal_move_mask, move_to_index
from chess_features import batch_boards_to_fused_token_ids

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transposition Table ──

TT_EXACT = 0
TT_LOWER = 1  # beta cutoff (fail-high)
TT_UPPER = 2  # fail-low

@dataclass
class TTEntry:
    key: int        # Zobrist hash
    depth: int
    score: float
    flag: int       # TT_EXACT, TT_LOWER, TT_UPPER
    best_move: chess.Move = None


class TranspositionTable:
    """Simple hash table for positions already searched."""
    def __init__(self, max_size=1_000_000):
        self.max_size = max_size
        self.table = {}

    def probe(self, board, depth):
        key = chess.polyglot.zobrist_hash(board)
        entry = self.table.get(key)
        if entry is None or entry.depth < depth:
            return None
        return entry

    def store(self, board, depth, score, flag, best_move=None):
        key = chess.polyglot.zobrist_hash(board)
        # Always replace (depth-preferred)
        existing = self.table.get(key)
        if existing is None or depth >= existing.depth:
            self.table[key] = TTEntry(key=key, depth=depth, score=score, flag=flag, best_move=best_move)
        # Evict if too large (simple: just clear)
        if len(self.table) > self.max_size:
            self.table.clear()

    def clear(self):
        self.table.clear()


# ── Policy-guided alpha-beta searcher ──

class PolicyAlphaBetaSearcher:
    """Alpha-beta search with neural network move ordering.
    
    Move ordering priority (from stockfish_md/architecture.md):
    1. TT best move (if available)
    2. Sorted by policy prior probability (descending)
    
    Pruning techniques:
    - Alpha-beta pruning (standard)
    - Late Move Reductions: moves ranked low by policy searched at depth-1
    - Null move pruning: skip a turn to get a quick bound (depth >= 3)
    - Stand-pat in quiescence (like Stockfish's quiescence search)
    """

    def __init__(self, model, device, max_depth=3, root_k=10, child_k=6,
                 use_null_move=True, use_lmr=True, use_quiescence=True,
                 quiescence_depth=2):
        self.model = model
        self.device = device
        self.max_depth = max_depth
        self.root_k = root_k      # how many moves to search at root
        self.child_k = child_k    # how many moves at child nodes
        self.use_null_move = use_null_move
        self.use_lmr = use_lmr
        self.use_quiescence = use_quiescence
        self.quiescence_depth = quiescence_depth
        self.tt = TranspositionTable()
        self.stats = {"nodes": 0, "tt_hits": 0, "tt_cutoffs": 0,
                      "null_cutoffs": 0, "lmr_researches": 0, "beta_cutoffs": 0}
        self._eval_cache = {}

    @torch.no_grad()
    def _evaluate_position(self, board):
        """Neural network evaluation. Returns score from white's perspective in [-1, 1]."""
        fen = board.fen()
        if fen in self._eval_cache:
            return self._eval_cache[fen]

        board_input = batch_boards_to_fused_token_ids([board], self.device)
        result = self.model(board_input)

        wdl = F.softmax(result["value_logits"][0].float(), dim=-1)
        # WDL is White-absolute: [P(W wins), P(draw), P(W loses)]
        # Convert to side-to-move perspective
        white_score = (wdl[0] - wdl[2]).item()
        score = white_score if board.turn == chess.WHITE else -white_score

        self._eval_cache[fen] = score
        if len(self._eval_cache) > 500_000:
            self._eval_cache.clear()
        return score

    @torch.no_grad()
    def _get_ordered_moves(self, board, k=None):
        """Get legal moves ordered by policy prior probability.
        
        Returns list of (move, prior_prob) tuples, sorted best-first.
        """
        board_input = batch_boards_to_fused_token_ids([board], self.device)
        result = self.model(board_input)
        logits = result["policy_logits"][0].float()
        mask = legal_move_mask(board).to(self.device)
        logits[~mask] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        legal_indices = mask.nonzero(as_tuple=True)[0]
        legal_probs = probs[legal_indices]

        # Sort by probability descending
        sorted_idx = legal_probs.argsort(descending=True)
        
        n = k if k else len(legal_indices)
        moves = []
        for i in range(min(n, len(sorted_idx))):
            idx = legal_indices[sorted_idx[i]].item()
            move = index_to_move(idx)
            prob = legal_probs[sorted_idx[i]].item()
            moves.append((move, prob))
        
        return moves

    def _is_capture(self, board, move):
        return board.is_capture(move)

    def _is_check(self, board, move):
        board.push(move)
        in_check = board.is_check()
        board.pop()
        return in_check

    def _quiescence(self, board, alpha, beta, depth=0):
        """Quiescence search: only search captures and checks to avoid horizon effect."""
        self.stats["nodes"] += 1

        if board.is_game_over(claim_draw=True):
            return self._terminal_score(board)

        # Stand pat: evaluate current position
        stand_pat = self._evaluate_position(board)
        if board.turn == chess.BLACK:
            stand_pat = -stand_pat
        # From side-to-move perspective
        stm_score = stand_pat if board.turn == chess.WHITE else -stand_pat

        if stm_score >= beta:
            return stm_score
        if stm_score > alpha:
            alpha = stm_score

        if depth >= self.quiescence_depth:
            return stm_score

        # Only search captures (and promotions)
        for move in board.legal_moves:
            if not (board.is_capture(move) or move.promotion):
                continue

            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, depth + 1)
            board.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def _terminal_score(self, board):
        """Score for terminal positions from side-to-move perspective."""
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if board.turn == chess.WHITE else 1.0
        return 0.0  # draw

    def _alpha_beta(self, board, depth, alpha, beta, is_root=False):
        """Alpha-beta search with policy-ordered moves.
        
        Returns score from side-to-move perspective.
        """
        self.stats["nodes"] += 1

        # Terminal
        if board.is_game_over(claim_draw=True):
            return self._terminal_score(board)

        # Depth 0 → leaf evaluation (or quiescence)
        if depth <= 0:
            if self.use_quiescence:
                return self._quiescence(board, alpha, beta)
            eval_score = self._evaluate_position(board)
            # Convert to side-to-move perspective
            return eval_score if board.turn == chess.WHITE else -eval_score

        # TT probe
        tt_entry = self.tt.probe(board, depth)
        tt_move = None
        if tt_entry is not None:
            self.stats["tt_hits"] += 1
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                if tt_entry.flag == TT_EXACT:
                    self.stats["tt_cutoffs"] += 1
                    return tt_entry.score
                elif tt_entry.flag == TT_LOWER:
                    alpha = max(alpha, tt_entry.score)
                elif tt_entry.flag == TT_UPPER:
                    beta = min(beta, tt_entry.score)
                if alpha >= beta:
                    self.stats["tt_cutoffs"] += 1
                    return tt_entry.score

        # Null move pruning (stockfish_md/architecture.md)
        # Skip if in check, at root, or shallow depth
        if (self.use_null_move and depth >= 3 and not is_root 
                and not board.is_check() and board.has_legal_en_passant is not True):
            # Make a null move (pass)
            board.push(chess.Move.null())
            null_score = -self._alpha_beta(board, depth - 3, -beta, -beta + 0.001)
            board.pop()
            if null_score >= beta:
                self.stats["null_cutoffs"] += 1
                return null_score

        # Get policy-ordered moves
        k = self.root_k if is_root else self.child_k
        ordered_moves = self._get_ordered_moves(board, k=k)

        # Put TT move first if available
        if tt_move is not None:
            ordered_moves = [(tt_move, 1.0)] + [(m, p) for m, p in ordered_moves if m != tt_move]

        best_score = -2.0
        best_move = None
        tt_flag = TT_UPPER  # assume fail-low until proven otherwise

        for move_idx, (move, prior) in enumerate(ordered_moves):
            # Late Move Reductions (stockfish_md)
            # Moves ranked low by policy (move_idx >= 3) and not captures/checks → reduced depth
            reduction = 0
            if (self.use_lmr and move_idx >= 3 and depth >= 2
                    and not self._is_capture(board, move)
                    and not self._is_check(board, move)):
                reduction = 1

            board.push(move)
            
            if reduction > 0:
                # Reduced depth search first
                score = -self._alpha_beta(board, depth - 1 - reduction, -beta, -alpha)
                # If it looks promising, re-search at full depth
                if score > alpha:
                    self.stats["lmr_researches"] += 1
                    score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
            else:
                score = -self._alpha_beta(board, depth - 1, -beta, -alpha)
            
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                tt_flag = TT_EXACT

            if alpha >= beta:
                tt_flag = TT_LOWER
                self.stats["beta_cutoffs"] += 1
                break

        # Store in TT
        if best_move:
            self.tt.store(board, depth, best_score, tt_flag, best_move)

        return best_score

    def search(self, board, depth=None):
        """Run iterative deepening alpha-beta search.
        
        Returns (best_move, info_dict).
        """
        depth = depth or self.max_depth
        self.stats = {"nodes": 0, "tt_hits": 0, "tt_cutoffs": 0,
                      "null_cutoffs": 0, "lmr_researches": 0, "beta_cutoffs": 0}

        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            return None, {"error": "no legal moves"}
        if len(legal_moves) == 1:
            return legal_moves[0], {"top_moves": [(legal_moves[0].uci(), "only")]}

        best_move = None
        best_score = -2.0

        # Iterative deepening (stockfish_md/architecture.md §1)
        for d in range(1, depth + 1):
            # Aspiration window (stockfish_md)
            if d >= 3 and best_score > -1.5:
                window = 0.15
                alpha = best_score - window
                beta = best_score + window
                score = self._search_root(board, d, alpha, beta)
                # Re-search with full window if outside aspiration
                if score <= alpha or score >= beta:
                    score = self._search_root(board, d, -2.0, 2.0)
            else:
                score = self._search_root(board, d, -2.0, 2.0)

            if score is not None:
                best_score = score

        # Get the best move from the final search
        ordered = self._get_ordered_moves(board, k=self.root_k)
        
        # Re-evaluate each top move at full depth to find the actual best
        move_scores = []
        for move, prior in ordered:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -2.0, 2.0)
            board.pop()
            move_scores.append((move, score, prior))

        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_move = move_scores[0][0]
        best_score = move_scores[0][1]

        # Build info
        top_moves = [(m.uci(), f"s={s:.3f}/p={p:.1%}") for m, s, p in move_scores[:5]]

        # WDL from value head
        board_input = batch_boards_to_fused_token_ids([board], self.device)
        result = self.model(board_input)
        wdl = F.softmax(result["value_logits"][0].float(), dim=-1).tolist()

        info = {
            "top_moves": top_moves,
            "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]},
            "search_stats": {
                "depth": depth,
                "best_score": best_score,
                **self.stats,
            },
        }
        return best_move, info

    def _search_root(self, board, depth, alpha, beta):
        """Root search to update TT and get score."""
        return self._alpha_beta(board, depth, alpha, beta, is_root=True)


# ── Drop-in move function ──

_SEARCHER = None
_SEARCH_DEPTH = 3

def get_policy_ab_move(model, board, device, temperature=0.0):
    """Drop-in replacement for get_model_move_generic with alpha-beta search."""
    global _SEARCHER
    if _SEARCHER is None or _SEARCHER.model is not model:
        _SEARCHER = PolicyAlphaBetaSearcher(model, device, max_depth=_SEARCH_DEPTH)
    return _SEARCHER.search(board)


# ── ELO Evaluation ──

def run_elo_eval(model, device, sf_path, sf_levels, n_games=30, time_limit=0.05):
    import chess.engine
    
    results = {}
    for sf_elo in sf_levels:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
        
        wins = draws = losses = 0
        for game_i in range(n_games):
            board = chess.Board()
            model_color = chess.WHITE if game_i % 2 == 0 else chess.BLACK
            
            while not board.is_game_over(claim_draw=True):
                if board.fullmove_number > 200:
                    break
                if board.turn == model_color:
                    move, info = get_policy_ab_move(model, board, device)
                    if move is None:
                        break
                else:
                    sf_result = engine.play(board, chess.engine.Limit(time=time_limit))
                    move = sf_result.move
                board.push(move)
            
            result = board.result(claim_draw=True)
            if result == "1-0":
                if model_color == chess.WHITE:
                    wins += 1
                else:
                    losses += 1
            elif result == "0-1":
                if model_color == chess.BLACK:
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1
        
        engine.quit()
        score = (wins + 0.5 * draws) / max(n_games, 1)
        results[sf_elo] = {"wins": wins, "draws": draws, "losses": losses, "score": score}
        print(f"  vs SF {sf_elo}: +{wins}={draws}-{losses} ({score:.1%})")
        
        if losses > n_games * 0.8:
            print(f"  Stopping — clearly below SF {sf_elo}")
            break
    
    return results


# ── Demo ──

def demo_game(model, device, depth=3):
    """Self-play demo showing search in action."""
    board = chess.Board()
    searcher = PolicyAlphaBetaSearcher(model, device, max_depth=depth)
    
    move_num = 0
    while not board.is_game_over(claim_draw=True) and move_num < 80:
        t0 = time.time()
        move, info = searcher.search(board)
        elapsed = time.time() - t0
        stats = info.get("search_stats", {})
        side = "W" if board.turn == chess.WHITE else "B"
        prefix = f"{board.fullmove_number}." if board.turn == chess.WHITE else "  ..."
        
        print(f"{prefix} {move.uci()} [{side}] "
              f"(score={stats.get('best_score', 0):.3f}, "
              f"nodes={stats.get('nodes', 0)}, "
              f"tt_cuts={stats.get('tt_cutoffs', 0)}, "
              f"β_cuts={stats.get('beta_cutoffs', 0)}, "
              f"{elapsed:.1f}s)")
        
        board.push(move)
        move_num += 1
        
        # Clear TT periodically
        if move_num % 20 == 0:
            searcher.tt.clear()
    
    print(f"\nResult: {board.result(claim_draw=True)}")


# ── Main ──

def main():
    global _SEARCH_DEPTH

    import argparse
    parser = argparse.ArgumentParser(description="exp104: Policy-guided alpha-beta search")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["elo", "demo", "sweep", "compare"], default="demo")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--root-k", type=int, default=10)
    parser.add_argument("--child-k", type=int, default=6)
    parser.add_argument("--sf-path", type=str, default="stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
    parser.add_argument("--sf-levels", type=str, default="1320,1450,1600,1750")
    parser.add_argument("--n-games", type=int, default=30)
    parser.add_argument("--no-null-move", action="store_true")
    parser.add_argument("--no-lmr", action="store_true")
    parser.add_argument("--no-quiescence", action="store_true")
    args = parser.parse_args()

    _SEARCH_DEPTH = args.depth

    print(f"Loading model from {args.checkpoint}...")
    from play import load_model
    model = load_model(args.checkpoint, DEVICE)
    model.eval()
    print(f"  Loaded. Device: {DEVICE}")

    if args.mode == "demo":
        demo_game(model, DEVICE, depth=args.depth)

    elif args.mode == "elo":
        sf_levels = [int(x) for x in args.sf_levels.split(",")]
        print(f"\nELO eval: depth={args.depth}, root_k={args.root_k}, child_k={args.child_k}")
        results = run_elo_eval(model, DEVICE, args.sf_path, sf_levels, n_games=args.n_games)

    elif args.mode == "sweep":
        # Test different depths
        global _SEARCHER
        sf_levels = [int(x) for x in args.sf_levels.split(",")]

        print("=== Depth Sweep ===")
        for depth in [1, 2, 3, 4]:
            _SEARCH_DEPTH = depth
            _SEARCHER = None  # force rebuild
            print(f"\n--- depth={depth} ---")
            results = run_elo_eval(model, DEVICE, args.sf_path, sf_levels[:3], n_games=20)

    elif args.mode == "compare":
        sf_levels = [int(x) for x in args.sf_levels.split(",")]

        # Greedy (depth=0, just policy)
        print("\n=== Greedy (no search) ===")
        _SEARCH_DEPTH = 0
        _SEARCHER = None
        # For depth=0, just use get_model_move_generic
        from elo_eval_latest import get_model_move_generic
        
        import chess.engine
        for sf_elo in sf_levels[:3]:
            engine = chess.engine.SimpleEngine.popen_uci(args.sf_path)
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
            wins = draws = losses = 0
            for gi in range(20):
                board = chess.Board()
                mc = chess.WHITE if gi % 2 == 0 else chess.BLACK
                while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
                    if board.turn == mc:
                        m, _ = get_model_move_generic(model, board, DEVICE)
                    else:
                        m = engine.play(board, chess.engine.Limit(time=0.05)).move
                    board.push(m)
                r = board.result(claim_draw=True)
                if r == "1-0": wins += (1 if mc == chess.WHITE else 0); losses += (0 if mc == chess.WHITE else 1)
                elif r == "0-1": losses += (1 if mc == chess.WHITE else 0); wins += (0 if mc == chess.WHITE else 1)
                else: draws += 1
            engine.quit()
            print(f"  Greedy vs SF {sf_elo}: +{wins}={draws}-{losses}")

        # Alpha-beta
        print(f"\n=== Alpha-Beta depth={args.depth} ===")
        _SEARCH_DEPTH = args.depth
        _SEARCHER = None
        results = run_elo_eval(model, DEVICE, args.sf_path, sf_levels[:3], n_games=20)


if __name__ == "__main__":
    main()
