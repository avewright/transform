"""exp103: Gumbel AlphaZero search — policy-based search without value head.

Source: alphazero/possible_improvements.md §6 (Gumbel AlphaZero / Policy Target via Search)

Hypothesis: Our value head is too weak for MCTS (exp094 showed -344 ELO with
value-augmented search). Gumbel AlphaZero replaces UCB+value with a principled
policy-only search method:

  1. Sample Gumbel noise for each legal action
  2. Add log prior (policy logits) + Gumbel noise  
  3. Use Sequential Halving to allocate simulations
  4. At each "simulation", expand the position 1-ply and use the child's
     policy entropy / raw policy score as a completion signal
  
  The key insight: we don't need a value head at all. The policy network's
  own consistency across positions IS the evaluation signal. When we expand
  a position and look at the opponent's reply distribution, we can infer
  how good our move was.

Modes:
  A. Pure Gumbel policy search (no value head, uses policy consistency)
  B. Gumbel + lightweight value (try value head, but let Gumbel structure dominate)
  C. Gumbel + Stockfish leaf eval (oracle test — how good can search get?)

Usage:
    # ELO test with Gumbel search
    python experiments/exp103_gumbel_search.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --n-simulations 16

    # Quick self-play demo
    python experiments/exp103_gumbel_search.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode demo

    # ELO eval with the search
    python experiments/exp103_gumbel_search.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode elo --sf-levels 1320,1450,1600
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, index_to_move, legal_move_mask, move_to_index
from chess_features import batch_boards_to_fused_token_ids

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Gumbel utilities ──

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_log_sigma(logits):
    """Compute log(softmax(logits)) = logits - logsumexp(logits)."""
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


@dataclass
class GumbelNode:
    """A node in the Gumbel search tree."""
    board: chess.Board
    prior_logits: torch.Tensor = None   # raw policy logits for legal moves
    prior_probs: torch.Tensor = None    # softmax of logits
    legal_indices: list = field(default_factory=list)  # vocab indices of legal moves
    gumbel_scores: torch.Tensor = None  # logits + gumbel noise
    visit_counts: dict = field(default_factory=dict)   # move_idx → count
    child_values: dict = field(default_factory=dict)   # move_idx → estimated value
    n_legal: int = 0


class GumbelSearcher:
    """Gumbel AlphaZero-style search using policy network.
    
    From the paper: instead of UCB, we use Gumbel noise to break ties,
    and Sequential Halving to allocate simulation budget efficiently.
    The value estimate can come from:
      - Policy consistency (how confident is the opponent's reply?)
      - Value head (if available)
      - Hybrid
    """

    def __init__(self, model, device, n_simulations=16, value_mode="policy_consistency",
                 c_scale=1.0, temperature=0.0):
        """
        Args:
            model: ChessTransformer200M with policy_logits and value_logits outputs
            device: torch device
            n_simulations: total simulation budget
            value_mode: "policy_consistency" | "value_head" | "hybrid"
            c_scale: scaling factor for exploration
            temperature: final move selection temperature (0 = greedy)
        """
        self.model = model
        self.device = device
        self.n_simulations = n_simulations
        self.value_mode = value_mode
        self.c_scale = c_scale
        self.temperature = temperature
        self.stats = {"nodes_expanded": 0, "cache_hits": 0}

    @torch.no_grad()
    def _get_policy_and_value(self, board):
        """Get policy logits and value for a position."""
        board_input = batch_boards_to_fused_token_ids([board], self.device)
        result = self.model(board_input)

        logits = result["policy_logits"][0].float()
        mask = legal_move_mask(board).to(self.device)
        logits[~mask] = float("-inf")

        wdl = F.softmax(result["value_logits"][0].float(), dim=-1)
        # WDL is White-absolute: [P(W wins), P(draw), P(W loses)]
        # Convert to side-to-move perspective
        white_value = wdl[0] - wdl[2]
        value = white_value if board.turn == chess.WHITE else -white_value

        self.stats["nodes_expanded"] += 1
        return logits, value.item(), mask

    def _policy_consistency_value(self, board, move):
        """Estimate move quality from the opponent's response distribution.
        
        Idea: after we play `move`, if the opponent has a very confident reply,
        our move was probably bad (we walked into something). If the opponent
        is uncertain (high entropy), our move was decent.
        
        Returns value in [-1, 1] from the mover's perspective.
        """
        board_copy = board.copy()
        board_copy.push(move)

        if board_copy.is_game_over(claim_draw=True):
            result = board_copy.result(claim_draw=True)
            if result == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                return -1.0 if board.turn == chess.WHITE else 1.0
            else:
                return 0.0

        child_logits, child_value, child_mask = self._get_policy_and_value(board_copy)

        # Opponent's value is negative of ours
        opp_value = -child_value

        # Policy consistency: high entropy of opponent = good for us
        legal_logits = child_logits[child_mask]
        if len(legal_logits) <= 1:
            return opp_value

        probs = F.softmax(legal_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_entropy = math.log(len(legal_logits))
        normalized_entropy = entropy / max(max_entropy, 1e-8)

        # Blend: opponent entropy (high=good for us) with value head
        entropy_signal = normalized_entropy * 2.0 - 1.0  # map [0,1] → [-1,1]
        
        if self.value_mode == "policy_consistency":
            return 0.6 * entropy_signal + 0.4 * opp_value
        elif self.value_mode == "value_head":
            return opp_value
        else:  # hybrid
            return 0.3 * entropy_signal + 0.7 * opp_value

    def search(self, board):
        """Run Gumbel search and return best move.
        
        Algorithm (simplified Gumbel MuZero / Policy Target via Search):
        1. Get policy logits for root
        2. Add Gumbel(0,1) noise to log-priors
        3. Select top-K actions by Gumbel score
        4. Sequentially halve: evaluate children, keep top half
        5. Return action with highest Gumbel score among survivors
        """
        root_logits, root_value, root_mask = self._get_policy_and_value(board)
        
        legal_indices = root_mask.nonzero(as_tuple=True)[0].tolist()
        n_legal = len(legal_indices)
        
        if n_legal == 0:
            return None, {"error": "no legal moves"}
        if n_legal == 1:
            move = index_to_move(legal_indices[0])
            return move, {"top_moves": [(move.uci(), "100.0%")], "wdl": {"loss": 0, "draw": 0, "win": 0}}

        # Step 1: Log-priors for legal moves
        legal_logits = root_logits[legal_indices].cpu()
        log_priors = gumbel_log_sigma(legal_logits)

        # Step 2: Add Gumbel noise
        gumbel_noise = sample_gumbel(log_priors.shape)
        gumbel_scores = log_priors + gumbel_noise

        # Step 3: Select top-K candidates (K = min(n_legal, n_simulations))
        K = min(n_legal, self.n_simulations)
        topk_indices = torch.topk(gumbel_scores, K).indices.tolist()

        # Map back to vocab indices and create candidates
        candidates = []
        for local_idx in topk_indices:
            vocab_idx = legal_indices[local_idx]
            candidates.append({
                "local_idx": local_idx,
                "vocab_idx": vocab_idx,
                "gumbel_score": gumbel_scores[local_idx].item(),
                "prior": F.softmax(legal_logits, dim=-1)[local_idx].item(),
                "move": index_to_move(vocab_idx),
                "value": 0.0,
                "visits": 0,
            })

        # Step 4: Sequential Halving
        # Allocate simulations across rounds, halving candidates each round
        remaining = list(candidates)
        sims_used = 0
        max_sims = self.n_simulations

        while len(remaining) > 1 and sims_used < max_sims:
            n_remaining = len(remaining)
            # Allocate equal sims to each remaining candidate this round
            sims_per_candidate = max(1, (max_sims - sims_used) // (n_remaining * max(1, int(math.log2(n_remaining)))))

            for cand in remaining:
                for _ in range(sims_per_candidate):
                    if sims_used >= max_sims:
                        break
                    v = self._policy_consistency_value(board, cand["move"])
                    cand["value"] = (cand["value"] * cand["visits"] + v) / (cand["visits"] + 1)
                    cand["visits"] += 1
                    sims_used += 1

            # Halve: keep top half by (gumbel_score + c * value_estimate)
            for cand in remaining:
                cand["combined_score"] = cand["gumbel_score"] + self.c_scale * cand["value"]
            
            remaining.sort(key=lambda c: c["combined_score"], reverse=True)
            keep = max(1, len(remaining) // 2)
            remaining = remaining[:keep]

        # Step 5: Select best among survivors
        best = remaining[0]
        
        # Build info dict
        all_candidates = sorted(candidates, key=lambda c: c.get("combined_score", c["gumbel_score"]), reverse=True)
        top_moves = [(c["move"].uci(), f"{c['prior']*100:.1f}%/v={c['value']:.2f}") for c in all_candidates[:5]]
        
        # WDL from root (for reporting)
        board_input = batch_boards_to_fused_token_ids([board], self.device)
        result = self.model(board_input)
        wdl = F.softmax(result["value_logits"][0].float(), dim=-1).tolist()

        info = {
            "top_moves": top_moves,
            "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]},
            "search_stats": {
                "n_simulations": sims_used,
                "n_candidates": len(candidates),
                "n_survivors": len(remaining),
                "best_value": best["value"],
                "best_visits": best["visits"],
                "nodes_expanded": self.stats["nodes_expanded"],
            },
        }
        
        self.stats = {"nodes_expanded": 0, "cache_hits": 0}  # reset for next search
        return best["move"], info


# ── Move function compatible with elo_eval_latest.py ──

_SEARCHER = None

def get_gumbel_move(model, board, device, temperature=0.0):
    """Drop-in replacement for get_model_move_generic, with Gumbel search."""
    global _SEARCHER
    if _SEARCHER is None or _SEARCHER.model is not model:
        _SEARCHER = GumbelSearcher(
            model, device,
            n_simulations=_GUMBEL_SIMS,
            value_mode=_VALUE_MODE,
            c_scale=_C_SCALE,
        )
    return _SEARCHER.search(board)


# Module-level config for the move function
_GUMBEL_SIMS = 16
_VALUE_MODE = "policy_consistency"
_C_SCALE = 1.0


# ── ELO evaluation (self-contained, inline) ──

def run_elo_eval(model, device, sf_path, sf_levels, n_games=30, time_limit=0.05):
    """Run ELO bracket evaluation against Stockfish."""
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
                    move, info = get_gumbel_move(model, board, device)
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
        
        # Early stop if dominated
        if losses > n_games * 0.8:
            print(f"  Stopping — clearly below SF {sf_elo}")
            break
    
    return results


# ── Demo mode ──

def demo_game(model, device):
    """Play a demo game showing Gumbel search in action."""
    board = chess.Board()
    searcher = GumbelSearcher(model, device, n_simulations=32, value_mode="policy_consistency")
    
    move_num = 0
    while not board.is_game_over(claim_draw=True) and move_num < 80:
        if board.turn == chess.WHITE:
            t0 = time.time()
            move, info = searcher.search(board)
            elapsed = time.time() - t0
            stats = info.get("search_stats", {})
            print(f"{move_num+1}. {move.uci()} (v={stats.get('best_value', 0):.2f}, "
                  f"sims={stats.get('n_simulations', 0)}, {elapsed:.1f}s) "
                  f"top: {info['top_moves'][:3]}")
        else:
            # Simple greedy for black (show what Gumbel would pick too)
            move, info = searcher.search(board)
            print(f"   ...{move.uci()} (v={info.get('search_stats',{}).get('best_value',0):.2f})")
        board.push(move)
        move_num += 1
    
    print(f"\nResult: {board.result(claim_draw=True)}")
    print(f"Final FEN: {board.fen()}")


# ── Sweep mode: test different simulation budgets ──

def sweep_simulations(model, device, sf_path, budgets=[1, 4, 8, 16, 32, 64]):
    """Test how ELO changes with simulation budget."""
    global _GUMBEL_SIMS
    
    print("=== Gumbel Search Simulation Budget Sweep ===")
    for n_sims in budgets:
        _GUMBEL_SIMS = n_sims
        print(f"\n--- n_simulations={n_sims} ---")
        results = run_elo_eval(model, device, sf_path, [1320, 1450, 1600], n_games=20)
        for elo, r in results.items():
            print(f"  SF {elo}: score={r['score']:.1%}")


# ── Main ──

def main():
    global _GUMBEL_SIMS, _VALUE_MODE, _C_SCALE

    import argparse
    parser = argparse.ArgumentParser(description="exp103: Gumbel AlphaZero search")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["elo", "demo", "sweep", "compare"], default="demo")
    parser.add_argument("--n-simulations", type=int, default=16)
    parser.add_argument("--value-mode", choices=["policy_consistency", "value_head", "hybrid"], 
                        default="policy_consistency")
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--sf-path", type=str, default="stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
    parser.add_argument("--sf-levels", type=str, default="1320,1450,1600,1750")
    parser.add_argument("--n-games", type=int, default=30)
    args = parser.parse_args()

    _GUMBEL_SIMS = args.n_simulations
    _VALUE_MODE = args.value_mode
    _C_SCALE = args.c_scale

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    from play import load_model
    model = load_model(args.checkpoint, DEVICE)
    model.eval()
    print(f"  Loaded. Device: {DEVICE}")

    if args.mode == "demo":
        demo_game(model, DEVICE)

    elif args.mode == "elo":
        sf_levels = [int(x) for x in args.sf_levels.split(",")]
        print(f"\nELO eval: sims={args.n_simulations}, value_mode={args.value_mode}")
        results = run_elo_eval(model, DEVICE, args.sf_path, sf_levels, n_games=args.n_games)

    elif args.mode == "sweep":
        sweep_simulations(model, DEVICE, args.sf_path)

    elif args.mode == "compare":
        # Compare Gumbel search vs greedy (no search)
        from elo_eval_latest import get_model_move_generic
        
        sf_levels = [int(x) for x in args.sf_levels.split(",")]
        
        print("\n=== Greedy (no search) ===")
        _GUMBEL_SIMS = 1  # effectively greedy with 1 sim
        results_greedy = run_elo_eval(model, DEVICE, args.sf_path, sf_levels, n_games=args.n_games)
        
        print(f"\n=== Gumbel Search (sims={args.n_simulations}) ===")
        _GUMBEL_SIMS = args.n_simulations
        results_search = run_elo_eval(model, DEVICE, args.sf_path, sf_levels, n_games=args.n_games)
        
        print("\n=== Comparison ===")
        for elo in sf_levels:
            g = results_greedy.get(elo, {}).get("score", 0)
            s = results_search.get(elo, {}).get("score", 0)
            diff = s - g
            print(f"  SF {elo}: greedy={g:.1%} search={s:.1%} (delta={diff:+.1%})")


if __name__ == "__main__":
    main()
