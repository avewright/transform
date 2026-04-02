"""exp105: Batched deterministic policy lookahead.

Source: Lessons from exp103 (Gumbel) and exp104 (alpha-beta) results.

FINDING: Gumbel search HURTS ELO (-25% at SF 1320). The problem is NOISE, not search.
The policy head (43% top-1, 76% top-3) is strong enough that adding Gumbel noise
promotes inferior moves. The value head is too weak for MCTS (-344 ELO in exp094).

This experiment takes a fundamentally different approach:
  1. NO noise — deterministic move selection
  2. BATCHED inference — evaluate all children in ONE forward pass 
  3. Policy-policy lookahead — use the POLICY head itself as the evaluation signal
  4. Practical speed — ~2-3 forward passes per move (vs 8-57 in exp103/104)

Algorithm (Policy Mirror Search):
  Given position P with side-to-move S:
  1. Get policy logits π(P). Select top-K legal moves {m1, ..., mK}
  2. For each mi, push mi to get positions {P1, ..., PK}
  3. BATCH evaluate all K positions in ONE forward pass → get opponent policies π(Pi)
  4. For each Pi, get opponent's top move mi' and confidence conf_i
  5. Push mi' to get {P1', ..., PK'} (our position after opponent's best reply)
  6. BATCH evaluate all K positions → get our policies π(Pi')
  7. Score each root move mi by: 
     score(mi) = α * original_prior(mi) 
               + β * our_confidence_after(Pi')
               - γ * opponent_confidence(Pi)
  8. Pick move with highest score

Why this works:
  - If we play a move and the opponent has a clear best reply AND we're still confident
    about our response → good move (we saw further into the game)
  - If we play a move and the opponent has a clear best reply but our response is uncertain
    → dangerous move (we walked into something)
  - If opponent has no clear reply → good move (opponent is confused)
  - NO noise, NO randomness — strictly improves on greedy when it works, falls back to
    greedy when lookahead is ambiguous

Speed: 3 forward passes per move (root + batch children + batch grandchildren)
       with K=8: ~1.5s per move (vs 0.4s greedy, ~5-8s Gumbel, ~30s+ alpha-beta)

Usage:
    python experiments/exp105_batched_policy_lookahead.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode demo
    python experiments/exp105_batched_policy_lookahead.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode elo --sf-levels 1320,1450,1600,1750
    python experiments/exp105_batched_policy_lookahead.py --checkpoint outputs/exp093_ema_curriculum_d8/ema_model.pt --mode quick-ab --sf-level 1320
"""

import math
import os
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, index_to_move, legal_move_mask, move_to_index
from chess_features import batch_boards_to_fused_token_ids

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyMirrorSearcher:
    """Deterministic batched policy lookahead.
    
    Looks 2 plies ahead using only the policy head:
    - Ply 0: our candidate moves (top-K from policy)
    - Ply 1: opponent's best reply to each (batched)
    - Ply 2: our confidence in the resulting position (batched)
    
    The "mirror" metaphor: we look at the REFLECTION of our move through
    the opponent's lens, then evaluate OUR position in the mirror.
    """

    def __init__(self, model, device, top_k=8, alpha=0.5, beta=0.3, gamma=0.2,
                 use_value=False, value_weight=0.3):
        """
        Args:
            model: ChessTransformer200M
            device: torch device
            top_k: number of candidate moves to evaluate
            alpha: weight for original policy prior
            beta: weight for our confidence after opponent's reply
            gamma: weight for opponent's confidence (negative signal)
            use_value: also use value head in scoring (careful — known to be weak)
            value_weight: how much to weight value head vs policy signals
        """
        self.model = model
        self.device = device
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_value = use_value
        self.value_weight = value_weight
        self.stats = {"root_evals": 0, "batch_evals": 0, "total_positions": 0}

    @torch.no_grad()
    def _batch_evaluate(self, boards):
        """Evaluate multiple boards in a single batched forward pass.
        
        Returns dict with:
            - policy_logits: (B, VOCAB_SIZE)
            - value_logits: (B, 3)  [loss, draw, win] from white's perspective
            - top_moves: list of (move, prob) for each board
            - confidences: (B,) max policy probability for each board
        """
        if not boards:
            return None

        board_input = batch_boards_to_fused_token_ids(boards, self.device)
        result = self.model(board_input)
        
        policy_logits = result["policy_logits"].float()  # (B, VOCAB_SIZE)
        value_logits = result["value_logits"].float()     # (B, 3)
        
        self.stats["batch_evals"] += 1
        self.stats["total_positions"] += len(boards)

        # Mask illegal moves and compute confidences
        confidences = []
        top_moves = []
        for i, board in enumerate(boards):
            mask = legal_move_mask(board).to(self.device)
            policy_logits[i][~mask] = float("-inf")
            
            probs = F.softmax(policy_logits[i], dim=-1)
            max_prob = probs.max().item()
            confidences.append(max_prob)
            
            # Top move
            top_idx = probs.argmax().item()
            top_move = index_to_move(top_idx)
            top_moves.append((top_move, max_prob))

        return {
            "policy_logits": policy_logits,
            "value_logits": value_logits,
            "top_moves": top_moves,
            "confidences": torch.tensor(confidences),
        }

    @torch.no_grad()
    def search(self, board):
        """Run 2-ply policy mirror search.
        
        Returns (best_move, info_dict).
        """
        self.stats = {"root_evals": 0, "batch_evals": 0, "total_positions": 0}
        
        legal = list(board.legal_moves)
        if len(legal) == 0:
            return None, {"error": "no legal moves"}
        if len(legal) == 1:
            return legal[0], {"top_moves": [(legal[0].uci(), "only")], 
                              "wdl": {"loss": 0, "draw": 0, "win": 0}}

        # Step 1: Root evaluation
        root_result = self._batch_evaluate([board])
        root_logits = root_result["policy_logits"][0]
        root_probs = F.softmax(root_logits, dim=-1)
        
        # Get WDL for reporting
        wdl = F.softmax(root_result["value_logits"][0], dim=-1).tolist()
        
        # Select top-K candidates
        mask = legal_move_mask(board).to(self.device)
        legal_indices = mask.nonzero(as_tuple=True)[0]
        legal_probs = root_probs[legal_indices]
        k = min(self.top_k, len(legal_indices))
        topk = legal_probs.topk(k)
        
        candidates = []
        child_boards = []
        
        for i in range(k):
            idx = legal_indices[topk.indices[i]].item()
            move = index_to_move(idx)
            prior = topk.values[i].item()
            
            # Push move to get child position
            child = board.copy()
            child.push(move)
            
            if child.is_game_over(claim_draw=True):
                # Terminal — score directly
                result_str = child.result(claim_draw=True)
                if result_str == "1-0":
                    term_score = 1.0 if board.turn == chess.WHITE else -1.0
                elif result_str == "0-1":
                    term_score = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    term_score = 0.0
                candidates.append({
                    "move": move,
                    "prior": prior,
                    "opp_conf": 0.0,
                    "our_conf": 1.0 if abs(term_score) > 0.5 else 0.3,
                    "value_score": term_score,
                    "terminal": True,
                })
            else:
                candidates.append({
                    "move": move,
                    "prior": prior,
                    "child_board": child,
                    "terminal": False,
                })
                child_boards.append(child)

        if not child_boards:
            # All moves are terminal (extremely rare)
            best = max(candidates, key=lambda c: c.get("value_score", 0))
            return best["move"], {"top_moves": [(c["move"].uci(), f"p={c['prior']:.1%}") for c in candidates[:5]],
                                   "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]}}

        # Step 2: Batch evaluate children (opponent's perspective)
        child_result = self._batch_evaluate(child_boards)
        
        # Build grandchild boards (opponent plays best reply)
        grandchild_boards = []
        child_idx = 0
        for cand in candidates:
            if cand["terminal"]:
                continue
            
            opp_move, opp_conf = child_result["top_moves"][child_idx]
            cand["opp_conf"] = opp_conf
            cand["opp_move"] = opp_move
            
            # Value from opponent's eval (for side-to-move)
            if self.use_value:
                v_logits = child_result["value_logits"][child_idx]
                v_probs = F.softmax(v_logits, dim=-1)
                # From our perspective: negate opponent's value assessment
                if board.turn == chess.WHITE:
                    # We're white, child is black's turn
                    # value_logits are from white's POV in our model
                    opp_value = (v_probs[2] - v_probs[0]).item()  # white's value = our value
                else:
                    opp_value = (v_probs[0] - v_probs[2]).item()  # we're black = invert
                cand["opp_value"] = opp_value

            # Push opponent's best reply
            gc = cand["child_board"].copy()
            gc.push(opp_move)
            
            if gc.is_game_over(claim_draw=True):
                result_str = gc.result(claim_draw=True)
                if result_str == "1-0":
                    cand["our_conf"] = 1.0 if board.turn == chess.WHITE else 0.0
                elif result_str == "0-1":
                    cand["our_conf"] = 0.0 if board.turn == chess.WHITE else 1.0
                else:
                    cand["our_conf"] = 0.3
                cand["gc_terminal"] = True
            else:
                grandchild_boards.append(gc)
                cand["gc_terminal"] = False
            
            child_idx += 1

        # Step 3: Batch evaluate grandchildren (our perspective again)
        if grandchild_boards:
            gc_result = self._batch_evaluate(grandchild_boards)
            
            gc_idx = 0
            for cand in candidates:
                if cand["terminal"] or cand.get("gc_terminal", False):
                    continue
                
                _, our_conf = gc_result["top_moves"][gc_idx]
                cand["our_conf"] = our_conf
                
                if self.use_value:
                    v_logits = gc_result["value_logits"][gc_idx]
                    v_probs = F.softmax(v_logits, dim=-1)
                    if board.turn == chess.WHITE:
                        cand["gc_value"] = (v_probs[2] - v_probs[0]).item()
                    else:
                        cand["gc_value"] = (v_probs[0] - v_probs[2]).item()
                
                gc_idx += 1

        # Step 4: Score each candidate
        for cand in candidates:
            if cand["terminal"]:
                cand["score"] = cand.get("value_score", 0) * 10.0  # strongly prefer winning moves
                continue
            
            prior = cand["prior"]
            opp_conf = cand.get("opp_conf", 0.5)
            our_conf = cand.get("our_conf", 0.5)
            
            # Policy-based score:
            # High prior + high our_conf after reply - high opponent conf = good
            policy_score = (self.alpha * prior + 
                          self.beta * our_conf - 
                          self.gamma * opp_conf)
            
            if self.use_value:
                # Blend with value head
                value_score = cand.get("gc_value", cand.get("opp_value", 0.0))
                cand["score"] = (1.0 - self.value_weight) * policy_score + self.value_weight * value_score
            else:
                cand["score"] = policy_score

        # Select best
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        
        # If top-1 greedy move is within epsilon of the best searched move, prefer the
        # prior (avoid changing moves when search signal is weak)
        greedy_cand = max(candidates, key=lambda c: c["prior"])
        if greedy_cand != best and best["score"] - greedy_cand["score"] < 0.02:
            best = greedy_cand  # fall back to greedy when search signal is ambiguous

        top_moves = [(c["move"].uci(), f"s={c['score']:.3f}/p={c['prior']:.1%}") 
                     for c in candidates[:5]]
        
        info = {
            "top_moves": top_moves,
            "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]},
            "search_stats": {
                "candidates": len(candidates),
                "forward_passes": self.stats["batch_evals"],
                "positions_evaluated": self.stats["total_positions"],
                "best_score": best["score"],
                "greedy_score": greedy_cand["score"],
                "changed_from_greedy": best["move"] != greedy_cand["move"],
            },
        }
        return best["move"], info


# ── Move function interface ──

_SEARCHER = None

def get_mirror_move(model, board, device, temperature=0.0):
    """Drop-in replacement for get_model_move_generic."""
    global _SEARCHER
    if _SEARCHER is None or _SEARCHER.model is not model:
        _SEARCHER = PolicyMirrorSearcher(model, device, top_k=_TOP_K, 
                                          alpha=_ALPHA, beta=_BETA, gamma=_GAMMA,
                                          use_value=_USE_VALUE)
    return _SEARCHER.search(board)

_TOP_K = 8
_ALPHA = 0.5
_BETA = 0.3
_GAMMA = 0.2
_USE_VALUE = False


# ── Game playing ──

def play_games(move_fn, label, sf_path, sf_elo, n_games=10, model=None):
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    
    w = d = l = 0
    total_time = 0
    changes = 0
    
    for gi in range(n_games):
        board = chess.Board()
        mc = chess.WHITE if gi % 2 == 0 else chess.BLACK
        moves = 0
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
            if board.turn == mc:
                t0 = time.time()
                m, info = move_fn(board)
                total_time += time.time() - t0
                if info.get("search_stats", {}).get("changed_from_greedy"):
                    changes += 1
                if m is None:
                    break
            else:
                m = engine.play(board, chess.engine.Limit(time=0.05)).move
            board.push(m)
            moves += 1
        result = board.result(claim_draw=True)
        side = "W" if mc == chess.WHITE else "B"
        if result == "1-0":
            if mc == chess.WHITE: w += 1
            else: l += 1
        elif result == "0-1":
            if mc == chess.BLACK: w += 1
            else: l += 1
        else:
            d += 1
        print(f"  {label} game {gi+1}: {result} ({side}) {moves} moves", flush=True)
    
    engine.quit()
    score = (w + 0.5 * d) / n_games
    avg_t = total_time / max(n_games, 1)
    print(f"{label} vs SF {sf_elo}: +{w}={d}-{l} ({score:.0%}) avg={avg_t:.1f}s/game changes={changes}", flush=True)
    return w, d, l, score


def main():
    global _TOP_K, _ALPHA, _BETA, _GAMMA, _USE_VALUE, _SEARCHER

    import argparse
    parser = argparse.ArgumentParser(description="exp105: Batched policy mirror search")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["demo", "elo", "quick-ab", "sweep"], default="demo")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for policy prior")
    parser.add_argument("--beta", type=float, default=0.3, help="weight for our post-reply confidence")
    parser.add_argument("--gamma", type=float, default=0.2, help="weight for opponent confidence (penalty)")
    parser.add_argument("--use-value", action="store_true", help="also use value head (risky)")
    parser.add_argument("--sf-path", type=str, default="stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")
    parser.add_argument("--sf-levels", type=str, default="1320,1450,1600,1750")
    parser.add_argument("--sf-level", type=int, default=1320)
    parser.add_argument("--n-games", type=int, default=10)
    args = parser.parse_args()

    _TOP_K = args.top_k
    _ALPHA = args.alpha
    _BETA = args.beta
    _GAMMA = args.gamma
    _USE_VALUE = args.use_value

    print(f"Loading model from {args.checkpoint}...", flush=True)
    from play import load_model
    model = load_model(args.checkpoint, DEVICE)
    model.eval()
    print(f"  Loaded on {DEVICE}. top_k={_TOP_K}, alpha={_ALPHA}, beta={_BETA}, gamma={_GAMMA}", flush=True)

    if args.mode == "demo":
        board = chess.Board()
        searcher = PolicyMirrorSearcher(model, DEVICE, top_k=args.top_k,
                                         alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                         use_value=args.use_value)
        move_num = 0
        while not board.is_game_over(claim_draw=True) and move_num < 80:
            t0 = time.time()
            move, info = searcher.search(board)
            elapsed = time.time() - t0
            stats = info.get("search_stats", {})
            side = "W" if board.turn == chess.WHITE else "B"
            pfx = f"{board.fullmove_number}." if board.turn == chess.WHITE else "  ..."
            changed = "*" if stats.get("changed_from_greedy") else " "
            print(f"{pfx} {move.uci()} [{side}]{changed} "
                  f"({elapsed:.1f}s, score={stats.get('best_score', 0):.3f}) "
                  f"{info['top_moves'][:3]}")
            board.push(move)
            move_num += 1
        print(f"\nResult: {board.result(claim_draw=True)}")

    elif args.mode == "quick-ab":
        # Quick A/B: greedy vs mirror search at one SF level
        from move_vocab import legal_move_mask as lmm
        
        @torch.no_grad()
        def greedy_fn(board):
            bi = batch_boards_to_fused_token_ids([board], DEVICE)
            r = model(bi)
            logits = r["policy_logits"][0].float()
            mask = lmm(board).to(DEVICE)
            logits[~mask] = float("-inf")
            move = index_to_move(logits.argmax().item())
            wdl = F.softmax(r["value_logits"][0].float(), dim=-1).tolist()
            return move, {"wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]}, "search_stats": {}}
        
        searcher = PolicyMirrorSearcher(model, DEVICE, top_k=args.top_k,
                                         alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                         use_value=args.use_value)
        def mirror_fn(board):
            return searcher.search(board)
        
        sf_elo = args.sf_level
        n = args.n_games
        
        print(f"\n=== GREEDY vs SF {sf_elo} ===", flush=True)
        gw, gd, gl, gscore = play_games(greedy_fn, "Greedy", args.sf_path, sf_elo, n)
        
        print(f"\n=== MIRROR (k={args.top_k}) vs SF {sf_elo} ===", flush=True)
        mw, md, ml, mscore = play_games(mirror_fn, f"Mirror-{args.top_k}", args.sf_path, sf_elo, n)
        
        print(f"\n=== SUMMARY vs SF {sf_elo} ===")
        print(f"  Greedy:  +{gw}={gd}-{gl} ({gscore:.0%})")
        print(f"  Mirror:  +{mw}={md}-{ml} ({mscore:.0%}) delta={mscore-gscore:+.0%}")

    elif args.mode == "elo":
        sf_levels = [int(x) for x in args.sf_levels.split(",")]
        searcher = PolicyMirrorSearcher(model, DEVICE, top_k=args.top_k,
                                         alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                         use_value=args.use_value)
        def mirror_fn(board):
            return searcher.search(board)
        for sf_elo in sf_levels:
            print(f"\n--- vs SF {sf_elo} ---", flush=True)
            w, d, l, score = play_games(mirror_fn, f"Mirror-{args.top_k}", args.sf_path, sf_elo, args.n_games)
            if l > args.n_games * 0.8:
                print("  Dominated — stopping.")
                break

    elif args.mode == "sweep":
        # Sweep alpha/beta/gamma parameters
        sf_elo = args.sf_level
        n = args.n_games
        
        configs = [
            (0.7, 0.2, 0.1, "high-prior"),
            (0.5, 0.3, 0.2, "balanced"),
            (0.3, 0.4, 0.3, "high-lookahead"),
            (0.5, 0.0, 0.0, "greedy-equiv"),  # should match greedy
            (0.0, 0.5, 0.5, "pure-lookahead"),
        ]
        
        for alpha, beta, gamma, label in configs:
            _ALPHA = alpha
            _BETA = beta
            _GAMMA = gamma
            _SEARCHER = None
            print(f"\n=== {label} (a={alpha}, b={beta}, g={gamma}) vs SF {sf_elo} ===", flush=True)
            
            searcher = PolicyMirrorSearcher(model, DEVICE, top_k=args.top_k,
                                             alpha=alpha, beta=beta, gamma=gamma)
            def mirror_fn(board, s=searcher):
                return s.search(board)
            play_games(mirror_fn, label, args.sf_path, sf_elo, n)


if __name__ == "__main__":
    main()
