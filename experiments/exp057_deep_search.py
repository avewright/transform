"""exp057: Deep search — 2-ply alpha-beta with trained value head vs Stockfish.

Hypothesis: 2-ply alpha-beta search using the exp055 trained value head will
beat 1-ply reranking at gameplay vs Stockfish, especially at SF depth 2 where
1-ply strategies collapsed to 0% score in exp054.

Rationale: exp054 showed value_rerank_k5 doubled the score at SF d1 (37.5% vs
18.8% policy-only). But 1-ply can't anticipate opponent responses. 2-ply
(model move → opponent move → value) should help against SF d2 by considering
the opponent's best reply.

This experiment does NOT train. It loads the exp055 joint checkpoint and plays
games using multi-ply search strategies.

Experiment contract:
  - Hypothesis: 2-ply search > 1-ply reranking at gameplay
  - Metric: W/D/L vs Stockfish d1, d2, d3
  - Seeds: fixed opening book (same 8 openings as exp054)
  - Data: none (inference only)
  - Runtime target: <10 minutes
  - Device: CUDA (RTX 2000 Ada, 16GB)
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids, board_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp057_deep_search")

CHECKPOINT_CANDIDATES = [
    Path("outputs/exp055_joint_policy_value/joint_medium_s42.pt"),
    Path("outputs/exp053_scaled_spatial/spatial_medium_s42.pt"),
]

SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
SF_DEPTHS = [1, 2, 3]
GAMES_PER_DEPTH = 8
MAX_MOVES = 200

# Fixed opening book (same as exp054 for comparison)
OPENINGS = [
    [],                                        # starting position
    ["e2e4", "e7e5"],                          # open game
    ["d2d4", "d7d5"],                          # closed game
    ["e2e4", "c7c5"],                          # Sicilian
    ["d2d4", "g8f6", "c2c4", "e7e6"],          # Nimzo-Indian setup
    ["e2e4", "e7e5", "g1f3", "b8c6"],          # Four Knights setup
    ["d2d4", "d7d5", "c2c4"],                  # Queen's Gambit
    ["e2e4", "e7e6"],                          # French
]


# =====================================================================
# Model architecture (matches exp053/055 ChessTransformerV2)
# =====================================================================

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for i in range(VOCAB_SIZE):
        uci = IDX_TO_UCI[i]
        from_sqs.append(chess.parse_square(uci[:2]))
        to_sqs.append(chess.parse_square(uci[2:4]))
        promo_types.append(promo_map.get(uci[4:5], 0))
    return (
        torch.tensor(from_sqs, dtype=torch.long),
        torch.tensor(to_sqs, dtype=torch.long),
        torch.tensor(promo_types, dtype=torch.long),
    )


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=256):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

    def forward(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(cls_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


def build_model(hidden_dim, num_layers, num_heads, encoder_dim=256,
                head_dim=256, dropout=0.0):

    class ChessTransformerSearch(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
            self.input_proj = nn.Linear(encoder_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, 68, hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.policy_head = SpatialPolicyHead(
                hidden_dim, n_ctx_tokens=4, head_dim=head_dim,
            )
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
            )
            self.hidden_dim = hidden_dim

        def forward(self, board_input):
            tokens = self.encoder(board_input)
            hidden = self.input_proj(tokens)
            B = hidden.shape[0]
            cls = self.cls_token.expand(B, -1, -1)
            hidden = torch.cat([cls, hidden], dim=1)
            hidden = hidden + self.pos_embed
            hidden = self.transformer(hidden)
            hidden = self.norm(hidden)
            cls_hidden = hidden[:, 0, :]
            policy_logits = self.policy_head(hidden, cls_hidden)
            value_logits = self.value_head(cls_hidden)
            return {
                "policy_logits": policy_logits,
                "value_logits": value_logits,
            }

        @torch.no_grad()
        def get_policy_and_value(self, board, device):
            """Return (policy_probs, value_score) for a single board."""
            self.eval()
            board_input = batch_boards_to_token_ids([board], device)
            result = self(board_input)
            logits = result["policy_logits"][0]
            mask = legal_move_mask(board).to(device)
            logits[~mask] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            wdl = F.softmax(result["value_logits"][0], dim=-1)
            value = (wdl[0] - wdl[2]).item()
            return probs, value

        @torch.no_grad()
        def get_values_batch(self, boards, device):
            """Batch value evaluation. Returns list of floats in [-1, 1]."""
            self.eval()
            board_input = batch_boards_to_token_ids(boards, device)
            result = self(board_input)
            wdl = F.softmax(result["value_logits"], dim=-1)
            return (wdl[:, 0] - wdl[:, 2]).cpu().tolist()

        @torch.no_grad()
        def get_policy_topk(self, board, device, k=10):
            """Return top-k (move, prob) pairs as list of (chess.Move, float)."""
            self.eval()
            board_input = batch_boards_to_token_ids([board], device)
            result = self(board_input)
            logits = result["policy_logits"][0]
            mask = legal_move_mask(board).to(device)
            logits[~mask] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            topk = probs.topk(k)
            moves = []
            for idx, p in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
                m = index_to_move(idx)
                if m is not None and m in board.legal_moves:
                    moves.append((m, p))
            return moves

    return ChessTransformerSearch()


# =====================================================================
# Search strategies
# =====================================================================

def strategy_policy_argmax(model, board, device, **kw):
    """Baseline: pure policy argmax."""
    probs, _ = model.get_policy_and_value(board, device)
    idx = probs.argmax().item()
    return IDX_TO_UCI[idx]


def strategy_value_rerank_k5(model, board, device, **kw):
    """1-ply value reranking over top-5 (exp054 best strategy)."""
    candidates = model.get_policy_topk(board, device, k=5)
    if not candidates:
        return strategy_policy_argmax(model, board, device)

    child_boards = []
    moves = []
    for m, p in candidates:
        child = board.copy()
        child.push(m)
        child_boards.append(child)
        moves.append(m)

    # Evaluate children (from opponent perspective), negate
    child_values = model.get_values_batch(child_boards, device)
    our_values = [-v for v in child_values]

    best_idx = max(range(len(our_values)), key=lambda i: our_values[i])
    return moves[best_idx].uci()


def strategy_alphabeta_2ply(model, board, device, top_k=5, **kw):
    """2-ply alpha-beta: model's top-k → opponent's top-k → value eval.

    For each of our top-k moves, consider the opponent's best response
    (from their top-k), and pick the move that maximizes our worst case.
    """
    our_candidates = model.get_policy_topk(board, device, k=top_k)
    if not our_candidates:
        return strategy_policy_argmax(model, board, device)

    best_move = None
    best_value = -2.0

    for our_move, our_prob in our_candidates:
        child = board.copy()
        child.push(our_move)

        if child.is_game_over():
            result = child.result()
            if result == "1-0":
                v = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                v = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                v = 0.0
            if v > best_value:
                best_value = v
                best_move = our_move
            continue

        # Opponent's responses (top-k from opponent's policy)
        opp_candidates = model.get_policy_topk(child, device, k=top_k)
        if not opp_candidates:
            # Opponent has no good moves — evaluate this position
            _, v = model.get_policy_and_value(child, device)
            v = -v  # negate (value is from side-to-move = opponent)
            if v > best_value:
                best_value = v
                best_move = our_move
            continue

        # For each opponent response, evaluate the resulting position
        grandchild_boards = []
        opp_moves = []
        for opp_move, opp_prob in opp_candidates:
            gc = child.copy()
            gc.push(opp_move)
            grandchild_boards.append(gc)
            opp_moves.append(opp_move)

        # Batch evaluate all grandchild positions
        gc_values = model.get_values_batch(grandchild_boards, device)
        # These values are from our perspective (it's our turn in grandchild)

        # Check for terminal grandchild positions
        for i, gc in enumerate(grandchild_boards):
            if gc.is_game_over():
                result = gc.result()
                if result == "1-0":
                    gc_values[i] = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    gc_values[i] = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    gc_values[i] = 0.0

        # Opponent picks their best response (minimizes our value)
        min_value = min(gc_values)

        if min_value > best_value:
            best_value = min_value
            best_move = our_move

    return best_move.uci() if best_move else strategy_policy_argmax(model, board, device)


def strategy_alphabeta_2ply_wide(model, board, device, **kw):
    """2-ply with wider candidate set (top-10 for us, top-5 for opponent)."""
    return strategy_alphabeta_2ply(model, board, device, top_k=10)


def strategy_alphabeta_3ply(model, board, device, top_k=5, **kw):
    """3-ply: our move → opp response → our reply → value eval.

    Uses narrower branching (top-3 at ply 2-3) for speed.
    """
    our_candidates = model.get_policy_topk(board, device, k=top_k)
    if not our_candidates:
        return strategy_policy_argmax(model, board, device)

    best_move = None
    best_value = -2.0

    for our_move, _ in our_candidates:
        child = board.copy()
        child.push(our_move)

        if child.is_game_over():
            result = child.result()
            if result == "1-0":
                v = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                v = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                v = 0.0
            if v > best_value:
                best_value = v
                best_move = our_move
            continue

        # Opponent's top-3 responses
        opp_candidates = model.get_policy_topk(child, device, k=3)
        if not opp_candidates:
            _, v = model.get_policy_and_value(child, device)
            v = -v
            if v > best_value:
                best_value = v
                best_move = our_move
            continue

        worst_for_us = 2.0  # opponent picks their best (worst for us)
        for opp_move, _ in opp_candidates:
            gc = child.copy()
            gc.push(opp_move)

            if gc.is_game_over():
                result = gc.result()
                if result == "1-0":
                    v = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    v = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    v = 0.0
                worst_for_us = min(worst_for_us, v)
                continue

            # Our top-3 replies
            our_replies = model.get_policy_topk(gc, device, k=3)
            if not our_replies:
                _, v = model.get_policy_and_value(gc, device)
                worst_for_us = min(worst_for_us, v)
                continue

            # Evaluate our best reply
            reply_boards = []
            for reply_move, _ in our_replies:
                rb = gc.copy()
                rb.push(reply_move)
                reply_boards.append(rb)

            reply_values = model.get_values_batch(reply_boards, device)

            # Check terminals
            for i, rb in enumerate(reply_boards):
                if rb.is_game_over():
                    result = rb.result()
                    if result == "1-0":
                        reply_values[i] = 1.0 if board.turn == chess.WHITE else -1.0
                    elif result == "0-1":
                        reply_values[i] = -1.0 if board.turn == chess.WHITE else 1.0
                    else:
                        reply_values[i] = 0.0

            # Opponent evaluates from their perspective — negate our reply values
            # After 3 plies it's opponent's turn, so values are from opp perspective
            opp_sees = [-v for v in reply_values]
            # Opponent's best continuation = min of our values
            best_reply = max(reply_values)  # we pick our best reply
            worst_for_us = min(worst_for_us, best_reply)

        if worst_for_us > best_value:
            best_value = worst_for_us
            best_move = our_move

    return best_move.uci() if best_move else strategy_policy_argmax(model, board, device)


# =====================================================================
# Game playing
# =====================================================================

def play_game(model, device, strategy_fn, sf_depth, opening_moves,
              strategy_kw=None):
    from stockfish import Stockfish
    sf = Stockfish(SF_PATH, depth=sf_depth)
    strategy_kw = strategy_kw or {}

    results = []
    for model_color in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        move_list = []

        for uci in opening_moves:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)
                move_list.append(uci)

        while not board.is_game_over() and len(move_list) < MAX_MOVES:
            if board.turn == model_color:
                move_uci = strategy_fn(model, board, device, **strategy_kw)
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    move = list(board.legal_moves)[0]
                    move_uci = move.uci()
            else:
                sf.set_fen_position(board.fen())
                move_uci = sf.get_best_move()
                move = chess.Move.from_uci(move_uci)

            board.push(move)
            move_list.append(move_uci)

        result = board.result()
        if result == "1-0":
            outcome = 1.0 if model_color == chess.WHITE else 0.0
        elif result == "0-1":
            outcome = 0.0 if model_color == chess.WHITE else 1.0
        elif result == "1/2-1/2":
            outcome = 0.5
        else:
            outcome = 0.5

        results.append({
            "model_color": "white" if model_color == chess.WHITE else "black",
            "outcome": outcome,
            "result": result,
            "num_moves": len(move_list),
            "termination": board.outcome().termination.name if board.outcome() else "max_moves",
        })

    return results


# =====================================================================
# Main
# =====================================================================

def load_checkpoint(device):
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            print(f"  Loading checkpoint: {path}")
            state = torch.load(path, map_location=device, weights_only=True)
            hidden_dim = state["cls_token"].shape[-1]
            layer_keys = [k for k in state if "transformer.layers" in k
                          and "self_attn.in_proj_weight" in k]
            num_layers = len(layer_keys)
            num_heads = 8
            print(f"  Detected: {hidden_dim}d, {num_layers}L, {num_heads}H")

            model = build_model(
                hidden_dim=hidden_dim, num_layers=num_layers,
                num_heads=num_heads, head_dim=256,
            ).to(device)

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {[k.split('.')[0] for k in missing[:3]]}...")
            return model, path.name
    raise FileNotFoundError("No checkpoint found.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp057_deep_search")
    print(f"Hypothesis: 2-ply alpha-beta > 1-ply reranking at gameplay")

    print(f"\n[1/2] Loading model...")
    model, ckpt_name = load_checkpoint(device)
    model.eval()

    # Strategies ordered by search depth
    strategies = {
        "policy_argmax": {"fn": strategy_policy_argmax, "kw": {}},
        "value_rerank_k5": {"fn": strategy_value_rerank_k5, "kw": {}},
        "alphabeta_2ply_k5": {"fn": strategy_alphabeta_2ply, "kw": {"top_k": 5}},
        "alphabeta_2ply_k10": {"fn": strategy_alphabeta_2ply_wide, "kw": {}},
    }

    results = {
        "experiment": "exp057_deep_search",
        "hypothesis": "2-ply alpha-beta > 1-ply reranking",
        "checkpoint": ckpt_name,
        "baseline": "exp054: value_rerank_k5 d1=W0/D6/L2, d2=W0/D1/L7",
        "sf_depths": SF_DEPTHS,
        "games_per_depth": GAMES_PER_DEPTH,
        "strategies": {},
    }

    for sname, sspec in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Strategy: {sname}")
        print(f"{'='*60}")

        strat_results = {}
        for sf_depth in SF_DEPTHS:
            print(f"\n  vs Stockfish depth {sf_depth}:")
            depth_results = []
            wins = draws = losses = 0
            total_moves = 0

            for g in range(GAMES_PER_DEPTH // 2):
                opening = OPENINGS[g % len(OPENINGS)]
                t_game = time.time()
                game_results = play_game(
                    model, device, sspec["fn"], sf_depth, opening,
                    strategy_kw=sspec["kw"],
                )
                game_time = time.time() - t_game
                for gr in game_results:
                    depth_results.append(gr)
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0:
                        wins += 1
                    elif gr["outcome"] == 0.5:
                        draws += 1
                    else:
                        losses += 1

                print(f"    Games {g*2+1}-{g*2+2}: "
                      f"W{wins}/D{draws}/L{losses} "
                      f"(avg {total_moves/(g*2+2):.0f}mv, {game_time:.1f}s)")

            avg_moves = total_moves / max(len(depth_results), 1)
            score = (wins + 0.5 * draws) / max(len(depth_results), 1)
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
                "avg_moves": round(avg_moves),
                "games": depth_results,
            }
            print(f"  d{sf_depth} total: W{wins}/D{draws}/L{losses} "
                  f"(score={score:.1%}, avg={avg_moves:.0f}mv)")

        results["strategies"][sname] = strat_results

    total_time = time.time() - t_start
    results["timing"] = {"total_s": round(total_time)}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS SUMMARY: exp057_deep_search")
    print(f" Checkpoint: {ckpt_name}")
    print(f"{'='*60}")
    print(f"\n  {'Strategy':<25} {'SF d1':>12} {'SF d2':>12} {'SF d3':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for sname, sdata in results["strategies"].items():
        d1 = sdata.get("d1", {})
        d2 = sdata.get("d2", {})
        d3 = sdata.get("d3", {})
        print(f"  {sname:<25} "
              f"W{d1.get('wins',0)}/D{d1.get('draws',0)}/L{d1.get('losses',0):>3} "
              f"W{d2.get('wins',0)}/D{d2.get('draws',0)}/L{d2.get('losses',0):>3} "
              f"W{d3.get('wins',0)}/D{d3.get('draws',0)}/L{d3.get('losses',0):>3}")
    print(f"\n  Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
