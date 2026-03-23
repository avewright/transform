"""exp054: Search baseline — policy + value + shallow search vs Stockfish.

Hypothesis: Adding 1-ply or 2-ply search on top of the best policy model
will beat policy-only at game play, even if the value head is untrained
(random). A trained value head (exp055) should then beat the untrained one.

This experiment does NOT train. It loads a trained checkpoint from exp052 or
exp053 and plays games against Stockfish at depths 1-3 using four strategies:
  A) policy argmax (baseline)
  B) policy top-k + 1-ply value reranking (untrained value head)
  C) policy top-k + 1-ply Stockfish eval (oracle ceiling)
  D) simple MCTS with policy prior + value backup (untrained)

Primary metric: win/draw/loss record vs Stockfish at each depth.
Secondary: mean game length, illegal move rate (should be 0 with masking).

Experiment contract:
  - Hypothesis: search > policy argmax at gameplay
  - Metric: W/D/L vs Stockfish d1, d2, d3
  - Seeds: fixed opening book (same 8 openings for all strategies)
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

OUTPUT_DIR = Path("outputs/exp054_search_baseline")

# Try to find the best checkpoint (prefer joint-trained for value head)
CHECKPOINT_CANDIDATES = [
    Path("outputs/exp055_joint_policy_value/joint_medium_s42.pt"),
    Path("outputs/exp055_joint_policy_value/joint_medium_s123.pt"),
    Path("outputs/exp053_scaled_spatial/spatial_medium_s42.pt"),
    Path("outputs/exp053_scaled_spatial/spatial_medium_s123.pt"),
    Path("outputs/exp052_head_comparison_v2/spatial_s42.pt"),
    Path("outputs/exp052_head_comparison_v2/spatial_s314.pt"),
]

SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
SF_DEPTHS = [1, 2, 3]
GAMES_PER_DEPTH = 8
TOP_K = 5              # for search strategies
MAX_MOVES = 200        # per game
SEARCH_DEPTH = 1       # default search ply

# Fixed opening book for reproducible comparison
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
# Model architecture (matches exp052/exp053 ChessTransformerV2)
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
    """Build ChessTransformerV2-compatible model with value head."""

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
                "cls_hidden": cls_hidden,
            }

        @torch.no_grad()
        def get_policy(self, board, device):
            """Return masked policy probs and top-k moves for a single board."""
            self.eval()
            board_input = batch_boards_to_token_ids([board], device)
            result = self(board_input)
            logits = result["policy_logits"][0]
            mask = legal_move_mask(board).to(device)
            logits[~mask] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            return probs

        @torch.no_grad()
        def get_value(self, board, device):
            """Return value scalar in [-1, 1] from side-to-move perspective.
            Uses WDL head: value = P(win) - P(loss)."""
            self.eval()
            board_input = batch_boards_to_token_ids([board], device)
            result = self(board_input)
            wdl = F.softmax(result["value_logits"][0], dim=-1)
            return (wdl[0] - wdl[2]).item()  # win - loss

        @torch.no_grad()
        def get_values_batch(self, boards, device):
            """Batch value evaluation. Returns list of floats in [-1, 1]."""
            self.eval()
            board_input = batch_boards_to_token_ids(boards, device)
            result = self(board_input)
            wdl = F.softmax(result["value_logits"], dim=-1)
            return ((wdl[:, 0] - wdl[:, 2]).cpu().tolist())

    return ChessTransformerSearch()


# =====================================================================
# Search strategies
# =====================================================================

def strategy_policy_argmax(model, board, device, **kw):
    """A) Pure policy argmax — no search."""
    probs = model.get_policy(board, device)
    idx = probs.argmax().item()
    return index_to_move(idx)


def strategy_value_rerank(model, board, device, top_k=TOP_K, **kw):
    """B) Top-k policy moves, 1-ply lookahead with value head reranking."""
    probs = model.get_policy(board, device)
    topk_idx = probs.topk(top_k).indices.cpu().tolist()

    child_boards = []
    valid_moves = []
    for idx in topk_idx:
        move = index_to_move(idx)
        if move is None:
            continue
        try:
            m = chess.Move.from_uci(move)
            if m not in board.legal_moves:
                continue
        except ValueError:
            continue
        child = board.copy()
        child.push(m)
        child_boards.append(child)
        valid_moves.append(m)

    if not valid_moves:
        # fallback to policy argmax
        return strategy_policy_argmax(model, board, device)

    # Evaluate children from opponent's perspective, negate
    child_values = model.get_values_batch(child_boards, device)
    # Negate because value is from side-to-move perspective (opponent after our move)
    our_values = [-v for v in child_values]

    best_idx = max(range(len(our_values)), key=lambda i: our_values[i])
    return valid_moves[best_idx].uci()


def strategy_sf_oracle(model, board, device, sf_engine=None, sf_depth=8, top_k=TOP_K, **kw):
    """C) Top-k policy moves, 1-ply lookahead with Stockfish eval (oracle ceiling)."""
    from stockfish import Stockfish

    if sf_engine is None:
        sf_engine = kw.get("_sf_engine")
    if sf_engine is None:
        sf_engine = Stockfish(SF_PATH, depth=sf_depth)

    probs = model.get_policy(board, device)
    topk_idx = probs.topk(top_k).indices.cpu().tolist()

    best_move = None
    best_eval = -99999

    for idx in topk_idx:
        move_uci = index_to_move(idx)
        if move_uci is None:
            continue
        try:
            m = chess.Move.from_uci(move_uci)
            if m not in board.legal_moves:
                continue
        except ValueError:
            continue

        child = board.copy()
        child.push(m)
        sf_engine.set_fen_position(child.fen())
        ev = sf_engine.get_evaluation()
        # Eval is from side-to-move (opponent); negate for us
        if ev["type"] == "cp":
            score = -ev["value"]
        elif ev["type"] == "mate":
            score = -10000 * (1 if ev["value"] > 0 else -1)
        else:
            score = 0

        if score > best_eval:
            best_eval = score
            best_move = move_uci

    return best_move or strategy_policy_argmax(model, board, device)


def strategy_mcts_simple(model, board, device, simulations=50, top_k=TOP_K,
                         c_puct=1.5, **kw):
    """D) Simple MCTS with policy prior + value backup."""
    probs = model.get_policy(board, device)
    topk = probs.topk(top_k)
    topk_idx = topk.indices.cpu().tolist()
    topk_probs = topk.values.cpu().tolist()

    # Build candidate moves
    moves = []
    prior = []
    for idx, p in zip(topk_idx, topk_probs):
        move_uci = index_to_move(idx)
        if move_uci is None:
            continue
        try:
            m = chess.Move.from_uci(move_uci)
            if m not in board.legal_moves:
                continue
        except ValueError:
            continue
        moves.append(m)
        prior.append(p)

    if not moves:
        return strategy_policy_argmax(model, board, device)

    # Normalize priors
    total_p = sum(prior)
    prior = [p / total_p for p in prior]

    n_visits = [0] * len(moves)
    total_value = [0.0] * len(moves)

    for _ in range(simulations):
        # UCB1 selection
        total_n = sum(n_visits) + 1
        ucb_scores = []
        for i in range(len(moves)):
            if n_visits[i] == 0:
                ucb_scores.append(float("inf"))
            else:
                q = total_value[i] / n_visits[i]
                u = c_puct * prior[i] * math.sqrt(total_n) / (1 + n_visits[i])
                ucb_scores.append(q + u)

        best_i = max(range(len(ucb_scores)), key=lambda i: ucb_scores[i])
        child = board.copy()
        child.push(moves[best_i])

        # Value backup (1-ply only — evaluate child position)
        if child.is_game_over():
            result = child.result()
            if result == "1-0":
                v = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                v = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                v = 0.0
        else:
            opp_value = model.get_value(child, device)
            v = -opp_value  # negate for our perspective

        n_visits[best_i] += 1
        total_value[best_i] += v

    # Pick most-visited move
    best_i = max(range(len(moves)), key=lambda i: n_visits[i])
    return moves[best_i].uci()


# =====================================================================
# Game playing
# =====================================================================

def play_game(model, device, strategy_fn, sf_depth, opening_moves,
              strategy_kw=None):
    """Play one game: model (white) vs Stockfish (black), then swap."""
    from stockfish import Stockfish

    sf = Stockfish(SF_PATH, depth=sf_depth)
    strategy_kw = strategy_kw or {}
    strategy_kw["_sf_engine"] = sf

    results = []
    for model_color in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        move_list = []

        # Play opening moves
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
                    # fallback to first legal move
                    move = list(board.legal_moves)[0]
                    move_uci = move.uci()
            else:
                sf.set_fen_position(board.fen())
                move_uci = sf.get_best_move()
                move = chess.Move.from_uci(move_uci)

            board.push(move)
            move_list.append(move_uci)

        # Determine result from model's perspective
        result = board.result()
        if result == "1-0":
            outcome = 1.0 if model_color == chess.WHITE else 0.0
        elif result == "0-1":
            outcome = 0.0 if model_color == chess.WHITE else 1.0
        elif result == "1/2-1/2":
            outcome = 0.5
        else:
            outcome = 0.5  # adjudication

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
    """Find and load the best available checkpoint."""
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            print(f"  Loading checkpoint: {path}")
            state = torch.load(path, map_location=device, weights_only=True)

            # Detect model size from checkpoint keys
            # Check hidden_dim from cls_token shape
            hidden_dim = state["cls_token"].shape[-1]
            # Check num_layers from transformer keys
            layer_keys = [k for k in state if "transformer.layers" in k and "self_attn.in_proj_weight" in k]
            num_layers = len(layer_keys)
            # Check num_heads from attention
            # For now, default to 8 heads
            num_heads = 8
            head_dim = 256

            print(f"  Detected: {hidden_dim}d, {num_layers}L, {num_heads}H")

            model = build_model(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
            ).to(device)

            # Load with strict=False to allow value_head mismatch
            # (checkpoint may not have value_head if from exp052)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys (expected — untrained): "
                      f"{[k.split('.')[0] for k in missing[:3]]}...")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:3]}...")

            return model, path.name
    raise FileNotFoundError("No checkpoint found. Run exp052 or exp053 first.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp054_search_baseline")
    print(f"Hypothesis: search > policy argmax at gameplay vs Stockfish")

    # Load model
    print(f"\n[1/3] Loading model...")
    model, ckpt_name = load_checkpoint(device)
    model.eval()

    strategies = {
        "policy_argmax": {"fn": strategy_policy_argmax, "kw": {}},
        "value_rerank_k5": {"fn": strategy_value_rerank, "kw": {"top_k": 5}},
        "value_rerank_k10": {"fn": strategy_value_rerank, "kw": {"top_k": 10}},
        "mcts_50": {"fn": strategy_mcts_simple, "kw": {"simulations": 50, "top_k": 10}},
    }

    # Only add SF oracle if we want the ceiling — it's slower
    # strategies["sf_oracle_k5"] = {"fn": strategy_sf_oracle, "kw": {"top_k": 5, "sf_depth": 8}}

    results = {
        "experiment": "exp054_search_baseline",
        "hypothesis": "search > policy argmax at gameplay",
        "checkpoint": ckpt_name,
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

            for g in range(GAMES_PER_DEPTH // 2):  # 2 games per opening (W+B)
                opening = OPENINGS[g % len(OPENINGS)]
                game_results = play_game(
                    model, device, sspec["fn"], sf_depth, opening,
                    strategy_kw=sspec["kw"],
                )
                for gr in game_results:
                    depth_results.append(gr)
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0:
                        wins += 1
                    elif gr["outcome"] == 0.5:
                        draws += 1
                    else:
                        losses += 1

                print(f"    Game {g*2+1}-{g*2+2}: "
                      f"W{wins}/D{draws}/L{losses} "
                      f"(avg {total_moves/(g*2+2):.0f} moves)")

            avg_moves = total_moves / max(len(depth_results), 1)
            score = (wins + 0.5 * draws) / max(len(depth_results), 1)
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
                "avg_moves": round(avg_moves),
                "games": depth_results,
            }
            print(f"  d{sf_depth} total: W{wins}/D{draws}/L{losses} "
                  f"(score={score:.1%}, avg={avg_moves:.0f} moves)")

        results["strategies"][sname] = strat_results

    # Final summary
    total_time = time.time() - t_start
    results["timing"] = {"total_s": round(total_time)}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS SUMMARY: exp054_search_baseline")
    print(f" Checkpoint: {ckpt_name}")
    print(f"{'='*60}")
    for sname, sdata in results["strategies"].items():
        print(f"\n  {sname}:")
        for dk, dv in sdata.items():
            print(f"    {dk}: W{dv['wins']}/D{dv['draws']}/L{dv['losses']} "
                  f"(score={dv['score']:.1%})")
    print(f"\n  Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
