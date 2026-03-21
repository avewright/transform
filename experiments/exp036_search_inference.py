"""exp036: Inference-time search — 1-ply lookahead with value head.

Hypothesis: Using the value head to evaluate positions 1 move ahead will
significantly improve game-playing strength vs Stockfish, even without
any additional training. The model already has a value head (3-class WDL)
that can guide move selection beyond pure policy argmax.

Background:
  - exp032: 51.4% accuracy, W0/D1/L7 vs SF d3 (policy argmax only)
  - The accuracy plateau (~51%) limits pure policy-based play
  - AlphaZero and Leela use MCTS with value + policy for much stronger play
  - Even 1-ply lookahead (evaluate top-K moves by resulting position) should help
  - Value head was trained on game outcomes → encodes positional understanding

Design:
  - Load exp032 checkpoint (51.4%, best game performance: W0/D1/L7)
  - Compare three inference strategies:
    A) Policy argmax (baseline) — pick highest-probability legal move
    B) 1-ply lookahead (top-5) — play each top-5 move, evaluate with value head, pick best
    C) 1-ply lookahead (top-10) — same but top-10 moves
  - Play 8 games vs SF d3 with each strategy
  - Also measure move prediction accuracy with lookahead

Primary metric: games vs SF d3
Time budget: ~10 min (no training, just inference)
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

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp036_search_inference")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_GAMES = 8
GAME_SF_DEPTH = 3
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42


# === Architecture (must match exp032) ===

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
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
    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer('from_sqs', from_sqs)
        self.register_buffer('to_sqs', to_sqs)
        self.register_buffer('promo_types', promo_types)

    def forward(self, hidden_states):
        sq_hidden = hidden_states[:, 3:67, :]
        global_hidden = hidden_states[:, 0, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(global_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformer(nn.Module):
    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 67, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(hidden_dim, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)

        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)

        return {"policy_logits": policy_logits, "value_logits": value_logits}

    @torch.no_grad()
    def predict_move(self, board):
        """Policy argmax (baseline)."""
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs

    @torch.no_grad()
    def evaluate_position(self, board):
        """Return value estimate for position from side-to-move perspective.

        Returns float in [-1, 1]: positive = good for side to move.
        WDL logits: 0=loss, 1=draw, 2=win (relative to side to move).
        """
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        result = self.forward(board_input)
        value_logits = result["value_logits"][0]  # (3,)
        probs = F.softmax(value_logits, dim=-1)
        # Value = win_prob - loss_prob ∈ [-1, 1]
        return (probs[2] - probs[0]).item()

    @torch.no_grad()
    def evaluate_positions_batch(self, boards):
        """Batch evaluate multiple positions."""
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_batch(boards, device)
        result = self.forward(board_input)
        value_logits = result["value_logits"]  # (B, 3)
        probs = F.softmax(value_logits, dim=-1)
        return (probs[:, 2] - probs[:, 0]).cpu().tolist()

    @torch.no_grad()
    def predict_with_search(self, board, top_k=5):
        """1-ply lookahead: evaluate top-K policy moves by resulting position.

        For each of the top-K policy moves:
          1. Make the move on a copy of the board
          2. Evaluate the resulting position with the value head
          3. Pick the move with the best value (from opponent's perspective, negated)

        Returns (best_move, move_info_list).
        """
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)

        # Get top-K legal moves
        topk_vals, topk_idxs = probs.topk(min(top_k, (mask.sum().item())), dim=-1)
        candidate_moves = []
        resulting_boards = []

        for idx in topk_idxs.cpu().tolist():
            move = index_to_move(idx)
            if move in board.legal_moves:
                candidate_moves.append(move)
                b_copy = board.copy()
                b_copy.push(move)
                resulting_boards.append(b_copy)

        if not candidate_moves:
            # Fallback to policy argmax
            return index_to_move(probs.argmax().item()), []

        # Batch evaluate resulting positions
        # The value is from the OPPONENT's perspective (since we just moved)
        # We want the position to be BAD for the opponent = GOOD for us
        opp_values = self.evaluate_positions_batch(resulting_boards)

        # We want to MINIMIZE opponent's value (maximize our advantage)
        # opp_value is [win_prob - loss_prob] from OPPONENT's perspective
        # So we want the most negative opp_value
        move_info = []
        for i, (move, opp_val) in enumerate(zip(candidate_moves, opp_values)):
            policy_prob = probs[move_to_index(move)].item()
            our_value = -opp_val  # Negate: opponent's loss is our gain
            move_info.append({
                "move": move,
                "policy_prob": policy_prob,
                "our_value": our_value,
                "opp_value": opp_val,
            })

        # Sort by our_value (highest = best for us)
        move_info.sort(key=lambda x: x["our_value"], reverse=True)
        best_move = move_info[0]["move"]

        return best_move, move_info

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Games ===

def play_game_vs_stockfish(model, sf_depth, model_color, device, max_moves=100,
                           search_mode="argmax", top_k=5):
    """Play a game vs Stockfish with configurable inference mode."""
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()

    search_changed = 0  # Count of moves where search chose differently from policy

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            if search_mode == "argmax":
                pred, _ = model.predict_move(board)
            else:
                pred, info = model.predict_with_search(board, top_k=top_k)
                # Count if search changed the move
                policy_move, _ = model.predict_move(board)
                if pred != policy_move:
                    search_changed += 1

            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None: break
            board.push(chess.Move.from_uci(sf_uci))

    result = board.result()
    winner = "white" if result == "1-0" else "black" if result == "0-1" else "draw"
    model_result = (
        "win" if (winner == "white" and model_color == chess.WHITE) or
                 (winner == "black" and model_color == chess.BLACK)
        else "loss" if winner != "draw" else "draw"
    )
    term = board.outcome().termination.name if board.outcome() else "max_moves"
    return {
        "model_color": "white" if model_color else "black",
        "result": result, "model_result": model_result,
        "moves": board.fullmove_number, "termination": term,
        "search_changed": search_changed,
    }


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp036_search_inference")
    print(f"Hypothesis: 1-ply value-head search improves game play")

    # --- Load checkpoint ---
    print("\n[1/2] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Quick value head diagnostic ---
    print("\n  Value head diagnostic on starting position:")
    board = chess.Board()
    val = model.evaluate_position(board)
    print(f"  Starting pos value: {val:+.3f} (should be ~0 for balanced)")

    # Check a few standard openings
    for uci_moves, desc in [
        (["e2e4"], "1.e4"),
        (["e2e4", "e7e5"], "1.e4 e5"),
        (["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "d1h5f7"], "Scholar's mate attempt"),
    ]:
        b = chess.Board()
        for m in uci_moves:
            try:
                b.push(chess.Move.from_uci(m))
            except:
                break
        val = model.evaluate_position(b)
        print(f"    {desc}: value={val:+.3f}")

    # --- Play games with different strategies ---
    strategies = [
        ("argmax", {"search_mode": "argmax", "top_k": 0}),
        ("search_top5", {"search_mode": "search", "top_k": 5}),
        ("search_top10", {"search_mode": "search", "top_k": 10}),
    ]

    all_results = {}
    print(f"\n[2/2] Playing {NUM_GAMES} games per strategy vs SF d{GAME_SF_DEPTH}")

    for strat_name, strat_kwargs in strategies:
        print(f"\n  Strategy: {strat_name}")
        game_results = []
        total_changed = 0

        for g in range(NUM_GAMES):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device,
                                       **strat_kwargs)
            game_results.append(r)
            total_changed += r.get("search_changed", 0)
            sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
            chg = f" (search changed {r['search_changed']} moves)" if r.get("search_changed", 0) else ""
            print(f"    Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv "
                  f"({r['termination']}){chg}")

        wins = sum(1 for r in game_results if r["model_result"] == "win")
        draws = sum(1 for r in game_results if r["model_result"] == "draw")
        losses = sum(1 for r in game_results if r["model_result"] == "loss")
        print(f"  Score: W{wins}/D{draws}/L{losses}")
        if total_changed > 0:
            print(f"  Total moves changed by search: {total_changed}")

        all_results[strat_name] = {
            "games": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
            "total_search_changed": total_changed,
        }

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp036_search_inference",
        "hypothesis": "1-ply value-head search improves game play",
        "primary_metric": "games vs SF d3 per strategy",
        "seed": SEED,
        "model": {"type": "chess_transformer", "layers": NUM_LAYERS,
                  "hidden": HIDDEN_DIM, "heads": NUM_HEADS,
                  "encoder_dim": ENCODER_DIM},
        "strategies": all_results,
        "comparison": {
            "exp032_games": "W0/D1/L7",
        },
        "timing": {"total_s": total_time},
        "device": str(device),
        "command": "python experiments/exp036_search_inference.py",
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp036_search_inference")
    print(f" Hypothesis: 1-ply value-head search improves game play")
    for name, r in all_results.items():
        s = r["score"]
        print(f"  {name:12s}: W{s['wins']}/D{s['draws']}/L{s['losses']} "
              f"(changed={r['total_search_changed']})")
    print(f" Baseline (exp032): W0/D1/L7")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
