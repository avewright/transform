"""exp014: MCTS Search at Inference — Multiply playing strength via tree search.

Hypothesis: Adding Monte Carlo Tree Search (MCTS) at inference time will
dramatically improve playing strength, even with a weak policy/value network.
A mediocre network + 200 MCTS sims should beat the same network's raw predictions.

Reference: Silver et al., "Mastering Chess and Shogi by Self-Play with a
General Reinforcement Learning Algorithm" (AlphaZero, 2017).
Every modern engine that beats Stockfish uses search + neural evaluation.

Approach:
  1. Load a trained ChessModel (from exp012b or exp013)
  2. Implement MCTS with:
     - Policy head → prior probabilities for move selection (PUCT)
     - Value head → leaf evaluation
     - Legal move masking at every node
  3. Play games against Stockfish at various depths with different sim budgets
  4. Measure: win/draw/loss, avg centipawn loss, positions evaluated/sec

Key parameters:
  - c_puct: Exploration constant (balances exploitation vs exploration)
  - num_simulations: Budget per move (50, 100, 200, 400)
  - temperature: Controls move selection from visit counts

Memory: ~4-6GB VRAM for model + search tree. Fits 8GB easily, 18GB no issue.
Time: ~5-10 min for 4 games at 100 sims/move.
"""

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, index_to_move, legal_move_mask
from config import Config

# --- Configuration ---
STOCKFISH_PATH = "stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR = Path("outputs/exp014_mcts")
POLICY_CACHE = Path("outputs/exp012_stockfish_supervised/labeled_data.json")

# MCTS
SIM_BUDGETS = [50, 100, 200]
C_PUCT = 1.5              # Exploration constant (AlphaZero uses ~1.25-2.5)
TEMPERATURE = 1.0          # For move selection from visit counts
TEMP_THRESHOLD = 15        # After this many moves, use temp=0 (greedy)

# Training (quick retrain to ensure both heads work)
TRAIN_POSITIONS = 5000
TRAIN_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42

# Evaluation games
NUM_GAMES = 4             # per sim budget
GAME_SF_DEPTH = 3         # opponent strength
MAX_GAME_MOVES = 150


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        "board", "parent", "move", "children",
        "visit_count", "value_sum", "prior",
        "is_expanded",
    ]

    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: dict[str, "MCTSNode"] = {}  # uci -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Mean value (from parent's perspective)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """Upper Confidence Bound score for selection."""
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Uses the policy head for move priors and the value head for leaf evaluation.
    """

    def __init__(
        self,
        chess_model: ChessModel,
        device: torch.device,
        c_puct: float = C_PUCT,
        num_simulations: int = 100,
    ):
        self.model = chess_model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.model.eval()

    @torch.no_grad()
    def _evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Evaluate a position: returns (move_priors, value).

        move_priors: dict mapping UCI → prior probability (only legal moves)
        value: float in [-1, 1] from side-to-move perspective
               (+1 = winning, -1 = losing, 0 = drawn)
        """
        if board.is_game_over():
            result = board.result()
            if result == "1/2-1/2":
                return {}, 0.0
            elif (result == "1-0" and board.turn == chess.WHITE) or \
                 (result == "0-1" and board.turn == chess.BLACK):
                return {}, 1.0  # Side to move just won (shouldn't happen)
            else:
                return {}, -1.0  # Side to move lost (checkmate)

        board_input = self.model.encoder.prepare_input(board, self.device)
        result = self.model(board_input)

        # Policy: softmax over legal moves
        logits = result["policy_logits"][0]
        mask = legal_move_mask(board).to(self.device)
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)

        move_priors = {}
        for move in board.legal_moves:
            uci = move.uci()
            if uci in UCI_TO_IDX:
                move_priors[uci] = probs[UCI_TO_IDX[uci]].item()

        # Normalize priors
        total = sum(move_priors.values())
        if total > 0:
            move_priors = {k: v / total for k, v in move_priors.items()}

        # Value: convert W/D/L logits to scalar [-1, 1]
        value_logits = result["value_logits"][0]
        value_probs = F.softmax(value_logits, dim=-1)
        # value_probs[0] = win, [1] = draw, [2] = loss
        value = value_probs[0].item() - value_probs[2].item()  # win - loss

        return move_priors, value

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node by following UCB scores down the tree."""
        while node.is_expanded and node.children:
            # Select child with highest UCB score
            best_child = max(
                node.children.values(),
                key=lambda c: c.ucb_score(self.c_puct),
            )
            node = best_child
        return node

    def _expand(self, node: MCTSNode) -> float:
        """Expand a leaf node and return its value."""
        move_priors, value = self._evaluate(node.board)

        if not move_priors:
            # Terminal node
            return value

        node.is_expanded = True
        for uci, prior in move_priors.items():
            move = chess.Move.from_uci(uci)
            child_board = node.board.copy()
            child_board.push(move)
            node.children[uci] = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior,
            )

        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value up the tree, alternating signs."""
        while node is not None:
            node.visit_count += 1
            # Value is from the perspective of the node's parent (who made the move)
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent

    def search(self, board: chess.Board) -> dict[str, int]:
        """Run MCTS and return visit counts for each move.

        Returns dict mapping UCI → visit count.
        """
        root = MCTSNode(board.copy())

        # Expand root
        self._expand(root)
        root.visit_count = 1

        for _ in range(self.num_simulations):
            # Select
            leaf = self._select(root)

            # Expand and evaluate
            value = self._expand(leaf)

            # Backpropagate
            self._backpropagate(leaf, value)

        # Collect visit counts
        visit_counts = {}
        for uci, child in root.children.items():
            visit_counts[uci] = child.visit_count

        return visit_counts

    def select_move(
        self,
        board: chess.Board,
        temperature: float = 1.0,
    ) -> chess.Move:
        """Run MCTS and select a move based on visit counts.

        temperature=0: select most-visited (greedy)
        temperature=1: sample proportional to visit counts
        temperature>1: more exploratory
        """
        visit_counts = self.search(board)

        if not visit_counts:
            # Fallback: random legal move
            return random.choice(list(board.legal_moves))

        if temperature == 0:
            best_uci = max(visit_counts, key=visit_counts.get)
            return chess.Move.from_uci(best_uci)

        # Temperature-scaled sampling
        ucis = list(visit_counts.keys())
        counts = [visit_counts[u] for u in ucis]
        total = sum(c ** (1.0 / temperature) for c in counts)
        probs = [(c ** (1.0 / temperature)) / total for c in counts]

        chosen = random.choices(ucis, weights=probs, k=1)[0]
        return chess.Move.from_uci(chosen)


def train_quick_model(device):
    """Quick-train a model on cached Stockfish data so both heads are useful."""
    print("Loading cached Stockfish labels...")
    with open(POLICY_CACHE) as f:
        cache = json.load(f)

    train_raw = cache["train"][:TRAIN_POSITIONS]

    print(f"Loading backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    print(f"  Trainable: {chess_model.trainable_params():,}")

    # Prepare training data
    train_data = []
    for e in train_raw:
        board = chess.Board(e["fen"])
        move = chess.Move.from_uci(e["uci"])
        # Convert eval to WDL
        if e["eval_type"] == "mate":
            v = e["eval_value"]
            wdl_idx = 0 if v > 0 else 2 if v < 0 else 1
        else:
            cp = e["eval_value"]
            if not board.turn:
                cp = -cp
            wdl_idx = 0 if cp > 100 else 2 if cp < -100 else 1
        train_data.append({"board": board, "move": move, "wdl_idx": wdl_idx})

    optimizer = torch.optim.AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    print(f"  Training {TRAIN_EPOCHS} epochs on {len(train_data)} positions...")
    for epoch in range(TRAIN_EPOCHS):
        chess_model.train()
        random.shuffle(train_data)
        total_loss = steps = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i : i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            move_targets = torch.tensor(
                [move_to_index(d["move"]) for d in chunk],
                dtype=torch.long, device=device,
            )
            value_targets = torch.tensor(
                [d["wdl_idx"] for d in chunk],
                dtype=torch.long, device=device,
            )

            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chess_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"    Epoch {epoch+1}: loss={total_loss/steps:.4f}")

    return chess_model


def play_game_raw(chess_model, sf_depth, model_color, device, max_moves=MAX_GAME_MOVES):
    """Play a game using RAW model predictions (no search)."""
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    chess_model.eval()

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = chess_model.predict_move(board)
            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None:
                break
            board.push(chess.Move.from_uci(sf_uci))

    return _game_result(board, model_color)


def play_game_mcts(mcts: MCTS, sf_depth, model_color, device, max_moves=MAX_GAME_MOVES):
    """Play a game using MCTS search."""
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()

    move_count = 0
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            # Use temperature=1 for first moves, then greedy
            temp = TEMPERATURE if move_count < TEMP_THRESHOLD else 0
            move = mcts.select_move(board, temperature=temp)
            board.push(move)
            move_count += 1
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None:
                break
            board.push(chess.Move.from_uci(sf_uci))

    return _game_result(board, model_color)


def _game_result(board: chess.Board, model_color) -> dict:
    """Extract game result."""
    result = board.result()
    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = "draw"

    model_result = (
        "win" if (winner == "white" and model_color == chess.WHITE)
              or (winner == "black" and model_color == chess.BLACK)
        else "loss" if winner != "draw" else "draw"
    )
    term = board.outcome().termination.name if board.outcome() else "max_moves"
    return {
        "model_color": "white" if model_color else "black",
        "result": result,
        "model_result": model_result,
        "moves": board.fullmove_number,
        "termination": term,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Train a model with both policy and value heads ----
    chess_model = train_quick_model(device)

    # ---- Benchmark: Raw model vs MCTS at various sim budgets ----
    print(f"\n{'=' * 60}")
    print(f" PLAYING GAMES vs Stockfish depth {GAME_SF_DEPTH}")
    print(f"{'=' * 60}")

    results = {"raw": [], "mcts": {}}

    # Raw model games
    print(f"\n--- RAW MODEL (no search) ---")
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        print(f"  Game {g+1}/{NUM_GAMES} (model={('white' if color else 'black')})...", end=" ")
        res = play_game_raw(chess_model, GAME_SF_DEPTH, color, device)
        results["raw"].append(res)
        print(f"{res['model_result']} in {res['moves']} moves ({res['termination']})")

    raw_wins = sum(1 for r in results["raw"] if r["model_result"] == "win")
    raw_draws = sum(1 for r in results["raw"] if r["model_result"] == "draw")
    raw_losses = sum(1 for r in results["raw"] if r["model_result"] == "loss")
    print(f"  Raw: {raw_wins}W / {raw_draws}D / {raw_losses}L")

    # MCTS games at different sim budgets
    for sims in SIM_BUDGETS:
        print(f"\n--- MCTS ({sims} simulations/move) ---")
        mcts = MCTS(chess_model, device, c_puct=C_PUCT, num_simulations=sims)
        results["mcts"][sims] = []

        for g in range(NUM_GAMES):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            print(f"  Game {g+1}/{NUM_GAMES} (model={('white' if color else 'black')})...", end=" ", flush=True)
            gt0 = time.time()
            res = play_game_mcts(mcts, GAME_SF_DEPTH, color, device)
            gt1 = time.time()
            res["time_seconds"] = round(gt1 - gt0, 1)
            results["mcts"][sims].append(res)
            print(f"{res['model_result']} in {res['moves']}m ({res['time_seconds']}s)")

        mcts_wins = sum(1 for r in results["mcts"][sims] if r["model_result"] == "win")
        mcts_draws = sum(1 for r in results["mcts"][sims] if r["model_result"] == "draw")
        mcts_losses = sum(1 for r in results["mcts"][sims] if r["model_result"] == "loss")
        print(f"  MCTS-{sims}: {mcts_wins}W / {mcts_draws}D / {mcts_losses}L")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f" SUMMARY: Search Impact on Playing Strength")
    print(f"{'=' * 60}")
    print(f"  vs Stockfish depth {GAME_SF_DEPTH}:")
    print(f"  Raw:       {raw_wins}W / {raw_draws}D / {raw_losses}L")
    for sims in SIM_BUDGETS:
        w = sum(1 for r in results["mcts"][sims] if r["model_result"] == "win")
        d = sum(1 for r in results["mcts"][sims] if r["model_result"] == "draw")
        l = sum(1 for r in results["mcts"][sims] if r["model_result"] == "loss")
        print(f"  MCTS-{sims:>3}: {w}W / {d}D / {l}L")

    # Save results
    elapsed = time.time() - t0
    output = {
        "experiment": "exp014_mcts",
        "hypothesis": "MCTS search multiplies playing strength of neural network",
        "sf_opponent_depth": GAME_SF_DEPTH,
        "c_puct": C_PUCT,
        "games_per_config": NUM_GAMES,
        "raw_results": results["raw"],
        "mcts_results": {str(k): v for k, v in results["mcts"].items()},
        "summary": {
            "raw": {"wins": raw_wins, "draws": raw_draws, "losses": raw_losses},
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    for sims in SIM_BUDGETS:
        w = sum(1 for r in results["mcts"][sims] if r["model_result"] == "win")
        d = sum(1 for r in results["mcts"][sims] if r["model_result"] == "draw")
        l = sum(1 for r in results["mcts"][sims] if r["model_result"] == "loss")
        output["summary"][f"mcts_{sims}"] = {"wins": w, "draws": d, "losses": l}

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
