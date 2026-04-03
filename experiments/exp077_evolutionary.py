"""exp077: Evolutionary Expert Iteration — population-based self-play.

Hypothesis: A population of model variants (via temperature diversity and/or
weight noise) playing a tournament creates a natural selection signal. Training
on the winning variants' moves produces a stronger next-generation model —
without RL gradients, without catastrophic forgetting.

Background:
  - exp033 REINFORCE: destroyed the model (gradient instability)
  - exp041 KL self-play: preserved accuracy but no signal (model draws itself)
  - exp038 Expert iteration: SF selects from model top-10, accuracy dropped
  - The 200M model beats SF ~1320 (87.5%) and holds 50% at ~1600.
  - Elo eval suggests the model is roughly 1500-1600 strength.

Key insight: Instead of using RL gradients to improve from game outcomes,
use game outcomes to SELECT which moves to train on (supervised learning).
This keeps training stable (CE loss, proven to work) while directing the
signal toward moves that actually win games.

Algorithm:
  1. Start with current best model (generation 0).
  2. Create a population of P variants via different sampling temperatures
     and/or small Gaussian noise on weights.
  3. Run a round-robin tournament (each variant vs each other).
  4. Collect (position, move) pairs from the TOP-K winning variants.
  5. Train the base model on these winning-move pairs (supervised CE loss)
     — interleaved with a fraction of the original training data to prevent
     catastrophic forgetting.
  6. The trained model becomes generation 1. Repeat.

Design choices:
  - Temperature diversity (0.0–0.6) is cheaper and more stable than weight noise.
  - Round-robin tournament within the population (no external engine needed).
  - Also play each variant vs Stockfish at Elo 1320 for absolute calibration.
  - Supervised training on winner moves only (no RL loss).
  - KL-style regularization: 80% winner moves + 20% original data replay.
  - Generations: 5 (can extend if improving).

Primary metric: Elo rating vs Stockfish (via win rate at 1320/1450/1600).
Secondary: move prediction accuracy on the standard eval set.
"""

import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import (
    VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX,
    index_to_move, legal_move_mask,
)

# ---- Paths ----
OUTPUT_DIR = Path("outputs/exp077_evolutionary")
CHECKPOINT_PATH = Path("outputs/hf/chess-transformer-200m-latest/best_model.pt")
STOCKFISH_PATH = Path("stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")

# ---- Population config ----
POPULATION_SIZE = 6           # Number of variants per generation
TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]  # One per variant
WEIGHT_NOISE_STD = 0.0       # 0 = temperature-only diversity (safer)
TOP_K_SELECT = 3             # Keep top-K winners' games for training
NUM_GENERATIONS = 5          # Evolutionary generations

# ---- Tournament config ----
GAMES_PER_MATCHUP = 4        # Games per pair (2 as white, 2 as black)
MAX_GAME_PLIES = 150         # Max plies before draw adjudication
OPENINGS = [
    [],                       # Startpos
    ["e2e4", "e7e5"],         # Open game
    ["d2d4", "d7d5"],         # Closed
    ["e2e4", "c7c5"],         # Sicilian
]

# ---- Stockfish calibration ----
SF_CALIBRATION_ELOS = [1320, 1450]
SF_GAMES_PER_LEVEL = 4       # 2 as white + 2 as black
SF_MOVETIME = 0.05           # Seconds per SF move

# ---- Training config ----
TRAIN_EPOCHS = 1             # Per generation (light touch)
TRAIN_BATCH = 4              # Tiny batches to fit 8GB VRAM (spatial head is memory-hungry)
TRAIN_ACCUM = 32             # Effective batch = 128
TRAIN_LR = 5e-6              # Very conservative LR
REPLAY_RATIO = 0.2           # Fraction of training batch from original data replay
MIN_WINNER_POSITIONS = 200   # Need at least this many positions to train
GRAD_CLIP = 1.0

# ---- Eval config ----
EVAL_POSITIONS = 500         # For move-prediction accuracy check (keep fast for 8GB GPU)

# ---- Model architecture (must match play.py / 200M) ----
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ===========================================================================
# Model definition (mirrors play.py ChessTransformer200M)
# ===========================================================================

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
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=512):
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


from chess_model import FusedBoardEncoder


class ChessTransformer200M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO, dropout=DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 3),
        )

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
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ===========================================================================
# Utility functions
# ===========================================================================

def log(msg: str, filepath=None):
    print(msg, flush=True)
    if filepath:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


LOG_FILE = None  # Set in main()


def encode_board(board: chess.Board, device: torch.device) -> dict:
    return batch_boards_to_fused_token_ids([board], device)


def encode_boards(boards: list[chess.Board], device: torch.device) -> dict:
    return batch_boards_to_fused_token_ids(boards, device)


def load_model(checkpoint_path, device):
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def add_weight_noise(model, std=0.001):
    """Add small Gaussian noise to all parameters (in-place)."""
    if std <= 0:
        return
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * std)


@torch.no_grad()
def get_move(model, board, device, temperature=0.0):
    """Get a move from the model. Returns (move, move_idx, log_prob)."""
    board_input = encode_board(board, device)
    result = model(board_input)
    logits = result["policy_logits"][0].float()

    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    if temperature <= 0:
        move_idx = logits.argmax().item()
        log_prob = 0.0
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[move_idx] + 1e-10).item()

    move = index_to_move(move_idx)
    return move, move_idx, log_prob


# ===========================================================================
# Game playing
# ===========================================================================

def play_game(model_w, model_b, device, temp_w=0.0, temp_b=0.0,
              opening=None, max_plies=150):
    """Play a game between two models (or same model with different temps).

    Returns dict with:
      result: "white", "black", "draw"
      positions_white: [(board_fen, move_uci)] for white's moves
      positions_black: [(board_fen, move_uci)] for black's moves
      n_plies: int
      termination: str
    """
    board = chess.Board()
    if opening:
        for uci in opening:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

    white_moves = []  # (fen, move_uci)
    black_moves = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        if board.turn == chess.WHITE:
            move, move_idx, _ = get_move(model_w, board, device, temp_w)
            white_moves.append((board.fen(), move.uci()))
        else:
            move, move_idx, _ = get_move(model_b, board, device, temp_b)
            black_moves.append((board.fen(), move.uci()))

        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result = "draw"
        termination = "adjudicated" if len(board.move_stack) >= max_plies else "draw_rule"
    elif outcome.winner == chess.WHITE:
        result = "white"
        termination = outcome.termination.name
    else:
        result = "black"
        termination = outcome.termination.name

    return {
        "result": result,
        "positions_white": white_moves,
        "positions_black": black_moves,
        "n_plies": len(board.move_stack),
        "termination": termination,
    }


def play_game_vs_sf(model, device, engine, sf_elo, model_color,
                    temperature=0.0, opening=None, max_plies=150):
    """Play a game of model vs Stockfish. Returns dict with result and moves."""
    board = chess.Board()
    if opening:
        for uci in opening:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

    model_moves = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        if board.turn == model_color:
            move, _, _ = get_move(model, board, device, temperature)
            model_moves.append((board.fen(), move.uci()))
        else:
            sf_result = engine.play(board, chess.engine.Limit(time=SF_MOVETIME))
            move = sf_result.move

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
        "score": score,
        "model_moves": model_moves,
        "n_plies": len(board.move_stack),
        "result": "win" if score == 1.0 else ("draw" if score == 0.5 else "loss"),
    }


# ===========================================================================
# Tournament
# ===========================================================================

def run_tournament(model, device, temperatures, openings, games_per_matchup=4,
                   max_plies=150):
    """Round-robin tournament among temperature variants of the same model.

    Returns:
      scores: dict[temp_idx] -> total points
      all_games: list of game records
      winner_positions: dict[temp_idx] -> list of (fen, move_uci) from winning games
    """
    n = len(temperatures)
    scores = defaultdict(float)
    winner_positions = defaultdict(list)
    all_games = []

    matchups = list(combinations(range(n), 2))
    total_games = len(matchups) * games_per_matchup
    log(f"  Tournament: {n} variants, {len(matchups)} matchups, "
        f"{total_games} games total", LOG_FILE)

    game_num = 0
    for i, j in matchups:
        ti, tj = temperatures[i], temperatures[j]
        # Play games_per_matchup games, alternating colors
        for g in range(games_per_matchup):
            opening = openings[g % len(openings)] if openings else None
            # Alternate who plays white
            if g % 2 == 0:
                w_idx, b_idx, tw, tb = i, j, ti, tj
            else:
                w_idx, b_idx, tw, tb = j, i, tj, ti

            game = play_game(model, model, device, temp_w=tw, temp_b=tb,
                             opening=opening, max_plies=max_plies)
            game_num += 1

            # Score
            if game["result"] == "white":
                scores[w_idx] += 1.0
                scores[b_idx] += 0.0
                # Collect winning side's positions
                winner_positions[w_idx].extend(game["positions_white"])
            elif game["result"] == "black":
                scores[w_idx] += 0.0
                scores[b_idx] += 1.0
                winner_positions[b_idx].extend(game["positions_black"])
            else:
                scores[w_idx] += 0.5
                scores[b_idx] += 0.5

            all_games.append({
                "white_idx": w_idx, "black_idx": b_idx,
                "temp_w": tw, "temp_b": tb,
                "result": game["result"],
                "n_plies": game["n_plies"],
                "termination": game["termination"],
            })

        if game_num % 10 == 0:
            log(f"    ...played {game_num}/{total_games} games", LOG_FILE)

    return dict(scores), all_games, dict(winner_positions)


# ===========================================================================
# Stockfish calibration
# ===========================================================================

def run_sf_calibration(model, device, temperature=0.0):
    """Play vs Stockfish at calibration Elos. Returns results dict."""
    if not STOCKFISH_PATH.exists():
        log(f"  SF not found at {STOCKFISH_PATH}, skipping calibration", LOG_FILE)
        return {}

    results = {}
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

    for sf_elo in SF_CALIBRATION_ELOS:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
        total_score = 0.0
        games = []

        for g in range(SF_GAMES_PER_LEVEL):
            opening = OPENINGS[g % len(OPENINGS)]
            model_color = chess.WHITE if g % 2 == 0 else chess.BLACK

            game = play_game_vs_sf(
                model, device, engine, sf_elo, model_color,
                temperature=temperature, opening=opening, max_plies=MAX_GAME_PLIES,
            )
            total_score += game["score"]
            games.append(game)

        results[sf_elo] = {
            "score": total_score / SF_GAMES_PER_LEVEL,
            "total": total_score,
            "games": SF_GAMES_PER_LEVEL,
            "detail": [g["result"] for g in games],
        }
        log(f"    vs SF {sf_elo}: {total_score}/{SF_GAMES_PER_LEVEL} "
            f"({total_score/SF_GAMES_PER_LEVEL:.1%})", LOG_FILE)

    engine.quit()
    return results


# ===========================================================================
# Training on winner moves
# ===========================================================================

def train_on_winner_moves(model, winner_positions, replay_positions, device,
                          epochs=1, batch_size=32, accum_steps=4, lr=5e-6):
    """Supervised training on (fen, move_uci) pairs from tournament winners.

    Interleaves replay_positions to prevent catastrophic forgetting.
    Returns training stats.
    """
    if len(winner_positions) < MIN_WINNER_POSITIONS:
        log(f"  Only {len(winner_positions)} winner positions "
            f"(need {MIN_WINNER_POSITIONS}), skipping training", LOG_FILE)
        return {"skipped": True, "reason": "insufficient_positions"}

    # Prepare data
    random.shuffle(winner_positions)

    # Add replay data
    if replay_positions and REPLAY_RATIO > 0:
        n_replay = max(1, int(len(winner_positions) * REPLAY_RATIO / (1 - REPLAY_RATIO)))
        replay_sample = random.sample(replay_positions, min(n_replay, len(replay_positions)))
        all_positions = winner_positions + replay_sample
        random.shuffle(all_positions)
        log(f"  Training data: {len(winner_positions)} winner + "
            f"{len(replay_sample)} replay = {len(all_positions)} total", LOG_FILE)
    else:
        all_positions = winner_positions
        log(f"  Training data: {len(all_positions)} winner positions", LOG_FILE)

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer.zero_grad()

    stats = {"loss": [], "n_positions": len(all_positions)}

    for epoch in range(epochs):
        random.shuffle(all_positions)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(all_positions), batch_size):
            batch = all_positions[i:i + batch_size]

            # Parse FENs and moves
            boards = []
            move_indices = []
            for fen, move_uci in batch:
                board = chess.Board(fen)
                if move_uci in UCI_TO_IDX:
                    boards.append(board)
                    move_indices.append(UCI_TO_IDX[move_uci])

            if not boards:
                continue

            # Forward pass
            board_input = encode_boards(boards, device)
            result = model(board_input)
            logits = result["policy_logits"]

            # Apply legal move masking
            targets = torch.tensor(move_indices, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)

            # Scale for gradient accumulation
            loss = loss / accum_steps
            loss.backward()

            if (n_batches + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            n_batches += 1

        # Flush any remaining gradients
        if n_batches % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        stats["loss"].append(avg_loss)
        log(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
            f"({n_batches} batches)", LOG_FILE)

    model.eval()
    return stats


# ===========================================================================
# Quick accuracy eval
# ===========================================================================

@torch.no_grad()
def quick_eval(model, device, n_positions=2000):
    """Quick eval on HF test positions. Returns accuracy and top-3."""
    try:
        from datasets import load_dataset
        log("  Loading eval dataset...", LOG_FILE)
        ds = load_dataset("avewright/chess-positions", split="test")
        ds = ds.shuffle(seed=42).select(range(min(n_positions, len(ds))))
        log(f"  Eval dataset: {len(ds)} positions loaded", LOG_FILE)
    except Exception as e:
        log(f"  Cannot load eval dataset: {e}", LOG_FILE)
        return {"accuracy": -1, "top3": -1}

    correct = 0
    top3_correct = 0
    total = 0

    batch_size = 4  # Small batches for 8GB VRAM
    n_batches_done = 0
    for start in range(0, len(ds), batch_size):
        batch = ds[start:start + batch_size]
        fens = batch["fen"]
        targets_uci = batch["best_move"]

        boards = [chess.Board(f) for f in fens]
        board_input = encode_boards(boards, device)
        result = model(board_input)
        logits = result["policy_logits"]

        for j, (board, target_uci) in enumerate(zip(boards, targets_uci)):
            if target_uci not in UCI_TO_IDX:
                continue

            mask = legal_move_mask(board).to(device)
            masked = logits[j].clone()
            masked[~mask] = float("-inf")

            target_idx = UCI_TO_IDX[target_uci]
            pred_idx = masked.argmax().item()

            total += 1
            if pred_idx == target_idx:
                correct += 1

            topk = torch.topk(masked, min(3, mask.sum().item())).indices.tolist()
            if target_idx in topk:
                top3_correct += 1

        n_batches_done += 1
        if n_batches_done % 50 == 0:
            log(f"    eval: {start+batch_size}/{len(ds)} positions...", LOG_FILE)

    acc = correct / max(total, 1)
    top3 = top3_correct / max(total, 1)
    return {"accuracy": acc, "top3": top3, "n_eval": total}


# ===========================================================================
# Replay data: collect positions from model self-play at temp=0
# (deterministic games provide "what the model already knows")
# ===========================================================================

def collect_replay_data(model, device, n_games=20):
    """Play greedy self-play games to collect the model's current knowledge."""
    positions = []
    for g in range(n_games):
        opening = OPENINGS[g % len(OPENINGS)]
        game = play_game(model, model, device,
                         temp_w=0.0, temp_b=0.0,
                         opening=opening, max_plies=MAX_GAME_PLIES)
        positions.extend(game["positions_white"])
        positions.extend(game["positions_black"])
    return positions


# ===========================================================================
# Main evolutionary loop
# ===========================================================================

def main():
    global LOG_FILE

    random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = OUTPUT_DIR / "exp077.log"

    log("=" * 70, LOG_FILE)
    log("exp077: Evolutionary Expert Iteration", LOG_FILE)
    log(f"Device: {DEVICE}", LOG_FILE)
    log(f"Population: {POPULATION_SIZE} variants, temps={TEMPERATURES}", LOG_FILE)
    log(f"Generations: {NUM_GENERATIONS}", LOG_FILE)
    log(f"Checkpoint: {CHECKPOINT_PATH}", LOG_FILE)
    log("=" * 70, LOG_FILE)

    # Load base model
    t0 = time.time()
    model = load_model(CHECKPOINT_PATH, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model loaded: {n_params/1e6:.0f}M params ({time.time()-t0:.1f}s)", LOG_FILE)

    # Baseline evaluation
    log("\n--- Generation 0 (baseline) ---", LOG_FILE)
    baseline_eval = quick_eval(model, DEVICE, EVAL_POSITIONS)
    log(f"  Baseline accuracy: {baseline_eval['accuracy']:.1%}, "
        f"top-3: {baseline_eval['top3']:.1%}", LOG_FILE)

    baseline_sf = run_sf_calibration(model, DEVICE, temperature=0.0)

    # Collect replay data from greedy self-play
    log("  Collecting replay data (greedy self-play)...", LOG_FILE)
    replay_data = collect_replay_data(model, DEVICE, n_games=20)
    log(f"  Replay buffer: {len(replay_data)} positions", LOG_FILE)

    # Store results across generations
    generation_results = [{
        "generation": 0,
        "eval": baseline_eval,
        "sf_calibration": baseline_sf,
        "tournament": None,
        "training": None,
    }]

    for gen in range(1, NUM_GENERATIONS + 1):
        gen_start = time.time()
        log(f"\n{'='*70}", LOG_FILE)
        log(f"--- Generation {gen}/{NUM_GENERATIONS} ---", LOG_FILE)

        # ---- 1. Tournament ----
        log(f"\n  Phase 1: Tournament ({POPULATION_SIZE} variants)...", LOG_FILE)
        t_tourn = time.time()
        scores, games, winner_positions = run_tournament(
            model, DEVICE, TEMPERATURES, OPENINGS,
            games_per_matchup=GAMES_PER_MATCHUP, max_plies=MAX_GAME_PLIES,
        )

        # Rank variants by score
        ranking = sorted(scores.items(), key=lambda x: -x[1])
        log(f"\n  Tournament results ({time.time()-t_tourn:.0f}s):", LOG_FILE)
        for idx, score in ranking:
            n_pos = len(winner_positions.get(idx, []))
            log(f"    Variant {idx} (temp={TEMPERATURES[idx]:.1f}): "
                f"score={score:.1f}, positions={n_pos}", LOG_FILE)

        # Game-level summary
        n_white_wins = sum(1 for g in games if g["result"] == "white")
        n_black_wins = sum(1 for g in games if g["result"] == "black")
        n_draws = sum(1 for g in games if g["result"] == "draw")
        log(f"  Game outcomes: {n_white_wins}W {n_black_wins}B {n_draws}D "
            f"(total {len(games)})", LOG_FILE)

        # ---- 2. Collect winner positions ----
        top_indices = [idx for idx, _ in ranking[:TOP_K_SELECT]]
        all_winner_positions = []
        for idx in top_indices:
            all_winner_positions.extend(winner_positions.get(idx, []))

        log(f"\n  Top-{TOP_K_SELECT} variants: {top_indices}", LOG_FILE)
        log(f"  Winner positions collected: {len(all_winner_positions)}", LOG_FILE)

        # Also add all positions from decisive games where EITHER side won
        # (the winning side's moves are valuable)
        decisive_positions = []
        for idx in range(POPULATION_SIZE):
            decisive_positions.extend(winner_positions.get(idx, []))
        log(f"  Total decisive positions (all variants): {len(decisive_positions)}", LOG_FILE)

        # Use top-K winner positions preferentially, but fall back to all decisive
        train_positions = all_winner_positions if len(all_winner_positions) >= MIN_WINNER_POSITIONS else decisive_positions

        # ---- 3. Stockfish calibration (pre-training) ----
        log(f"\n  Phase 2: SF calibration (pre-training)...", LOG_FILE)
        sf_pre = run_sf_calibration(model, DEVICE, temperature=0.0)

        # ---- 4. Train on winner moves ----
        log(f"\n  Phase 3: Training on winner moves...", LOG_FILE)
        t_train = time.time()
        train_stats = train_on_winner_moves(
            model, train_positions, replay_data, DEVICE,
            epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH,
            accum_steps=TRAIN_ACCUM, lr=TRAIN_LR,
        )
        log(f"  Training done ({time.time()-t_train:.0f}s)", LOG_FILE)

        # ---- 5. Evaluate ----
        log(f"\n  Phase 4: Evaluation...", LOG_FILE)
        gen_eval = quick_eval(model, DEVICE, EVAL_POSITIONS)
        log(f"  Accuracy: {gen_eval['accuracy']:.1%} "
            f"(baseline: {baseline_eval['accuracy']:.1%}, "
            f"Δ={gen_eval['accuracy']-baseline_eval['accuracy']:+.1%})", LOG_FILE)
        log(f"  Top-3: {gen_eval['top3']:.1%} "
            f"(baseline: {baseline_eval['top3']:.1%})", LOG_FILE)

        sf_post = run_sf_calibration(model, DEVICE, temperature=0.0)

        # ---- 6. Update replay buffer with new self-play ----
        log("  Refreshing replay buffer...", LOG_FILE)
        new_replay = collect_replay_data(model, DEVICE, n_games=10)
        # Keep a sliding window of replay data
        replay_data = replay_data[-2000:] + new_replay
        log(f"  Replay buffer: {len(replay_data)} positions", LOG_FILE)

        # ---- 7. Save checkpoint ----
        ckpt_path = OUTPUT_DIR / f"gen{gen}_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "generation": gen,
            "eval": gen_eval,
        }, ckpt_path)
        log(f"  Checkpoint saved: {ckpt_path}", LOG_FILE)

        gen_time = time.time() - gen_start
        log(f"\n  Generation {gen} complete in {gen_time:.0f}s", LOG_FILE)

        gen_result = {
            "generation": gen,
            "eval": gen_eval,
            "sf_calibration_pre": sf_pre,
            "sf_calibration_post": sf_post,
            "tournament": {
                "scores": {str(k): v for k, v in scores.items()},
                "ranking": [(idx, TEMPERATURES[idx], s) for idx, s in ranking],
                "n_games": len(games),
                "outcomes": {"white": n_white_wins, "black": n_black_wins, "draw": n_draws},
            },
            "training": train_stats,
            "n_winner_positions": len(train_positions),
            "time_s": gen_time,
        }
        generation_results.append(gen_result)

        # Save running results
        results_path = OUTPUT_DIR / "results.json"
        with open(results_path, "w") as f:
            json.dump(generation_results, f, indent=2, default=str)

        # ---- Early stopping: if accuracy dropped > 3pp, abort ----
        if gen_eval["accuracy"] < baseline_eval["accuracy"] - 0.03:
            log(f"\n  WARNING: Accuracy dropped {gen_eval['accuracy']-baseline_eval['accuracy']:+.1%} "
                f"from baseline. Stopping early.", LOG_FILE)
            break

    # ---- Final summary ----
    log(f"\n{'='*70}", LOG_FILE)
    log("FINAL SUMMARY", LOG_FILE)
    log(f"{'='*70}", LOG_FILE)

    for r in generation_results:
        gen = r["generation"]
        acc = r["eval"]["accuracy"]
        top3 = r["eval"]["top3"]
        delta = acc - baseline_eval["accuracy"]
        sf_str = ""
        sf_data = r.get("sf_calibration_post") or r.get("sf_calibration", {})
        for elo, data in sorted(sf_data.items()):
            sf_str += f" SF{elo}={data['score']:.1%}"
        log(f"  Gen {gen}: acc={acc:.1%} (Δ={delta:+.1%}) top3={top3:.1%}{sf_str}", LOG_FILE)

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)", LOG_FILE)

    # Save final results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(generation_results, f, indent=2, default=str)
    log(f"Results saved to {results_path}", LOG_FILE)


if __name__ == "__main__":
    main()
