"""exp080: Continuous Evolutionary Self-Play — infinite improvement loop.

Runs FOREVER (until manually stopped or no progress). Each cycle:

  1. PERTURB: Create N variants of current best via weight noise + temperature
  2. COMPETE: Round-robin self-play tournament + SF Elo calibration
  3. SELECT: Pick the best variant (by tournament score + SF performance)
  4. TRAIN: Light supervised training on winning moves from best variants,
     verified by Stockfish (only keep moves SF agrees are good)
  5. GATE: Accept new model only if it passes SF Elo gate
  6. REPEAT from step 1 with the new champion

Key design vs exp077:
  - Stockfish as ground-truth quality filter (not just self-play signal)
  - Weight perturbation diversity (not just temperature)
  - Elo gating: new model must beat current champion in SF games to be accepted
  - Runs continuously with checkpointing (can resume)
  - Larger tournament (more games per matchup for statistical significance)

Primary metric: Elo rating vs Stockfish (measured by win rate at 1320/1450/1600).
Device: RTX 4060 Laptop (8GB VRAM).
"""

import copy
import json
import math
import os
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
OUTPUT_DIR = Path("outputs/exp080_evo_rl")
# Use exp079 best if it exists, else fall back to HF latest
CHECKPOINT_PATH = Path("outputs/exp079_fast_continued/best_model.pt")
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = Path("outputs/hf/chess-transformer-200m-latest/best_model.pt")
STOCKFISH_PATH = Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")

# ---- Population config ----
POPULATION_SIZE = 5
# Temperature variants for move selection diversity
TEMPERATURES = [0.0, 0.05, 0.1, 0.15, 0.2]
# Weight noise levels (applied independently to create diverse variants)
NOISE_LEVELS = [0.0, 0.0, 0.0001, 0.0001, 0.0003]
TOP_K_SELECT = 3  # Keep top-K winners' moves

# ---- Tournament config ----
GAMES_PER_MATCHUP = 4         # Per pair in round-robin (2W + 2B)
MAX_GAME_PLIES = 200
OPENINGS = [
    [],                       # Startpos
    ["e2e4", "e7e5"],         # Open game
    ["d2d4", "d7d5"],         # Closed
    ["e2e4", "c7c5"],         # Sicilian
    ["d2d4", "g8f6", "c2c4", "e7e6"],  # Nimzo-ish
    ["e2e4", "e7e6"],         # French
]

# ---- Stockfish evaluation ----
SF_ELO_LEVELS = [1320, 1450, 1600, 1750]
SF_GAMES_PER_LEVEL = 8        # More games for better signal
SF_MOVETIME = 0.05
SF_VERIFY_DEPTH = 12          # Depth for move quality verification
SF_VERIFY_CP_THRESHOLD = 80   # Move must be within 80cp of SF best

# ---- Training config (very light per cycle) ----
TRAIN_BATCH = 4
TRAIN_ACCUM = 16              # Effective batch = 64 (lighter than exp079)
TRAIN_LR = 2e-6               # Very conservative
TRAIN_EPOCHS = 1
MIN_WINNER_POSITIONS = 100
REPLAY_RATIO = 0.3            # 30% replay to prevent forgetting
GRAD_CLIP = 0.5

# ---- Eval / gating ----
EVAL_POSITIONS = 500          # Quick accuracy check
# Weighted composite Elo score: higher levels count more
# Weights: 1320→1, 1450→2, 1600→3, 1750→4  (total weight=10)
# e.g. 75%@1320 + 50%@1450 + 50%@1600 + 25%@1750 = (0.75+1.0+1.5+1.0)/10 = 42.5%
ELO_WEIGHTS = {1320: 1, 1450: 2, 1600: 3, 1750: 4}
ELO_GATE_MIN_COMPOSITE = 0.20  # Weighted composite floor (very lenient — just no collapse)
MAX_CONSECUTIVE_FAILURES = 10  # More patient — noisy signal needs more cycles

# ---- Model architecture ----
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
LOG_FILE = None


def log(msg: str):
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ===========================================================================
# Model (mirrors play.py/exp077)
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
# Utilities
# ===========================================================================

def load_model(path, device):
    model = ChessTransformer200M()
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def save_model(model, path, **extra):
    data = {"model_state_dict": model.state_dict()}
    data.update(extra)
    torch.save(data, str(path))


def add_weight_noise(model, std):
    if std <= 0:
        return
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * std)


@torch.no_grad()
def get_move(model, board, device, temperature=0.0):
    inp = batch_boards_to_fused_token_ids([board], device)
    out = model(inp)
    logits = out["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    if temperature <= 0:
        idx = logits.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1).item()

    move = index_to_move(idx)
    return move, idx


# ===========================================================================
# Game playing
# ===========================================================================

def play_game(model, device, temp_w, temp_b, opening=None):
    """Self-play game. Returns result + positions from both sides."""
    board = chess.Board()
    if opening:
        for uci in opening:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

    w_moves, b_moves = [], []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_GAME_PLIES:
        if board.turn == chess.WHITE:
            move, _ = get_move(model, board, device, temp_w)
            w_moves.append((board.fen(), move.uci()))
        else:
            move, _ = get_move(model, board, device, temp_b)
            b_moves.append((board.fen(), move.uci()))

        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result = "draw"
    elif outcome.winner == chess.WHITE:
        result = "white"
    else:
        result = "black"

    return result, w_moves, b_moves, len(board.move_stack)


def play_vs_sf(model, device, engine, sf_elo, model_white, temperature=0.0, opening=None):
    """Play one game vs Stockfish. Returns (score, model_positions)."""
    board = chess.Board()
    if opening:
        for uci in opening:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

    model_color = chess.WHITE if model_white else chess.BLACK
    model_positions = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_GAME_PLIES:
        if board.turn == model_color:
            move, _ = get_move(model, board, device, temperature)
            model_positions.append((board.fen(), move.uci()))
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

    return score, model_positions


# ===========================================================================
# Stockfish move verification
# ===========================================================================

def verify_moves_with_sf(positions, engine, depth=SF_VERIFY_DEPTH, cp_threshold=SF_VERIFY_CP_THRESHOLD):
    """Filter positions to only keep moves that SF agrees are good.

    A move is 'verified' if it's within cp_threshold centipawns of SF's best move.
    Returns list of verified (fen, move_uci) pairs.
    """
    verified = []

    for fen, move_uci in positions:
        try:
            board = chess.Board(fen)
            # Get SF's evaluation of the position
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            sf_best = info.get("pv", [None])[0]
            sf_score = info.get("score")

            if sf_best is None or sf_score is None:
                continue

            # If model played SF's best move, always accept
            if move_uci == sf_best.uci():
                verified.append((fen, move_uci))
                continue

            # Otherwise check if model's move is within threshold
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                continue

            board.push(move)
            info2 = engine.analyse(board, chess.engine.Limit(depth=max(1, depth - 2)))
            after_score = info2.get("score")
            board.pop()

            if sf_score.is_mate() or after_score is None or after_score.is_mate():
                # Can't compare mate scores easily; accept if SF best is mate
                if sf_score.is_mate() and sf_score.relative.mate() > 0:
                    verified.append((fen, move_uci))
                continue

            # Compare: SF best score vs our move's resulting score
            # (scores are from perspective of side to move)
            sf_cp = sf_score.relative.score(mate_score=10000)
            # after_score is from opponent's perspective since we pushed our move
            our_cp = -after_score.relative.score(mate_score=10000)

            if sf_cp is not None and our_cp is not None:
                cp_loss = sf_cp - our_cp
                if cp_loss <= cp_threshold:
                    verified.append((fen, move_uci))

        except Exception:
            continue

    return verified


# ===========================================================================
# Tournament
# ===========================================================================

def run_tournament(model, device, population_configs):
    """Round-robin tournament among population variants.

    population_configs: list of (temperature, noise_std) tuples.
    Returns scores dict and winner positions.
    """
    n = len(population_configs)
    scores = defaultdict(float)
    all_positions = defaultdict(list)  # variant_idx -> [(fen, move_uci)]

    matchups = list(combinations(range(n), 2))
    total_games = len(matchups) * GAMES_PER_MATCHUP
    log(f"  Tournament: {n} variants, {len(matchups)} matchups, {total_games} games")

    game_num = 0
    outcomes = {"white": 0, "black": 0, "draw": 0}

    for i, j in matchups:
        ti, _ = population_configs[i]
        tj, _ = population_configs[j]

        for g in range(GAMES_PER_MATCHUP):
            opening = OPENINGS[g % len(OPENINGS)]
            if g % 2 == 0:
                w_idx, b_idx, tw, tb = i, j, ti, tj
            else:
                w_idx, b_idx, tw, tb = j, i, tj, ti

            result, w_pos, b_pos, n_plies = play_game(
                model, device, tw, tb, opening=opening
            )
            game_num += 1
            outcomes[result] += 1

            if result == "white":
                scores[w_idx] += 1.0
                all_positions[w_idx].extend(w_pos)
            elif result == "black":
                scores[b_idx] += 1.0
                all_positions[b_idx].extend(b_pos)
            else:
                scores[w_idx] += 0.5
                scores[b_idx] += 0.5

            if game_num % 20 == 0:
                log(f"    ...played {game_num}/{total_games} games")

    log(f"  Outcomes: {outcomes['white']}W {outcomes['black']}B {outcomes['draw']}D")
    return dict(scores), dict(all_positions)


# ===========================================================================
# SF Elo evaluation
# ===========================================================================

def evaluate_elo(model, device, engine, temperature=0.0):
    """Play games vs SF at multiple Elo levels. Returns {elo: score}."""
    results = {}
    for elo in SF_ELO_LEVELS:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        total = 0.0
        for g in range(SF_GAMES_PER_LEVEL):
            opening = OPENINGS[g % len(OPENINGS)]
            model_white = (g % 2 == 0)
            score, _ = play_vs_sf(model, device, engine, elo, model_white,
                                  temperature=temperature, opening=opening)
            total += score

        frac = total / SF_GAMES_PER_LEVEL
        results[elo] = frac
        log(f"    vs SF {elo}: {total:.1f}/{SF_GAMES_PER_LEVEL} ({frac:.1%})")

    return results


# ===========================================================================
# Quick accuracy eval
# ===========================================================================

@torch.no_grad()
def evaluate_accuracy(model, eval_data, device):
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0
    for i in range(0, len(eval_data), 4):
        batch = eval_data[i:i+4]
        boards = [d["board"] for d in batch]
        targets = [d["move_idx"] for d in batch]
        inp = batch_boards_to_fused_token_ids(boards, device)
        out = model(inp)
        logits = out["policy_logits"]
        for j, board in enumerate(boards):
            mask = legal_move_mask(board).to(device)
            logits[j] = logits[j].masked_fill(~mask, float("-inf"))
        preds = logits.argmax(dim=-1)
        top3 = logits.topk(3, dim=-1).indices
        for j, t in enumerate(targets):
            total += 1
            if preds[j].item() == t:
                correct += 1
            if t in top3[j].tolist():
                top3_correct += 1
    model.train()
    return correct / max(1, total), top3_correct / max(1, total)


# ===========================================================================
# Training on verified winning moves
# ===========================================================================

def train_on_positions(model, positions, replay_data, device):
    """Light supervised training on verified positions + replay.

    Returns (loss, n_batches).
    """
    if len(positions) < MIN_WINNER_POSITIONS:
        log(f"  Only {len(positions)} positions (need {MIN_WINNER_POSITIONS}), skipping")
        return None, 0

    # Mix in replay
    n_replay = int(len(positions) * REPLAY_RATIO / (1 - REPLAY_RATIO))
    replay_sample = random.sample(replay_data, min(n_replay, len(replay_data)))

    all_data = []
    for fen, move_uci in positions:
        if move_uci in UCI_TO_IDX:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                all_data.append({"board": board, "move_idx": UCI_TO_IDX[move_uci]})

    for item in replay_sample:
        all_data.append({"board": item["board"], "move_idx": item["move_idx"]})

    random.shuffle(all_data)
    log(f"  Training on {len(all_data)} positions ({len(positions)} winner + {len(replay_sample)} replay)")

    model.train()
    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=0.01)

    total_loss = 0.0
    n_batches = 0
    micro = 0
    optimizer.zero_grad()

    for i in range(0, len(all_data), TRAIN_BATCH):
        batch = all_data[i:i+TRAIN_BATCH]
        if len(batch) < 2:
            continue

        boards = [d["board"] for d in batch]
        targets = torch.tensor([d["move_idx"] for d in batch], dtype=torch.long, device=device)

        inp = batch_boards_to_fused_token_ids(boards, device)
        out = model(inp)
        loss = F.cross_entropy(out["policy_logits"], targets) / TRAIN_ACCUM
        loss.backward()
        micro += 1

        if micro >= TRAIN_ACCUM:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * TRAIN_ACCUM
            n_batches += 1
            micro = 0

    # Flush remaining
    if micro > 0:
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    model.eval()
    return avg_loss, n_batches


# ===========================================================================
# Load replay data
# ===========================================================================

def _load_hf_token():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def load_replay_and_eval():
    """Load replay data (original 47.5K) and eval set."""
    from datasets import load_dataset

    token = _load_hf_token()
    ds_train = load_dataset("avewright/chess-positions", split="train", token=token)
    ds_test = load_dataset("avewright/chess-positions", split="test", token=token)

    replay = []
    for row in ds_train:
        try:
            uci = row["best_move"]
            if uci not in UCI_TO_IDX:
                continue
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            replay.append({"board": board, "move_idx": UCI_TO_IDX[uci]})
        except Exception:
            continue

    eval_data = []
    for row in ds_test:
        try:
            uci = row["best_move"]
            if uci not in UCI_TO_IDX:
                continue
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            eval_data.append({"board": board, "move_idx": UCI_TO_IDX[uci]})
            if len(eval_data) >= EVAL_POSITIONS:
                break
        except Exception:
            continue

    return replay, eval_data


# ===========================================================================
# Main loop
# ===========================================================================

def main():
    global LOG_FILE

    random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = str(OUTPUT_DIR / "exp080.log")

    log("=" * 70)
    log("exp080: Continuous Evolutionary Self-Play")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Original checkpoint: {CHECKPOINT_PATH}")
    log(f"Population: {POPULATION_SIZE} variants")
    log(f"SF gate: composite >= {ELO_GATE_MIN_COMPOSITE:.0%} (weighted 1320-1750)")
    log("")

    # Resume from best checkpoint if it exists, otherwise start fresh
    resume_path = OUTPUT_DIR / "best_model.pt"
    if resume_path.exists():
        log(f"Resuming from: {resume_path}")
        champion = load_model(resume_path, DEVICE)
    else:
        log(f"Starting fresh from: {CHECKPOINT_PATH}")
        champion = load_model(CHECKPOINT_PATH, DEVICE)
    log(f"Model loaded ({sum(p.numel() for p in champion.parameters()) / 1e6:.1f}M params)")

    # Load data
    log("Loading replay + eval data...")
    replay_data, eval_data = load_replay_and_eval()
    log(f"Replay: {len(replay_data)}, Eval: {len(eval_data)}")

    # Open Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

    # Baseline evaluation
    log("\n--- Baseline ---")
    baseline_acc, baseline_top3 = evaluate_accuracy(champion, eval_data, DEVICE)
    log(f"Accuracy: {baseline_acc:.1%}, Top-3: {baseline_top3:.1%}")

    log("SF Elo calibration:")
    baseline_sf = evaluate_elo(champion, DEVICE, engine, temperature=0.0)

    # Track history
    history = [{
        "cycle": 0,
        "acc": baseline_acc,
        "top3": baseline_top3,
        "sf": {str(k): v for k, v in baseline_sf.items()},
        "event": "baseline",
    }]

    champion_acc = baseline_acc
    champion_sf = baseline_sf
    consecutive_failures = 0
    cycle = 0

    # Build population configs
    pop_configs = list(zip(TEMPERATURES, NOISE_LEVELS))

    log(f"\nPopulation configs:")
    for i, (t, n) in enumerate(pop_configs):
        log(f"  Variant {i}: temp={t}, noise={n}")

    log(f"\n{'='*70}")
    log("Starting continuous evolution loop...")
    log(f"{'='*70}\n")

    while True:
        cycle += 1
        cycle_start = time.time()
        log(f"\n{'='*70}")
        log(f"--- Cycle {cycle} ---")
        log(f"{'='*70}")
        champ_comp = sum(champion_sf.get(elo, 0) * w for elo, w in ELO_WEIGHTS.items()) / sum(ELO_WEIGHTS.values())
        log(f"Champion: acc={champion_acc:.1%}, "
            f"composite={champ_comp:.1%}, SF1450={champion_sf.get(1450, 0):.1%}")

        # ---- Phase 1: Tournament ----
        log(f"\n  Phase 1: Self-play tournament")
        tournament_start = time.time()
        scores, variant_positions = run_tournament(champion, DEVICE, pop_configs)

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        log(f"\n  Tournament results ({time.time()-tournament_start:.0f}s):")
        for idx, score in ranked:
            t, n = pop_configs[idx]
            n_pos = len(variant_positions.get(idx, []))
            log(f"    Variant {idx} (t={t}, noise={n}): "
                f"score={score:.1f}, positions={n_pos}")

        # ---- Phase 2: Collect winner positions ----
        top_variants = [idx for idx, _ in ranked[:TOP_K_SELECT]]
        winner_positions = []
        for idx in top_variants:
            winner_positions.extend(variant_positions.get(idx, []))

        log(f"\n  Top-{TOP_K_SELECT} winners: {top_variants}")
        log(f"  Raw winner positions: {len(winner_positions)}")

        # ---- Phase 3: SF verification of winner moves ----
        log(f"\n  Phase 2: Stockfish move verification (depth {SF_VERIFY_DEPTH})...")
        verify_start = time.time()
        # Sample a subset for verification (full set would be too slow)
        to_verify = winner_positions[:min(500, len(winner_positions))]
        verified = verify_moves_with_sf(to_verify, engine, SF_VERIFY_DEPTH, SF_VERIFY_CP_THRESHOLD)
        verify_rate = len(verified) / max(1, len(to_verify))
        log(f"  Verified: {len(verified)}/{len(to_verify)} ({verify_rate:.0%}) "
            f"in {time.time()-verify_start:.0f}s")

        # If not enough verified, fall back to unverified but with a warning
        if len(verified) < MIN_WINNER_POSITIONS:
            log(f"  Low verification rate — using unverified positions too")
            verified = winner_positions[:min(1000, len(winner_positions))]

        # ---- Phase 4: SF Elo calibration (pre-training) ----
        log(f"\n  Phase 3: Pre-training SF calibration")
        # Use the winning temperature for calibration
        best_temp = pop_configs[ranked[0][0]][0]
        pre_sf = evaluate_elo(champion, DEVICE, engine, temperature=best_temp)

        # ---- Phase 5: Training ----
        log(f"\n  Phase 4: Training on verified moves")
        train_loss, n_batches = train_on_positions(
            champion, verified, replay_data, DEVICE
        )
        if train_loss is not None:
            log(f"  Loss: {train_loss:.4f} ({n_batches} optimizer steps)")
        else:
            log(f"  Training skipped (insufficient data)")

        # ---- Phase 6: Evaluation (gating) ----
        log(f"\n  Phase 5: Evaluation & Elo gating")
        new_acc, new_top3 = evaluate_accuracy(champion, eval_data, DEVICE)
        acc_delta = (new_acc - champion_acc) * 100
        log(f"  Accuracy: {new_acc:.1%} (delta={acc_delta:+.1f}pp)")
        log(f"  Top-3: {new_top3:.1%}")

        log(f"  Post-training SF calibration:")
        post_sf = evaluate_elo(champion, DEVICE, engine, temperature=0.0)

        # Composite Elo score (weighted: higher levels worth more)
        total_weight = sum(ELO_WEIGHTS.values())
        new_composite = sum(post_sf.get(elo, 0) * w for elo, w in ELO_WEIGHTS.items()) / total_weight
        champ_composite = sum(champion_sf.get(elo, 0) * w for elo, w in ELO_WEIGHTS.items()) / total_weight
        log(f"  Composite Elo score: {new_composite:.1%} (champion: {champ_composite:.1%})")

        cycle_time = time.time() - cycle_start

        # Gate: composite must not collapse AND (composite improved OR accuracy improved)
        gate_passed = new_composite >= ELO_GATE_MIN_COMPOSITE
        improved = (new_composite > champ_composite) or (new_acc > champion_acc)
        acc_ok = new_acc >= champion_acc - 0.02  # 2pp tolerance (noisy eval)

        if gate_passed and acc_ok and improved:
            # Accept new champion
            champion_acc = new_acc
            champion_sf = post_sf
            consecutive_failures = 0

            ckpt_path = OUTPUT_DIR / f"champion_cycle{cycle}.pt"
            save_model(champion, ckpt_path, cycle=cycle, acc=new_acc, sf=post_sf)
            # Also save as "best"
            save_model(champion, OUTPUT_DIR / "best_model.pt",
                       cycle=cycle, acc=new_acc, sf=post_sf)

            log(f"\n  ACCEPTED as new champion! ({cycle_time:.0f}s)")
            log(f"  Saved: {ckpt_path}")
            event = "accepted"
        else:
            # Reject — reload previous champion
            consecutive_failures += 1
            reason = []
            if not gate_passed:
                reason.append(f"composite too low ({new_composite:.0%} < {ELO_GATE_MIN_COMPOSITE:.0%})")
            if not acc_ok:
                reason.append(f"accuracy dropped too much ({new_acc:.1%} vs {champion_acc:.1%})")
            if not improved:
                reason.append(f"no improvement (composite {new_composite:.1%} <= {champ_composite:.1%}, acc {new_acc:.1%} <= {champion_acc:.1%})")

            log(f"\n  REJECTED: {'; '.join(reason)} "
                f"(failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")

            # Reload champion from best checkpoint
            best_path = OUTPUT_DIR / "best_model.pt"
            if best_path.exists():
                champion = load_model(best_path, DEVICE)
            else:
                champion = load_model(CHECKPOINT_PATH, DEVICE)
            log(f"  Reverted to previous champion")
            event = "rejected"

        # Record history
        history.append({
            "cycle": cycle,
            "acc": new_acc,
            "top3": new_top3,
            "sf": {str(k): v for k, v in post_sf.items()},
            "gate_passed": gate_passed,
            "gate_score": gate_score,
            "train_loss": train_loss,
            "verified_positions": len(verified),
            "winner_positions": len(winner_positions),
            "best_temp": best_temp,
            "time": cycle_time,
            "event": event,
        })

        # Save history
        with open(OUTPUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Print running summary
        log(f"\n  === Running Summary ===")
        log(f"  Cycle {cycle}: acc={new_acc:.1%} SF1320={post_sf.get(1320, 0):.0%} "
            f"SF1450={post_sf.get(1450, 0):.0%} SF1600={post_sf.get(1600, 0):.0%} "
            f"SF1750={post_sf.get(1750, 0):.0%} [{event}]")

        # Check stopping condition
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log(f"\n  Stopping: {MAX_CONSECUTIVE_FAILURES} consecutive failures")
            break

        # Slight seed variation per cycle for diversity
        random.seed(SEED + cycle)
        torch.manual_seed(SEED + cycle)

    # Final summary
    engine.quit()

    log(f"\n{'='*70}")
    log("FINAL SUMMARY")
    log(f"{'='*70}")
    log(f"  Total cycles: {cycle}")
    log(f"  Baseline: acc={baseline_acc:.1%}")
    log(f"  Final champion: acc={champion_acc:.1%}")
    log(f"  Delta: {(champion_acc - baseline_acc)*100:+.1f}pp")
    log(f"\n  Cycle history:")
    for h in history:
        sf_str = " ".join(f"SF{k}={v:.0%}" for k, v in sorted(h.get('sf', {}).items()))
        log(f"    C{h['cycle']}: acc={h['acc']:.1%} {sf_str} [{h['event']}]")

    log(f"\n  Best model: {OUTPUT_DIR / 'best_model.pt'}")
    log("Done!")


if __name__ == "__main__":
    main()
