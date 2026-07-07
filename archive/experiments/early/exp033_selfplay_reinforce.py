"""exp033: Self-play REINFORCE — improve policy via game outcomes.

Hypothesis: Fine-tuning with REINFORCE from self-play game outcomes will
improve game-playing strength (wins vs SF) even if supervised accuracy
doesn't increase, because the model learns from actual game dynamics
rather than imitating noisy human moves.

Background:
  - exp032: 51.4% accuracy, 78.4% top3, W0/D1/L7 vs SF d3
  - The ~51% ceiling appears to be data quality (amateur game labels)
  - Self-play can provide unlimited, self-consistent training signal
  - REINFORCE: R * ∇log π(a|s) where R = game outcome
  - With value baseline: advantage = R - V(s) reduces variance

Design:
  - Load exp032 checkpoint (51.4%)
  - 20 generations of: play 30 self-play games → REINFORCE update
  - Temperature=0.5 for diverse but reasonable play
  - LR=1e-5, advantage = outcome - V(s)
  - Also update value head with MSE on game outcomes
  - Evaluate supervised accuracy + games vs SF after training

Primary metric: games vs SF d3 (wins/draws), supervised accuracy
Time budget: ~10-15 min (self-play is fast with our tiny model)
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
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp033_selfplay_reinforce")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Self-play config
NUM_GENERATIONS = 20
GAMES_PER_GEN = 30
TEMPERATURE = 0.8  # Higher temperature for more diverse play & decisive games
MAX_GAME_MOVES = 100  # Shorter games to avoid endless draws
POLICY_LR = 1e-5
VALUE_LR = 1e-4
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01  # Encourage exploration
MAX_BATCH = 64  # Max positions per gradient sub-batch (spatial head is memory-hungry)

# Eval config
NUM_EVAL = 1000
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

    def forward(self, board_input, move_targets=None, value_targets=None, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)

        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)

        result = {"policy_logits": policy_logits, "value_logits": value_logits}

        device = board_input["piece_ids"].device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            total_loss = total_loss + F.cross_entropy(policy_logits, move_targets)
        if value_targets is not None:
            total_loss = total_loss + 0.5 * F.cross_entropy(value_logits, value_targets)
        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def predict_move(self, board):
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
    def sample_move(self, board, temperature=0.5):
        """Sample a move with temperature (for self-play exploration)."""
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        # Numerical stability: subtract max of legal logits before temperature scaling
        legal_logits = logits[mask]
        if legal_logits.numel() == 0:
            # No legal moves — shouldn't happen, but fallback
            idx = 0
            probs = torch.zeros_like(logits)
        else:
            logits_shifted = logits - legal_logits.max()
            probs = F.softmax(logits_shifted / max(temperature, 0.01), dim=-1)
            # Clamp to avoid numerical issues
            probs = probs.clamp(min=0.0)
            prob_sum = probs.sum()
            if prob_sum <= 0 or prob_sum.isnan():
                # Fallback: uniform over legal moves
                probs = mask.float()
                probs = probs / probs.sum()
            else:
                probs = probs / prob_sum
            idx = torch.multinomial(probs, 1).item()
        return index_to_move(idx), idx, probs

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Self-play game collection ===

def play_selfplay_game(model, temperature=0.5, max_moves=150):
    """Play one self-play game, collecting (board, move_idx, side) for each move.

    Returns: (trajectory, outcome, termination, n_moves)
    - trajectory: list of (board_copy, move_idx, is_white_turn)
    - outcome: "1-0", "0-1", "1/2-1/2"
    """
    board = chess.Board()
    trajectory = []

    for _ in range(max_moves):
        if board.is_game_over():
            break
        if not list(board.legal_moves):
            break

        is_white = board.turn == chess.WHITE
        board_copy = board.copy()

        _, move_idx, _ = model.sample_move(board, temperature)

        trajectory.append((board_copy, move_idx, is_white))
        board.push(index_to_move(move_idx))

    # Determine outcome
    if board.is_checkmate():
        outcome = "0-1" if board.turn == chess.WHITE else "1-0"
        termination = "checkmate"
    elif board.is_stalemate():
        outcome = "1/2-1/2"
        termination = "stalemate"
    elif board.is_repetition():
        outcome = "1/2-1/2"
        termination = "repetition"
    elif board.can_claim_fifty_moves():
        outcome = "1/2-1/2"
        termination = "fifty_moves"
    else:
        # Max moves reached — adjudicate by material
        material = _material_balance(board)
        if material >= 3:  # White up a minor piece or more
            outcome = "1-0"
            termination = "adjudication"
        elif material <= -3:
            outcome = "0-1"
            termination = "adjudication"
        else:
            outcome = "1/2-1/2"
            termination = "max_moves"

    return trajectory, outcome, termination, len(trajectory)


_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_balance(board):
    """Material advantage for White (positive) or Black (negative)."""
    bal = 0
    for sq, piece in board.piece_map().items():
        val = _PIECE_VALUES.get(piece.piece_type, 0)
        bal += val if piece.color == chess.WHITE else -val
    return bal


def play_generation(model, num_games, temperature, max_moves):
    """Play a set of self-play games, return aggregated training data."""
    all_positions = []  # (board, move_idx, reward)
    stats = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "total_moves": 0,
             "terminations": {}}

    for _ in range(num_games):
        traj, outcome, term, n_moves = play_selfplay_game(
            model, temperature, max_moves)

        stats[outcome] = stats.get(outcome, 0) + 1
        stats["total_moves"] += n_moves
        stats["terminations"][term] = stats["terminations"].get(term, 0) + 1

        # Assign rewards based on game outcome
        if outcome == "1-0":
            white_reward, black_reward = 1.0, -1.0
        elif outcome == "0-1":
            white_reward, black_reward = -1.0, 1.0
        else:
            white_reward, black_reward = 0.0, 0.0  # draws give no signal

        for board, move_idx, is_white in traj:
            reward = white_reward if is_white else black_reward
            all_positions.append((board, move_idx, reward))

    return all_positions, stats


# === REINFORCE update ===

def reinforce_update(model, optimizer, positions, device, max_batch=64):
    """One REINFORCE gradient step on collected self-play positions.

    Processes in sub-batches to avoid OOM from spatial policy head.

    For each position:
      advantage = reward - V(board)  [reduces variance]
      policy_loss = -advantage * log π(action|board)
      value_loss = CE(V(board), WDL_target)
      entropy_loss = -entropy(π)

    Returns dict of loss components.
    """
    if not positions:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "n_pos": 0}

    # Filter out draws (reward=0) — they provide no gradient signal
    decisive_pos = [(b, m, r) for b, m, r in positions if r != 0.0]
    if not decisive_pos:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0,
                "n_pos": len(positions), "n_decisive": 0}

    # Subsample if too many positions (keep training fast)
    if len(decisive_pos) > max_batch * 4:
        decisive_pos = random.sample(decisive_pos, max_batch * 4)

    model.train()
    optimizer.zero_grad()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_sub_batches = 0

    # Process in sub-batches
    for start in range(0, len(decisive_pos), max_batch):
        batch = decisive_pos[start:start + max_batch]
        boards = [p[0] for p in batch]
        move_indices = torch.tensor([p[1] for p in batch],
                                    dtype=torch.long, device=device)
        rewards = torch.tensor([p[2] for p in batch],
                               dtype=torch.float32, device=device)

        # Forward pass
        board_input = batch_boards_to_token_ids(boards, device)
        result = model(board_input)
        policy_logits = result["policy_logits"]  # (B, VOCAB_SIZE)
        value_logits = result["value_logits"]    # (B, 3) — WDL

        # Compute value estimate from WDL
        value_probs = F.softmax(value_logits, dim=-1)
        value_estimate = value_probs[:, 2] - value_probs[:, 0]  # ∈ [-1, 1]

        # Advantage
        advantage = (rewards - value_estimate.detach())
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy loss with legal move masking
        for j, board in enumerate(boards):
            mask = legal_move_mask(board).to(device)
            policy_logits[j, ~mask] = float("-inf")

        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, move_indices.unsqueeze(1)).squeeze(1)
        policy_loss = -(advantage * action_log_probs).mean()

        # Value loss
        value_targets = (rewards + 1).long()  # -1→0, 0→1, +1→2
        value_loss = F.cross_entropy(value_logits, value_targets)

        # Entropy
        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).nan_to_num(0.0).mean()

        # Scale loss by sub-batch fraction
        n_sub = len(batch)
        n_total = len(decisive_pos)
        scale = n_sub / n_total

        loss = scale * (POLICY_WEIGHT * policy_loss +
                        VALUE_WEIGHT * value_loss -
                        ENTROPY_WEIGHT * entropy)
        loss.backward()

        total_policy_loss += policy_loss.item() * scale
        total_value_loss += value_loss.item() * scale
        total_entropy += entropy.item() * scale
        n_sub_batches += 1

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "policy_loss": total_policy_loss,
        "value_loss": total_value_loss,
        "entropy": total_entropy,
        "n_pos": len(positions),
        "n_decisive": len(decisive_pos),
    }


# === Evaluation ===

def build_old_move_mapping():
    PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    def inside(idx, f_from): return 0 <= idx < 64 and abs((idx % 8) - f_from) <= 2
    ucis = set()
    for f in range(64):
        ff = f % 8
        for d in (+8, -8, +1, -1):
            t = f + d
            while inside(t, ff): ucis.add(chess.Move(f, t).uci()); t += d
        for d in (+9, -9, +7, -7):
            t = f + d
            while inside(t, ff): ucis.add(chess.Move(f, t).uci()); t += d
        for off in (+17, +15, +10, +6, -6, -10, -15, -17):
            t = f + off
            if inside(t, ff): ucis.add(chess.Move(f, t).uci())
    for f in range(64):
        file_, rank = f % 8, f // 8
        if rank == 6:
            for df in (-9, -8, -7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES: ucis.add(chess.Move(f, t, promotion=p).uci())
        if rank == 1:
            for df in (+9, +8, +7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES: ucis.add(chess.Move(f, t, promotion=p).uci())
    return sorted(ucis)


_PIECE2CHESS = {
    1: (chess.PAWN, chess.WHITE), 2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE), 4: (chess.ROOK, chess.WHITE),
    5: (chess.QUEEN, chess.WHITE), 6: (chess.KING, chess.WHITE),
    7: (chess.PAWN, chess.BLACK), 8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK), 10: (chess.ROOK, chess.BLACK),
    11: (chess.QUEEN, chess.BLACK), 12: (chess.KING, chess.BLACK),
}


def hf_sample_to_board(board_flat, turn_raw):
    board = chess.Board(fen=None)
    board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8; col = idx % 8
            rank = 7 - row; sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = bool(turn_raw)
    return board


def prepare_hf_data(dataset, old_sorted_uci, n, offset=0):
    data, skipped = [], 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n: break
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX: skipped += 1; continue
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError: skipped += 1; continue
        if move not in board.legal_moves: skipped += 1; continue
        winner = s["winner"]
        if winner == 0: vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK): vt = 2
        else: vt = 0
        data.append({"board": board, "move": move, "value_target": vt})
    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


def evaluate_accuracy(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, target_idx in enumerate(targets):
                total += 1
                if preds[j] == target_idx: correct += 1
                if target_idx in top3s[j]: top3_correct += 1
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "top3_accuracy": round(top3_correct / max(total, 1), 4),
        "total": total,
    }


def play_game_vs_stockfish(model, sf_depth, model_color, device, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = model.predict_move(board)
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
    }


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp033_selfplay_reinforce")
    print(f"Hypothesis: REINFORCE from self-play outcomes improves game strength")

    # --- Load checkpoint ---
    print("\n[1/4] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")
    print(f"  Trainable: {model.trainable_params():,}")

    # --- Load eval data for supervised accuracy comparison ---
    print("\n[2/4] Loading eval data...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                offset=len(ds) - NUM_EVAL * 3)

    # Baseline eval before self-play
    baseline_eval = evaluate_accuracy(model, eval_data, device)
    print(f"  Baseline: acc={baseline_eval['accuracy']:.1%} "
          f"top3={baseline_eval['top3_accuracy']:.1%}")

    # --- Self-play training loop ---
    print(f"\n[3/4] Self-play REINFORCE: {NUM_GENERATIONS} generations × "
          f"{GAMES_PER_GEN} games")

    optimizer = AdamW(model.parameters(), lr=POLICY_LR, weight_decay=0.01)
    generation_history = []

    for gen in range(NUM_GENERATIONS):
        gen_t0 = time.time()

        # Play self-play games
        positions, game_stats = play_generation(
            model, GAMES_PER_GEN, TEMPERATURE, MAX_GAME_MOVES)

        # REINFORCE update
        update_stats = reinforce_update(
            model, optimizer, positions, device, MAX_BATCH)

        # Quick eval every 5 generations
        eval_result = None
        if (gen + 1) % 5 == 0 or gen == NUM_GENERATIONS - 1:
            eval_result = evaluate_accuracy(model, eval_data, device)

        gen_time = time.time() - gen_t0
        gen_record = {
            "generation": gen + 1,
            "games": game_stats,
            "update": update_stats,
            "eval": eval_result,
            "time_s": round(gen_time, 1),
        }
        generation_history.append(gen_record)

        white_wins = game_stats.get("1-0", 0)
        black_wins = game_stats.get("0-1", 0)
        draws = game_stats.get("1/2-1/2", 0)
        avg_moves = game_stats["total_moves"] / max(GAMES_PER_GEN, 1)
        terms = game_stats["terminations"]

        acc_str = f" acc={eval_result['accuracy']:.1%}" if eval_result else ""
        print(f"  Gen {gen+1:2d}: W{white_wins}/B{black_wins}/D{draws} "
              f"avg_moves={avg_moves:.0f} "
              f"pl={update_stats['policy_loss']:.4f} "
              f"vl={update_stats['value_loss']:.4f} "
              f"ent={update_stats['entropy']:.2f} "
              f"decisive={update_stats.get('n_decisive', 0)}"
              f"{acc_str} [{gen_time:.1f}s]")

        # Print termination breakdown
        if terms:
            term_str = " ".join(f"{k}={v}" for k, v in sorted(terms.items()))
            print(f"         terms: {term_str}")

    # --- Final evaluation ---
    print(f"\n[4/4] Final evaluation")
    final_eval = evaluate_accuracy(model, eval_data, device)
    print(f"  Supervised: acc={final_eval['accuracy']:.1%} "
          f"top3={final_eval['top3_accuracy']:.1%}")
    print(f"  (Baseline: acc={baseline_eval['accuracy']:.1%} "
          f"top3={baseline_eval['top3_accuracy']:.1%})")

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
        },
        "accuracy": final_eval["accuracy"],
    }, OUTPUT_DIR / "best_checkpoint.pt")

    # Play games vs Stockfish
    print(f"\n  Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv "
              f"({r['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")

    # --- Save results ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp033_selfplay_reinforce",
        "hypothesis": "REINFORCE from self-play improves game strength",
        "primary_metric": "games vs SF d3 + supervised accuracy",
        "seed": SEED,
        "config": {
            "num_generations": NUM_GENERATIONS,
            "games_per_gen": GAMES_PER_GEN,
            "temperature": TEMPERATURE,
            "max_game_moves": MAX_GAME_MOVES,
            "policy_lr": POLICY_LR,
            "value_lr": VALUE_LR,
            "policy_weight": POLICY_WEIGHT,
            "value_weight": VALUE_WEIGHT,
            "entropy_weight": ENTROPY_WEIGHT,
            "max_batch": MAX_BATCH,
        },
        "model": {"type": "chess_transformer", "layers": NUM_LAYERS,
                  "hidden": HIDDEN_DIM, "heads": NUM_HEADS,
                  "encoder_dim": ENCODER_DIM,
                  "trainable": model.trainable_params()},
        "results": {
            "baseline_eval": baseline_eval,
            "final_eval": final_eval,
            "acc_delta": round(final_eval["accuracy"] - baseline_eval["accuracy"], 4),
        },
        "selfplay": {
            "generation_history": generation_history,
            "total_games": NUM_GENERATIONS * GAMES_PER_GEN,
        },
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "comparison": {
            "exp032_acc": 0.514,
            "exp032_games": "W0/D1/L7",
        },
        "timing": {"total_s": total_time},
        "device": str(device),
        "command": "python experiments/exp033_selfplay_reinforce.py",
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp033_selfplay_reinforce")
    print(f" Hypothesis: REINFORCE from self-play improves game strength")
    print(f" Baseline: acc={baseline_eval['accuracy']:.1%} "
          f"top3={baseline_eval['top3_accuracy']:.1%}")
    print(f" After self-play: acc={final_eval['accuracy']:.1%} "
          f"top3={final_eval['top3_accuracy']:.1%}")
    print(f" Delta: {final_eval['accuracy'] - baseline_eval['accuracy']:+.1%}")
    print(f" Games vs SF d{GAME_SF_DEPTH}: W{wins}/D{draws}/L{losses} "
          f"(exp032: W0/D1/L7)")
    print(f" Total self-play games: {NUM_GENERATIONS * GAMES_PER_GEN}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
