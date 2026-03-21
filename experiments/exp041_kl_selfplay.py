"""exp041: KL-Constrained Self-Play — prevent catastrophic forgetting.

Hypothesis: exp033 REINFORCE destroyed the model because unrestricted gradient
updates pushed the policy arbitrarily far from the pretrained distribution.
A KL penalty against the original policy keeps updates conservative, preventing
catastrophic forgetting while still improving from self-play signal.

Background:
  - exp032: 51.4%, W0/D1/L7 (baseline)
  - exp033: Vanilla REINFORCE → 0.8% (DESTROYED in 1 generation)
  - exp033 root cause: ±1 rewards with no KL constraint = catastrophic forgetting
  - PPO / RLHF use KL penalties for exactly this reason

Design:
  - Load exp032 checkpoint as both policy AND reference policy (frozen)
  - Play 50 games of self-play at temperature=0.3 (low noise)
  - REINFORCE update with KL penalty: loss = -R*log(p) + β*KL(p||p_ref)
  - β=0.1 (KL coefficient), small LR=1e-6
  - Only 3 generations to stay within time budget
  - Evaluate on human test + games vs SF d3

Primary metric: human accuracy retention + games vs SF d3
Time budget: ~10 min (3 gens × 50 games + training)
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

OUTPUT_DIR = Path("outputs/exp041_kl_selfplay")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Self-play config
NUM_GENERATIONS = 3
GAMES_PER_GEN = 50
MAX_MOVES = 80
TEMPERATURE = 0.3  # Low temperature for less randomness
KL_COEFF = 0.1     # KL penalty coefficient
SP_LR = 1e-6       # Very conservative LR
MAX_BATCH = 64      # Sub-batch for spatial head memory

# Eval config
NUM_EVAL = 1000
NUM_GAMES = 8
GAME_SF_DEPTH = 3
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42


# === Architecture ===

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
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(hidden_dim, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
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
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs


# === Self-Play ===

def sample_move(model, board, temperature, device):
    """Sample a move from policy with temperature."""
    model.eval()
    board_input = model.encoder.prepare_input(board, device)
    mask = legal_move_mask(board).to(device)
    with torch.no_grad():
        result = model.forward(board_input)
    logits = result["policy_logits"][0]
    logits[~mask] = float("-inf")

    # Temperature-scaled sampling with numerical stability
    legal_logits = logits[mask]
    if legal_logits.numel() == 0:
        return None, None, None
    legal_logits = legal_logits - legal_logits.max()
    scaled = legal_logits / max(temperature, 0.01)
    probs = F.softmax(scaled, dim=-1)
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum()

    try:
        idx_in_legal = torch.multinomial(probs, 1).item()
    except RuntimeError:
        idx_in_legal = 0

    legal_indices = mask.nonzero(as_tuple=True)[0]
    move_idx = legal_indices[idx_in_legal].item()
    move = index_to_move(move_idx)
    log_prob = torch.log(probs[idx_in_legal])
    return move, move_idx, log_prob


def play_self_play_game(model, device, temperature, max_moves):
    """Play one self-play game, recording positions/moves/log_probs for both sides."""
    board = chess.Board()
    trajectory = []  # (board_copy, move_idx, log_prob, side_to_move)

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        move, move_idx, log_prob = sample_move(model, board, temperature, device)
        if move is None or move not in board.legal_moves:
            break
        trajectory.append({
            "board": board.copy(),
            "move_idx": move_idx,
            "log_prob": log_prob,
            "side": board.turn,
        })
        board.push(move)

    # Determine outcome
    result = board.result()
    if result == "1-0":
        white_reward, black_reward = 1.0, -1.0
    elif result == "0-1":
        white_reward, black_reward = -1.0, 1.0
    else:
        # Draw or incomplete: check material
        white_reward, black_reward = 0.0, 0.0

    # Assign rewards to trajectory positions
    for t in trajectory:
        t["reward"] = white_reward if t["side"] == chess.WHITE else black_reward

    return trajectory, result


def compute_kl_constrained_loss(model, ref_model, trajectories, device, kl_coeff):
    """Compute REINFORCE loss with KL penalty against reference policy.
    
    loss = -reward * log_prob(current) + kl_coeff * KL(current || reference)
    where KL is computed over legal moves only.
    """
    # Collect unique boards and their data
    boards = [t["board"] for t in trajectories]
    move_idxs = [t["move_idx"] for t in trajectories]
    rewards = torch.tensor([t["reward"] for t in trajectories],
                           dtype=torch.float32, device=device)

    # Normalize rewards
    if rewards.std() > 0:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_reinforce_loss = torch.tensor(0.0, device=device)
    total_kl_loss = torch.tensor(0.0, device=device)
    n_positions = 0

    # Process in sub-batches
    for i in range(0, len(boards), MAX_BATCH):
        chunk_boards = boards[i:i + MAX_BATCH]
        chunk_move_idxs = move_idxs[i:i + MAX_BATCH]
        chunk_rewards = rewards[i:i + MAX_BATCH]
        batch_size = len(chunk_boards)

        batch_input = batch_boards_to_token_ids(chunk_boards, device)

        # Current policy
        model.train()
        curr_result = model(batch_input)
        curr_logits = curr_result["policy_logits"]

        # Reference policy
        with torch.no_grad():
            ref_result = ref_model(batch_input)
            ref_logits = ref_result["policy_logits"]

        for j in range(batch_size):
            mask = legal_move_mask(chunk_boards[j]).to(device)
            if mask.sum() < 2:
                continue

            curr_legal = curr_logits[j][mask]
            ref_legal = ref_logits[j][mask]

            curr_log_probs = F.log_softmax(curr_legal, dim=-1)
            ref_log_probs = F.log_softmax(ref_legal, dim=-1)
            curr_probs = curr_log_probs.exp()

            # REINFORCE: -reward * log_prob(chosen move)
            legal_indices = mask.nonzero(as_tuple=True)[0]
            move_pos = (legal_indices == chunk_move_idxs[j]).nonzero(as_tuple=True)[0]
            if move_pos.numel() == 0:
                continue
            chosen_log_prob = curr_log_probs[move_pos[0]]
            reinforce = -chunk_rewards[j] * chosen_log_prob

            # KL divergence: sum p_curr * (log p_curr - log p_ref)
            kl = (curr_probs * (curr_log_probs - ref_log_probs)).sum()

            total_reinforce_loss = total_reinforce_loss + reinforce
            total_kl_loss = total_kl_loss + kl
            n_positions += 1

    if n_positions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

    avg_reinforce = total_reinforce_loss / n_positions
    avg_kl = total_kl_loss / n_positions
    total_loss = avg_reinforce + kl_coeff * avg_kl
    return total_loss, avg_reinforce.item(), avg_kl.item()


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
        data.append({"board": board, "move": move})
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
    print(f"Experiment: exp041_kl_selfplay")

    # --- Load models ---
    print("\n[1/4] Loading exp032 checkpoint...")
    def make_model():
        return ChessTransformer(
            encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
        ).to(device)

    model = make_model()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # Reference model (frozen)
    ref_model = make_model()
    ref_model.load_state_dict(ckpt["model_state"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Baseline ---
    print("\n[2/4] Baseline evaluation...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                offset=len(ds) - NUM_EVAL * 3)
    baseline = evaluate_accuracy(model, eval_data, device)
    print(f"  Baseline: acc={baseline['accuracy']:.1%} top3={baseline['top3_accuracy']:.1%}")

    # --- Self-play generations ---
    print(f"\n[3/4] KL-Constrained Self-Play: {NUM_GENERATIONS} gens × "
          f"{GAMES_PER_GEN} games, β={KL_COEFF}, lr={SP_LR}")

    optimizer = AdamW(model.parameters(), lr=SP_LR, weight_decay=0.0)
    gen_history = []

    for gen in range(NUM_GENERATIONS):
        gen_t0 = time.time()
        print(f"\n  Generation {gen+1}/{NUM_GENERATIONS}")

        # Play self-play games
        all_trajectories = []
        outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        total_positions = 0

        for g in range(GAMES_PER_GEN):
            traj, result = play_self_play_game(model, device, TEMPERATURE, MAX_MOVES)
            all_trajectories.extend(traj)
            outcomes[result] = outcomes.get(result, 0) + 1
            total_positions += len(traj)

        print(f"    Games: {outcomes}, Positions: {total_positions}")

        # Filter to decisive positions only
        decisive = [t for t in all_trajectories if abs(t["reward"]) > 0]
        print(f"    Decisive positions: {len(decisive)}/{total_positions}")

        if len(decisive) < 10:
            print(f"    Too few decisive positions, skipping update")
            gen_history.append({
                "gen": gen + 1, "outcomes": outcomes, "positions": total_positions,
                "decisive": len(decisive), "updated": False,
            })
            continue

        # REINFORCE + KL update
        optimizer.zero_grad()
        loss, reinforce_loss, kl_loss = compute_kl_constrained_loss(
            model, ref_model, decisive, device, KL_COEFF
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    NaN/Inf loss, skipping update")
            gen_history.append({
                "gen": gen + 1, "outcomes": outcomes, "positions": total_positions,
                "decisive": len(decisive), "updated": False, "nan": True,
            })
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Evaluate
        eval_result = evaluate_accuracy(model, eval_data, device)
        gen_info = {
            "gen": gen + 1,
            "outcomes": outcomes,
            "positions": total_positions,
            "decisive": len(decisive),
            "reinforce_loss": round(reinforce_loss, 6),
            "kl_loss": round(kl_loss, 6),
            "total_loss": round(loss.item(), 6),
            "accuracy": eval_result["accuracy"],
            "top3_accuracy": eval_result["top3_accuracy"],
            "updated": True,
            "time_s": round(time.time() - gen_t0),
        }
        gen_history.append(gen_info)
        print(f"    Loss: reinforce={reinforce_loss:.6f} kl={kl_loss:.6f} "
              f"total={loss.item():.6f}")
        print(f"    Acc: {eval_result['accuracy']:.1%} "
              f"(baseline: {baseline['accuracy']:.1%}) "
              f"[{gen_info['time_s']}s]")

    # --- Play games vs SF ---
    print(f"\n[4/4] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
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
    print(f"  Score: W{wins}/D{draws}/L{losses}")

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp041_kl_selfplay",
        "hypothesis": "KL constraint prevents catastrophic forgetting during self-play",
        "seed": SEED,
        "config": {
            "generations": NUM_GENERATIONS, "games_per_gen": GAMES_PER_GEN,
            "temperature": TEMPERATURE, "kl_coeff": KL_COEFF, "lr": SP_LR,
        },
        "baseline": baseline,
        "gen_history": gen_history,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "comparison": {"exp032": "51.4% W0/D1/L7", "exp033_vanilla": "DESTROYED 0.8%"},
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp041_kl_selfplay")
    print(f" KL-constrained REINFORCE: β={KL_COEFF}, lr={SP_LR}")
    print(f" Baseline: {baseline['accuracy']:.1%}")
    for h in gen_history:
        if h.get("updated"):
            print(f"   Gen{h['gen']}: acc={h['accuracy']:.1%} "
                  f"reinforce={h['reinforce_loss']:.6f} kl={h['kl_loss']:.6f}")
        else:
            print(f"   Gen{h['gen']}: skipped ({h.get('decisive',0)} decisive)")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
