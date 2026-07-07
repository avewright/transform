"""exp022: Stockfish Value Head + Alpha-Beta Search

Hypothesis: Training the value head on Stockfish centipawn evaluations (instead
of game W/D/L) and combining with alpha-beta search will dramatically improve
game performance against Stockfish.

Context:
  - exp021: 1-ply search with game-outcome value head = 0/6 vs SF d3
  - The value head trained on W/D/L game outcomes can't distinguish positions
  - We have 10K positions with SF d8 cp evals in sf_labels_10k_d8.jsonl
  - We have a pre-trained spatial policy model (36.5% acc) from exp020

Plan:
  1. Load pre-trained spatial model from exp020 checkpoint
  2. Fine-tune ONLY the value head on SF cp → win-probability mapping
  3. Test with alpha-beta search at depths 1, 2, 3
  4. Play games vs SF d3

Time budget: ~10 min
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
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move
from config import Config

OUTPUT_DIR = Path("outputs/exp022_search")
CHECKPOINT = Path("outputs/exp020_scaled_spatial/best_checkpoint.pt")
SF_DATA = Path("data/sf_labels_10k_d8.jsonl")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_GAMES = 8
GAME_SF_DEPTH = 3
SEED = 42

# Value head training
VALUE_EPOCHS = 5
VALUE_BATCH = 64
VALUE_LR = 1e-3


# === Spatial Model (same as exp019/exp020) ===

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


class SpatialChessModel(nn.Module):
    def __init__(self, qwen_model, encoder, encoder_dim=256, freeze_backbone=True):
        super().__init__()
        if hasattr(qwen_model, 'model') and hasattr(qwen_model, 'lm_head'):
            base_model = qwen_model.model
        else:
            base_model = qwen_model
        self.hidden_size = getattr(base_model.config, 'hidden_size', 1024)
        self.encoder = encoder
        self.input_proj = nn.Linear(encoder_dim, self.hidden_size)
        self.backbone = base_model
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.policy_head = SpatialPolicyHead(self.hidden_size, head_dim=256)
        # Better value head: predict scalar eval (win probability)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # Output in [-1, 1] = loss to win
        )

    def forward(self, board_input, move_targets=None, value_targets=None, **kw):
        tokens = self.encoder(board_input)
        embeds = self.input_proj(tokens)
        backbone_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = self.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()
        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_pred = self.value_head(global_hidden).squeeze(-1)  # (B,)
        result = {"policy_logits": policy_logits, "value_pred": value_pred}
        device = board_input["piece_ids"].device if isinstance(board_input, dict) else board_input.device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            total_loss = total_loss + F.cross_entropy(policy_logits, move_targets)
        if value_targets is not None:
            value_loss = F.mse_loss(value_pred, value_targets)
            result["value_loss"] = value_loss.item()
            total_loss = total_loss + value_loss
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
    def evaluate_position(self, board):
        """Evaluate position, returns value in [-1, 1] from side-to-move perspective."""
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        result = self.forward(board_input)
        return result["value_pred"].item()

    @torch.no_grad()
    def evaluate_positions_batch(self, boards):
        """Batch position evaluation."""
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_batch(boards, device)
        tokens = self.encoder(board_input)
        embeds = self.input_proj(tokens)
        backbone_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = self.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()
        global_hidden = hidden[:, 0, :]
        values = self.value_head(global_hidden).squeeze(-1)
        return values.cpu().tolist()

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Search ===

def search_move(model, board, depth=1, top_k=10):
    """Alpha-beta search using the model's value head.

    Uses iterative deepening with policy-based move ordering.
    """
    model.eval()
    device = next(model.parameters()).device

    if depth == 0:
        _, probs = model.predict_move(board)
        return index_to_move(probs.argmax().item()), 0.0

    # Get policy ordering for legal moves
    board_input = model.encoder.prepare_input(board, device)
    mask = legal_move_mask(board).to(device)
    with torch.no_grad():
        result = model.forward(board_input)
    logits = result["policy_logits"][0]
    logits[~mask] = float("-inf")
    probs = F.softmax(logits, dim=-1)

    # Get top-K moves for search (policy-based ordering)
    k = min(top_k, mask.sum().item())
    top_idxs = probs.topk(k).indices.tolist()
    candidates = [index_to_move(idx) for idx in top_idxs]
    candidates = [m for m in candidates if m in board.legal_moves]

    if not candidates:
        candidates = list(board.legal_moves)[:top_k]

    best_move = candidates[0]
    best_value = -999.0

    for move in candidates:
        b = board.copy()
        b.push(move)

        if b.is_game_over():
            outcome = b.outcome()
            if outcome and outcome.winner is not None:
                # Current player just moved and won
                child_value = -1.0  # Loss for the opponent (side to move)
            else:
                child_value = 0.0  # Draw
        elif depth == 1:
            # Leaf: evaluate from opponent's perspective
            child_value = model.evaluate_position(b)
        else:
            # Recurse: opponent's best reply
            _, child_value = search_move(model, b, depth=depth - 1, top_k=top_k)

        # Negate: what's good for opponent is bad for us
        our_value = -child_value

        if our_value > best_value:
            best_value = our_value
            best_move = move

    return best_move, best_value


# === Centipawn → Win Probability ===

def cp_to_winprob(cp, ply=30):
    """Convert centipawn to win probability using logistic function."""
    k = -0.00368208  # Stockfish's internal conversion constant
    return 1.0 / (1.0 + math.exp(k * cp))


def cp_to_value(cp, ply=30):
    """Convert centipawn to value in [-1, 1] for the side to move."""
    wp = cp_to_winprob(cp, ply)
    return 2.0 * wp - 1.0  # Map [0,1] → [-1,1]


# === Data Loading ===

def load_sf_data(path, max_n=None):
    """Load SF-labeled data for value head training."""
    data = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            board = chess.Board(entry["fen"])
            cp = entry["best_cp"]
            # Normalize cp from white's perspective to side-to-move's perspective
            if not board.turn:  # If black to move, flip sign
                cp = -cp
            value = cp_to_value(cp)
            data.append({"board": board, "value": value, "fen": entry["fen"]})
            if max_n and len(data) >= max_n:
                break
    return data


def play_game_vs_stockfish(model, sf_depth, model_color, device, search_depth=0,
                           search_top_k=10, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            if search_depth > 0:
                pred, _ = search_move(model, board, depth=search_depth, top_k=search_top_k)
            else:
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Build model and load policy weights ---
    print("\n[1/4] Building model and loading policy checkpoint...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    encoder = LearnedBoardEncoder(embed_dim=256)
    chess_model = SpatialChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)

    # Load pre-trained policy weights (encoder + spatial policy head)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    loaded = 0
    model_dict = dict(chess_model.named_parameters())
    for name, tensor in ckpt["model_state"].items():
        if name in model_dict and "value_head" not in name:
            model_dict[name].data.copy_(tensor.to(device))
            loaded += 1
    print(f"  Loaded {loaded} policy parameter tensors from checkpoint")

    # --- Load SF data for value training ---
    print("\n[2/4] Loading Stockfish evaluation data...")
    sf_data = load_sf_data(SF_DATA)
    random.shuffle(sf_data)
    val_split = int(0.9 * len(sf_data))
    train_data = sf_data[:val_split]
    eval_data = sf_data[val_split:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # --- Train value head only ---
    print(f"\n[3/4] Fine-tuning value head on SF evaluations ({VALUE_EPOCHS} epochs)")

    # Freeze everything except value head
    for name, param in chess_model.named_parameters():
        if "value_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    value_params = [p for p in chess_model.parameters() if p.requires_grad]
    print(f"  Value head params: {sum(p.numel() for p in value_params):,}")

    optimizer = AdamW(value_params, lr=VALUE_LR, weight_decay=0.01)

    for epoch in range(VALUE_EPOCHS):
        chess_model.train()
        random.shuffle(train_data)
        total_loss = steps = 0
        for i in range(0, len(train_data), VALUE_BATCH):
            chunk = train_data[i:i + VALUE_BATCH]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor([d["value"] for d in chunk], dtype=torch.float32, device=device)
            batch_input = batch_boards_to_token_ids(boards, device)
            result = chess_model(batch_input, value_targets=targets)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)

        # Evaluate value head accuracy on held-out data
        chess_model.eval()
        correct_sign = total_eval = 0
        mse_sum = 0
        with torch.no_grad():
            for i in range(0, len(eval_data), VALUE_BATCH):
                chunk = eval_data[i:i + VALUE_BATCH]
                boards = [d["board"] for d in chunk]
                targets = [d["value"] for d in chunk]
                values = chess_model.evaluate_positions_batch(boards)
                for pred, target in zip(values, targets):
                    mse_sum += (pred - target) ** 2
                    if (pred > 0 and target > 0) or (pred < 0 and target < 0) or \
                       (abs(pred) < 0.1 and abs(target) < 0.1):
                        correct_sign += 1
                    total_eval += 1

        mse = mse_sum / max(total_eval, 1)
        sign_acc = correct_sign / max(total_eval, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{VALUE_EPOCHS}: loss={avg_loss:.4f} "
              f"mse={mse:.4f} sign_acc={sign_acc:.1%} [{elapsed:.0f}s]")

    # --- Play games with different search depths ---
    print(f"\n[4/4] Playing games vs SF d{GAME_SF_DEPTH}")

    all_game_results = {}
    for search_depth in [0, 1, 2]:
        label = f"depth-{search_depth}" if search_depth > 0 else "policy-only"
        games_to_play = NUM_GAMES if search_depth <= 1 else 4  # Fewer games for deeper search (slow)
        print(f"\n  --- {label} ({games_to_play} games) ---")

        games = []
        for g in range(games_to_play):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            r = play_game_vs_stockfish(
                chess_model, GAME_SF_DEPTH, color, device,
                search_depth=search_depth, search_top_k=8,
            )
            games.append(r)
            sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
            print(f"    Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv ({r['termination']})")

        w = sum(1 for r in games if r["model_result"] == "win")
        d = sum(1 for r in games if r["model_result"] == "draw")
        l = sum(1 for r in games if r["model_result"] == "loss")
        print(f"    Score: W{w}/D{d}/L{l}")
        all_game_results[label] = {"games": games, "score": {"wins": w, "draws": d, "losses": l}}

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp022_search",
        "hypothesis": "SF-trained value head + search improves gameplay",
        "seed": SEED,
        "value_training": {
            "data": str(SF_DATA), "train_n": len(train_data), "eval_n": len(eval_data),
            "epochs": VALUE_EPOCHS,
        },
        "game_results": all_game_results,
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    for label, data in all_game_results.items():
        s = data["score"]
        print(f"  {label}: W{s['wins']}/D{s['draws']}/L{s['losses']}")
    print(f"  Time: {total_time:.0f}s")
    print(f"  Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
