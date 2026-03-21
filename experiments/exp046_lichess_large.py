"""exp046: Large-scale Lichess 2200+ data — train from scratch.

Hypothesis: exp045 showed our model only gets 21.6% on 2860+ ELO positions.
With enough high-quality data (100K+ positions from 2200+ players), we can
train a model that plays stronger chess, measured by both high-ELO accuracy
AND games vs Stockfish.

Data plan:
  - Download from 189 top Lichess players (blitz/rapid/classical/bullet top-50)
  - 300 games per player, both players must be 2000+
  - Target: 200K+ positions
  - Save to disk for reuse

Training plan:
  - Train from SCRATCH on Lichess-only data (not fine-tune from HF-trained model)
  - 8L/512d/8h architecture (same as exp032)
  - Higher epochs since fresh training
  - Evaluate on Lichess eval set AND games vs SF

Time budget: ~10 min for data, longer for training (maybe split across sessions)
"""

import json
import math
import random
import sys
import time
import io
from pathlib import Path

import chess
import chess.pgn
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp046_lichess_large")
DATA_FILE = Path("outputs/exp046_lichess_large/lichess_2200plus.jsonl")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Data config
MIN_RATING = 2000       # Both players must be 2000+
GAMES_PER_PLAYER = 300  # Download more per player
MIN_PLY = 10            # Skip first 5 moves (opening book)
MAX_POSITIONS_PER_GAME = 40
TARGET_POSITIONS = 200000

# Training config
EPOCHS = 6
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 3e-4
WARMUP_STEPS = 300
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3
NUM_EVAL = 2000


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


# === Lichess Download ===

def get_top_players():
    """Fetch top 50 players from each time control."""
    all_users = set()
    for perf in ['blitz', 'rapid', 'classical', 'bullet']:
        try:
            r = requests.get(f'https://lichess.org/api/player/top/50/{perf}',
                           headers={'Accept': 'application/vnd.lichess.v3+json'},
                           timeout=10)
            if r.status_code == 200:
                for u in r.json().get('users', []):
                    all_users.add(u['username'])
        except Exception:
            pass
    return sorted(all_users)


def download_games(username, max_games=300, perf_type="blitz,rapid,classical"):
    """Download PGN games from a Lichess user."""
    url = f"https://lichess.org/api/games/user/{username}"
    params = {"max": max_games, "rated": "true", "perfType": perf_type}
    headers = {"Accept": "application/x-chess-pgn"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=60, stream=True)
        if r.status_code == 200:
            return r.text
        return ""
    except Exception:
        return ""


def extract_positions(pgn_text, min_rating, min_ply, max_per_game):
    """Extract positions from PGN, returning list of dicts."""
    positions = []
    pgn_io = io.StringIO(pgn_text)
    n_games = 0
    n_filtered = 0

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        n_games += 1

        try:
            w_elo = int(game.headers.get("WhiteElo", "0"))
            b_elo = int(game.headers.get("BlackElo", "0"))
        except ValueError:
            continue

        if w_elo < min_rating or b_elo < min_rating:
            n_filtered += 1
            continue

        board = game.board()
        game_pos = []
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= min_ply:
                uci = move.uci()
                if uci in UCI_TO_IDX and move in board.legal_moves:
                    game_pos.append({
                        "fen": board.fen(),
                        "move": uci,
                        "ply": ply,
                        "white_elo": w_elo,
                        "black_elo": b_elo,
                    })
            board.push(move)

        if len(game_pos) > max_per_game:
            game_pos = random.sample(game_pos, max_per_game)
        positions.extend(game_pos)

    return positions, n_games, n_filtered


def load_or_download_data():
    """Load cached data or download fresh from Lichess."""
    if DATA_FILE.exists():
        print(f"  Loading cached data from {DATA_FILE}...")
        positions = []
        with open(DATA_FILE) as f:
            for line in f:
                positions.append(json.loads(line))
        print(f"  Loaded {len(positions)} cached positions")
        return positions

    print(f"  Downloading fresh data from Lichess...")
    players = get_top_players()
    print(f"  Found {len(players)} top players")

    all_positions = []
    dl_t0 = time.time()

    for idx, player in enumerate(players):
        if len(all_positions) >= TARGET_POSITIONS:
            break

        pgn_text = download_games(player, max_games=GAMES_PER_PLAYER)
        if not pgn_text:
            continue

        positions, n_games, n_filtered = extract_positions(
            pgn_text, MIN_RATING, MIN_PLY, MAX_POSITIONS_PER_GAME
        )
        all_positions.extend(positions)

        if (idx + 1) % 10 == 0 or len(positions) > 0:
            elapsed = time.time() - dl_t0
            print(f"  [{idx+1}/{len(players)}] {player}: {n_games} games, "
                  f"+{len(positions)} pos → total {len(all_positions)} "
                  f"[{elapsed:.0f}s]")

        # Rate limiting: Lichess asks for 1 request per second
        time.sleep(0.5)

    dl_time = time.time() - dl_t0
    print(f"\n  Downloaded {len(all_positions)} positions in {dl_time:.0f}s")

    # Save to disk
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        for pos in all_positions:
            f.write(json.dumps(pos) + "\n")
    print(f"  Saved to {DATA_FILE}")

    return all_positions


def positions_to_training_data(positions):
    """Convert saved positions (fen/move dicts) to training-ready (board, move)."""
    data = []
    for pos in positions:
        try:
            board = chess.Board(pos["fen"])
            move = chess.Move.from_uci(pos["move"])
            if move in board.legal_moves and pos["move"] in UCI_TO_IDX:
                data.append({"board": board, "move": move})
        except Exception:
            continue
    return data


# === Eval ===

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
    print(f"Experiment: exp046_lichess_large")

    # --- Data ---
    print("\n[1/5] Loading/downloading Lichess data...")
    raw_positions = load_or_download_data()

    # ELO analysis
    elos = [max(p["white_elo"], p["black_elo"]) for p in raw_positions]
    avg_elo = sum(elos) / len(elos) if elos else 0
    print(f"  Total positions: {len(raw_positions)}")
    print(f"  Avg max ELO: {avg_elo:.0f}")

    # Convert to training data
    all_data = positions_to_training_data(raw_positions)
    random.shuffle(all_data)
    print(f"  Valid training samples: {len(all_data)}")

    # Split
    eval_data = all_data[:NUM_EVAL]
    train_data = all_data[NUM_EVAL:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # --- Build model from scratch ---
    print(f"\n[2/5] Building model from scratch...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # --- Train ---
    print(f"\n[3/5] Training: {EPOCHS} epochs on {len(train_data)} positions")
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_data) // (BATCH_SIZE * ACCUM_STEPS)) * EPOCHS
    warmup_steps = min(WARMUP_STEPS, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_acc = 0.0
    best_state = None
    epoch_history = []

    for epoch in range(EPOCHS):
        ep_t0 = time.time()
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor([move_to_index(d["move"]) for d in chunk],
                                   device=device)
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            loss = F.cross_entropy(logits, targets) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (i // BATCH_SIZE + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 100 == 0:
                    avg = total_loss / (i // BATCH_SIZE + 1)
                    print(f"    Step {step_count}: loss={avg:.4f}")

        if (len(train_data) // BATCH_SIZE) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_loss = total_loss / max(len(train_data) // BATCH_SIZE, 1)
        eval_result = evaluate_accuracy(model, eval_data, device)
        ep_time = time.time() - ep_t0

        ep_info = {
            "epoch": epoch + 1,
            "loss": round(ep_loss, 4),
            "accuracy": eval_result["accuracy"],
            "top3_accuracy": eval_result["top3_accuracy"],
            "time_s": round(ep_time),
        }
        epoch_history.append(ep_info)
        print(f"  Epoch {epoch+1}: loss={ep_loss:.4f} "
              f"acc={eval_result['accuracy']:.1%} "
              f"top3={eval_result['top3_accuracy']:.1%} [{ep_time:.0f}s]")

        if eval_result['accuracy'] > best_acc:
            best_acc = eval_result['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ** New best: {best_acc:.1%} **")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # --- Also evaluate exp032 on Lichess data for comparison ---
    print(f"\n[4/5] Comparing with exp032 checkpoint on Lichess eval...")
    exp032_path = Path("outputs/exp032_continue_training/best_checkpoint.pt")
    if exp032_path.exists():
        model_032 = ChessTransformer(
            encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
        ).to(device)
        ckpt = torch.load(exp032_path, map_location=device, weights_only=True)
        model_032.load_state_dict(ckpt["model_state"])
        exp032_lichess = evaluate_accuracy(model_032, eval_data, device)
        print(f"  exp032 on Lichess eval: acc={exp032_lichess['accuracy']:.1%} "
              f"top3={exp032_lichess['top3_accuracy']:.1%}")
        del model_032
    else:
        exp032_lichess = {"accuracy": 0, "top3_accuracy": 0}

    # --- Games ---
    print(f"\n[5/5] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
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

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "epoch": EPOCHS,
        "accuracy": best_acc,
    }, OUTPUT_DIR / "best_checkpoint.pt")

    results = {
        "experiment": "exp046_lichess_large",
        "hypothesis": "Training from scratch on high-ELO Lichess data",
        "seed": SEED,
        "data": {
            "total_positions": len(raw_positions),
            "train_positions": len(train_data),
            "eval_positions": len(eval_data),
            "avg_elo": round(avg_elo),
        },
        "exp032_on_lichess": exp032_lichess,
        "training": epoch_history,
        "best_accuracy": best_acc,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp046_lichess_large")
    print(f" Data: {len(train_data)} train, {len(eval_data)} eval, avg ELO {avg_elo:.0f}")
    print(f" exp032 on Lichess: {exp032_lichess['accuracy']:.1%}")
    for h in epoch_history:
        print(f"   Ep{h['epoch']}: loss={h['loss']:.4f} "
              f"acc={h['accuracy']:.1%} top3={h['top3_accuracy']:.1%}")
    print(f" Best: {best_acc:.1%}")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
