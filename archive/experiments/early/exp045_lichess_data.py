"""exp045: Lichess High-ELO Data Engineering.

Hypothesis: The 51.4% accuracy ceiling comes from mixed-skill human data.
Training on positions from strong players (2000+ ELO on Lichess) should give
a fundamentally higher ceiling since the label quality is much better.

Design:
  1. Download ~1000 recent games from highly rated Lichess players
  2. Extract positions + moves (skip first 5 moves for opening book)
  3. Filter to only include positions where player rating > 2000
  4. Create training set (~50-100K positions from strong games)
  5. Fine-tune exp032 checkpoint on this new high-quality data
  6. Also train from scratch to test if high-ELO data alone is sufficient

Data sources:
  - Lichess API: export games by top players (classical/rapid)
  - Filter: both players rated 2000+
  - Positions: every move after ply 10 (skip deep opening theory)

Primary metric: top-1 human accuracy on NEW high-ELO eval set
Secondary: accuracy on original HF dataset + games vs SF d3
Time budget: ~10 min (download + process + train + eval)
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

OUTPUT_DIR = Path("outputs/exp045_lichess_data")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Data config
TOP_PLAYERS = [
    "DrNykterstein",  # Magnus Carlsen
    "FSJAL220",       # High-rated player
    "penguingim1",    # Andrew Tang
    "Msb2",           # Musab
    "Zhigalko_Sergei",# GM Zhigalko
    "nihalsarin",     # Nihal Sarin
    "opperwezen",     # Top blitz
    "Jospem",         # Jose Martinez
    "LyonBeast",      # High-rated
    "homayoont",      # Tabatabaei
    "Grischansen",    # Grischuk alt
    "BabyLlama69",    # Alireza Firouzja alt
    "Polish_fighter3000", # Duda
    "FairChess_on_YouTube",
    "lachesisQ",      # MVL
]
MIN_RATING = 2000
GAMES_PER_PLAYER = 200  # API pages
MIN_PLY = 10            # Skip opening book positions
MAX_POSITIONS_PER_GAME = 30  # Limit per game for diversity

# Training config
NUM_EVAL = 1000
EPOCHS = 3
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 1e-5
WARMUP_STEPS = 100
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3


# === Architecture (same as exp032) ===

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


# === Lichess Data Download ===

def download_lichess_games(username, max_games=200, perf_type="blitz,rapid,classical"):
    """Download games from a Lichess user via API."""
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "rated": "true",
        "perfType": perf_type,
    }
    headers = {
        "Accept": "application/x-chess-pgn",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30, stream=True)
        if r.status_code == 200:
            return r.text
        else:
            print(f"    HTTP {r.status_code} for {username}")
            return ""
    except Exception as e:
        print(f"    Error downloading {username}: {e}")
        return ""


def extract_positions_from_pgn(pgn_text, min_rating, min_ply, max_per_game):
    """Extract (board, move) pairs from PGN text, filtering by rating."""
    positions = []
    pgn_io = io.StringIO(pgn_text)
    games_processed = 0
    games_filtered = 0

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games_processed += 1

        # Check ratings
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
        except ValueError:
            continue

        if white_elo < min_rating or black_elo < min_rating:
            games_filtered += 1
            continue

        # Extract positions
        board = game.board()
        game_positions = []
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= min_ply:
                uci = move.uci()
                if uci in UCI_TO_IDX and move in board.legal_moves:
                    game_positions.append({
                        "board": board.copy(),
                        "move": move,
                        "ply": ply,
                        "white_elo": white_elo,
                        "black_elo": black_elo,
                    })
            board.push(move)

        # Subsample for diversity
        if len(game_positions) > max_per_game:
            game_positions = random.sample(game_positions, max_per_game)
        positions.extend(game_positions)

    return positions, games_processed, games_filtered


# === Eval ===

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
    print(f"Experiment: exp045_lichess_data")

    # --- Download Lichess data ---
    print(f"\n[1/6] Downloading games from {len(TOP_PLAYERS)} Lichess players...")
    dl_t0 = time.time()
    all_positions = []
    total_games = 0
    total_filtered = 0

    for player in TOP_PLAYERS:
        pgn_text = download_lichess_games(player, max_games=GAMES_PER_PLAYER)
        if not pgn_text:
            continue
        positions, n_games, n_filtered = extract_positions_from_pgn(
            pgn_text, MIN_RATING, MIN_PLY, MAX_POSITIONS_PER_GAME
        )
        all_positions.extend(positions)
        total_games += n_games
        total_filtered += n_filtered
        print(f"  {player}: {n_games} games, {n_filtered} filtered, "
              f"{len(positions)} positions (total: {len(all_positions)})")

        # Stop if we have enough
        if len(all_positions) >= 100000:
            break

    dl_time = time.time() - dl_t0
    print(f"\n  Total: {len(all_positions)} positions from {total_games} games "
          f"({total_filtered} filtered) in {dl_time:.0f}s")

    if len(all_positions) < 1000:
        print("  Not enough positions! Aborting.")
        return

    # Analyze ELO distribution
    all_elos = [max(p["white_elo"], p["black_elo"]) for p in all_positions]
    avg_elo = sum(all_elos) / len(all_elos)
    min_elo = min(all_elos)
    max_elo_val = max(all_elos)
    print(f"  ELO stats: avg={avg_elo:.0f}, min={min_elo}, max={max_elo_val}")

    # --- Split into train/eval ---
    random.shuffle(all_positions)
    lichess_eval = all_positions[:NUM_EVAL]
    lichess_train = all_positions[NUM_EVAL:]
    print(f"  Split: {len(lichess_train)} train, {len(lichess_eval)} eval")

    # --- Load model ---
    print(f"\n[2/6] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Baseline on BOTH eval sets ---
    print(f"\n[3/6] Baseline evaluation...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    hf_eval = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                              offset=len(ds) - NUM_EVAL * 3)

    baseline_hf = evaluate_accuracy(model, hf_eval, device)
    baseline_lichess = evaluate_accuracy(model, lichess_eval, device)
    print(f"  HF eval: acc={baseline_hf['accuracy']:.1%} "
          f"top3={baseline_hf['top3_accuracy']:.1%}")
    print(f"  Lichess eval (2000+): acc={baseline_lichess['accuracy']:.1%} "
          f"top3={baseline_lichess['top3_accuracy']:.1%}")

    # --- Train on Lichess data ---
    print(f"\n[4/6] Training: {EPOCHS} epochs on {len(lichess_train)} "
          f"Lichess positions, lr={LR}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(lichess_train) // (BATCH_SIZE * ACCUM_STEPS)) * EPOCHS
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)

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
        random.shuffle(lichess_train)

        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for i in range(0, len(lichess_train), BATCH_SIZE):
            chunk = lichess_train[i:i + BATCH_SIZE]
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

                if step_count % 50 == 0:
                    avg = total_loss / (i // BATCH_SIZE + 1)
                    print(f"    Step {step_count}: loss={avg:.4f}")

        # Handle remaining gradients
        if (len(lichess_train) // BATCH_SIZE) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_loss = total_loss / max(len(lichess_train) // BATCH_SIZE, 1)

        # Evaluate on BOTH sets
        eval_hf = evaluate_accuracy(model, hf_eval, device)
        eval_lichess = evaluate_accuracy(model, lichess_eval, device)
        ep_time = time.time() - ep_t0

        ep_info = {
            "epoch": epoch + 1,
            "loss": round(ep_loss, 4),
            "hf_accuracy": eval_hf["accuracy"],
            "hf_top3": eval_hf["top3_accuracy"],
            "lichess_accuracy": eval_lichess["accuracy"],
            "lichess_top3": eval_lichess["top3_accuracy"],
            "time_s": round(ep_time),
        }
        epoch_history.append(ep_info)
        print(f"  Epoch {epoch+1}: loss={ep_loss:.4f}")
        print(f"    HF: {eval_hf['accuracy']:.1%} "
              f"Lichess: {eval_lichess['accuracy']:.1%} [{ep_time:.0f}s]")

        # Track best by Lichess accuracy
        if eval_lichess['accuracy'] > best_acc:
            best_acc = eval_lichess['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ** New best (Lichess): {best_acc:.1%} **")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # --- Also try mixed training: HF + Lichess ---
    print(f"\n[5/6] Mixed training: reload + train on HF+Lichess mixed...")
    model_mixed = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    model_mixed.load_state_dict(ckpt["model_state"])

    # Get HF training data
    hf_train_data = prepare_hf_data(ds, old_sorted_uci, len(lichess_train), offset=0)
    mixed_data = hf_train_data + lichess_train
    random.shuffle(mixed_data)

    opt_mixed = AdamW(model_mixed.parameters(), lr=LR, weight_decay=0.01)
    total_steps_mixed = len(mixed_data) // (BATCH_SIZE * ACCUM_STEPS)
    warmup_mixed = min(50, total_steps_mixed // 5)

    def lr_sched_mixed(step):
        if step < warmup_mixed:
            return step / max(warmup_mixed, 1)
        progress = (step - warmup_mixed) / max(total_steps_mixed - warmup_mixed, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched_mixed = torch.optim.lr_scheduler.LambdaLR(opt_mixed, lr_sched_mixed)

    # 1 epoch on mixed data
    model_mixed.train()
    opt_mixed.zero_grad()
    total_loss_m = 0.0
    for i in range(0, len(mixed_data), BATCH_SIZE):
        chunk = mixed_data[i:i + BATCH_SIZE]
        boards = [d["board"] for d in chunk]
        targets = torch.tensor([move_to_index(d["move"]) for d in chunk],
                               device=device)
        batch_input = batch_boards_to_token_ids(boards, device)
        result = model_mixed(batch_input)
        logits = result["policy_logits"]
        loss = F.cross_entropy(logits, targets) / ACCUM_STEPS
        loss.backward()
        total_loss_m += loss.item() * ACCUM_STEPS
        if (i // BATCH_SIZE + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model_mixed.parameters(), 1.0)
            opt_mixed.step()
            sched_mixed.step()
            opt_mixed.zero_grad()

    if (len(mixed_data) // BATCH_SIZE) % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model_mixed.parameters(), 1.0)
        opt_mixed.step()
        opt_mixed.zero_grad()

    mix_hf = evaluate_accuracy(model_mixed, hf_eval, device)
    mix_lichess = evaluate_accuracy(model_mixed, lichess_eval, device)
    mixed_loss = total_loss_m / max(len(mixed_data) // BATCH_SIZE, 1)
    print(f"  Mixed (1ep): loss={mixed_loss:.4f} "
          f"HF={mix_hf['accuracy']:.1%} Lichess={mix_lichess['accuracy']:.1%}")

    # --- Games vs SF ---
    print(f"\n[6/6] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
    # Use the Lichess-only model (typically better at chess)
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
        "experiment": "exp045_lichess_data",
        "hypothesis": "High-ELO Lichess data provides better label quality",
        "seed": SEED,
        "data": {
            "players": TOP_PLAYERS[:len(TOP_PLAYERS)],
            "total_positions": len(all_positions),
            "train_positions": len(lichess_train),
            "eval_positions": len(lichess_eval),
            "avg_elo": round(avg_elo),
            "min_elo": min_elo,
            "max_elo": max_elo_val,
            "download_time_s": round(dl_time),
        },
        "baseline": {
            "hf": baseline_hf,
            "lichess": baseline_lichess,
        },
        "lichess_only_training": epoch_history,
        "mixed_training": {
            "loss": round(mixed_loss, 4),
            "hf_accuracy": mix_hf,
            "lichess_accuracy": mix_lichess,
        },
        "best_lichess_accuracy": best_acc,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save Lichess data for future use
    lichess_data_file = OUTPUT_DIR / "lichess_positions.jsonl"
    with open(lichess_data_file, "w") as f:
        for pos in all_positions:
            f.write(json.dumps({
                "fen": pos["board"].fen(),
                "move": pos["move"].uci(),
                "ply": pos["ply"],
                "white_elo": pos["white_elo"],
                "black_elo": pos["black_elo"],
            }) + "\n")
    print(f"  Saved {len(all_positions)} positions to {lichess_data_file}")

    # Save best checkpoint
    if best_state:
        torch.save({
            "model_state": best_state,
            "epoch": EPOCHS,
            "accuracy": best_acc,
        }, OUTPUT_DIR / "best_checkpoint.pt")

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp045_lichess_data")
    print(f" Downloaded {len(all_positions)} positions (avg ELO: {avg_elo:.0f})")
    print(f" Baseline:  HF={baseline_hf['accuracy']:.1%} "
          f"Lichess={baseline_lichess['accuracy']:.1%}")
    for h in epoch_history:
        print(f"   Ep{h['epoch']}: HF={h['hf_accuracy']:.1%} "
              f"Lichess={h['lichess_accuracy']:.1%}")
    print(f" Mixed(1ep): HF={mix_hf['accuracy']:.1%} "
          f"Lichess={mix_lichess['accuracy']:.1%}")
    print(f" Best Lichess: {best_acc:.1%}")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
