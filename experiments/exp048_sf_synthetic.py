"""exp048: Synthetic SF-labeled dataset — train on perfectly labeled data.

Hypothesis: Training on positions labeled by Stockfish best-move provides
higher quality supervision than training on human moves (even 2600+ ELO).
Unlike exp046 (human moves from ~2598 ELO players), this uses SF depth 8
as ground truth, so every label is the objectively best move.

Data pipeline:
  1. Generate 200K diverse positions via random+structured play
  2. Label each with SF depth 8 best move (+ eval + top3)
  3. Cache to JSONL for reuse
  4. Two training approaches:
     A) From scratch on SF data
     B) Fine-tune exp032 (HF pretrained) on SF data
  5. Evaluate on BOTH SF-labeled eval AND Lichess eval AND HF eval

Key difference from exp043 (SF distillation): here we have MUCH more data
(200K vs 5K) and train from scratch, not just fine-tune.

Time budget: ~23 min for labeling (cached), ~20 min training, ~5 min eval/games
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

OUTPUT_DIR = Path("outputs/exp048_sf_synthetic")
DATA_FILE = Path("outputs/exp048_sf_synthetic/sf_200k_d8.jsonl")
HF_CHECKPOINT = Path("outputs/exp032_continue_training/best_checkpoint.pt")
LICHESS_DATA = Path("outputs/exp046_lichess_large/lichess_2200plus.jsonl")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Data config
NUM_POSITIONS = 200000
SF_DEPTH = 8
SF_THREADS = 4

# Training config
EPOCHS_SCRATCH = 4
EPOCHS_FINETUNE = 2
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR_SCRATCH = 3e-4
LR_FINETUNE = 5e-5
WARMUP_STEPS = 500
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3
NUM_EVAL = 3000


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


# === Position Generation ===

def generate_diverse_positions(n, seed=42):
    """Generate diverse chess positions using stratified random play.

    Strategy:
    - Opening-like: 4-15 ply random play (25%)
    - Middlegame: 16-40 ply (45%)
    - Endgame: 41-80 ply (20%)
    - Extra diversity: 10-30 ply (10%)

    Returns list of chess.Board objects, deduplicated.
    """
    random.seed(seed)
    seen = set()
    positions = []

    ply_ranges = [
        (4, 15, 0.25),    # opening
        (16, 40, 0.45),   # middlegame
        (41, 80, 0.20),   # endgame
        (10, 30, 0.10),   # extra diversity
    ]

    for min_ply, max_ply, fraction in ply_ranges:
        target = int(n * fraction * 1.1)  # Generate 10% extra for dedup
        collected = 0
        attempts = 0
        while collected < target and attempts < target * 30:
            board = chess.Board()
            ply = random.randint(min_ply, max_ply)
            for _ in range(ply):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))
            if not board.is_game_over() and list(board.legal_moves):
                # Normalize for dedup
                key = board.board_fen() + (" w " if board.turn else " b ") + board.castling_xfen()
                if key not in seen:
                    seen.add(key)
                    positions.append(board.copy())
                    collected += 1
            attempts += 1

    random.shuffle(positions)
    return positions[:n]


def label_positions_with_sf(positions, depth, threads, output_path):
    """Label positions with SF best move. Saves incrementally to JSONL.

    Returns list of dicts with fen, best_move, eval_cp, phase.
    """
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=depth,
        parameters={"Threads": threads, "Hash": 256},
    )

    # Resume support
    start_idx = 0
    if output_path.exists():
        with open(output_path) as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= len(positions):
            print(f"  Already labeled {start_idx} positions, loading from cache")
            labeled = []
            with open(output_path) as f:
                for line in f:
                    labeled.append(json.loads(line))
            return labeled
        print(f"  Resuming from position {start_idx}/{len(positions)}")

    labeled = []
    # Load any already-labeled entries
    if start_idx > 0:
        with open(output_path) as f:
            for line in f:
                labeled.append(json.loads(line))

    t0 = time.time()
    mode = "a" if start_idx > 0 else "w"

    with open(output_path, mode) as f:
        for i in range(start_idx, len(positions)):
            board = positions[i]
            try:
                sf.set_fen_position(board.fen())
                best_move = sf.get_best_move()

                if best_move and best_move in UCI_TO_IDX:
                    ev = sf.get_evaluation()
                    eval_cp = ev.get("value", 0) if ev.get("type") == "cp" else (
                        10000 if ev.get("value", 0) > 0 else -10000
                    )

                    # Classify phase
                    material = 0
                    for sq in chess.SQUARES:
                        piece = board.piece_at(sq)
                        if piece and piece.piece_type != chess.KING:
                            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                                    chess.ROOK: 5, chess.QUEEN: 9}
                            material += vals.get(piece.piece_type, 0)
                    if board.fullmove_number <= 10 and material >= 60:
                        phase = "opening"
                    elif material <= 20:
                        phase = "endgame"
                    else:
                        phase = "middlegame"

                    entry = {
                        "fen": board.fen(),
                        "best_move": best_move,
                        "eval_cp": eval_cp,
                        "phase": phase,
                        "num_legal": len(list(board.legal_moves)),
                    }
                    f.write(json.dumps(entry) + "\n")
                    labeled.append(entry)

            except Exception:
                continue

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - start_idx) / elapsed
                remaining = (len(positions) - i - 1) / rate if rate > 0 else 0
                print(f"    Labeled {i+1}/{len(positions)} "
                      f"({rate:.0f} pos/s, ~{remaining/60:.1f} min left)")

    total_time = time.time() - t0
    print(f"  Labeled {len(labeled)} positions in {total_time:.0f}s "
          f"({len(labeled)/max(total_time,1):.0f} pos/s)")

    return labeled


def load_cached_data(path):
    """Load JSONL data file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# === Training Data Conversion ===

def sf_data_to_training(entries):
    """Convert SF-labeled entries to training format."""
    data = []
    for entry in entries:
        try:
            board = chess.Board(entry["fen"])
            move_uci = entry["best_move"]
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves and move_uci in UCI_TO_IDX:
                data.append({"board": board, "move": move})
        except Exception:
            continue
    return data


def lichess_data_to_training(entries):
    """Convert Lichess JSONL entries to training format."""
    data = []
    for entry in entries:
        try:
            board = chess.Board(entry["fen"])
            move = chess.Move.from_uci(entry["move"])
            if move in board.legal_moves and entry["move"] in UCI_TO_IDX:
                data.append({"board": board, "move": move})
        except Exception:
            continue
    return data


# === Training ===

def train_model(model, train_data, eval_data, device, epochs, lr, warmup,
                label=""):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_data) // (BATCH_SIZE * ACCUM_STEPS)) * epochs
    warmup_steps = min(warmup, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
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

                if step_count % 200 == 0:
                    avg = total_loss / (i // BATCH_SIZE + 1)
                    print(f"    [{label}] Step {step_count}: loss={avg:.4f}")

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
        history.append(ep_info)
        print(f"  [{label}] Epoch {epoch+1}: loss={ep_loss:.4f} "
              f"acc={eval_result['accuracy']:.1%} "
              f"top3={eval_result['top3_accuracy']:.1%} [{ep_time:.0f}s]")

        if eval_result['accuracy'] > best_acc:
            best_acc = eval_result['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ** New best: {best_acc:.1%} **")

    if best_state:
        model.load_state_dict(best_state)

    return history, best_acc


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


def play_games(model, device, label=""):
    print(f"\n  [{label}] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"    Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv "
              f"({r['termination']})")
    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")
    print(f"    Score: W{wins}/D{draws}/L{losses}")
    return game_results, {"wins": wins, "draws": draws, "losses": losses}


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp048_sf_synthetic")

    # === STEP 1: Generate or load data ===
    print(f"\n[1/7] Generating/loading SF-labeled data...")

    if DATA_FILE.exists():
        print(f"  Loading cached data from {DATA_FILE}...")
        sf_entries = load_cached_data(DATA_FILE)
        print(f"  Loaded {len(sf_entries)} cached entries")
    else:
        print(f"  Generating {NUM_POSITIONS} diverse positions...")
        gen_t0 = time.time()
        positions = generate_diverse_positions(NUM_POSITIONS, seed=SEED)
        gen_time = time.time() - gen_t0
        print(f"  Generated {len(positions)} positions in {gen_time:.1f}s")

        # Phase distribution
        phases = {"opening": 0, "middlegame": 0, "endgame": 0}
        for b in positions[:1000]:  # Sample
            material = sum(
                {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}.get(b.piece_at(sq).piece_type, 0)
                for sq in chess.SQUARES if b.piece_at(sq) and b.piece_at(sq).piece_type != chess.KING
            )
            if b.fullmove_number <= 10 and material >= 60:
                phases["opening"] += 1
            elif material <= 20:
                phases["endgame"] += 1
            else:
                phases["middlegame"] += 1
        print(f"  Phase distribution (sample 1K): {phases}")

        print(f"  Labeling with SF depth {SF_DEPTH}...")
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        sf_entries = label_positions_with_sf(positions, SF_DEPTH, SF_THREADS, DATA_FILE)

    # Convert to training data
    all_sf_data = sf_data_to_training(sf_entries)
    random.shuffle(all_sf_data)
    print(f"  Valid SF training samples: {len(all_sf_data)}")

    # Split
    sf_eval = all_sf_data[:NUM_EVAL]
    sf_train = all_sf_data[NUM_EVAL:]
    print(f"  SF Train: {len(sf_train)}, SF Eval: {len(sf_eval)}")

    # Phase stats
    phase_counts = {}
    for e in sf_entries:
        p = e.get("phase", "unknown")
        phase_counts[p] = phase_counts.get(p, 0) + 1
    print(f"  Phase breakdown: {phase_counts}")

    # Also load Lichess eval set for cross-comparison
    lichess_eval = None
    if LICHESS_DATA.exists():
        print(f"  Loading Lichess eval set for cross-comparison...")
        lichess_raw = load_cached_data(LICHESS_DATA)
        lichess_all = lichess_data_to_training(lichess_raw)
        random.seed(SEED)
        random.shuffle(lichess_all)
        lichess_eval = lichess_all[:NUM_EVAL]
        print(f"  Lichess eval: {len(lichess_eval)} positions")

    # === STEP 2: From-scratch training on SF data ===
    print(f"\n[2/7] Approach A: From scratch on {len(sf_train)} SF-labeled positions...")
    model_scratch = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    total_params = sum(p.numel() for p in model_scratch.parameters())
    print(f"  Total params: {total_params:,}")

    history_scratch, best_scratch = train_model(
        model_scratch, sf_train, sf_eval, device,
        epochs=EPOCHS_SCRATCH, lr=LR_SCRATCH, warmup=WARMUP_STEPS,
        label="scratch",
    )

    # Cross-eval on Lichess
    scratch_lichess = None
    if lichess_eval:
        scratch_lichess = evaluate_accuracy(model_scratch, lichess_eval, device)
        print(f"  [scratch] On Lichess eval: acc={scratch_lichess['accuracy']:.1%} "
              f"top3={scratch_lichess['top3_accuracy']:.1%}")

    torch.save({
        "model_state": model_scratch.state_dict(),
        "accuracy": best_scratch,
    }, OUTPUT_DIR / "scratch_checkpoint.pt")

    # === STEP 3: Fine-tune exp032 on SF data ===
    print(f"\n[3/7] Approach B: Loading HF pretrained model (exp032)...")
    model_ft = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)

    hf_baseline_sf = {"accuracy": 0, "top3_accuracy": 0}
    hf_baseline_lichess = {"accuracy": 0, "top3_accuracy": 0}

    if HF_CHECKPOINT.exists():
        ckpt = torch.load(HF_CHECKPOINT, map_location=device, weights_only=True)
        model_ft.load_state_dict(ckpt["model_state"])
        hf_baseline_sf = evaluate_accuracy(model_ft, sf_eval, device)
        print(f"  exp032 baseline on SF eval: acc={hf_baseline_sf['accuracy']:.1%} "
              f"top3={hf_baseline_sf['top3_accuracy']:.1%}")
        if lichess_eval:
            hf_baseline_lichess = evaluate_accuracy(model_ft, lichess_eval, device)
            print(f"  exp032 baseline on Lichess eval: acc={hf_baseline_lichess['accuracy']:.1%} "
                  f"top3={hf_baseline_lichess['top3_accuracy']:.1%}")
    else:
        print("  WARNING: No exp032 checkpoint found")

    print(f"\n[4/7] Fine-tuning HF model on SF data...")
    history_ft, best_ft = train_model(
        model_ft, sf_train, sf_eval, device,
        epochs=EPOCHS_FINETUNE, lr=LR_FINETUNE, warmup=200,
        label="finetune",
    )

    ft_lichess = None
    if lichess_eval:
        ft_lichess = evaluate_accuracy(model_ft, lichess_eval, device)
        print(f"  [finetune] On Lichess eval: acc={ft_lichess['accuracy']:.1%} "
              f"top3={ft_lichess['top3_accuracy']:.1%}")

    torch.save({
        "model_state": model_ft.state_dict(),
        "accuracy": best_ft,
    }, OUTPUT_DIR / "finetune_checkpoint.pt")

    # === STEP 5: Games ===
    print(f"\n[5/7] Playing games...")
    games_scratch, score_scratch = play_games(model_scratch, device, "scratch")
    games_ft, score_ft = play_games(model_ft, device, "finetune")

    # === STEP 6: Also eval exp032 for comparison ===
    print(f"\n[6/7] Baseline comparison summary...")
    print(f"  exp032 on SF eval: {hf_baseline_sf['accuracy']:.1%}")
    print(f"  exp032 on Lichess eval: {hf_baseline_lichess['accuracy']:.1%}")
    print(f"  Scratch on SF eval: {best_scratch:.1%}")
    if scratch_lichess:
        print(f"  Scratch on Lichess eval: {scratch_lichess['accuracy']:.1%}")
    print(f"  Finetune on SF eval: {best_ft:.1%}")
    if ft_lichess:
        print(f"  Finetune on Lichess eval: {ft_lichess['accuracy']:.1%}")

    # === STEP 7: Save ===
    total_time = time.time() - t0

    results = {
        "experiment": "exp048_sf_synthetic",
        "hypothesis": "SF-labeled synthetic data provides better training signal than human moves",
        "seed": SEED,
        "data": {
            "total_sf_positions": len(sf_entries),
            "train_positions": len(sf_train),
            "eval_positions": len(sf_eval),
            "phase_breakdown": phase_counts,
        },
        "baselines": {
            "exp032_on_sf_eval": hf_baseline_sf,
            "exp032_on_lichess_eval": hf_baseline_lichess,
        },
        "approach_A_scratch": {
            "training": history_scratch,
            "best_sf_accuracy": best_scratch,
            "lichess_accuracy": scratch_lichess,
            "games": {"results": games_scratch, "score": score_scratch},
        },
        "approach_B_finetune": {
            "training": history_ft,
            "best_sf_accuracy": best_ft,
            "lichess_accuracy": ft_lichess,
            "games": {"results": games_ft, "score": score_ft},
        },
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp048_sf_synthetic")
    print(f" Data: {len(sf_train)} SF-labeled train, {len(sf_eval)} eval")
    print(f" Phase breakdown: {phase_counts}")
    print(f"\n Baselines:")
    print(f"   exp032 on SF eval: {hf_baseline_sf['accuracy']:.1%}")
    print(f"   exp032 on Lichess: {hf_baseline_lichess['accuracy']:.1%}")
    print(f"\n Approach A (from scratch, {EPOCHS_SCRATCH} epochs):")
    for h in history_scratch:
        print(f"   Ep{h['epoch']}: loss={h['loss']:.4f} "
              f"acc={h['accuracy']:.1%} top3={h['top3_accuracy']:.1%}")
    print(f"   Best SF acc: {best_scratch:.1%}")
    if scratch_lichess:
        print(f"   Lichess acc: {scratch_lichess['accuracy']:.1%}")
    print(f"   Games: W{score_scratch['wins']}/D{score_scratch['draws']}/L{score_scratch['losses']}")
    print(f"\n Approach B (HF pretrained → SF finetune, {EPOCHS_FINETUNE} epochs):")
    for h in history_ft:
        print(f"   Ep{h['epoch']}: loss={h['loss']:.4f} "
              f"acc={h['accuracy']:.1%} top3={h['top3_accuracy']:.1%}")
    print(f"   Best SF acc: {best_ft:.1%}")
    if ft_lichess:
        print(f"   Lichess acc: {ft_lichess['accuracy']:.1%}")
    print(f"   Games: W{score_ft['wins']}/D{score_ft['draws']}/L{score_ft['losses']}")
    print(f"\n Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
