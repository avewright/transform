"""exp044: Hard Example Mining — train on the model's near-misses.

Hypothesis: The model has 51.4% top-1 and 78.4% top-3 — meaning ~27% of
positions have the correct answer in the model's top-3 but NOT top-1.
These "near-miss" positions are highest-leverage: a small gradient nudge
could promote the correct move from rank 2-3 to rank 1. Training
specifically on these should maximize accuracy improvement per sample.

Background:
  - Every approach failed to break 51.4% — data quality seems to be the issue
  - Key insight: 27% of positions are near-misses (correct in top3, not top1)
  - These positions are "learnable" — the model already partly understands them
  - Classic curriculum learning: focus on boundary examples

Design:
  1. Score 100K training positions with exp032 model
  2. Select positions where target is in top-3 but not top-1 (~27K)
  3. Fine-tune from exp032 on these hard examples only
  4. Very conservative: LR=5e-6, 3 epochs on ~27K
  5. Evaluate on full (unfiltered) eval set

Primary metric: top-1 accuracy on human moves
Time budget: ~6 min (filtering + 3 small epochs + eval + games)
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

OUTPUT_DIR = Path("outputs/exp044_hard_examples")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Config
NUM_CANDIDATES = 100000  # Positions to score for hard example mining
NUM_EVAL = 1000
EPOCHS = 3
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 5e-6           # Very conservative
WARMUP_STEPS = 50
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3


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


# === Data ===

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


def mine_hard_examples(model, data, device, batch_size=256):
    """Find positions where correct move is in top-3 but NOT top-1.
    
    Returns: (hard_examples, stats)
    """
    model.eval()
    hard = []
    easy = 0  # correct in top-1
    hard_count = 0  # correct in top-3 but not top-1
    impossible = 0  # not in top-3

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]

            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")

            preds = logits.argmax(dim=-1).cpu().tolist()
            top3 = logits.topk(3, dim=-1).indices.cpu().tolist()

            for j, target_idx in enumerate(targets):
                if preds[j] == target_idx:
                    easy += 1
                elif target_idx in top3[j]:
                    hard.append(chunk[j])
                    hard_count += 1
                else:
                    impossible += 1

            if (i + batch_size) % 10000 < batch_size:
                total = easy + hard_count + impossible
                print(f"    Scored {total}: easy={easy} hard={hard_count} "
                      f"impossible={impossible}")

    stats = {
        "total": easy + hard_count + impossible,
        "easy_top1": easy,
        "hard_top3_not_top1": hard_count,
        "impossible_not_top3": impossible,
        "easy_pct": round(easy / (easy + hard_count + impossible), 3),
        "hard_pct": round(hard_count / (easy + hard_count + impossible), 3),
        "impossible_pct": round(impossible / (easy + hard_count + impossible), 3),
    }
    return hard, stats


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
    print(f"Experiment: exp044_hard_examples")

    # --- Load model ---
    print("\n[1/5] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Load data ---
    print("\n[2/5] Loading data...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                offset=len(ds) - NUM_EVAL * 3)
    baseline = evaluate_accuracy(model, eval_data, device)
    print(f"  Baseline: acc={baseline['accuracy']:.1%} "
          f"top3={baseline['top3_accuracy']:.1%}")

    # Candidate data for mining (different from eval)
    print(f"\n  Loading {NUM_CANDIDATES} candidate positions...")
    candidates = prepare_hf_data(ds, old_sorted_uci, NUM_CANDIDATES, offset=0)
    print(f"  Got {len(candidates)} candidates")

    # --- Mine hard examples ---
    print(f"\n[3/5] Mining hard examples (top3-not-top1)...")
    mine_t0 = time.time()
    hard_examples, mining_stats = mine_hard_examples(model, candidates, device)
    mine_time = time.time() - mine_t0
    print(f"  Mining stats: {mining_stats}")
    print(f"  Found {len(hard_examples)} hard examples in {mine_time:.0f}s")

    if len(hard_examples) < 100:
        print("  Too few hard examples! Aborting.")
        return

    # --- Train on hard examples ---
    print(f"\n[4/5] Training on {len(hard_examples)} hard examples: "
          f"{EPOCHS} epochs, lr={LR}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(hard_examples) // (BATCH_SIZE * ACCUM_STEPS)) * EPOCHS
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_acc = baseline['accuracy']
    best_state = None
    epoch_history = []

    for epoch in range(EPOCHS):
        ep_t0 = time.time()
        model.train()
        random.shuffle(hard_examples)

        total_loss = 0.0
        optimizer.zero_grad()

        for i in range(0, len(hard_examples), BATCH_SIZE):
            chunk = hard_examples[i:i + BATCH_SIZE]
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

        # Handle remaining gradients
        if (len(hard_examples) // BATCH_SIZE) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_loss = total_loss / max(len(hard_examples) // BATCH_SIZE, 1)
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
              f"top3={eval_result['top3_accuracy']:.1%} "
              f"[{ep_time:.0f}s]")

        if eval_result['accuracy'] > best_acc:
            best_acc = eval_result['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ** New best: {best_acc:.1%} **")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best model: {best_acc:.1%}")
    else:
        print(f"  No improvement over baseline {baseline['accuracy']:.1%}")

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
    results = {
        "experiment": "exp044_hard_examples",
        "hypothesis": "Training on near-miss positions (top3-not-top1) improves accuracy",
        "seed": SEED,
        "config": {
            "num_candidates": NUM_CANDIDATES, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS, "lr": LR,
        },
        "mining": mining_stats,
        "baseline": baseline,
        "training": epoch_history,
        "best_accuracy": best_acc,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time, "mining_s": mine_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp044_hard_examples")
    print(f" Mined {len(hard_examples)} hard examples from {len(candidates)}")
    print(f" Mining breakdown: {mining_stats}")
    print(f" Baseline: {baseline['accuracy']:.1%}")
    print(f" Best: {best_acc:.1%}")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
