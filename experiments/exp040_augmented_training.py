"""exp040: Data Augmentation via Horizontal Board Flip + Label Smoothing.

Hypothesis: Board flip augmentation provides free data diversity. Fine-tuning
exp032 checkpoint on augmented data with label smoothing may push past 51.4%.

Background:
  - exp032 (8L, 460K×10ep): 51.4% acc, W0/D1/L7 (CEILING)
  - exp039 (12L, 460K×6ep): 50.1% — capacity NOT the bottleneck
  - Horizontal flip (a↔h file mirror) is strategically valid
  - Label smoothing epsilon=0.1 regularizes overconfident predictions

Design:
  - Start from exp032 checkpoint (51.4%)
  - Take 100K positions → augment to ~200K with horizontal flip
  - Fine-tune 2 epochs with label smoothing, LR=1e-5
  - Evaluate on ORIGINAL (non-flipped) test positions only + games

Primary metric: accuracy on real positions + games vs SF d3
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
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp040_augmented_training")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN_RAW = 100_000  # 100K raw → ~200K augmented
NUM_EVAL = 1000
EPOCHS = 2
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 1e-5  # Conservative for fine-tuning
WARMUP_STEPS = 100
LABEL_SMOOTHING = 0.1
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3


# === Board Flip Augmentation ===

_FILE_MIRROR = {
    'a': 'h', 'b': 'g', 'c': 'f', 'd': 'e',
    'e': 'd', 'f': 'c', 'g': 'b', 'h': 'a',
}


def flip_board(board):
    """Create horizontally mirrored board (swap a↔h files)."""
    flipped = chess.Board(fen=None)
    flipped.clear()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            file_ = chess.square_file(sq)
            rank = chess.square_rank(sq)
            new_file = 7 - file_  # Mirror: 0↔7, 1↔6, etc.
            new_sq = chess.square(new_file, rank)
            flipped.set_piece_at(new_sq, piece)
    flipped.turn = board.turn
    return flipped


def flip_uci(uci_str):
    """Mirror a UCI move string horizontally."""
    result = []
    for c in uci_str:
        if c in _FILE_MIRROR:
            result.append(_FILE_MIRROR[c])
        else:
            result.append(c)
    return ''.join(result)


def flip_move(move):
    """Mirror a chess.Move horizontally."""
    flipped_uci = flip_uci(move.uci())
    return chess.Move.from_uci(flipped_uci)


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

    def forward(self, board_input, move_targets=None, value_targets=None,
                label_smoothing=0.0, **kw):
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
            total_loss = total_loss + F.cross_entropy(
                policy_logits, move_targets, label_smoothing=label_smoothing
            )
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

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        winner = s["winner"]
        if winner == 0: vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK): vt = 2
        else: vt = 0
        data.append({"board": board, "move": move, "value_target": vt})
    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


def augment_with_flip(data):
    """Double the dataset by adding horizontally flipped positions."""
    augmented = list(data)  # Keep originals
    flipped_count = 0
    skip_count = 0
    for d in data:
        flipped_board = flip_board(d["board"])
        flipped_move = flip_move(d["move"])
        # Verify the flipped move is in the new vocabulary and legal
        flipped_uci = flipped_move.uci()
        if flipped_uci not in UCI_TO_IDX:
            skip_count += 1
            continue
        if flipped_move not in flipped_board.legal_moves:
            skip_count += 1
            continue
        augmented.append({
            "board": flipped_board,
            "move": flipped_move,
            "value_target": d["value_target"],
        })
        flipped_count += 1
    print(f"  Augmented: {len(data)} → {len(augmented)} "
          f"(+{flipped_count} flipped, {skip_count} skipped)")
    return augmented


# === Training ===

def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        if len(chunk) < batch_size // 2: continue
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk],
                                    dtype=torch.long, device=device)
        value_targets = torch.tensor([d["value_target"] for d in chunk],
                                     dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


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
    print(f"Experiment: exp040_augmented_training")
    print(f"Fine-tuning from exp032, label_smoothing={LABEL_SMOOTHING}")

    # --- Load checkpoint ---
    print(f"\n[1/4] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")
    print(f"  Parameters: {model.trainable_params():,}")

    # --- Load data ---
    print(f"\n[2/4] Loading dataset and augmenting...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()

    # Eval data: REAL positions only (NOT augmented)
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                offset=len(ds) - NUM_EVAL * 3)

    # Baseline on real positions
    baseline = evaluate_accuracy(model, eval_data, device)
    print(f"  Baseline (real positions): acc={baseline['accuracy']:.1%} "
          f"top3={baseline['top3_accuracy']:.1%}")

    # Training data: subset + augment
    train_data_raw = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN_RAW, offset=0)
    train_data = augment_with_flip(train_data_raw)

    # --- Training ---
    print(f"\n[3/4] Fine-tuning {EPOCHS} epochs on {len(train_data):,} positions "
          f"(batch={BATCH_SIZE}x{ACCUM_STEPS}, lr={LR}, "
          f"label_smoothing={LABEL_SMOOTHING})")

    params = list(model.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // (BATCH_SIZE * ACCUM_STEPS) + 1)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    history = []
    best_acc = baseline["accuracy"]
    best_state = None
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0
        optimizer.zero_grad()
        epoch_t0 = time.time()

        for b_idx, (batch_input, move_targets, value_targets) in enumerate(batches):
            result = model(batch_input, move_targets=move_targets,
                          value_targets=value_targets,
                          label_smoothing=LABEL_SMOOTHING)
            (result["loss"] / ACCUM_STEPS).backward()
            ep_loss += result["loss"].item()
            steps += 1

            if (b_idx + 1) % ACCUM_STEPS == 0 or b_idx == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if steps % 200 == 0:
                print(f"    step {steps}/{len(batches)} loss={ep_loss/steps:.4f}")

        epoch_time = time.time() - epoch_t0
        # Evaluate on REAL positions only
        eval_result = evaluate_accuracy(model, eval_data, device)
        history.append({
            "epoch": epoch + 1,
            "loss": round(ep_loss / max(steps, 1), 4),
            "accuracy": eval_result["accuracy"],
            "top3_accuracy": eval_result["top3_accuracy"],
            "time_s": round(epoch_time),
        })
        print(f"  Epoch {epoch+1}: loss={ep_loss/steps:.4f} "
              f"acc={eval_result['accuracy']:.1%} "
              f"top3={eval_result['top3_accuracy']:.1%} "
              f"[{epoch_time:.0f}s]")

        if eval_result["accuracy"] > best_acc:
            best_acc = eval_result["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state": best_state,
                "epoch": epoch + 1,
                "accuracy": best_acc,
                "config": {
                    "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
                },
            }, OUTPUT_DIR / "best_checkpoint.pt")
            print(f"    -> New best! Saved checkpoint (acc={best_acc:.1%})")

    # --- Load best and play games ---
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

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
        "experiment": "exp040_augmented_training",
        "hypothesis": "Board flip augmentation + label smoothing from exp032 checkpoint",
        "primary_metric": "accuracy on real positions + games vs SF d3",
        "eval_note": "Evaluated on REAL (non-flipped) positions only",
        "seed": SEED,
        "baseline": baseline,
        "model": {
            "layers": NUM_LAYERS, "hidden": HIDDEN_DIM, "heads": NUM_HEADS,
            "params": model.trainable_params(),
        },
        "training": {
            "raw_data": len(train_data_raw),
            "augmented_data": len(train_data),
            "epochs": EPOCHS,
            "effective_epochs": EPOCHS * 2,
            "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
            "lr": LR, "warmup": WARMUP_STEPS,
            "label_smoothing": LABEL_SMOOTHING,
            "history": history,
        },
        "best_accuracy": best_acc,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "comparison": {
            "exp024_8L_3ep": {"acc": 0.487, "games": "W0/D2/L6"},
            "exp031_8L_6ep": {"acc": 0.512, "games": "W0/D1/L7"},
            "exp032_8L_10ep": {"acc": 0.514, "games": "W0/D1/L7"},
        },
        "timing": {"total_s": total_time},
        "device": str(device),
        "command": "python experiments/exp040_augmented_training.py",
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp040_augmented_training")
    print(f" Fine-tuned from exp032, label_smoothing={LABEL_SMOOTHING}")
    print(f" Data: {len(train_data_raw):,} raw → {len(train_data):,} augmented × {EPOCHS}ep")
    print(f" Baseline: {baseline['accuracy']:.1%} (real positions)")
    for h in history:
        print(f"   Ep{h['epoch']}: acc={h['accuracy']:.1%} "
              f"top3={h['top3_accuracy']:.1%} loss={h['loss']:.4f} [{h['time_s']}s]")
    print(f" Best: {best_acc:.1%}")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Baselines: 8L/6ep=51.2% W0/D1/L7, 8L/10ep=51.4%")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
