"""exp035: Mixed Human+SF training — inject SF signal without forgetting.

Hypothesis: Training on a 90/10 mix of human and SF-labeled data will improve
game-playing strength without losing supervised accuracy, because the model
retains its human-learned policy while gradually incorporating SF signal.

Background:
  - exp032: 51.4% human accuracy, W0/D1/L7 vs SF d3
  - exp033: REINFORCE destroyed model (0.8%) — catastrophic forgetting
  - exp034: Pure SF distill (50K) improved SF acc but worsened games (W0/D0/L8)
  - Human-SF agreement: only 37.4% — targets are very different
  - Key insight: need to MIX human + SF data, not replace one with the other

Design:
  - Load exp032 checkpoint (51.4%)
  - Load 460K human-labeled training data (existing pipeline)
  - Load 50K SF-labeled data (from exp034, re-compute here)
  - Train 2 epochs: each batch is 90% human samples + 10% SF samples
  - LR=3e-5 (moderate — not too aggressive)
  - Evaluate on both human and SF test sets + games vs SF d3

Primary metric: games vs SF d3 + human accuracy retention
Time budget: ~10-15 min
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

OUTPUT_DIR = Path("outputs/exp035_mixed_training")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Training config
NUM_SF_LABEL = 50_000
SF_DEPTH = 8
EPOCHS = 2
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 3e-5
WARMUP_STEPS = 300
SF_MIX_RATIO = 0.10  # 10% of each batch is SF data

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


def prepare_hf_boards(dataset, n, offset=0):
    boards = []
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(boards) >= n: break
        s = dataset[i]
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            if list(board.legal_moves):
                boards.append(board)
        except (ValueError, IndexError):
            continue
    return boards


def label_positions_with_stockfish(boards, sf_depth=8, batch_print=5000):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 128})
    labeled = []
    skipped = 0
    t0 = time.time()
    for i, board in enumerate(boards):
        sf.set_fen_position(board.fen())
        sf_uci = sf.get_best_move()
        if sf_uci is None:
            skipped += 1; continue
        try:
            move = chess.Move.from_uci(sf_uci)
        except (ValueError, chess.InvalidMoveError):
            skipped += 1; continue
        if move not in board.legal_moves:
            skipped += 1; continue
        if sf_uci not in UCI_TO_IDX:
            skipped += 1; continue
        eval_info = sf.get_evaluation()
        cp = eval_info.get("value", 0) if eval_info["type"] == "cp" else \
             (10000 if eval_info["value"] > 0 else -10000)
        if cp > 100: vt = 2
        elif cp < -100: vt = 0
        else: vt = 1
        labeled.append({"board": board, "move": move, "value_target": vt})
        if (i + 1) % batch_print == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(boards)} labeled ({len(labeled)} kept) [{elapsed:.0f}s]")
    print(f"  SF labeling: {len(labeled)}/{len(boards)} [{time.time()-t0:.0f}s]")
    return labeled


# === Training with mixed batches ===

def make_mixed_batches(human_data, sf_data, batch_size, sf_ratio, device):
    """Create batches with sf_ratio fraction of SF samples, rest human."""
    random.shuffle(human_data)
    random.shuffle(sf_data)
    
    sf_per_batch = max(1, int(batch_size * sf_ratio))
    human_per_batch = batch_size - sf_per_batch
    
    batches = []
    h_idx = 0
    s_idx = 0
    
    while h_idx + human_per_batch <= len(human_data):
        # Get human samples
        h_chunk = human_data[h_idx:h_idx + human_per_batch]
        h_idx += human_per_batch
        
        # Get SF samples (wrap around if needed)
        if s_idx + sf_per_batch > len(sf_data):
            random.shuffle(sf_data)
            s_idx = 0
        s_chunk = sf_data[s_idx:s_idx + sf_per_batch]
        s_idx += sf_per_batch
        
        chunk = h_chunk + s_chunk
        random.shuffle(chunk)  # Mix within batch
        
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
    print(f"Experiment: exp035_mixed_training")
    print(f"Hypothesis: 90% human + 10% SF data retains acc while improving play")

    # --- Load checkpoint ---
    print("\n[1/5] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Load data ---
    print("\n[2/5] Loading datasets...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()

    # Eval data
    eval_human = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                 offset=len(ds) - NUM_EVAL * 3)

    # Full training data (human labels)
    max_train = len(ds) - NUM_EVAL * 3
    train_human = prepare_hf_data(ds, old_sorted_uci, max_train, offset=0)

    # Baseline
    baseline = evaluate_accuracy(model, eval_human, device)
    print(f"  Baseline: acc={baseline['accuracy']:.1%} top3={baseline['top3_accuracy']:.1%}")

    # --- SF labeling ---
    print(f"\n[3/5] Labeling {NUM_SF_LABEL} positions with SF d{SF_DEPTH}...")
    # Use different positions than training (from middle of dataset)
    sf_boards = prepare_hf_boards(ds, NUM_SF_LABEL, offset=100_000)
    train_sf = label_positions_with_stockfish(sf_boards, SF_DEPTH)

    # SF eval
    eval_boards = prepare_hf_boards(ds, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    eval_sf = label_positions_with_stockfish(eval_boards, SF_DEPTH, batch_print=10000)
    baseline_sf = evaluate_accuracy(model, eval_sf, device)
    print(f"  Baseline SF: acc={baseline_sf['accuracy']:.1%} "
          f"top3={baseline_sf['top3_accuracy']:.1%}")

    # --- Mixed training ---
    print(f"\n[4/5] Mixed training: {len(train_human):,} human + {len(train_sf):,} SF "
          f"({SF_MIX_RATIO:.0%} SF ratio), {EPOCHS} epochs")

    params = list(model.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_human) // (BATCH_SIZE * ACCUM_STEPS) + 1)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    history = []
    best_human_acc = 0
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        batches = make_mixed_batches(train_human, train_sf, BATCH_SIZE,
                                     SF_MIX_RATIO, device)
        ep_loss = steps = 0
        optimizer.zero_grad()

        for b_idx, (batch_input, move_targets, value_targets) in enumerate(batches):
            result = model(batch_input, move_targets=move_targets,
                          value_targets=value_targets)
            (result["loss"] / ACCUM_STEPS).backward()
            ep_loss += result["loss"].item()
            steps += 1

            if (b_idx + 1) % ACCUM_STEPS == 0 or b_idx == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if steps % 500 == 0:
                print(f"    step {steps}/{len(batches)} loss={ep_loss/steps:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = ep_loss / max(steps, 1)
        ev_human = evaluate_accuracy(model, eval_human, device)
        ev_sf = evaluate_accuracy(model, eval_sf, device)

        if ev_human["accuracy"] > best_human_acc:
            best_human_acc = ev_human["accuracy"]
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
                },
                "epoch": epoch + 1,
                "accuracy": ev_human["accuracy"],
            }, OUTPUT_DIR / "best_checkpoint.pt")

        history.append({
            "epoch": epoch + 1, "loss": round(avg_loss, 4),
            "human_acc": ev_human["accuracy"],
            "human_top3": ev_human["top3_accuracy"],
            "sf_acc": ev_sf["accuracy"],
            "sf_top3": ev_sf["top3_accuracy"],
        })

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} "
              f"human={ev_human['accuracy']:.1%} sf={ev_sf['accuracy']:.1%} "
              f"[{elapsed:.0f}s]")

    # --- Games ---
    print(f"\n[5/5] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}")
    ckpt = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model_state"])

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

    final_human = evaluate_accuracy(model, eval_human, device)
    final_sf = evaluate_accuracy(model, eval_sf, device)

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp035_mixed_training",
        "hypothesis": "90% human + 10% SF mix improves play without forgetting",
        "primary_metric": "human accuracy + games vs SF d3",
        "seed": SEED,
        "data": {"human_train": len(train_human), "sf_train": len(train_sf),
                 "sf_depth": SF_DEPTH, "sf_mix_ratio": SF_MIX_RATIO},
        "model": {"type": "chess_transformer", "layers": NUM_LAYERS,
                  "hidden": HIDDEN_DIM, "heads": NUM_HEADS,
                  "encoder_dim": ENCODER_DIM,
                  "trainable": model.trainable_params()},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                     "accum_steps": ACCUM_STEPS, "lr": LR,
                     "warmup_steps": WARMUP_STEPS},
        "results": {
            "baseline_human": baseline,
            "baseline_sf": baseline_sf,
            "final_human": final_human,
            "final_sf": final_sf,
            "best_human_acc": best_human_acc,
            "history": history,
        },
        "comparison": {
            "exp032_acc": 0.514,
            "human_delta": round(final_human["accuracy"] - 0.514, 4),
        },
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time},
        "device": str(device),
        "command": "python experiments/exp035_mixed_training.py",
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp035_mixed_training")
    print(f" Hypothesis: 90% human + 10% SF mix")
    print(f" Baseline: human={baseline['accuracy']:.1%} SF={baseline_sf['accuracy']:.1%}")
    print(f" Final:    human={final_human['accuracy']:.1%} "
          f"SF={final_sf['accuracy']:.1%}")
    print(f" Human delta: {final_human['accuracy'] - baseline['accuracy']:+.1%}")
    print(f" SF delta: {final_sf['accuracy'] - baseline_sf['accuracy']:+.1%}")
    for h in history:
        print(f"   Epoch {h['epoch']}: loss={h['loss']:.4f} "
              f"human={h['human_acc']:.1%} sf={h['sf_acc']:.1%}")
    print(f" Games vs SF d{GAME_SF_DEPTH}: W{wins}/D{draws}/L{losses} "
          f"(exp032: W0/D1/L7)")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
