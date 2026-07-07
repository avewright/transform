"""exp025: Continue chess transformer training — more epochs from exp024 checkpoint.

Hypothesis: exp024 loss was still declining at epoch 3 (3.11→2.50→2.33) and
accuracy was climbing (43.4→46.5→48.7). Training for 3 more epochs should
push accuracy higher and improve game results.

Background:
  - exp024: Chess Transformer 8L/512d, 460K×3ep → 48.7% acc, W0/D2/L6
  - Loss not converged at epoch 3
  - Model can draw SF d3 by repetition but can't win

Changes from exp024:
  - Load exp024 best checkpoint and continue training
  - 3 more epochs (epochs 4-6 effective)
  - Lower starting LR (1e-4) with cosine decay — we're fine-tuning now
  - Also test with soft-target augmentation: add temperature-smoothed label noise

Primary metric: accuracy + game results vs SF d3
Time budget: ~30 min (same as exp024 epoch cost)
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

OUTPUT_DIR = Path("outputs/exp025_continue_training")
CHECKPOINT_PATH = Path("outputs/exp024_scaled_chess_transformer/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_EVAL = 1000
EPOCHS = 3  # Epochs 4-6 effective
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 1e-4  # Lower LR for continuation
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 10  # More games for better signal
GAME_SF_DEPTH = 3


# === Architecture (identical to exp024) ===

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
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input, move_targets=None, value_targets=None, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)

        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_pred = self.value_head(global_hidden).squeeze(-1)

        result = {"policy_logits": policy_logits, "value_pred": value_pred}

        device = board_input["piece_ids"].device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            total_loss = total_loss + F.cross_entropy(policy_logits, move_targets)
        if value_targets is not None:
            wdl_logits = torch.stack([
                -value_pred, torch.zeros_like(value_pred), value_pred,
            ], dim=-1)
            total_loss = total_loss + 0.5 * F.cross_entropy(wdl_logits, value_targets)
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


# === Data (same as exp024) ===

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


# === Training ===

def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], dtype=torch.long, device=device)
        value_targets = torch.tensor([d["value_target"] for d in chunk], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_accuracy(model, eval_data, device, n=None, batch_size=128):
    model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
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
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def play_game_vs_stockfish(model, sf_depth, model_color, device, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[1/4] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    print(f"  Dataset: {len(ds):,} samples")

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    max_train = len(ds) - NUM_EVAL * 3
    train_data = prepare_hf_data(ds, old_sorted_uci, max_train, offset=0)

    # --- Load checkpoint ---
    print(f"\n[2/4] Loading exp024 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=0.1,
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")
    trainable = model.trainable_params()
    print(f"  Trainable: {trainable:,}")

    pre = evaluate_accuracy(model, eval_data, device)
    print(f"  Pre-continue: acc={pre['accuracy']:.1%} top3={pre['top3_accuracy']:.1%}")

    # --- Setup optimizer ---
    params = list(model.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // (BATCH_SIZE * ACCUM_STEPS) + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- Train ---
    start_epoch = ckpt["epoch"]
    print(f"\n[3/4] Continuing training: epochs {start_epoch+1}-{start_epoch+EPOCHS} "
          f"on {len(train_data)} positions (batch={BATCH_SIZE}x{ACCUM_STEPS}, lr={LR})")
    history = []
    best_acc = ckpt["accuracy"]
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0
        optimizer.zero_grad()
        for b_idx, (batch_input, move_targets, value_targets) in enumerate(batches):
            result = model(batch_input, move_targets=move_targets, value_targets=value_targets)
            (result["loss"] / ACCUM_STEPS).backward()
            ep_loss += result["loss"].item()
            steps += 1
            if (b_idx + 1) % ACCUM_STEPS == 0 or b_idx == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if steps % 400 == 0:
                print(f"    step {steps}/{len(batches)} loss={ep_loss/steps:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_accuracy(model, eval_data, device)
        elapsed = time.time() - t0
        effective_epoch = start_epoch + epoch + 1
        history.append({**ev, "loss": avg_loss, "effective_epoch": effective_epoch})
        improved = ev["accuracy"] > best_acc
        if improved:
            best_acc = ev["accuracy"]
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
                },
                "epoch": effective_epoch, "accuracy": ev["accuracy"],
            }, OUTPUT_DIR / "best_checkpoint.pt")
        print(f"  Epoch {effective_epoch}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
              f"{'*BEST*' if improved else ''} [{elapsed:.0f}s]")

    final = evaluate_accuracy(model, eval_data, device)
    print(f"\n  Best: {best_acc:.1%}  Final: {final['accuracy']:.1%} / top3={final['top3_accuracy']:.1%}")
    print(f"  vs exp024 (3ep): 48.7%")

    # --- Play games with best model ---
    if (OUTPUT_DIR / "best_checkpoint.pt").exists():
        ckpt2 = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt2["model_state"])
        print(f"  Loaded best checkpoint (epoch {ckpt2['epoch']}, acc={ckpt2['accuracy']:.1%})")
    
    print(f"\n[4/4] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv ({r['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp025_continue_training",
        "hypothesis": "More training epochs improve accuracy and game survival",
        "seed": SEED,
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "model": {
            "type": "chess_transformer",
            "layers": NUM_LAYERS, "hidden": HIDDEN_DIM, "heads": NUM_HEADS,
            "trainable": trainable,
        },
        "training": {
            "continuation_from": "exp024",
            "start_epoch": start_epoch, "additional_epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS, "lr": LR,
        },
        "results": {
            "pre_continue": pre,
            "best_acc": best_acc, "final": final, "history": history,
            "exp024_baseline": 0.487,
        },
        "games": game_results,
        "game_score": {"wins": wins, "draws": draws, "losses": losses},
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Chess Transformer (continued: epochs {start_epoch+1}-{start_epoch+EPOCHS})")
    print(f"  Accuracy: {best_acc:.1%} best / {final['accuracy']:.1%} final")
    print(f"  vs exp024 (3ep): 48.7%")
    print(f"  vs SF d{GAME_SF_DEPTH}: W{wins}/D{draws}/L{losses}")
    print(f"  Time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
