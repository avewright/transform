"""exp020: Scaled Spatial Head — Push spatial policy to its limits.

Hypothesis: The spatial head's steep loss decline (3.68→2.91→2.61) at epoch 3
indicates it's far from saturated. More epochs and more data will push accuracy
significantly above 36.5%.

Context:
  - exp019: Spatial 36.5% acc / 61.4% top3 at 50K × 3ep (still declining)
  - Standard head plateaus at 25% by epoch 2
  - Spatial loss at epoch 3 was 2.61 vs standard's 4.55

Plan:
  1. Train spatial head on 200K positions for 5 epochs
  2. Eval every epoch to track scaling curve
  3. Play games vs SF d3

Primary metric: top-1 accuracy
Time budget: ~30 min (5ep × 200K at batch 64)
"""

import json
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

OUTPUT_DIR = Path("outputs/exp020_scaled_spatial")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN = 200_000
NUM_EVAL = 500
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
NUM_GAMES = 6
GAME_SF_DEPTH = 3


# === Spatial Policy Head (from exp019) ===

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
    def __init__(self, hidden_size: int, head_dim: int = 256):
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.ReLU(), nn.Linear(256, 3),
        )

    def forward(self, board_input, move_targets=None, value_targets=None, legal_masks=None):
        tokens = self.encoder(board_input)
        embeds = self.input_proj(tokens)
        backbone_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = self.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()
        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)
        result = {"policy_logits": policy_logits, "value_logits": value_logits}
        device = board_input["piece_ids"].device if isinstance(board_input, dict) else board_input.device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            result["policy_loss"] = policy_loss
            total_loss = total_loss + policy_loss
        if value_targets is not None:
            value_loss = F.cross_entropy(value_logits, value_targets)
            result["value_loss"] = value_loss
            total_loss = total_loss + 0.5 * value_loss
        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def predict_move(self, board: chess.Board):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        best_idx = probs.argmax().item()
        return index_to_move(best_idx), probs

    @torch.no_grad()
    def predict_value(self, board: chess.Board):
        """Predict W/D/L probabilities for a position."""
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        result = self.forward(board_input)
        return F.softmax(result["value_logits"][0], dim=-1)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === HF Dataset ===

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


# === Training Utils ===

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


def evaluate_accuracy(model, eval_data, device, n=None, batch_size=64):
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
    moves_played = []
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = model.predict_move(board)
            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
            moves_played.append(pred.uci())
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None: break
            board.push(chess.Move.from_uci(sf_uci))
            moves_played.append(sf_uci)
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
    print("\n[1/3] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    print(f"  Dataset: {len(ds):,} samples")

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)

    # --- Build spatial model ---
    print(f"\n[2/3] Building Spatial model")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = SpatialChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    trainable = chess_model.trainable_params()
    print(f"  Trainable: {trainable:,}")

    pre = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre['accuracy']:.1%}")

    params = [p for p in chess_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- Train ---
    print(f"\n[3/3] Training on {len(train_data)} positions, {EPOCHS} epochs")
    history = []
    best_state = None
    best_acc = 0
    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0
        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += result["loss"].item()
            steps += 1
        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_accuracy(chess_model, eval_data, device, n=200)
        elapsed = time.time() - t0
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            # Save best checkpoint (just trainable params)
            best_state = {
                name: param.cpu().clone()
                for name, param in chess_model.named_parameters()
                if param.requires_grad
            }

    # Restore best checkpoint
    if best_state:
        for name, param in chess_model.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name].to(device))
        print(f"\n  Restored best checkpoint (acc={best_acc:.1%})")

    # Full eval
    final = evaluate_accuracy(chess_model, eval_data, device)
    print(f"  Final (full eval): acc={final['accuracy']:.1%} top3={final['top3_accuracy']:.1%}")

    # --- Play games ---
    print(f"\n  Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(chess_model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']} moves ({r['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")

    # --- Save ---
    total_time = time.time() - t0

    # Save model checkpoint
    torch.save({
        "model_state": {
            name: param.cpu()
            for name, param in chess_model.named_parameters()
            if param.requires_grad
        },
        "config": {
            "encoder_dim": ENCODER_DIM,
            "head_type": "spatial",
            "head_dim": 256,
        },
    }, OUTPUT_DIR / "best_checkpoint.pt")

    results = {
        "experiment": "exp020_scaled_spatial",
        "hypothesis": "Spatial head scales with more data and epochs beyond 36.5%",
        "primary_metric": "top-1 accuracy",
        "seed": SEED,
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "model": {"trainable": trainable, "head_type": "spatial"},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "results": {
            "pre_train": pre,
            "best_acc": best_acc,
            "final": final,
            "history": history,
            "prior_best": {"exp019_spatial_50k": 0.365, "exp013_standard_50k": 0.25},
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
    print(f"  Spatial (200K, {EPOCHS}ep): {best_acc:.1%} best / {final['accuracy']:.1%} final")
    print(f"  Prior best (50K):      36.5%")
    print(f"  vs SF d{GAME_SF_DEPTH}: W{wins}/D{draws}/L{losses}")
    print(f"  Time: {total_time:.0f}s")
    print(f"  Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
