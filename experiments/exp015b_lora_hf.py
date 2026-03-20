"""exp015b: LoRA Backbone Adaptation with HF Dataset

Hypothesis: LoRA fine-tuning of Qwen3 attention layers (q_proj, v_proj) will
outperform a frozen backbone when trained on 20K game-play positions, because
the backbone can adapt its attention patterns to chess board structure.

Based on exp015 design, adapted to use HF dataset (no Stockfish cache needed).

Comparison: Train two identical models on same 20K data:
  A) Frozen backbone (baseline) — only encoder + heads train
  B) LoRA backbone — encoder + heads + LoRA adapters train

LoRA adds ~1.8M params (rank=16, 28 layers × {q,v}_proj).
Total trainable: ~9.2M (A) vs ~11M (B).

Time budget: ~10 min (2 × 3 epochs × 20K positions).
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
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp015b_lora_hf")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN = 20_000
NUM_EVAL = 500
EPOCHS = 3
BATCH_SIZE = 32  # smaller for LoRA (more activations stored for backward)
LR = 1e-3
LR_LORA = 5e-5
ENCODER_DIM = 256
SEED = 42
NUM_GAMES = 4
GAME_SF_DEPTH = 3

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]


# === LoRA ===

class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out.to(result.dtype)


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float, targets: list[str]) -> int:
    """Apply LoRA to target linear layers. Returns number of params added."""
    lora_params = 0
    modules_dict = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        for target in targets:
            if target in name and isinstance(module, nn.Linear):
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = modules_dict[parts[0]]
                    attr_name = parts[1]
                else:
                    parent = model
                    attr_name = name

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

    return lora_params


# === HF Dataset Pipeline (from exp013) ===

def build_old_move_mapping():
    ROOK_DIRS = (+8, -8, +1, -1)
    BISHOP_DIRS = (+9, -9, +7, -7)
    KNIGHT_OFFS = (+17, +15, +10, +6, -6, -10, -15, -17)
    PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    def inside(board_idx, file_from):
        return 0 <= board_idx < 64 and abs((board_idx % 8) - file_from) <= 2

    ucis = set()
    for f in range(64):
        f_file = f % 8
        for d in ROOK_DIRS:
            t = f + d
            while inside(t, f_file):
                ucis.add(chess.Move(f, t).uci()); t += d
        for d in BISHOP_DIRS:
            t = f + d
            while inside(t, f_file):
                ucis.add(chess.Move(f, t).uci()); t += d
        for off in KNIGHT_OFFS:
            t = f + off
            if inside(t, f_file):
                ucis.add(chess.Move(f, t).uci())
    for f in range(64):
        file_ = f % 8; rank = f // 8
        if rank == 6:
            for df in (-9, -8, -7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES:
                        ucis.add(chess.Move(f, t, promotion=p).uci())
        if rank == 1:
            for df in (+9, +8, +7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES:
                        ucis.add(chess.Move(f, t, promotion=p).uci())
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
    data = []
    skipped = 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n:
            break
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX:
            skipped += 1; continue
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError:
            skipped += 1; continue
        if move not in board.legal_moves:
            skipped += 1; continue

        winner = s["winner"]
        if winner == 0:
            vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK):
            vt = 2
        else:
            vt = 0
        data.append({"board": board, "move": move, "value_target": vt})
    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


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


def evaluate_accuracy(chess_model, eval_data, device, n=None, batch_size=64):
    chess_model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = chess_model(batch_input)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, target_idx in enumerate(targets):
                total += 1
                if preds[j] == target_idx:
                    correct += 1
                if target_idx in top3s[j]:
                    top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def play_game_vs_stockfish(chess_model, sf_depth, model_color, device, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    chess_model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = chess_model.predict_move(board)
            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None:
                break
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


# === Training ===

def train_model(chess_model, train_data, eval_data, device, epochs, param_groups, label):
    optimizer = AdamW(param_groups, weight_decay=0.01)
    total_steps = epochs * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0

        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in param_groups for p in group["params"]], 1.0
            )
            optimizer.step()
            scheduler.step()
            ep_loss += result["loss"].item()
            steps += 1

        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_accuracy(chess_model, eval_data, device, n=200)
        elapsed = time.time() - t0
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})
        print(f"  [{label}] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")

    return history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load HF dataset ---
    print("\n[1/6] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    print(f"  Dataset: {len(ds):,} samples, vocab: {len(old_sorted_uci)} moves")

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)

    # --- Variant A: Frozen backbone (baseline) ---
    print(f"\n{'='*60}")
    print(f" [2/6] BASELINE: Frozen backbone, {EPOCHS} epochs on {len(train_data)} positions")
    print(f"{'='*60}")

    cfg = Config()
    full_model_a, _ = load_base_model(cfg)
    full_model_a = full_model_a.to(device)

    encoder_a = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_a = ChessModel(full_model_a, encoder=encoder_a, freeze_backbone=True).to(device)
    trainable_a = model_a.trainable_params()
    print(f"  Trainable: {trainable_a:,}")

    pre_a = evaluate_accuracy(model_a, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_a['accuracy']:.1%}")

    params_a = [{"params": [p for p in model_a.parameters() if p.requires_grad], "lr": LR}]
    history_a = train_model(model_a, train_data, eval_data, device, EPOCHS, params_a, "Frozen")

    # Full eval
    final_a = evaluate_accuracy(model_a, eval_data, device)
    print(f"  Final: acc={final_a['accuracy']:.1%} top3={final_a['top3_accuracy']:.1%}")

    # Free GPU
    del model_a, encoder_a, full_model_a
    torch.cuda.empty_cache()

    # --- Variant B: LoRA backbone ---
    print(f"\n{'='*60}")
    print(f" [3/6] LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)  # Reset seed for fair comparison

    full_model_b, _ = load_base_model(cfg)
    full_model_b = full_model_b.to(device)

    encoder_b = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_b = ChessModel(full_model_b, encoder=encoder_b, freeze_backbone=True).to(device)

    lora_count = apply_lora(model_b.backbone, rank=LORA_RANK, alpha=LORA_ALPHA,
                            dropout=LORA_DROPOUT, targets=LORA_TARGETS)
    # Move newly-created LoRA params to GPU
    model_b = model_b.to(device)
    trainable_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"  LoRA params: {lora_count:,}")
    print(f"  Total trainable: {trainable_b:,}")

    pre_b = evaluate_accuracy(model_b, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_b['accuracy']:.1%}")

    lora_params_list = []
    other_params_list = []
    for name, p in model_b.named_parameters():
        if p.requires_grad:
            if "lora_" in name:
                lora_params_list.append(p)
            else:
                other_params_list.append(p)

    params_b = [
        {"params": other_params_list, "lr": LR},
        {"params": lora_params_list, "lr": LR_LORA},
    ]
    history_b = train_model(model_b, train_data, eval_data, device, EPOCHS, params_b, "LoRA")

    final_b = evaluate_accuracy(model_b, eval_data, device)
    print(f"  Final: acc={final_b['accuracy']:.1%} top3={final_b['top3_accuracy']:.1%}")

    # --- Compare ---
    print(f"\n{'='*60}")
    print(f" [4/6] COMPARISON")
    print(f"{'='*60}")

    best_a = max(h["accuracy"] for h in history_a)
    best_b = max(h["accuracy"] for h in history_b)
    diff = best_b - best_a

    print(f"  Frozen:  best={best_a:.1%}  final={final_a['accuracy']:.1%}  top3={final_a['top3_accuracy']:.1%}  params={trainable_a:,}")
    print(f"  LoRA:    best={best_b:.1%}  final={final_b['accuracy']:.1%}  top3={final_b['top3_accuracy']:.1%}  params={trainable_b:,}")
    winner = "LoRA" if diff > 0.005 else "Frozen" if diff < -0.005 else "TIE"
    print(f"  Winner: {winner} (delta: {diff:+.1%})")

    # --- Play games with LoRA model ---
    print(f"\n{'='*60}")
    print(f" [5/6] Playing {NUM_GAMES} games vs Stockfish depth={GAME_SF_DEPTH}")
    print(f"{'='*60}")

    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(model_b, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']} moves ({r['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")
    score = wins + 0.5 * draws

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" [6/6] SUMMARY")
    print(f"{'='*60}")
    print(f"  Data: {len(train_data)} train, {len(eval_data)} eval (HF game-play moves)")
    print(f"  Frozen: {best_a:.1%} (best) / {final_a['accuracy']:.1%} (final)")
    print(f"  LoRA:   {best_b:.1%} (best) / {final_b['accuracy']:.1%} (final)")
    print(f"  Delta:  {diff:+.1%} ({winner})")
    print(f"  LoRA vs SF d{GAME_SF_DEPTH}: {score}/{NUM_GAMES} (W{wins}/D{draws}/L{losses})")
    print(f"  Total time: {total_time:.0f}s")

    results = {
        "experiment": "exp015b_lora_hf",
        "hypothesis": "LoRA adapts frozen backbone to chess with HF data",
        "data": {"train": len(train_data), "eval": len(eval_data), "source": "HF game-play"},
        "frozen": {
            "trainable": trainable_a,
            "best_acc": best_a,
            "final": final_a,
            "history": history_a,
        },
        "lora": {
            "rank": LORA_RANK, "alpha": LORA_ALPHA, "targets": LORA_TARGETS,
            "lora_params": lora_count, "trainable": trainable_b,
            "best_acc": best_b,
            "final": final_b,
            "history": history_b,
        },
        "comparison": {"winner": winner, "delta": diff},
        "games": game_results,
        "game_score": {"wins": wins, "draws": draws, "losses": losses},
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
