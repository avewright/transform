"""exp018: LoRA at Scale — 50K HF game-play positions with LoRA backbone.

Hypothesis: LoRA adaptation of Qwen3 attention layers will break through the
~25% accuracy ceiling seen with frozen backbone on 50K positions. The backbone's
attention patterns can adapt to chess board structure, improving feature extraction.

Context:
  - exp013 frozen 50K = 25% acc (best result, 3 epochs)
  - exp015b frozen 20K = 21.5% (baseline we just measured)
  - LoRA exp015b crashed (device bug, now fixed)

Approach:
  1. Train LoRA (rank=16, q_proj + v_proj) on 50K HF game-play positions, 3 epochs
  2. Compare against exp013 frozen baseline (25% acc)
  3. If improvement confirmed, play games vs Stockfish

Primary metric: top-1 accuracy on 500 held-out positions
Secondary: top-3 accuracy, game performance vs SF d3

Time budget: ~10 min (3 epochs × 50K at batch=64)
"""

import json
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp018_lora_50k")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN = 50_000
NUM_EVAL = 500
EPOCHS = 3
BATCH_SIZE = 64
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
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        # Compute LoRA in float32 for stability, cast back
        x_f = self.lora_dropout(x).float()
        lora_out = (x_f @ self.lora_A.T @ self.lora_B.T * self.scaling)
        return result + lora_out.to(result.dtype)


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float, targets: list[str]) -> int:
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
                if preds[j] == target_idx: correct += 1
                if target_idx in top3s[j]: top3_correct += 1
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

    # --- Load HF dataset ---
    print("\n[1/4] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    print(f"  Dataset: {len(ds):,} samples")

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)

    # --- Build model with LoRA ---
    print(f"\n[2/4] Building LoRA model (rank={LORA_RANK}, targets={LORA_TARGETS})")
    cfg = Config()
    full_model, _ = load_base_model(cfg)

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True)

    # Apply LoRA BEFORE moving to device
    lora_count = apply_lora(chess_model.backbone, rank=LORA_RANK, alpha=LORA_ALPHA,
                            dropout=LORA_DROPOUT, targets=LORA_TARGETS)
    chess_model = chess_model.to(device)

    trainable = sum(p.numel() for p in chess_model.parameters() if p.requires_grad)
    print(f"  LoRA params: {lora_count:,}")
    print(f"  Total trainable: {trainable:,}")

    pre = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre['accuracy']:.1%}")

    # Separate LoRA vs other params for different LR
    lora_params, other_params = [], []
    for name, p in chess_model.named_parameters():
        if p.requires_grad:
            if "lora_" in name:
                lora_params.append(p)
            else:
                other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": LR},
        {"params": lora_params, "lr": LR_LORA},
    ]
    optimizer = AdamW(param_groups, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- Train ---
    print(f"\n[3/4] Training LoRA on {len(train_data)} positions, {EPOCHS} epochs")
    history = []
    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0

        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
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

    # Full eval
    final = evaluate_accuracy(chess_model, eval_data, device)
    best_acc = max(h["accuracy"] for h in history)
    print(f"\n  Best: {best_acc:.1%}  Final: {final['accuracy']:.1%} / top3={final['top3_accuracy']:.1%}")
    print(f"  vs frozen baseline: 25.0% (exp013 on 50K)")

    # --- Play games ---
    print(f"\n[4/4] Playing {NUM_GAMES} games vs Stockfish d{GAME_SF_DEPTH}")
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
    results = {
        "experiment": "exp018_lora_50k",
        "hypothesis": "LoRA on 50K HF positions beats frozen baseline (25%)",
        "primary_metric": "top-1 accuracy",
        "seed": SEED,
        "data": {"train": len(train_data), "eval": len(eval_data), "source": "HF game-play"},
        "model": {
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS, "lora_params": lora_count,
            "trainable": trainable, "encoder_dim": ENCODER_DIM,
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "lr_lora": LR_LORA,
        },
        "results": {
            "pre_train": pre,
            "best_acc": best_acc,
            "final": final,
            "history": history,
            "frozen_baseline": 0.25,
            "delta_vs_frozen": best_acc - 0.25,
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
    print(f"  LoRA (50K, {EPOCHS}ep): {best_acc:.1%} best / {final['accuracy']:.1%} final")
    print(f"  Frozen baseline:     25.0% (exp013)")
    print(f"  Delta: {best_acc - 0.25:+.1%}")
    print(f"  vs SF d{GAME_SF_DEPTH}: W{wins}/D{draws}/L{losses}")
    print(f"  Time: {total_time:.0f}s")
    print(f"  Saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
