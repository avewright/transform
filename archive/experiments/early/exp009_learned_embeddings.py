"""exp009: Learned Embeddings Encoder vs CNN Encoder (head-to-head).

Hypothesis: A learned-embedding encoder (piece + color projection + position
+ game-state context tokens) is at least as effective as the CNN encoder from
exp008, with ~20x fewer encoder parameters.

Design:
  - Same self-distillation pipeline as exp008 (label with text model, train)
  - Train both CNN and Learned encoders on the SAME labeled data
  - Compare accuracy, legal rate, and loss curves

Time budget: ~10 minutes.
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

from chess_features import (
    board_to_planes, batch_boards_to_planes,
    batch_boards_to_token_ids,
)
from chess_model import BoardEncoder, LearnedBoardEncoder, ChessModel
from constrained import build_token_text_map
from model import load_base_model
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    legal_move_mask, move_to_index,
)
from selfplay import generate_move
from config import Config

OUTPUT_DIR = Path("outputs/exp009_learned_embeddings")
BOARD_ENCODING = "grid_compact"
NUM_TRAIN = 500
NUM_EVAL = 100
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
ENCODER_DIM = 256
SEED = 42


def generate_random_positions(n: int, min_ply: int = 4, max_ply: int = 60) -> list[chess.Board]:
    positions = []
    attempts = 0
    while len(positions) < n and attempts < n * 5:
        board = chess.Board()
        ply = random.randint(min_ply, max_ply)
        for _ in range(ply):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        if not board.is_game_over() and len(list(board.legal_moves)) > 0:
            positions.append(board.copy())
        attempts += 1
    return positions


def label_with_teacher(
    positions: list[chess.Board],
    teacher_model: nn.Module,
    tokenizer,
    token_texts: dict,
    device: torch.device,
) -> list[tuple[chess.Board, chess.Move]]:
    labeled = []
    for i, board in enumerate(positions):
        move, _ = generate_move(
            teacher_model, tokenizer, board,
            temperature=0.3,
            constrained=True,
            token_texts=token_texts,
            board_encoding=BOARD_ENCODING,
        )
        if move is not None and move in board.legal_moves:
            labeled.append((board, move))
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(positions)}")
    return labeled


def train_model(
    chess_model: ChessModel,
    train_data: list[tuple[chess.Board, chess.Move]],
    eval_data: list[tuple[chess.Board, chess.Move]],
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    label: str = "",
) -> list[dict]:
    """Train and evaluate a ChessModel, return per-epoch metrics."""
    optimizer = AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    history = []

    for epoch in range(epochs):
        chess_model.train()
        random.shuffle(train_data)
        epoch_loss = 0.0
        steps = 0

        for i in range(0, len(train_data), batch_size):
            chunk = train_data[i:i + batch_size]
            boards = [b for b, m in chunk]
            moves = [m for b, m in chunk]

            board_input = chess_model.encoder.prepare_batch(boards, device)
            targets = torch.tensor(
                [move_to_index(m) for m in moves], dtype=torch.long, device=device
            )

            optimizer.zero_grad()
            result = chess_model(board_input, move_targets=targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in chess_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            epoch_loss += result["loss"].item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)

        # Eval
        metrics = evaluate_model(chess_model, eval_data, device)
        metrics["loss"] = avg_loss
        history.append(metrics)

        print(
            f"    [{label}] Epoch {epoch+1}/{epochs}: "
            f"loss={avg_loss:.4f} acc={metrics['accuracy']:.1%} "
            f"legal={metrics['legal_rate']:.1%}"
        )

    return history


def evaluate_model(
    chess_model: ChessModel,
    eval_data: list[tuple[chess.Board, chess.Move]],
    device: torch.device,
) -> dict:
    chess_model.eval()
    correct = legal = total = 0

    with torch.no_grad():
        for board, target_move in eval_data:
            pred_move, probs = chess_model.predict_move(board)
            total += 1
            if pred_move in board.legal_moves:
                legal += 1
            if pred_move == target_move:
                correct += 1

    return {
        "accuracy": correct / max(total, 1),
        "legal_rate": legal / max(total, 1),
        "total": total,
        "correct": correct,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    total_start = time.time()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load teacher model ===
    print("Loading teacher model (Qwen3-0.6B)...")
    full_model, tokenizer = load_base_model(cfg)
    full_model = full_model.to(device).eval()

    # === Generate and label data ===
    print(f"Generating positions: {NUM_TRAIN} train + {NUM_EVAL} eval...")
    train_boards = generate_random_positions(NUM_TRAIN)
    eval_boards = generate_random_positions(NUM_EVAL)

    print(f"Labeling with teacher (constrained decoding)...")
    token_texts = build_token_text_map(tokenizer)
    label_start = time.time()
    train_data = label_with_teacher(train_boards, full_model, tokenizer, token_texts, device)
    eval_data = label_with_teacher(eval_boards, full_model, tokenizer, token_texts, device)
    label_time = time.time() - label_start
    print(f"  Labeled: train={len(train_data)}, eval={len(eval_data)} ({label_time:.0f}s)")

    # === Build both models ===
    print("\n" + "=" * 60)
    print(" Building two ChessModels (same backbone, different encoders)")
    print("=" * 60)

    # Model A: CNN encoder (same as exp008)
    cnn_encoder = BoardEncoder(embed_dim=ENCODER_DIM, num_blocks=4)
    cnn_model = ChessModel(full_model, encoder=cnn_encoder, freeze_backbone=True).to(device)
    cnn_enc_params = sum(p.numel() for p in cnn_encoder.parameters())

    # Model B: Learned embedding encoder
    learned_encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    learned_model = ChessModel(full_model, encoder=learned_encoder, freeze_backbone=True).to(device)
    learned_enc_params = sum(p.numel() for p in learned_encoder.parameters())

    print(f"  CNN encoder:     {cnn_enc_params:>10,} params | total trainable: {cnn_model.trainable_params():,}")
    print(f"  Learned encoder: {learned_enc_params:>10,} params | total trainable: {learned_model.trainable_params():,}")

    # === Pre-training eval ===
    print("\nPre-training eval:")
    cnn_pre = evaluate_model(cnn_model, eval_data, device)
    learned_pre = evaluate_model(learned_model, eval_data, device)
    print(f"  CNN:     acc={cnn_pre['accuracy']:.1%} legal={cnn_pre['legal_rate']:.1%}")
    print(f"  Learned: acc={learned_pre['accuracy']:.1%} legal={learned_pre['legal_rate']:.1%}")

    # === Train both ===
    print(f"\n{'='*60}")
    print(f" Training ({len(train_data)} examples, {EPOCHS} epochs, lr={LR})")
    print(f"{'='*60}")

    print("\n  --- CNN Encoder ---")
    train_start = time.time()
    cnn_history = train_model(cnn_model, train_data, eval_data, device, label="CNN")
    cnn_train_time = time.time() - train_start

    print("\n  --- Learned Encoder ---")
    train_start = time.time()
    learned_history = train_model(learned_model, train_data, eval_data, device, label="Learned")
    learned_train_time = time.time() - train_start

    # === Final comparison ===
    print(f"\n{'='*60}")
    print(f" exp009 RESULTS")
    print(f"{'='*60}")

    cnn_final = cnn_history[-1]
    learned_final = learned_history[-1]

    print(f"  {'':20s} {'CNN':>10s} {'Learned':>10s}")
    print(f"  {'Encoder params':20s} {cnn_enc_params:>10,} {learned_enc_params:>10,}")
    print(f"  {'Trainable params':20s} {cnn_model.trainable_params():>10,} {learned_model.trainable_params():>10,}")
    print(f"  {'Final accuracy':20s} {cnn_final['accuracy']:>10.1%} {learned_final['accuracy']:>10.1%}")
    print(f"  {'Final legal rate':20s} {cnn_final['legal_rate']:>10.1%} {learned_final['legal_rate']:>10.1%}")
    print(f"  {'Final loss':20s} {cnn_final['loss']:>10.4f} {learned_final['loss']:>10.4f}")
    print(f"  {'Best accuracy':20s} {max(h['accuracy'] for h in cnn_history):>10.1%} {max(h['accuracy'] for h in learned_history):>10.1%}")
    print(f"  {'Train time (s)':20s} {cnn_train_time:>10.0f} {learned_train_time:>10.0f}")

    # Test on known positions
    test_positions = [
        ("Starting", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Italian setup", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")),
    ]

    print("\n  Move predictions:")
    for name, board in test_positions:
        cnn_move, _ = cnn_model.predict_move(board)
        learned_move, _ = learned_model.predict_move(board)
        print(f"    {name:15s}: CNN={cnn_move.uci():6s} Learned={learned_move.uci()}")

    # Save results
    results = {
        "experiment": "exp009_learned_embeddings",
        "hypothesis": "Learned embedding encoder matches CNN with 20x fewer encoder params",
        "data": {"train": len(train_data), "eval": len(eval_data), "seed": SEED},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "cnn": {
            "encoder_params": cnn_enc_params,
            "trainable_params": cnn_model.trainable_params(),
            "pre_eval": cnn_pre,
            "history": cnn_history,
            "train_time_s": cnn_train_time,
        },
        "learned": {
            "encoder_params": learned_enc_params,
            "trainable_params": learned_model.trainable_params(),
            "pre_eval": learned_pre,
            "history": learned_history,
            "train_time_s": learned_train_time,
        },
        "label_time_s": label_time,
        "total_time_s": time.time() - total_start,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
