"""exp008: Board Encoder Architecture — proof of concept.

Hypothesis: A learned board encoder feeding Qwen's backbone via inputs_embeds
can learn to predict chess moves, proving the architecture is viable.

Phase 1 — Architecture test: forward + backward pass, gradient flow check
Phase 2 — Self-distillation: use base Qwen's text-based chess knowledge to
           generate move labels, train the encoder model on those labels
Phase 3 — Evaluate: measure legal move rate and move agreement with teacher

Time budget: ~8 minutes.
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

from chess_features import board_to_planes, batch_boards_to_planes
from chess_model import ChessModel
from constrained import build_token_text_map
from model import load_base_model
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    legal_move_mask, move_to_index,
)
from selfplay import generate_move
from config import Config

OUTPUT_DIR = Path("outputs/exp008_board_encoder")
BOARD_ENCODING = "grid_compact"
NUM_TRAIN_POSITIONS = 200   # positions for training
NUM_EVAL_POSITIONS = 50     # positions for evaluation
TRAIN_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3        # higher LR for encoder + heads (backbone frozen)


def generate_random_positions(n: int, min_ply: int = 4, max_ply: int = 60) -> list[chess.Board]:
    """Generate diverse positions by playing random legal moves."""
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
    """Use text-based model to generate move labels for positions."""
    labeled = []
    for i, board in enumerate(positions):
        move, _ = generate_move(
            teacher_model, tokenizer, board,
            temperature=0.3,  # low temp for best-move approximation
            constrained=True,
            token_texts=token_texts,
            board_encoding=BOARD_ENCODING,
        )
        if move is not None and move in board.legal_moves:
            labeled.append((board, move))
        if (i + 1) % 50 == 0:
            print(f"    Labeled {i+1}/{len(positions)}")
    return labeled


def make_batches(
    data: list[tuple[chess.Board, chess.Move]],
    batch_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create training batches: (planes, move_targets, legal_masks)."""
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [b for b, m in chunk]
        moves = [m for b, m in chunk]

        planes = batch_boards_to_planes(boards).to(device)
        targets = torch.tensor(
            [move_to_index(m) for m in moves], dtype=torch.long, device=device
        )
        masks = torch.stack([legal_move_mask(b) for b in boards]).to(device)
        batches.append((planes, targets, masks))
    return batches


def evaluate_model(
    chess_model: ChessModel,
    eval_data: list[tuple[chess.Board, chess.Move]],
    device: torch.device,
) -> dict:
    """Evaluate move prediction accuracy and legal move rate."""
    chess_model.eval()
    correct = 0
    legal = 0
    total = 0

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
        "legal": legal,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Phase 1: Architecture Test ===
    print("=" * 60)
    print(" Phase 1: Architecture Test")
    print("=" * 60)

    print(f"Loading Qwen3-0.6B...")
    full_model, tokenizer = load_base_model(cfg)
    full_model = full_model.to(device)

    print(f"Building ChessModel...")
    chess_model = ChessModel(
        full_model,
        encoder_dim=256,
        encoder_blocks=4,
        freeze_backbone=True,
    )
    chess_model = chess_model.to(device)

    print(f"  Total params:     {chess_model.total_params():,}")
    print(f"  Trainable params: {chess_model.trainable_params():,}")
    print(f"  Move vocab size:  {VOCAB_SIZE}")

    # Forward pass test
    print("\n  Forward pass test...")
    test_board = chess.Board()
    test_planes = board_to_planes(test_board).unsqueeze(0).to(device)
    test_target = torch.tensor([UCI_TO_IDX["e2e4"]], device=device)
    result = chess_model(test_planes, move_targets=test_target)
    print(f"    policy_logits shape: {result['policy_logits'].shape}")
    print(f"    value_logits shape:  {result['value_logits'].shape}")
    print(f"    loss: {result['loss'].item():.4f}")

    # Backward pass test
    print("  Backward pass test...")
    result["loss"].backward()

    # Check gradient flow
    encoder_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in chess_model.encoder.parameters()
    )
    proj_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in chess_model.input_proj.parameters()
    )
    head_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in chess_model.policy_head.parameters()
    )
    backbone_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in chess_model.backbone.parameters()
    )
    print(f"    Encoder gradients:  {'OK' if encoder_grad else 'MISSING'}")
    print(f"    Projection grads:   {'OK' if proj_grad else 'MISSING'}")
    print(f"    Policy head grads:  {'OK' if head_grad else 'MISSING'}")
    print(f"    Backbone gradients: {'frozen (expected)' if not backbone_grad else 'ACTIVE'}")

    # Predict a move from starting position
    chess_model.eval()
    pred_move, probs = chess_model.predict_move(chess.Board())
    top5_idx = probs.topk(5).indices.cpu().tolist()
    top5_moves = [(IDX_TO_UCI[i], f"{probs[i].item():.3f}") for i in top5_idx]
    print(f"\n  Starting position prediction (untrained):")
    print(f"    Best: {pred_move.uci()}")
    print(f"    Top 5: {top5_moves}")
    print(f"    Legal: {pred_move in chess.Board().legal_moves}")

    # === Phase 2: Self-Distillation Training ===
    print(f"\n{'='*60}")
    print(f" Phase 2: Self-Distillation ({NUM_TRAIN_POSITIONS} positions, {TRAIN_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Generate positions
    print("  Generating random positions...")
    phase2_start = time.time()
    train_boards = generate_random_positions(NUM_TRAIN_POSITIONS)
    eval_boards = generate_random_positions(NUM_EVAL_POSITIONS)
    print(f"    Train: {len(train_boards)}, Eval: {len(eval_boards)}")

    # Label with teacher (text-based model)
    print("  Labeling with teacher model...")
    token_texts = build_token_text_map(tokenizer)
    # Use the full CausalLM model for text-based move generation
    full_model.eval()
    train_data = label_with_teacher(train_boards, full_model, tokenizer, token_texts, device)
    eval_data = label_with_teacher(eval_boards, full_model, tokenizer, token_texts, device)
    print(f"    Labeled: train={len(train_data)}, eval={len(eval_data)}")
    label_time = time.time() - phase2_start

    # Pre-training eval
    print("\n  Pre-training eval:")
    pre_eval = evaluate_model(chess_model, eval_data, device)
    print(f"    Accuracy: {pre_eval['accuracy']:.1%}  Legal: {pre_eval['legal_rate']:.1%}")

    # Train
    print(f"\n  Training...")
    chess_model.train()
    optimizer = AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    train_start = time.time()
    for epoch in range(TRAIN_EPOCHS):
        batches = make_batches(train_data, BATCH_SIZE, device)
        epoch_loss = 0.0
        epoch_steps = 0

        for planes, targets, masks in batches:
            optimizer.zero_grad()
            result = chess_model(planes, move_targets=targets)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in chess_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1

        avg = epoch_loss / max(epoch_steps, 1)

        # Quick eval every epoch
        epoch_eval = evaluate_model(chess_model, eval_data, device)
        print(
            f"    Epoch {epoch+1}/{TRAIN_EPOCHS}: "
            f"loss={avg:.4f} acc={epoch_eval['accuracy']:.1%} "
            f"legal={epoch_eval['legal_rate']:.1%}"
        )
    train_time = time.time() - train_start

    # === Phase 3: Final Evaluation ===
    print(f"\n{'='*60}")
    print(f" Phase 3: Final Evaluation")
    print(f"{'='*60}")

    post_eval = evaluate_model(chess_model, eval_data, device)
    print(f"  Accuracy: {post_eval['accuracy']:.1%} (was {pre_eval['accuracy']:.1%})")
    print(f"  Legal:    {post_eval['legal_rate']:.1%} (was {pre_eval['legal_rate']:.1%})")

    # Test on specific positions
    test_positions = [
        ("Starting", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Italian setup", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")),
    ]

    print("\n  Move predictions on known positions:")
    for name, board in test_positions:
        pred, probs = chess_model.predict_move(board)
        top3_idx = probs.topk(3).indices.cpu().tolist()
        top3 = [(IDX_TO_UCI[i], f"{probs[i].item():.2f}") for i in top3_idx]
        is_legal = pred in board.legal_moves
        print(f"    {name}: {pred.uci()} ({'legal' if is_legal else 'ILLEGAL'}) top3={top3}")

    total_time = time.time() - total_start

    # === Summary ===
    print(f"\n{'='*60}")
    print(f" exp008: Board Encoder - RESULTS")
    print(f"{'='*60}")
    print(f"  Architecture: CNN({4} blocks) -> Qwen3-0.6B(frozen) -> policy head")
    print(f"  Trainable: {chess_model.trainable_params():,} / {chess_model.total_params():,}")
    print(f"  Move vocab: {VOCAB_SIZE}")
    print(f"  Training: {len(train_data)} examples, {TRAIN_EPOCHS} epochs, lr={LEARNING_RATE}")
    print(f"  Pre-train:  acc={pre_eval['accuracy']:.1%} legal={pre_eval['legal_rate']:.1%}")
    print(f"  Post-train: acc={post_eval['accuracy']:.1%} legal={post_eval['legal_rate']:.1%}")
    print(f"  Time: label={label_time:.0f}s train={train_time:.0f}s total={total_time:.0f}s")

    results = {
        "experiment": "exp008_board_encoder",
        "hypothesis": "Learned board encoder + Qwen backbone can predict chess moves",
        "architecture": {
            "encoder_dim": 256,
            "encoder_blocks": 4,
            "hidden_size": chess_model.hidden_size,
            "move_vocab_size": VOCAB_SIZE,
            "trainable_params": chess_model.trainable_params(),
            "total_params": chess_model.total_params(),
        },
        "training": {
            "num_positions": len(train_data),
            "epochs": TRAIN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
        },
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "timing": {
            "label_s": label_time,
            "train_s": train_time,
            "total_s": total_time,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
