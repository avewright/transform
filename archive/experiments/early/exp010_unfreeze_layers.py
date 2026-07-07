"""exp010: Partial Backbone Unfreezing with Learned Encoder.

Hypothesis: Unfreezing the top N layers of Qwen3's backbone improves
move prediction accuracy, because the pretrained features benefit from
task-specific adaptation.

Design:
  - Learned embedding encoder (from exp009)
  - One labeling pass, three training runs:
      (a) Fully frozen backbone (baseline, same as exp009)
      (b) Top 4 layers unfrozen
      (c) Top 8 layers unfrozen
  - Same data, same seed, same hyperparams

Time budget: ~15 min (labeling ~8 min + 3x training ~2 min each).
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
from constrained import build_token_text_map
from model import load_base_model
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index
from selfplay import generate_move
from config import Config

OUTPUT_DIR = Path("outputs/exp010_unfreeze_layers")
CACHE_FILE = OUTPUT_DIR / "labeled_data.json"
BOARD_ENCODING = "grid_compact"
NUM_TRAIN = 500
NUM_EVAL = 100
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
ENCODER_DIM = 256
SEED = 42

# How many top layers to unfreeze in each variant
UNFREEZE_VARIANTS = [0, 4, 8]


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


def save_labeled_data(
    data: list[tuple[chess.Board, chess.Move]], path: Path, split: str,
):
    """Save labeled data as JSON (FEN + UCI)."""
    entries = [{"fen": b.fen(), "uci": m.uci()} for b, m in data]
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing[split] = entries
    with open(path, "w") as f:
        json.dump(existing, f, indent=1)


def load_labeled_data(
    path: Path, split: str,
) -> list[tuple[chess.Board, chess.Move]]:
    """Load labeled data from JSON cache."""
    with open(path) as f:
        entries = json.load(f)[split]
    return [
        (chess.Board(e["fen"]), chess.Move.from_uci(e["uci"]))
        for e in entries
    ]


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
            temperature=0.3, constrained=True,
            token_texts=token_texts, board_encoding=BOARD_ENCODING,
        )
        if move is not None and move in board.legal_moves:
            labeled.append((board, move))
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(positions)}")
    return labeled


def freeze_backbone_selective(chess_model: ChessModel, unfreeze_top_n: int):
    """Freeze all backbone layers, then unfreeze the top N."""
    # First freeze everything
    for param in chess_model.backbone.parameters():
        param.requires_grad = False

    if unfreeze_top_n == 0:
        return

    # Qwen3Model has .layers (ModuleList of decoder layers) and .norm
    layers = chess_model.backbone.layers
    total_layers = len(layers)
    start = max(0, total_layers - unfreeze_top_n)

    # Unfreeze top N layers
    for layer in layers[start:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Always unfreeze final norm
    if hasattr(chess_model.backbone, 'norm'):
        for param in chess_model.backbone.norm.parameters():
            param.requires_grad = True


def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [b for b, m in chunk]
        moves = [m for b, m in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        targets = torch.tensor(
            [move_to_index(m) for m in moves], dtype=torch.long, device=device,
        )
        batches.append((batch_input, targets))
    return batches


def evaluate_model(chess_model, eval_data, device):
    chess_model.eval()
    correct = legal = total = 0
    with torch.no_grad():
        for board, target_move in eval_data:
            pred_move, _ = chess_model.predict_move(board)
            total += 1
            if pred_move in board.legal_moves:
                legal += 1
            if pred_move == target_move:
                correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "legal_rate": legal / max(total, 1),
        "total": total,
    }


def train_variant(
    qwen_model: nn.Module,
    train_data: list[tuple[chess.Board, chess.Move]],
    eval_data: list[tuple[chess.Board, chess.Move]],
    unfreeze_top_n: int,
    device: torch.device,
) -> dict:
    """Train a LearnedEncoder + ChessModel with given unfreezing."""
    label = f"unfreeze={unfreeze_top_n}"

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model = ChessModel(qwen_model, encoder=encoder, freeze_backbone=True).to(device)

    # Selectively unfreeze
    freeze_backbone_selective(model, unfreeze_top_n)
    trainable = model.trainable_params()
    total = model.total_params()
    print(f"\n  [{label}] trainable={trainable:,} / {total:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR if unfreeze_top_n == 0 else LR * 0.1,  # lower LR when unfreezing backbone
        weight_decay=0.01,
    )

    history = []
    train_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        epoch_loss = 0.0
        steps = 0

        for batch_input, targets in batches:
            optimizer.zero_grad()
            result = model(batch_input, move_targets=targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            epoch_loss += result["loss"].item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        metrics = evaluate_model(model, eval_data, device)
        metrics["loss"] = avg_loss
        history.append(metrics)

        print(
            f"    [{label}] Epoch {epoch+1}/{EPOCHS}: "
            f"loss={avg_loss:.4f} acc={metrics['accuracy']:.1%} "
            f"legal={metrics['legal_rate']:.1%}"
        )

    train_time = time.time() - train_start
    best_acc = max(h["accuracy"] for h in history)

    return {
        "unfreeze_top_n": unfreeze_top_n,
        "trainable_params": trainable,
        "history": history,
        "train_time_s": train_time,
        "best_accuracy": best_acc,
        "final_accuracy": history[-1]["accuracy"],
        "final_loss": history[-1]["loss"],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    total_start = time.time()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading Qwen3-0.6B...")
    full_model, tokenizer = load_base_model(cfg)
    full_model = full_model.to(device)

    # === Get labeled data (cache to avoid relabeling) ===
    if CACHE_FILE.exists():
        print(f"Loading cached labels from {CACHE_FILE}...")
        train_data = load_labeled_data(CACHE_FILE, "train")
        eval_data = load_labeled_data(CACHE_FILE, "eval")
        label_time = 0.0
    else:
        print(f"Generating {NUM_TRAIN} + {NUM_EVAL} positions...")
        train_boards = generate_random_positions(NUM_TRAIN)
        eval_boards = generate_random_positions(NUM_EVAL)

        print("Labeling with teacher...")
        full_model.eval()
        token_texts = build_token_text_map(tokenizer)
        label_start = time.time()
        train_data = label_with_teacher(train_boards, full_model, tokenizer, token_texts, device)
        eval_data = label_with_teacher(eval_boards, full_model, tokenizer, token_texts, device)
        label_time = time.time() - label_start
        print(f"  Labeled: train={len(train_data)}, eval={len(eval_data)} ({label_time:.0f}s)")

        # Cache for reuse
        save_labeled_data(train_data, CACHE_FILE, "train")
        save_labeled_data(eval_data, CACHE_FILE, "eval")
        print(f"  Cached to {CACHE_FILE}")

    print(f"  Data: train={len(train_data)}, eval={len(eval_data)}")

    # === Run all variants ===
    total_layers = len(full_model.model.layers)
    print(f"\nQwen3 has {total_layers} transformer layers")

    print(f"\n{'='*60}")
    print(f" Training variants: unfreeze_top_n = {UNFREEZE_VARIANTS}")
    print(f"{'='*60}")

    results = []
    for n_unfreeze in UNFREEZE_VARIANTS:
        torch.manual_seed(SEED)  # Reset for fair comparison
        variant_result = train_variant(
            full_model, train_data, eval_data, n_unfreeze, device,
        )
        results.append(variant_result)

    # === Summary ===
    print(f"\n{'='*60}")
    print(f" exp010 RESULTS — Partial Backbone Unfreezing")
    print(f"{'='*60}")
    print(f"  {'Unfreeze':>10s} {'Trainable':>12s} {'Best Acc':>10s} {'Final Acc':>10s} {'Final Loss':>11s} {'Time(s)':>8s}")
    for r in results:
        print(
            f"  {r['unfreeze_top_n']:>10d} "
            f"{r['trainable_params']:>12,} "
            f"{r['best_accuracy']:>10.1%} "
            f"{r['final_accuracy']:>10.1%} "
            f"{r['final_loss']:>11.4f} "
            f"{r['train_time_s']:>8.0f}"
        )

    # Test positions
    print("\n  Move predictions (last variant = unfreeze top 8):")
    test_positions = [
        ("Starting", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Italian", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")),
    ]
    # The last model trained is still in memory
    # (We'd need to keep refs to compare, but for summary just use the last)

    # Save
    output = {
        "experiment": "exp010_unfreeze_layers",
        "hypothesis": "Unfreezing top N backbone layers improves accuracy",
        "data": {"train": len(train_data), "eval": len(eval_data), "seed": SEED},
        "total_backbone_layers": total_layers,
        "variants": results,
        "label_time_s": label_time,
        "total_time_s": time.time() - total_start,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
