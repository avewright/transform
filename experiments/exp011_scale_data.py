"""exp011: Scaling data with fast batched labeling.

Hypothesis: More training data (5000 positions vs 500) will break the ~48%
accuracy ceiling observed in exp009/exp010. The bottleneck is labeling speed,
not model capacity (exp010 showed unfreezing doesn't help).

Key innovation: Batch greedy generation (no constrained decoding) — process
8 positions at once, discard invalid outputs. ~5-10x faster labeling.

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
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from data import fen_to_prompt
from model import load_base_model
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index
from config import Config

OUTPUT_DIR = Path("outputs/exp011_scale_data")
CACHE_FILE = OUTPUT_DIR / "labeled_data.json"
BOARD_ENCODING = "grid_compact"
NUM_TRAIN = 5000
NUM_EVAL = 500
EPOCHS = 15
BATCH_SIZE = 32
LR = 5e-4
ENCODER_DIM = 256
SEED = 42
LABEL_BATCH_SIZE = 8


def generate_random_positions(n: int, min_ply: int = 4, max_ply: int = 80) -> list[chess.Board]:
    """Generate diverse positions with wider ply range for better coverage."""
    positions = []
    seen_fens = set()
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
            fen_key = board.board_fen()  # deduplicate by piece placement
            if fen_key not in seen_fens:
                seen_fens.add(fen_key)
                positions.append(board.copy())
        attempts += 1
    return positions


def batch_label_greedy(
    positions: list[chess.Board],
    model: nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = LABEL_BATCH_SIZE,
) -> list[tuple[chess.Board, chess.Move]]:
    """Fast batched greedy labeling — no constrained decoding.

    Processes `batch_size` positions simultaneously. Discards any output
    that isn't a valid legal move. Typically ~70-80% hit rate.
    """
    labeled = []
    total = len(positions)

    # Ensure left-padding for generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_boards = positions[i:i + batch_size]
            prompts = [
                fen_to_prompt(b.fen(), encoding=BOARD_ENCODING)
                for b in batch_boards
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,  # greedy — deterministic, fast
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode each output
            for j, board in enumerate(batch_boards):
                prompt_len = inputs["attention_mask"][j].sum().item()
                new_tokens = gen_ids[j, prompt_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                uci_str = text.strip().replace(" ", "").lower()[:5]

                # Try 5-char then 4-char UCI
                for length in [5, 4]:
                    candidate = uci_str[:length]
                    try:
                        move = chess.Move.from_uci(candidate)
                        if move in board.legal_moves:
                            labeled.append((board, move))
                            break
                    except (ValueError, chess.InvalidMoveError):
                        pass

            if (i + batch_size) % 500 <= batch_size:
                done = min(i + batch_size, total)
                rate = len(labeled) / max(done, 1)
                print(f"      {done}/{total} (hit rate: {rate:.0%})")

    tokenizer.padding_side = orig_padding_side
    return labeled


def save_labeled_data(data, path, split):
    entries = [{"fen": b.fen(), "uci": m.uci()} for b, m in data]
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing[split] = entries
    with open(path, "w") as f:
        json.dump(existing, f, indent=1)


def load_labeled_data(path, split):
    with open(path) as f:
        entries = json.load(f)[split]
    return [
        (chess.Board(e["fen"]), chess.Move.from_uci(e["uci"]))
        for e in entries
    ]


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
        "correct": correct,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    total_start = time.time()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Qwen3-0.6B...")
    full_model, tokenizer = load_base_model(cfg)
    full_model = full_model.to(device)

    # === Get labeled data ===
    if CACHE_FILE.exists():
        print(f"Loading cached labels from {CACHE_FILE}...")
        train_data = load_labeled_data(CACHE_FILE, "train")
        eval_data = load_labeled_data(CACHE_FILE, "eval")
        label_time = 0.0
    else:
        # Generate extra positions to compensate for <100% hit rate
        over_generate = 1.4  # expect ~70% hit rate
        print(f"Generating {int(NUM_TRAIN*over_generate)} + {int(NUM_EVAL*over_generate)} positions...")
        train_boards = generate_random_positions(int(NUM_TRAIN * over_generate))
        eval_boards = generate_random_positions(int(NUM_EVAL * over_generate))

        print(f"Fast batched labeling (batch_size={LABEL_BATCH_SIZE})...")
        label_start = time.time()
        train_data = batch_label_greedy(train_boards, full_model, tokenizer, device)
        eval_data = batch_label_greedy(eval_boards, full_model, tokenizer, device)
        label_time = time.time() - label_start

        # Trim to target size
        train_data = train_data[:NUM_TRAIN]
        eval_data = eval_data[:NUM_EVAL]
        print(f"  Labeled: train={len(train_data)}, eval={len(eval_data)} ({label_time:.0f}s)")

        save_labeled_data(train_data, CACHE_FILE, "train")
        save_labeled_data(eval_data, CACHE_FILE, "eval")
        print(f"  Cached to {CACHE_FILE}")

    print(f"  Data: train={len(train_data)}, eval={len(eval_data)}")

    # === Build model ===
    print(f"\n{'='*60}")
    print(f" Training (learned encoder, frozen backbone)")
    print(f"{'='*60}")

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    print(f"  Trainable: {chess_model.trainable_params():,} / {chess_model.total_params():,}")

    # === Pre-training eval ===
    pre_eval = evaluate_model(chess_model, eval_data[:100], device)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} legal={pre_eval['legal_rate']:.1%}")

    # === Train with cosine LR schedule ===
    optimizer = AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    train_start = time.time()

    for epoch in range(EPOCHS):
        chess_model.train()
        random.shuffle(train_data)
        epoch_loss = 0.0
        steps = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [b for b, m in chunk]
            moves = [m for b, m in chunk]

            batch_input = batch_boards_to_token_ids(boards, device)
            targets = torch.tensor(
                [move_to_index(m) for m in moves], dtype=torch.long, device=device,
            )

            optimizer.zero_grad()
            result = chess_model(batch_input, move_targets=targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in chess_model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += result["loss"].item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        metrics = evaluate_model(chess_model, eval_data[:100], device)
        metrics["loss"] = avg_loss
        metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(metrics)

        print(
            f"  Epoch {epoch+1}/{EPOCHS}: "
            f"loss={avg_loss:.4f} acc={metrics['accuracy']:.1%} "
            f"legal={metrics['legal_rate']:.1%} lr={metrics['lr']:.2e}"
        )

    train_time = time.time() - train_start

    # === Final eval on FULL eval set ===
    print(f"\n{'='*60}")
    print(f" Final Evaluation (full {len(eval_data)} eval positions)")
    print(f"{'='*60}")

    final_eval = evaluate_model(chess_model, eval_data, device)
    print(f"  Accuracy: {final_eval['accuracy']:.1%}")
    print(f"  Legal:    {final_eval['legal_rate']:.1%}")
    print(f"  Best epoch acc: {max(h['accuracy'] for h in history):.1%}")

    # Test positions
    test_positions = [
        ("Starting", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Italian", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")),
        ("Sicilian", chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")),
    ]

    print("\n  Move predictions:")
    for name, board in test_positions:
        pred, probs = chess_model.predict_move(board)
        top3_idx = probs.topk(3).indices.cpu().tolist()
        top3 = [(IDX_TO_UCI[i], f"{probs[i].item():.2f}") for i in top3_idx]
        is_legal = pred in board.legal_moves
        print(f"    {name:15s}: {pred.uci()} ({'legal' if is_legal else 'ILLEGAL'}) top3={top3}")

    total_time = time.time() - total_start

    # === Summary ===
    print(f"\n{'='*60}")
    print(f" exp011: RESULTS")
    print(f"{'='*60}")
    print(f"  Data: {len(train_data)} train, {len(eval_data)} eval")
    print(f"  Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR} (cosine)")
    print(f"  Pre-train:  acc={pre_eval['accuracy']:.1%}")
    print(f"  Best epoch: acc={max(h['accuracy'] for h in history):.1%}")
    print(f"  Final eval: acc={final_eval['accuracy']:.1%} legal={final_eval['legal_rate']:.1%}")
    print(f"  Time: label={label_time:.0f}s train={train_time:.0f}s total={total_time:.0f}s")

    results = {
        "experiment": "exp011_scale_data",
        "hypothesis": "5000 positions breaks 48% accuracy ceiling",
        "data": {
            "train": len(train_data),
            "eval": len(eval_data),
            "seed": SEED,
            "labeling": "batch_greedy",
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "scheduler": "cosine",
        },
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "best_epoch_acc": max(h["accuracy"] for h in history),
        "history": history,
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
