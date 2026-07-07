"""exp017: Data Scaling Law — How accuracy scales with training data volume.

Hypothesis: Accuracy follows a power-law scaling with data volume. By measuring
the scaling curve at 1K, 5K, 10K, 20K positions, we can predict performance
at 100K-1M and decide optimal data investment for the GPU cluster.

Reference: Kaplan et al., "Scaling Laws for Neural Language Models" (2020).
Also: Ruoss et al. (2024) trained on 10M games to reach 2895 Elo.

Approach:
  1. Use the cached 20K Stockfish-labeled positions
  2. Train identical models at 4 data volumes: 1K, 5K, 10K, 20K
  3. Same architecture, epochs, hyperparams for each
  4. Plot accuracy vs data volume → fit power law
  5. Extrapolate: how much data do we need for 30%? 50%?

This directly answers: "Should we generate more data on the GPU cluster?"

Memory: ~3GB VRAM per model. Sequential runs, fits 8GB.
Time: ~10 min (4 models × ~2.5 min each).
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
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, move_to_index, legal_move_mask
from config import Config

# --- Configuration ---
OUTPUT_DIR = Path("outputs/exp017_scaling_law")
CACHE_FILE = Path("outputs/exp012_stockfish_supervised/labeled_data.json")

DATA_VOLUMES = [1000, 5000, 10000, 20000]
EVAL_SIZE = 500  # fixed eval set across all runs

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42


def prepare_data(labeled):
    data = []
    for e in labeled:
        board = chess.Board(e["fen"])
        move = chess.Move.from_uci(e["uci"])
        if e["eval_type"] == "mate":
            v = e["eval_value"]
            wdl_idx = 0 if v > 0 else 2 if v < 0 else 1
        else:
            cp = e["eval_value"]
            if not board.turn:
                cp = -cp
            wdl_idx = 0 if cp > 100 else 2 if cp < -100 else 1
        data.append({"board": board, "move": move, "wdl_idx": wdl_idx})
    return data


def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], dtype=torch.long, device=device)
        value_targets = torch.tensor([d["wdl_idx"] for d in chunk], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_model(chess_model, eval_data, device, n=None):
    chess_model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for entry in subset:
            board, target = entry["board"], entry["move"]
            pred, probs = chess_model.predict_move(board)
            total += 1
            if pred == target:
                correct += 1
            top3 = probs.topk(min(3, probs.shape[0])).indices.cpu().tolist()
            if move_to_index(target) in top3:
                top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def train_and_evaluate(full_model, train_data, eval_data, device):
    """Train a fresh model and return best accuracy + history."""
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)

    params = [p for p in chess_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0

        for batch_input, move_targets, value_targets in batches:
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            steps += 1

        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_model(chess_model, eval_data, device)
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})

    # Clean up
    del chess_model, encoder, optimizer, scheduler
    torch.cuda.empty_cache()

    return history


def fit_power_law(xs, ys):
    """Fit y = a * x^b using log-log linear regression.

    Returns (a, b, r_squared).
    """
    import math
    n = len(xs)
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]

    mean_lx = sum(log_x) / n
    mean_ly = sum(log_y) / n

    ss_xx = sum((lx - mean_lx) ** 2 for lx in log_x)
    ss_xy = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
    ss_yy = sum((ly - mean_ly) ** 2 for ly in log_y)

    if ss_xx == 0:
        return 0, 0, 0

    b = ss_xy / ss_xx
    log_a = mean_ly - b * mean_lx
    a = math.exp(log_a)

    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0

    return a, b, r_squared


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all data
    print("Loading cached Stockfish labels...")
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    all_train = prepare_data(cache["train"])
    eval_data = prepare_data(cache["eval"][:EVAL_SIZE])
    print(f"  Available train: {len(all_train)}, Eval: {len(eval_data)}")

    # Load backbone once
    print("Loading Qwen3-0.6B backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # Run scaling experiment
    scaling_results = []
    for n in DATA_VOLUMES:
        actual_n = min(n, len(all_train))
        print(f"\n{'=' * 60}")
        print(f" DATA VOLUME: {actual_n:,} positions")
        print(f"{'=' * 60}")

        train_subset = all_train[:actual_n]
        history = train_and_evaluate(full_model, train_subset, eval_data, device)

        best_acc = max(h["accuracy"] for h in history)
        best_top3 = max(h["top3_accuracy"] for h in history)
        final = history[-1]

        print(f"  Best: acc={best_acc:.1%} top3={best_top3:.1%}")
        print(f"  Final: acc={final['accuracy']:.1%} loss={final['loss']:.4f}")

        scaling_results.append({
            "data_volume": actual_n,
            "best_accuracy": best_acc,
            "best_top3": best_top3,
            "final": final,
            "history": history,
        })

    # Fit power law
    xs = [r["data_volume"] for r in scaling_results]
    ys = [r["best_accuracy"] for r in scaling_results]

    a, b, r2 = fit_power_law(xs, ys)

    print(f"\n{'=' * 60}")
    print(f" SCALING LAW FIT")
    print(f"{'=' * 60}")
    print(f"  Accuracy ≈ {a:.6f} × N^{b:.4f}")
    print(f"  R² = {r2:.4f}")

    # Extrapolate
    print(f"\n  Extrapolations:")
    for target_n in [50_000, 100_000, 500_000, 1_000_000]:
        pred = a * (target_n ** b)
        print(f"    {target_n:>10,} positions → {pred:.1%} predicted accuracy")

    # What data do we need for target accuracies?
    print(f"\n  Data needed for target accuracy:")
    for target_acc in [0.20, 0.30, 0.50]:
        if a > 0 and b != 0:
            needed = (target_acc / a) ** (1.0 / b)
            print(f"    {target_acc:.0%} accuracy → {needed:,.0f} positions needed")

    # Save
    elapsed = time.time() - t0
    results = {
        "experiment": "exp017_scaling_law",
        "hypothesis": "Accuracy follows power-law scaling with data volume",
        "data_volumes": DATA_VOLUMES,
        "eval_size": EVAL_SIZE,
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "power_law_fit": {"a": a, "b": b, "r_squared": r2,
                          "formula": f"accuracy = {a:.6f} * N^{b:.4f}"},
        "scaling_results": scaling_results,
        "extrapolations": {
            str(n): round(a * (n ** b), 4)
            for n in [50_000, 100_000, 500_000, 1_000_000]
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
