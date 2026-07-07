"""exp013: HuggingFace Dataset Data Scaling

Hypothesis: Training on 50K game-play positions from the pre-existing HF dataset
(avewright/chess-dataset-production-1968) will substantially improve move prediction
accuracy compared to the 5K Stockfish-labeled baseline (exp012b: 14.2%).

Key insight from previous chess_engine repo: A pre-built dataset of ~475K labeled
positions from Lichess games exists. While these are game-play moves (not
Stockfish-optimal), they provide 10x more training data.

Limitations vs exp012b:
- Moves are from game-play, not Stockfish-optimal (noisier labels)
- No castling/en passant metadata (dataset only stores piece positions + turn)
- Value labels are game-level winner, not position-level evaluation

Expected: Data volume should still dominate despite noisier labels.
Target: > 20% top-1 accuracy (up from 14.2% with 5K Stockfish positions).
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
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp013_hf_dataset_scale")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN = 50_000
NUM_EVAL = 500
EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
GRAD_ACCUM = 1
ENCODER_DIM = 256
SEED = 42
NUM_GAMES = 4
GAME_SF_DEPTH = 3


# ---------------------------------------------------------------------------
# Reconstruct the 1508-move mapping from the old chess_engine repo
# ---------------------------------------------------------------------------

def build_old_move_mapping():
    """Reconstruct the 1508-move mapping used in avewright/chess-dataset-production-1968."""
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
                ucis.add(chess.Move(f, t).uci())
                t += d
        for d in BISHOP_DIRS:
            t = f + d
            while inside(t, f_file):
                ucis.add(chess.Move(f, t).uci())
                t += d
        for off in KNIGHT_OFFS:
            t = f + off
            if inside(t, f_file):
                ucis.add(chess.Move(f, t).uci())

    for f in range(64):
        file_ = f % 8
        rank = f // 8
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


# Old piece encoding -> (piece_type, color) for chess.py
_PIECE2CHESS = {
    1: (chess.PAWN, chess.WHITE), 2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE), 4: (chess.ROOK, chess.WHITE),
    5: (chess.QUEEN, chess.WHITE), 6: (chess.KING, chess.WHITE),
    7: (chess.PAWN, chess.BLACK), 8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK), 10: (chess.ROOK, chess.BLACK),
    11: (chess.QUEEN, chess.BLACK), 12: (chess.KING, chess.BLACK),
}


def hf_sample_to_board(board_flat, turn_raw):
    """Convert HF dataset sample to chess.Board.

    Args:
        board_flat: list of 64 ints (row-major, rank 8 first)
        turn_raw: int from dataset (1=WHITE, 0=BLACK per chess.py convention)
    """
    board = chess.Board(fen=None)
    board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8
            col = idx % 8
            rank = 7 - row
            sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    # In the old code: int(chess.WHITE)=1, int(chess.BLACK)=0
    board.turn = bool(turn_raw)
    return board


def prepare_hf_data(dataset, old_sorted_uci, n, offset=0):
    """Convert HF dataset samples to training data.

    Returns list of dicts with 'board' (chess.Board), 'move' (chess.Move),
    and 'value_target' (0=loss, 1=draw, 2=win from side-to-move perspective).
    """
    data = []
    skipped = 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n:
            break
        s = dataset[i]
        board = hf_sample_to_board(s["board"], s["turn"])
        old_uci = old_sorted_uci[s["move_id"]]

        # Map old UCI to chess.Move
        try:
            move = chess.Move.from_uci(old_uci)
        except ValueError:
            skipped += 1
            continue

        # Verify legality (some fail due to missing castling/ep)
        if move not in board.legal_moves:
            skipped += 1
            continue

        # Map to our 5504 vocab
        if old_uci not in UCI_TO_IDX:
            skipped += 1
            continue

        # Value label: game winner -> side-to-move perspective
        # winner: 0=draw, 1=white, 2=black
        winner = s["winner"]
        if winner == 0:
            value_target = 1  # draw
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK):
            value_target = 2  # win for side to move
        else:
            value_target = 0  # loss for side to move

        data.append({"board": board, "move": move, "value_target": value_target})

    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        moves = [d["move"] for d in chunk]
        values = [d["value_target"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor(
            [move_to_index(m) for m in moves], dtype=torch.long, device=device
        )
        value_targets = torch.tensor(values, dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_accuracy(chess_model, eval_data, device, n=None, batch_size=128):
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
            logits = result["policy_logits"]  # (B, VOCAB_SIZE)

            # Mask illegal moves per board
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
    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = "draw"

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

    # --- Load HF dataset and build move mapping ---
    print("\n[1/5] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    print(f"  Dataset: {len(ds):,} samples")

    old_sorted_uci = build_old_move_mapping()
    print(f"  Old move vocab: {len(old_sorted_uci)} moves")

    # --- Prepare data ---
    print(f"\n[2/5] Preparing {NUM_TRAIN} train + {NUM_EVAL} eval samples...")
    # Use different game ranges for train/eval to avoid data leakage
    # Shuffle by selecting from spread-out indices
    indices = list(range(len(ds)))
    random.shuffle(indices)

    # Prepare eval data from the end of the dataset
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    # Prepare train data from the beginning
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)

    print(f"  Final: train={len(train_data)}, eval={len(eval_data)}")

    # --- Load model ---
    print("\n[3/5] Loading model...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    trainable = chess_model.trainable_params()
    total_p = chess_model.total_params()
    print(f"  Trainable: {trainable:,} / {total_p:,} total")

    # Pre-train eval
    pre_eval = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    # --- Train ---
    print(f"\n{'='*60}")
    print(f" [4/5] Training: {len(train_data)} positions, {EPOCHS} epochs, batch={BATCH_SIZE}")
    print(f"{'='*60}")

    optimizer = AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = EPOCHS * (len(train_data) // (BATCH_SIZE * GRAD_ACCUM) + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    train_start = time.time()

    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = ep_ploss = ep_vloss = 0.0
        steps = 0
        optimizer.zero_grad()

        for bi, (batch_input, move_targets, value_targets) in enumerate(batches):
            result = chess_model(
                batch_input, move_targets=move_targets, value_targets=value_targets
            )
            (result["loss"] / GRAD_ACCUM).backward()
            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in chess_model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            ep_ploss += result.get("policy_loss", torch.tensor(0)).item()
            ep_vloss += result.get("value_loss", torch.tensor(0)).item()
            ep_loss += result["loss"].item()
            steps += 1

        avg_p = ep_ploss / max(steps, 1)
        avg_v = ep_vloss / max(steps, 1)

        metrics = evaluate_accuracy(chess_model, eval_data, device, n=200)
        metrics["policy_loss"] = avg_p
        metrics["value_loss"] = avg_v
        history.append(metrics)

        elapsed = time.time() - train_start
        print(
            f"  Epoch {epoch+1}/{EPOCHS}: p_loss={avg_p:.4f} v_loss={avg_v:.4f} "
            f"acc={metrics['accuracy']:.1%} top3={metrics['top3_accuracy']:.1%} "
            f"[{elapsed:.0f}s]"
        )

    train_time = time.time() - train_start

    # --- Final eval ---
    print(f"\n{'='*60}")
    print(f" Final Eval ({len(eval_data)} positions)")
    print(f"{'='*60}")
    final_eval = evaluate_accuracy(chess_model, eval_data, device)
    best_acc = max(h["accuracy"] for h in history)
    print(f"  Accuracy: {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  Best epoch: {best_acc:.1%}")

    # --- Play games ---
    print(f"\n{'='*60}")
    print(f" [5/5] Playing {NUM_GAMES} games vs Stockfish depth={GAME_SF_DEPTH}")
    print(f"{'='*60}")

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
    score = wins + 0.5 * draws
    print(f"\n  Score: {score}/{NUM_GAMES} (W={wins} D={draws} L={losses})")

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Data: {len(train_data)} train, {len(eval_data)} eval (game-play moves)")
    print(f"  Training: {EPOCHS} epochs, {train_time:.0f}s")
    print(f"  Pre-train acc: {pre_eval['accuracy']:.1%}")
    print(f"  Final acc:     {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  vs SF depth {GAME_SF_DEPTH}: {score}/{NUM_GAMES} (W{wins}/D{draws}/L{losses})")
    print(f"  Total time: {total_time:.0f}s")

    # Compare to baseline
    print(f"\n  Comparison to exp012b (5K Stockfish d10):")
    print(f"    exp012b: 14.2% acc, top3: 31.0%, vs SF d3: 0/4")
    print(f"    exp013:  {final_eval['accuracy']:.1%} acc, top3: {final_eval['top3_accuracy']:.1%}, vs SF d3: {score}/{NUM_GAMES}")

    results = {
        "experiment": "exp013_hf_dataset_scale",
        "hypothesis": "50K game-play positions from HF dataset will outperform 5K Stockfish positions",
        "data": {
            "source": "avewright/chess-dataset-production-1968",
            "train": len(train_data),
            "eval": len(eval_data),
            "label_type": "game-play moves (not Stockfish)",
            "limitations": "no castling/ep metadata, noisy value labels",
        },
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "encoder_dim": ENCODER_DIM},
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "best_epoch_acc": best_acc,
        "history": history,
        "games": game_results,
        "game_score": {"wins": wins, "draws": draws, "losses": losses, "total": NUM_GAMES},
        "timing": {"train_s": train_time, "total_s": total_time},
        "baseline_comparison": {
            "exp012b_acc": 0.142,
            "exp012b_top3": 0.31,
            "exp012b_sf_score": "0/4",
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
