"""exp012b: Quick Stockfish-Supervised Training (reuses exp012 cache).

Hypothesis: Even 5K positions with 5 epochs (vs 20K/20 epochs) should
show meaningful Stockfish-move agreement. Then play 4 quick games.

Uses cached labels from exp012. Target: < 5 min total.
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
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from config import Config

STOCKFISH_PATH = "stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR = Path("outputs/exp012b_quick_stockfish")
CACHE_FILE = Path("outputs/exp012_stockfish_supervised/labeled_data.json")

NUM_TRAIN = 5000
NUM_EVAL = 500
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
NUM_GAMES = 4
GAME_SF_DEPTH = 3


def cp_to_wdl(cp, ply=30):
    k = 1.0 / (111.7 + 0.5 * max(0, ply))
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0, 0.5 - 0.5 * abs(win - 0.5) * 2)
    total = win + draw + loss
    return win / total, draw / total, loss / total


def prepare_data(labeled):
    data = []
    for e in labeled:
        board = chess.Board(e["fen"])
        move = chess.Move.from_uci(e["uci"])
        if e["eval_type"] == "mate":
            v = e["eval_value"]
            wdl = (1, 0, 0) if v > 0 else (0, 0, 1) if v < 0 else (0, 1, 0)
        else:
            cp = e["eval_value"]
            if not board.turn:
                cp = -cp
            wdl = cp_to_wdl(cp, board.fullmove_number * 2)
        data.append({"board": board, "move": move, "wdl": wdl})
    return data


def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        moves = [d["move"] for d in chunk]
        wdls = [d["wdl"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(m) for m in moves], dtype=torch.long, device=device)
        value_targets = torch.tensor([max(range(3), key=lambda x: w[x]) for w in wdls], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_accuracy(chess_model, eval_data, device, n=None):
    chess_model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = legal = top3_correct = total = 0
    with torch.no_grad():
        for entry in subset:
            board, target = entry["board"], entry["move"]
            pred, probs = chess_model.predict_move(board)
            total += 1
            if pred in board.legal_moves:
                legal += 1
            if pred == target:
                correct += 1
            top3 = probs.topk(min(3, probs.shape[0])).indices.cpu().tolist()
            if move_to_index(target) in top3:
                top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "legal_rate": legal / max(total, 1),
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
    return {"model_color": "white" if model_color else "black",
            "result": result, "model_result": model_result,
            "moves": board.fullmove_number, "termination": term}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cached Stockfish labels
    print(f"Loading cached labels from {CACHE_FILE}...")
    with open(CACHE_FILE) as f:
        cache = json.load(f)

    train_data = prepare_data(cache["train"][:NUM_TRAIN])
    eval_data = prepare_data(cache["eval"][:NUM_EVAL])
    print(f"  train={len(train_data)}, eval={len(eval_data)}")

    # Load backbone
    print("Loading Qwen3-0.6B...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # Build model
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    print(f"  Trainable: {chess_model.trainable_params():,}")

    # Pre-train eval
    pre_eval = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    # Train
    print(f"\n{'='*60}")
    print(f" Training: {len(train_data)} positions, {EPOCHS} epochs, batch={BATCH_SIZE}")
    print(f"{'='*60}")

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
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = ep_ploss = ep_vloss = 0.0
        steps = 0

        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in chess_model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
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

        print(f"  Epoch {epoch+1}/{EPOCHS}: p_loss={avg_p:.4f} v_loss={avg_v:.4f} "
              f"acc={metrics['accuracy']:.1%} top3={metrics['top3_accuracy']:.1%}")

    train_time = time.time() - train_start

    # Final eval on full eval set
    print(f"\n{'='*60}")
    print(f" Final Eval ({len(eval_data)} positions)")
    print(f"{'='*60}")
    final_eval = evaluate_accuracy(chess_model, eval_data, device)
    best_acc = max(h["accuracy"] for h in history)
    print(f"  Accuracy: {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  Best epoch: {best_acc:.1%}")

    # Play games
    print(f"\n{'='*60}")
    print(f" Playing {NUM_GAMES} games vs Stockfish depth={GAME_SF_DEPTH}")
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

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Data: {len(train_data)} train, {len(eval_data)} eval (Stockfish depth 10)")
    print(f"  Training: {EPOCHS} epochs, {train_time:.0f}s")
    print(f"  Pre-train acc: {pre_eval['accuracy']:.1%}")
    print(f"  Final acc:     {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  vs SF depth {GAME_SF_DEPTH}: {score}/{NUM_GAMES} (W{wins}/D{draws}/L{losses})")
    print(f"  Total time: {total_time:.0f}s")

    results = {
        "experiment": "exp012b_quick_stockfish",
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "best_epoch_acc": best_acc,
        "history": history,
        "games": game_results,
        "game_score": {"wins": wins, "draws": draws, "losses": losses, "total": NUM_GAMES},
        "timing": {"train_s": train_time, "total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
