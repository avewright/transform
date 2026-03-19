"""exp012: Stockfish-Supervised Training — Road to Beating Stockfish.

Hypothesis: Training on 50K+ Stockfish-labeled positions with both
policy (best move) and value (evaluation) targets will produce a model
far stronger than self-distillation from the text model.

Pipeline:
  1. Generate diverse positions (random play + varied ply depths)
  2. Label with Stockfish depth-12 (best move + centipawn eval)
  3. Convert evals to W/D/L soft targets using sigmoid scaling
  4. Train learned-embedding encoder + frozen Qwen backbone
  5. Evaluate: accuracy vs Stockfish, then play actual games

Time budget: ~10 min (labeling ~3min at 30pos/s, training ~5min).
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, move_to_index, legal_move_mask
from config import Config

STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
OUTPUT_DIR = Path("outputs/exp012_stockfish_supervised")
CACHE_FILE = OUTPUT_DIR / "labeled_data.json"

# Data
NUM_TRAIN = 20000
NUM_EVAL = 2000
SF_DEPTH = 10
SF_THREADS = 4

# Training
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
ENCODER_DIM = 256
SEED = 42

# Eval games
NUM_GAMES = 10
GAME_SF_DEPTH = 5  # opponent Stockfish at low depth (beatable target)


def generate_diverse_positions(n: int) -> list[chess.Board]:
    """Generate diverse positions with stratified ply distribution."""
    positions = []
    seen = set()

    # Strategy: mix of opening (4-15), middlegame (16-40), endgame (41-80)
    ply_ranges = [
        (4, 15, 0.3),    # 30% opening positions
        (16, 40, 0.45),   # 45% middlegame
        (41, 80, 0.25),   # 25% endgame
    ]

    for min_ply, max_ply, fraction in ply_ranges:
        target = int(n * fraction)
        attempts = 0
        while len([p for p in positions if True]) < len(positions) + target and attempts < target * 5:
            board = chess.Board()
            ply = random.randint(min_ply, max_ply)
            for _ in range(ply):
                if board.is_game_over():
                    break
                move = random.choice(list(board.legal_moves))
                board.push(move)
            if not board.is_game_over() and len(list(board.legal_moves)) > 0:
                key = board.board_fen() + (" w" if board.turn else " b")
                if key not in seen:
                    seen.add(key)
                    positions.append(board.copy())
                    if len(positions) >= n:
                        break
            attempts += 1
            if len(positions) >= n:
                break

    random.shuffle(positions)
    return positions[:n]


def label_with_stockfish_fast(
    positions: list[chess.Board],
    depth: int = SF_DEPTH,
) -> list[dict]:
    """Label positions with Stockfish best move and evaluation.

    Returns list of {fen, best_move_uci, eval_cp, eval_type}.
    Uses the stockfish Python package for speed.
    """
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=depth,
        parameters={"Threads": SF_THREADS, "Hash": 256},
    )

    labeled = []
    for i, board in enumerate(positions):
        fen = board.fen()
        sf.set_fen_position(fen)
        best_uci = sf.get_best_move()
        if best_uci is None:
            continue

        # Validate move
        try:
            move = chess.Move.from_uci(best_uci)
            if move not in board.legal_moves:
                continue
        except (ValueError, chess.InvalidMoveError):
            continue

        ev = sf.get_evaluation()
        labeled.append({
            "fen": fen,
            "uci": best_uci,
            "eval_type": ev.get("type", "cp"),
            "eval_value": ev.get("value", 0),
        })

        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - label_start_time)
            print(f"    {i+1}/{len(positions)} ({rate:.0f} pos/s)")

    return labeled


def cp_to_wdl(cp: int, ply: int = 30) -> tuple[float, float, float]:
    """Convert centipawn eval to (win, draw, loss) probabilities.

    Uses the same formula as Lichess/Leela: sigmoid scaling.
    Positive cp = white advantage.
    """
    # Adjust scaling by game phase (earlier = more draws)
    k = 1.0 / (111.7 + 0.5 * max(0, ply))  # from LC0 WDL model approximation
    win_prob = 1.0 / (1.0 + math.exp(-k * cp))
    loss_prob = 1.0 - win_prob
    # Simple draw model: higher near 0 cp
    draw_prob = max(0, 0.5 - 0.5 * abs(win_prob - 0.5) * 2)
    # Normalize
    total = win_prob + draw_prob + loss_prob
    return win_prob / total, draw_prob / total, loss_prob / total


def prepare_training_data(
    labeled: list[dict],
) -> list[dict]:
    """Convert labeled data to training-ready format with board objects."""
    data = []
    for entry in labeled:
        board = chess.Board(entry["fen"])
        move = chess.Move.from_uci(entry["uci"])

        # Value target: convert eval to WDL
        if entry["eval_type"] == "mate":
            mate_val = entry["eval_value"]
            if mate_val > 0:
                wdl = (1.0, 0.0, 0.0)
            elif mate_val < 0:
                wdl = (0.0, 0.0, 1.0)
            else:
                wdl = (0.0, 1.0, 0.0)
        else:
            cp = entry["eval_value"]
            # Flip for black's perspective
            if not board.turn:  # black to move
                cp = -cp
            wdl = cp_to_wdl(cp, ply=board.fullmove_number * 2)

        data.append({
            "board": board,
            "move": move,
            "wdl": wdl,  # (win, draw, loss) from side-to-move perspective
            "cp": entry.get("eval_value", 0),
        })

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
        move_targets = torch.tensor(
            [move_to_index(m) for m in moves], dtype=torch.long, device=device,
        )
        # WDL as soft targets (win=2, draw=1, loss=0 for cross_entropy)
        # But we use the index of max as hard target for simplicity
        value_targets = torch.tensor(
            [max(range(3), key=lambda x: w[x]) for w in wdls],
            dtype=torch.long, device=device,
        )  # 0=win, 1=draw, 2=loss

        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_accuracy(chess_model, eval_data, device, n=None):
    """Evaluate move prediction accuracy."""
    chess_model.eval()
    if n:
        eval_data = eval_data[:n]
    correct = legal = total = 0
    top3_correct = 0

    with torch.no_grad():
        for entry in eval_data:
            board = entry["board"]
            target_move = entry["move"]
            pred_move, probs = chess_model.predict_move(board)
            total += 1
            if pred_move in board.legal_moves:
                legal += 1
            if pred_move == target_move:
                correct += 1
            # Top-3 accuracy
            mask = legal_move_mask(board).to(device)
            top3_moves = probs.topk(min(3, probs.shape[0])).indices.cpu().tolist()
            target_idx = move_to_index(target_move)
            if target_idx in top3_moves:
                top3_correct += 1

    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "legal_rate": legal / max(total, 1),
        "total": total,
    }


def play_game_vs_stockfish(
    chess_model: ChessModel,
    sf_depth: int,
    model_color: chess.Color,
    device: torch.device,
    max_moves: int = 150,
) -> dict:
    """Play a full game: model vs Stockfish."""
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=sf_depth,
        parameters={"Threads": 2, "Hash": 64},
    )

    board = chess.Board()
    move_log = []
    chess_model.eval()

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            # Model's move
            pred_move, probs = chess_model.predict_move(board)
            if pred_move not in board.legal_moves:
                # Fallback: pick random legal
                pred_move = random.choice(list(board.legal_moves))
            move_log.append(("model", pred_move.uci()))
            board.push(pred_move)
        else:
            # Stockfish's move
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None:
                break
            sf_move = chess.Move.from_uci(sf_uci)
            move_log.append(("stockfish", sf_uci))
            board.push(sf_move)

    # Determine result
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
        else "loss" if winner != "draw"
        else "draw"
    )

    return {
        "model_color": "white" if model_color == chess.WHITE else "black",
        "result": result,
        "model_result": model_result,
        "moves": len(move_log),
        "termination": board.outcome().termination.name if board.outcome() else "max_moves",
    }


def main():
    global label_start_time
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    total_start = time.time()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Qwen backbone ===
    print("Loading Qwen3-0.6B backbone...")
    full_model, tokenizer = load_base_model(cfg)
    full_model = full_model.to(device)

    # === Generate and label with Stockfish ===
    if CACHE_FILE.exists():
        print(f"Loading cached Stockfish labels from {CACHE_FILE}...")
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        train_labeled = cache["train"]
        eval_labeled = cache["eval"]
        label_time = 0.0
    else:
        over_gen = 1.1  # slight over-generation
        n_gen_train = int(NUM_TRAIN * over_gen)
        n_gen_eval = int(NUM_EVAL * over_gen)

        print(f"Generating {n_gen_train + n_gen_eval} diverse positions...")
        gen_start = time.time()
        all_boards = generate_diverse_positions(n_gen_train + n_gen_eval)
        train_boards = all_boards[:n_gen_train]
        eval_boards = all_boards[n_gen_train:]
        gen_time = time.time() - gen_start
        print(f"  Generated {len(all_boards)} in {gen_time:.1f}s")

        print(f"Labeling with Stockfish (depth={SF_DEPTH})...")
        label_start_time = time.time()
        train_labeled = label_with_stockfish_fast(train_boards, depth=SF_DEPTH)
        eval_labeled = label_with_stockfish_fast(eval_boards, depth=SF_DEPTH)
        label_time = time.time() - label_start_time

        # Trim
        train_labeled = train_labeled[:NUM_TRAIN]
        eval_labeled = eval_labeled[:NUM_EVAL]
        print(f"  Labeled: train={len(train_labeled)}, eval={len(eval_labeled)} ({label_time:.0f}s)")

        # Cache
        with open(CACHE_FILE, "w") as f:
            json.dump({"train": train_labeled, "eval": eval_labeled}, f)
        print(f"  Cached to {CACHE_FILE}")

    # === Prepare data ===
    train_data = prepare_training_data(train_labeled)
    eval_data = prepare_training_data(eval_labeled)
    print(f"  Data ready: train={len(train_data)}, eval={len(eval_data)}")

    # Stats
    cp_values = [d["cp"] for d in train_data if d["cp"] is not None]
    if cp_values:
        print(f"  Eval distribution: mean={sum(cp_values)/len(cp_values):.0f}cp "
              f"std={torch.tensor(cp_values, dtype=torch.float).std().item():.0f}cp")

    # === Build model ===
    print(f"\n{'='*60}")
    print(f" Building ChessModel (learned encoder, frozen backbone)")
    print(f"{'='*60}")

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    trainable = chess_model.trainable_params()
    total = chess_model.total_params()
    print(f"  Trainable: {trainable:,} / {total:,}")

    # === Pre-training eval ===
    pre_eval = evaluate_accuracy(chess_model, eval_data, device, n=500)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%} legal={pre_eval['legal_rate']:.1%}")

    # === Train ===
    print(f"\n{'='*60}")
    print(f" Training ({len(train_data)} examples, {EPOCHS} epochs, batch={BATCH_SIZE})")
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
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        steps = 0

        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = chess_model(
                batch_input,
                move_targets=move_targets,
                value_targets=value_targets,
            )
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in chess_model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()

            epoch_policy_loss += result.get("policy_loss", torch.tensor(0)).item()
            epoch_value_loss += result.get("value_loss", torch.tensor(0)).item()
            steps += 1

        avg_p = epoch_policy_loss / max(steps, 1)
        avg_v = epoch_value_loss / max(steps, 1)

        # Eval every 2 epochs (saves time on 2K eval set)
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            metrics = evaluate_accuracy(chess_model, eval_data, device, n=500)
        else:
            metrics = history[-1].copy() if history else {"accuracy": 0, "top3_accuracy": 0, "legal_rate": 1}

        metrics["policy_loss"] = avg_p
        metrics["value_loss"] = avg_v
        metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(metrics)

        print(
            f"  Epoch {epoch+1:2d}/{EPOCHS}: "
            f"p_loss={avg_p:.4f} v_loss={avg_v:.4f} "
            f"acc={metrics['accuracy']:.1%} top3={metrics['top3_accuracy']:.1%} "
            f"legal={metrics['legal_rate']:.1%}"
        )

    train_time = time.time() - train_start

    # === Final evaluation ===
    print(f"\n{'='*60}")
    print(f" Final Evaluation")
    print(f"{'='*60}")

    final_eval = evaluate_accuracy(chess_model, eval_data, device)
    print(f"  Accuracy: {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  Legal:    {final_eval['legal_rate']:.1%}")
    print(f"  Best epoch acc: {max(h['accuracy'] for h in history):.1%}")

    # === Play games vs Stockfish ===
    print(f"\n{'='*60}")
    print(f" Playing {NUM_GAMES} games vs Stockfish (depth={GAME_SF_DEPTH})")
    print(f"{'='*60}")

    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        result = play_game_vs_stockfish(
            chess_model, GAME_SF_DEPTH, color, device,
        )
        game_results.append(result)
        symbol = {"win": "W", "loss": "L", "draw": "D"}[result["model_result"]]
        print(f"  Game {g+1}: {result['model_color']} {symbol} in {result['moves']} moves ({result['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")
    score = wins + 0.5 * draws
    print(f"\n  Score: {score}/{NUM_GAMES} (W={wins} D={draws} L={losses})")

    # Test positions
    test_positions = [
        ("Starting", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
        ("Italian", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")),
        ("Sicilian", chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")),
    ]

    print("\n  Move predictions vs Stockfish:")
    from stockfish import Stockfish as SF
    sf_check = SF(path=STOCKFISH_PATH, depth=SF_DEPTH)
    for name, board in test_positions:
        pred, probs = chess_model.predict_move(board)
        sf_check.set_fen_position(board.fen())
        sf_best = sf_check.get_best_move()
        match = "MATCH" if pred.uci() == sf_best else "diff"
        top3_idx = probs.topk(3).indices.cpu().tolist()
        top3 = [IDX_TO_UCI[i] for i in top3_idx]
        sf_in_top3 = "YES" if sf_best in top3 else "no"
        print(f"    {name:15s}: model={pred.uci()} sf={sf_best} [{match}] sf_in_top3={sf_in_top3}")

    total_time = time.time() - total_start

    # === Summary ===
    print(f"\n{'='*60}")
    print(f" exp012 RESULTS — Stockfish Supervised Training")
    print(f"{'='*60}")
    print(f"  Data: {len(train_data)} train (SF depth={SF_DEPTH}), {len(eval_data)} eval")
    print(f"  Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR} cosine")
    print(f"  Final: acc={final_eval['accuracy']:.1%} top3={final_eval['top3_accuracy']:.1%}")
    print(f"  vs SF depth-{GAME_SF_DEPTH}: {score}/{NUM_GAMES} ({wins}W {draws}D {losses}L)")
    print(f"  Time: label={label_time:.0f}s train={train_time:.0f}s total={total_time:.0f}s")

    results = {
        "experiment": "exp012_stockfish_supervised",
        "hypothesis": "Stockfish-labeled training far surpasses self-distillation",
        "data": {
            "train": len(train_data),
            "eval": len(eval_data),
            "sf_depth": SF_DEPTH,
            "seed": SEED,
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
        "games_vs_stockfish": {
            "opponent_depth": GAME_SF_DEPTH,
            "results": game_results,
            "score": f"{score}/{NUM_GAMES}",
            "wins": wins,
            "draws": draws,
            "losses": losses,
        },
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
