"""exp014: Full HF Dataset Single-Epoch Scaling

Hypothesis: Training on all ~475K positions for a single epoch will push accuracy
further than 3 epochs on 50K (~25% acc). Each new position provides more signal
than re-seeing the same position.

Motivation:
- exp013 showed plateau at epoch 2-3 with 50K data: 25% acc, 45% top3
- We have 475K samples we aren't using
- At 475K samples / 64 batch = ~7400 steps ≈ 1 long epoch

Expected: > 28% top-1 accuracy from raw data scale.
Time budget: ~8 min for one epoch (extrapolating from batch=64 throughput).
"""

import json
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp014_full_hf_1epoch")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

BATCH_SIZE = 64
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
NUM_EVAL = 500
NUM_GAMES = 4
GAME_SF_DEPTH = 3
LOG_EVERY = 200  # log training loss every N batches


# --- Reuse move mapping and board conversion from exp013 ---

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


def streaming_batches(dataset, old_sorted_uci, batch_size, device):
    """Yield batches directly from dataset without materializing all boards."""
    boards = []
    moves = []
    values = []
    skipped = 0

    for i in range(len(dataset)):
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX:
            skipped += 1
            continue

        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError:
            skipped += 1
            continue

        if move not in board.legal_moves:
            skipped += 1
            continue

        winner = s["winner"]
        if winner == 0:
            value_target = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK):
            value_target = 2
        else:
            value_target = 0

        boards.append(board)
        moves.append(move)
        values.append(value_target)

        if len(boards) == batch_size:
            batch_input = batch_boards_to_token_ids(boards, device)
            move_targets = torch.tensor(
                [move_to_index(m) for m in moves], dtype=torch.long, device=device
            )
            value_targets = torch.tensor(values, dtype=torch.long, device=device)
            yield batch_input, move_targets, value_targets
            boards, moves, values = [], [], []

    # Final partial batch
    if boards:
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor(
            [move_to_index(m) for m in moves], dtype=torch.long, device=device
        )
        value_targets = torch.tensor(values, dtype=torch.long, device=device)
        yield batch_input, move_targets, value_targets


def prepare_eval_data(dataset, old_sorted_uci, n, offset=0):
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
    return data


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load HF dataset ---
    print("\n[1/5] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    print(f"  Dataset: {len(ds):,} samples")

    old_sorted_uci = build_old_move_mapping()
    print(f"  Old move vocab: {len(old_sorted_uci)} moves")

    # Use last chunk for eval, rest for train (streamed)
    eval_data = prepare_eval_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    print(f"  Eval: {len(eval_data)} positions")

    # --- Load model ---
    print("\n[2/5] Loading model...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    trainable = chess_model.trainable_params()
    print(f"  Trainable: {trainable:,} / {chess_model.total_params():,} total")

    pre_eval = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    # --- Train: single pass over full dataset ---
    print(f"\n{'='*60}")
    print(f" [3/5] Training: ~{len(ds):,} positions, 1 epoch, batch={BATCH_SIZE}")
    print(f"{'='*60}")

    optimizer = AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    est_steps = len(ds) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=est_steps)

    chess_model.train()
    total_loss = total_ploss = total_vloss = 0.0
    steps = 0
    train_start = time.time()

    for batch_input, move_targets, value_targets in streaming_batches(
        ds, old_sorted_uci, BATCH_SIZE, device
    ):
        optimizer.zero_grad()
        result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in chess_model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        total_ploss += result.get("policy_loss", torch.tensor(0)).item()
        total_vloss += result.get("value_loss", torch.tensor(0)).item()
        total_loss += result["loss"].item()
        steps += 1

        if steps % LOG_EVERY == 0:
            elapsed = time.time() - train_start
            avg_p = total_ploss / steps
            rate = steps * BATCH_SIZE / elapsed
            print(f"    step {steps}/{est_steps}: p_loss={avg_p:.4f} "
                  f"[{elapsed:.0f}s, {rate:.0f} pos/s]")

    train_time = time.time() - train_start
    avg_p = total_ploss / max(steps, 1)
    avg_v = total_vloss / max(steps, 1)
    total_positions = steps * BATCH_SIZE
    print(f"  Done: {total_positions:,} positions, {steps} steps, {train_time:.0f}s")
    print(f"  Avg p_loss={avg_p:.4f} v_loss={avg_v:.4f}")

    # --- Final eval ---
    print(f"\n{'='*60}")
    print(f" [4/5] Final Eval ({len(eval_data)} positions)")
    print(f"{'='*60}")
    final_eval = evaluate_accuracy(chess_model, eval_data, device)
    print(f"  Accuracy: {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")

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
    print(f"  Data: {total_positions:,} train (streamed), {len(eval_data)} eval")
    print(f"  Training: 1 epoch, {train_time:.0f}s ({total_positions/train_time:.0f} pos/s)")
    print(f"  Pre-train acc: {pre_eval['accuracy']:.1%}")
    print(f"  Final acc:     {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  vs SF depth {GAME_SF_DEPTH}: {score}/{NUM_GAMES} (W{wins}/D{draws}/L{losses})")
    print(f"  Total time: {total_time:.0f}s")
    print(f"\n  Comparison:")
    print(f"    exp012b (5K SF):   14.2% acc, 31% top3")
    print(f"    exp013  (50K HF):  25.0% acc, 45% top3")
    print(f"    exp014  (475K HF): {final_eval['accuracy']:.1%} acc, {final_eval['top3_accuracy']:.1%} top3")

    results = {
        "experiment": "exp014_full_hf_1epoch",
        "hypothesis": "Single epoch over 475K positions outperforms 3 epochs on 50K",
        "data": {
            "source": "avewright/chess-dataset-production-1968",
            "train_positions": total_positions,
            "eval": len(eval_data),
            "train_steps": steps,
        },
        "training": {"epochs": 1, "batch_size": BATCH_SIZE, "lr": LR},
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "avg_policy_loss": avg_p,
        "avg_value_loss": avg_v,
        "games": game_results,
        "game_score": {"wins": wins, "draws": draws, "losses": losses, "total": NUM_GAMES},
        "timing": {"train_s": train_time, "total_s": total_time, "pos_per_s": total_positions / train_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
