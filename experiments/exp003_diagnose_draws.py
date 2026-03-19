"""Experiment 003: Diagnose all-draws problem.

Hypothesis: Games always draw because (a) the model plays too similarly 
regardless of perturbation, and (b) max_moves is too low for natural 
checkmates. We need to understand whether the model can ever produce 
decisive games, and if not, what adjudication method would give us a 
useful selection signal.

Metrics:
  - Termination reason distribution across games
  - Average pieces remaining at game end
  - Material imbalance at max_moves (can be used as adjudication)
  - Time per move
"""

import sys
import time

import chess
import torch

sys.path.insert(0, ".")

from constrained import build_token_text_map
from selfplay import generate_move
from transformers import AutoModelForCausalLM, AutoTokenizer


def count_material(board: chess.Board) -> dict:
    """Count material for both sides."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
    }
    white_mat = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values() if p.color == chess.WHITE
    )
    black_mat = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values() if p.color == chess.BLACK
    )
    return {"white": white_mat, "black": black_mat, "diff": white_mat - black_mat}


def play_diagnostic_game(model, tokenizer, token_texts, max_moves=200, temperature=0.8):
    board = chess.Board()
    moves = []
    move_times = []

    for i in range(max_moves):
        if board.is_game_over():
            break
        t0 = time.time()
        move, _ = generate_move(
            model, tokenizer, board, temperature=temperature,
            constrained=True, token_texts=token_texts,
        )
        move_times.append(time.time() - t0)
        if move is None:
            break
        moves.append(board.san(move))
        board.push(move)

    # Classify ending
    if board.is_checkmate():
        reason = "checkmate"
    elif board.is_stalemate():
        reason = "stalemate"
    elif board.can_claim_fifty_moves():
        reason = "fifty_moves"
    elif board.is_repetition(3):
        reason = "repetition"
    elif board.is_insufficient_material():
        reason = "insufficient"
    else:
        reason = "max_moves"

    mat = count_material(board)
    avg_time = sum(move_times) / len(move_times) if move_times else 0

    return {
        "num_moves": len(moves),
        "reason": reason,
        "material": mat,
        "pieces_remaining": len(board.piece_map()),
        "avg_move_time": avg_time,
        "total_time": sum(move_times),
        "last_10_moves": moves[-10:] if len(moves) >= 10 else moves,
        "fen": board.fen(),
    }


def main():
    print("=== Experiment 003: Diagnose All-Draws ===\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    print("Building token map...")
    token_texts = build_token_text_map(tokenizer)

    # Play games at different settings
    configs = [
        {"max_moves": 150, "temperature": 0.8, "label": "default"},
        {"max_moves": 150, "temperature": 1.2, "label": "high_temp"},
        {"max_moves": 150, "temperature": 0.5, "label": "low_temp"},
    ]

    for cfg in configs:
        print(f"\n--- Config: {cfg['label']} (max_moves={cfg['max_moves']}, temp={cfg['temperature']}) ---")
        results = []
        for g in range(2):
            r = play_diagnostic_game(
                model, tokenizer, token_texts,
                max_moves=cfg["max_moves"], temperature=cfg["temperature"],
            )
            results.append(r)
            mat_str = f"W={r['material']['white']} B={r['material']['black']} diff={r['material']['diff']:+d}"
            print(
                f"  Game {g+1}: {r['num_moves']} moves, {r['reason']}, "
                f"{r['pieces_remaining']} pieces, material=[{mat_str}], "
                f"{r['avg_move_time']:.2f}s/move, {r['total_time']:.0f}s total"
            )
            print(f"    Last moves: {' '.join(r['last_10_moves'])}")

        # Summary
        avg_moves = sum(r["num_moves"] for r in results) / len(results)
        avg_pieces = sum(r["pieces_remaining"] for r in results) / len(results)
        avg_mat_diff = sum(abs(r["material"]["diff"]) for r in results) / len(results)
        reasons = [r["reason"] for r in results]
        print(f"  Summary: avg_moves={avg_moves:.0f}, avg_pieces={avg_pieces:.0f}, "
              f"avg_|mat_diff|={avg_mat_diff:.1f}, reasons={reasons}")

    print("\n=== Diagnosis Complete ===")
    print("\nKey question: Is material imbalance at game end large enough")
    print("to use as an adjudication signal for evolution?")


if __name__ == "__main__":
    main()
