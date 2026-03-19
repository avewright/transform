"""Evaluation harness for the chess model.

Tests model on chess positions by:
  1. Move prediction accuracy (does top-1 match Stockfish best move?)
  2. Legal move rate (is the predicted move actually legal in the position?)
  3. Ensemble accuracy (majority vote over top K perturbations)
"""

import chess
import torch
from tqdm import tqdm

from data import ChessPosition, fen_to_prompt, tokenize_positions
from randopt import PerturbationResult, apply_perturbation, remove_perturbation, ensemble_predict


def is_legal_uci(fen: str, uci_move: str) -> bool:
    """Check if a UCI move string is legal in the given position."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return move in board.legal_moves
    except (ValueError, chess.InvalidMoveError):
        return False


@torch.no_grad()
def evaluate_single_model(
    model,
    tokenizer,
    positions: list[ChessPosition],
    batch_size: int = 32,
    max_new_tokens: int = 6,
    use_attnres: bool = False,
) -> dict:
    """Evaluate a single model (no ensemble) on chess positions.

    Returns dict with accuracy, legal_rate, and per-position details.
    """
    model.eval()
    correct = 0
    legal = 0
    total = 0
    details = []

    for i in tqdm(range(0, len(positions), batch_size), desc="Evaluating"):
        batch_pos = positions[i : i + batch_size]

        # Build prompts
        prompts = [fen_to_prompt(p.fen) for p in batch_pos]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        # Generate
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Decode new tokens
        new_ids = gen_ids[:, inputs["input_ids"].shape[1]:]

        for j, pos in enumerate(batch_pos):
            pred_text = tokenizer.decode(new_ids[j], skip_special_tokens=True).strip()
            pred_move = pred_text.replace(" ", "").lower()[:5].rstrip()

            is_correct = pred_move == pos.best_move_uci.lower()
            is_legal = is_legal_uci(pos.fen, pred_move)

            if is_correct:
                correct += 1
            if is_legal:
                legal += 1
            total += 1

            details.append({
                "fen": pos.fen,
                "target": pos.best_move_uci,
                "predicted": pred_move,
                "correct": is_correct,
                "legal": is_legal,
            })

    return {
        "accuracy": correct / max(total, 1),
        "legal_rate": legal / max(total, 1),
        "correct": correct,
        "legal": legal,
        "total": total,
        "details": details,
    }


@torch.no_grad()
def evaluate_ensemble(
    model,
    tokenizer,
    positions: list[ChessPosition],
    top_k_results: list[PerturbationResult],
    batch_size: int = 16,
    max_new_tokens: int = 6,
) -> dict:
    """Evaluate the RandOpt ensemble (majority vote over top K) on chess positions.

    Returns dict with accuracy, legal_rate, and per-position details.
    """
    model.eval()
    correct = 0
    legal = 0
    total = 0
    details = []

    for i in tqdm(range(0, len(positions), batch_size), desc="Ensemble eval"):
        batch_pos = positions[i : i + batch_size]

        prompts = [fen_to_prompt(p.fen) for p in batch_pos]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256,
        )
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Ensemble prediction (majority vote)
        pred_moves = ensemble_predict(
            model, input_ids, attention_mask,
            top_k_results, tokenizer, max_new_tokens,
        )

        for j, pos in enumerate(batch_pos):
            pred_move = pred_moves[j]
            is_correct = pred_move == pos.best_move_uci.lower()
            is_legal = is_legal_uci(pos.fen, pred_move)

            if is_correct:
                correct += 1
            if is_legal:
                legal += 1
            total += 1

            details.append({
                "fen": pos.fen,
                "target": pos.best_move_uci,
                "predicted": pred_move,
                "correct": is_correct,
                "legal": is_legal,
            })

    return {
        "accuracy": correct / max(total, 1),
        "legal_rate": legal / max(total, 1),
        "correct": correct,
        "legal": legal,
        "total": total,
        "details": details,
    }


def print_eval_results(results: dict, label: str = "Evaluation"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    print(f" Accuracy:   {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f" Legal rate: {results['legal_rate']:.4f} ({results['legal']}/{results['total']})")
    print(f"{'='*50}\n")

    # Show some examples
    print("Sample predictions:")
    for d in results["details"][:10]:
        status = "✓" if d["correct"] else ("~" if d["legal"] else "✗")
        print(f"  [{status}] {d['fen'][:40]}... → pred={d['predicted']} target={d['target']}")
