"""exp007: Expert Iteration — self-play data collection + fine-tuning.

Hypothesis: Fine-tuning on moves from winning games (gradient-based) will
improve chess play more effectively than random weight perturbation.

Design:
  Phase 1 — Data collection: play 8 games (model vs self), collect
            (position, winning_move) pairs from the winning side.
  Phase 2 — Fine-tune: train on those pairs for 2 epochs, lr=2e-5.
  Phase 3 — Evaluate: play 8 games (fine-tuned vs original), measure score.

Metric: head-to-head score of fine-tuned model vs original.
Time budget: ~8 minutes total.
"""

import copy
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

from config import Config, SelfPlayConfig
from constrained import build_token_text_map
from data import fen_to_prompt
from model import load_base_model
from selfplay import Candidate, play_game, play_match

OUTPUT_DIR = Path("outputs/exp007_expert_iteration")
BOARD_ENCODING = "grid_compact"
MAX_MOVES = 80
ADJ_MATERIAL = 5
TEMPERATURE = 0.8
DATA_GAMES = 8          # games for data collection
EVAL_GAMES = 8          # games for evaluation
TRAIN_EPOCHS = 2
TRAIN_LR = 2e-5
MAX_TRAIN_SEQ_LEN = 128  # positions are short


def collect_selfplay_data(model, tokenizer, token_texts, device):
    """Play games and collect (prompt, move_uci) pairs from winning side."""
    print("\n=== Phase 1: Self-Play Data Collection ===")
    training_pairs = []
    game_stats = {"decisive": 0, "draws": 0, "total": 0}

    for g in range(DATA_GAMES):
        # Play game with no perturbation (model vs itself)
        result = play_game(
            model, tokenizer,
            white_noise=None, black_noise=None,
            white_name="self_w", black_name="self_b",
            max_moves=MAX_MOVES,
            temperature=TEMPERATURE,
            constrained=True,
            token_texts=token_texts,
            board_encoding=BOARD_ENCODING,
            adjudicate_material=ADJ_MATERIAL,
        )

        game_stats["total"] += 1
        outcome_label = f"{result.outcome} ({result.termination}, {result.num_moves}m)"

        if result.outcome == "1/2-1/2":
            game_stats["draws"] += 1
            print(f"  Game {g+1}: {outcome_label} -> skip (draw)")
            continue

        game_stats["decisive"] += 1

        # Determine winning color
        winner_is_white = result.outcome == "1-0"

        # Replay to collect positions + moves for the winning side
        board = chess.Board()
        for i, san_move in enumerate(result.pgn_moves):
            is_white_turn = board.turn == chess.WHITE
            if is_white_turn == winner_is_white:
                fen = board.fen()
                move = board.parse_san(san_move)
                uci = move.uci()
                training_pairs.append((fen, uci))
            board.push_san(san_move)

        n_moves = sum(1 for i, m in enumerate(result.pgn_moves)
                      if (i % 2 == 0) == winner_is_white)
        print(f"  Game {g+1}: {outcome_label} -> collected {n_moves} winning moves")

    print(f"\n  Summary: {game_stats['decisive']} decisive / {game_stats['total']} total")
    print(f"  Training pairs: {len(training_pairs)}")
    return training_pairs, game_stats


def build_training_batch(pairs, tokenizer, device, batch_size=4):
    """Convert (fen, uci_move) pairs into training batches.

    Training target: given the position prompt, predict the UCI move tokens.
    Loss is computed only on the move tokens (not the prompt).
    """
    batches = []
    random.shuffle(pairs)

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        texts = []
        for fen, uci in batch_pairs:
            prompt = fen_to_prompt(fen, encoding=BOARD_ENCODING)
            # Full text = prompt + " " + move
            texts.append(f"{prompt} {uci}")

        # Tokenize with padding
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TRAIN_SEQ_LEN,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Build labels: -100 for prompt tokens, actual ids for move tokens
        labels = input_ids.clone()

        # For each example, find where the move starts
        for j, (fen, uci) in enumerate(batch_pairs):
            prompt = fen_to_prompt(fen, encoding=BOARD_ENCODING)
            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_TRAIN_SEQ_LEN)["input_ids"]
            prompt_len = len(prompt_ids)
            # Mask prompt tokens in labels
            labels[j, :prompt_len] = -100
            # Also mask padding
            labels[j, attention_mask[j] == 0] = -100

        batches.append((input_ids, attention_mask, labels))

    return batches


def fine_tune(model, pairs, tokenizer, device):
    """Fine-tune model on (position, move) pairs."""
    print(f"\n=== Phase 2: Fine-Tuning ({len(pairs)} examples, {TRAIN_EPOCHS} epochs) ===")

    model.train()
    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=0.01)

    total_loss = 0.0
    total_steps = 0

    for epoch in range(TRAIN_EPOCHS):
        batches = build_training_batch(pairs, tokenizer, device, batch_size=4)
        epoch_loss = 0.0
        epoch_steps = 0

        for input_ids, attention_mask, labels in batches:
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        total_loss += epoch_loss
        total_steps += epoch_steps
        print(f"  Epoch {epoch+1}/{TRAIN_EPOCHS}: loss={avg_loss:.4f} ({epoch_steps} steps)")

    model.eval()
    avg_total = total_loss / max(total_steps, 1)
    print(f"  Training complete: avg_loss={avg_total:.4f}")
    return avg_total


def evaluate_vs_original(
    finetuned_model, original_state_dict, tokenizer, token_texts, device,
):
    """Play finetuned model vs original, using perturbation swap trick.

    The finetuned model is our 'champion'. We compute the noise as
    (finetuned_weights - original_weights) and use it as the perturbation.
    """
    print(f"\n=== Phase 3: Evaluation ({EVAL_GAMES} games) ===")

    # Compute the weight delta: fine_tuned - original
    noise = {}
    for name, param in finetuned_model.named_parameters():
        if name in original_state_dict:
            delta = param.data - original_state_dict[name].to(device)
            if delta.abs().max() > 0:
                noise[name] = delta

    print(f"  Weight delta: {len(noise)} changed tensors")
    delta_norm = sum(d.norm().item() ** 2 for d in noise.values()) ** 0.5
    print(f"  L2 norm of delta: {delta_norm:.6f}")

    # Now: finetuned_model currently IS the finetuned weights.
    # The "original" can be played by applying -noise (subtracting the delta).
    # But it's easier to just load from state_dict. Let's use the perturbation framework.

    # Reset model to original weights, then finetuned is noise-applied
    for name, param in finetuned_model.named_parameters():
        if name in original_state_dict:
            param.data.copy_(original_state_dict[name].to(device))

    # Now: model = original weights. fine_tuned_noise applied = finetuned model.
    finetuned = Candidate(name="finetuned", noise=noise)
    original = Candidate(name="original", noise=None)

    score_ft, score_orig, games = play_match(
        finetuned_model, tokenizer,
        finetuned, original,
        games=EVAL_GAMES,
        max_moves=MAX_MOVES,
        temperature=TEMPERATURE,
        log_games=True,
        constrained=True,
        token_texts=token_texts,
        board_encoding=BOARD_ENCODING,
        adjudicate_material=ADJ_MATERIAL,
    )

    # Tally W/D/L
    for g_idx, game in enumerate(games):
        ft_is_white = g_idx % 2 == 0
        if game.outcome == "1-0":
            if ft_is_white:
                finetuned.wins += 1; original.losses += 1
            else:
                original.wins += 1; finetuned.losses += 1
        elif game.outcome == "0-1":
            if ft_is_white:
                finetuned.losses += 1; original.wins += 1
            else:
                original.losses += 1; finetuned.wins += 1
        else:
            finetuned.draws += 1; original.draws += 1

    result_label = (
        "FINETUNED WINS" if score_ft > score_orig else
        "TIE" if score_ft == score_orig else
        "ORIGINAL WINS"
    )

    print(f"\n  Result: {result_label}")
    print(f"  Score: finetuned={score_ft:.1f} original={score_orig:.1f}")
    print(f"  W/D/L(ft): {finetuned.wins}/{finetuned.draws}/{finetuned.losses}")
    print(f"  W/D/L(orig): {original.wins}/{original.draws}/{original.losses}")

    game_details = []
    for g_idx, game in enumerate(games):
        game_details.append({
            "outcome": game.outcome,
            "termination": game.termination,
            "num_moves": game.num_moves,
        })

    return {
        "score_finetuned": score_ft,
        "score_original": score_orig,
        "result": result_label,
        "ft_wdl": [finetuned.wins, finetuned.draws, finetuned.losses],
        "orig_wdl": [original.wins, original.draws, original.losses],
        "delta_l2_norm": delta_norm,
        "games": game_details,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    print(f"Model: {cfg.model.name_or_path}")
    model, tokenizer = load_base_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Building token-text map...")
    token_texts = build_token_text_map(tokenizer)
    print(f"Token map: {len(token_texts)} usable tokens")

    # Save original weights for comparison
    print("Saving original weights snapshot...")
    original_state_dict = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
    }

    total_start = time.time()

    # Phase 1: Collect data
    model.eval()
    phase1_start = time.time()
    pairs, game_stats = collect_selfplay_data(model, tokenizer, token_texts, device)
    phase1_time = time.time() - phase1_start

    if len(pairs) < 5:
        print("\nNot enough decisive games for training. Aborting.")
        return

    # Phase 2: Fine-tune
    phase2_start = time.time()
    avg_loss = fine_tune(model, pairs, tokenizer, device)
    phase2_time = time.time() - phase2_start

    # Phase 3: Evaluate
    model.eval()
    phase3_start = time.time()
    eval_results = evaluate_vs_original(
        model, original_state_dict, tokenizer, token_texts, device,
    )
    phase3_time = time.time() - phase3_start

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  exp007: Expert Iteration — RESULTS")
    print(f"{'='*60}")
    print(f"  Data: {game_stats['decisive']}/{game_stats['total']} decisive games -> {len(pairs)} training pairs")
    print(f"  Training: {TRAIN_EPOCHS} epochs, lr={TRAIN_LR}, avg_loss={avg_loss:.4f}")
    print(f"  Eval: {eval_results['result']}")
    print(f"    finetuned={eval_results['score_finetuned']:.1f} original={eval_results['score_original']:.1f}")
    print(f"    W/D/L(ft)={eval_results['ft_wdl']}")
    print(f"  Weight delta L2: {eval_results['delta_l2_norm']:.6f}")
    print(f"  Time: data={phase1_time:.0f}s train={phase2_time:.0f}s eval={phase3_time:.0f}s total={total_time:.0f}s")

    # Save results
    results = {
        "experiment": "exp007_expert_iteration",
        "hypothesis": "Fine-tuning on winning moves improves play vs original",
        "model": cfg.model.name_or_path,
        "config": {
            "board_encoding": BOARD_ENCODING,
            "max_moves": MAX_MOVES,
            "adjudicate_material": ADJ_MATERIAL,
            "temperature": TEMPERATURE,
            "data_games": DATA_GAMES,
            "eval_games": EVAL_GAMES,
            "train_epochs": TRAIN_EPOCHS,
            "train_lr": TRAIN_LR,
        },
        "data_collection": {
            "total_games": game_stats["total"],
            "decisive_games": game_stats["decisive"],
            "training_pairs": len(pairs),
        },
        "training": {
            "avg_loss": avg_loss,
        },
        "evaluation": eval_results,
        "timing": {
            "data_collection_s": phase1_time,
            "training_s": phase2_time,
            "evaluation_s": phase3_time,
            "total_s": total_time,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
