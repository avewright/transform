"""Experiment 004: Compare board encoding strategies.

Hypothesis: Structured board encodings (grid, squares) will produce
more coherent moves than raw FEN because the spatial layout is explicit.

Metrics:
  - Token count per prompt (context cost)
  - Move quality on a fixed set of positions (greedy pick sensibility)
  - Generation speed (tokens/sec)
  - Agreement between encodings (do they pick the same move?)
"""

import sys
import time

import chess
import torch

sys.path.insert(0, ".")

from constrained import build_token_text_map, make_legal_move_processor
from data import fen_to_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed test positions: opening, middlegame, endgame, tactical
TEST_POSITIONS = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "starting"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "after_1e4"),
    ("r1bqkb1r/pppppppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "italian"),
    ("r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8", "middlegame"),
    ("8/5pk1/5p1p/8/3R4/6PP/5PK1/8 w - - 0 40", "rook_endgame"),
    ("r1b1k2r/pppp1ppp/2n2n2/2b1p2q/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6", "complex"),
]

ENCODINGS = ["fen", "grid", "grid_compact", "squares"]


def main():
    print("=== Experiment 004: Board Encoding Comparison ===\n")

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
    print()

    # --- Show encoding examples for starting position ---
    print("=== Encoding Samples (starting position) ===\n")
    fen0 = TEST_POSITIONS[0][0]
    for enc in ENCODINGS:
        prompt = fen_to_prompt(fen0, encoding=enc)
        toks = tokenizer(prompt, return_tensors="pt")
        n_tokens = toks["input_ids"].shape[1]
        print(f"--- {enc} ({n_tokens} tokens) ---")
        print(prompt)
        print()

    # --- Compare move generation across encodings and positions ---
    print("=== Move Generation Comparison ===\n")
    print(f"{'Position':<15} {'Encoding':<14} {'Tokens':<8} {'Move':<8} {'Time(s)':<8}")
    print("-" * 60)

    results = {}  # (pos_label, encoding) -> move

    for fen, label in TEST_POSITIONS:
        board = chess.Board(fen)
        for enc in ENCODINGS:
            prompt = fen_to_prompt(fen, encoding=enc)
            toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            n_tokens = toks["input_ids"].shape[1]
            inputs = {k: v.to("cuda") for k, v in toks.items()}
            prompt_length = inputs["input_ids"].shape[1]

            processor = make_legal_move_processor(board, tokenizer, prompt_length, token_texts=token_texts)

            t0 = time.time()
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=6,
                    do_sample=False,  # greedy for reproducibility
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=processor,
                )
            elapsed = time.time() - t0

            new_text = tokenizer.decode(gen_ids[0, prompt_length:], skip_special_tokens=True).strip()
            uci = new_text.replace(" ", "").lower()[:5].rstrip()

            # Validate
            move_str = uci
            for length in [5, 4]:
                candidate = uci[:length]
                try:
                    move = chess.Move.from_uci(candidate)
                    if move in board.legal_moves:
                        move_str = candidate
                        break
                except (ValueError, chess.InvalidMoveError):
                    pass

            results[(label, enc)] = move_str
            print(f"{label:<15} {enc:<14} {n_tokens:<8} {move_str:<8} {elapsed:<8.3f}")

        print()

    # --- Agreement analysis ---
    print("=== Agreement Matrix ===\n")
    print(f"{'Position':<15}", end="")
    for enc in ENCODINGS:
        print(f" {enc:<14}", end="")
    print("  Agree?")
    print("-" * 80)

    total_agree = 0
    for fen, label in TEST_POSITIONS:
        moves = [results[(label, enc)] for enc in ENCODINGS]
        agree = len(set(moves)) == 1
        if agree:
            total_agree += 1
        print(f"{label:<15}", end="")
        for m in moves:
            print(f" {m:<14}", end="")
        print(f"  {'YES' if agree else 'NO'}")

    print(f"\nTotal agreement: {total_agree}/{len(TEST_POSITIONS)} positions")

    # --- Token cost summary ---
    print("\n=== Token Cost Summary ===\n")
    for enc in ENCODINGS:
        tok_counts = []
        for fen, label in TEST_POSITIONS:
            prompt = fen_to_prompt(fen, encoding=enc)
            toks = tokenizer(prompt, return_tensors="pt")
            tok_counts.append(toks["input_ids"].shape[1])
        avg = sum(tok_counts) / len(tok_counts)
        mn, mx = min(tok_counts), max(tok_counts)
        print(f"  {enc:<14} avg={avg:.0f} tokens  range=[{mn}, {mx}]")


if __name__ == "__main__":
    main()
