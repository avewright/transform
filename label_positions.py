"""Offline Stockfish labeling pipeline — the "dataset factory".

Generates a cached corpus of chess positions with:
  - FEN (normalized, deduplicated)
  - Stockfish eval for ALL legal moves (action-values)
  - Best move UCI
  - WDL estimation from eval
  - Game phase bucket (opening/middlegame/endgame)

Usage:
  python label_positions.py --num 5000 --depth 8 --output data/sf_labels_5k_d8.jsonl
  python label_positions.py --num 50000 --depth 8 --output data/sf_labels_50k_d8.jsonl --threads 4

Output format (JSONL, one JSON object per line):
  {
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "phase": "opening",
    "best_uci": "e7e5",
    "best_cp": -48,
    "move_values": [
      {"uci": "e7e5", "cp": -48, "type": "cp"},
      {"uci": "d7d5", "cp": -52, "type": "cp"},
      ...
    ],
    "wdl": [0.43, 0.14, 0.43],
    "num_legal": 20
  }
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import chess


STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"


def classify_phase(board: chess.Board) -> str:
    """Classify position as opening/middlegame/endgame by material + move number."""
    material = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            material += vals.get(piece.piece_type, 0)
    if board.fullmove_number <= 10 and material >= 60:
        return "opening"
    elif material <= 20:
        return "endgame"
    return "middlegame"


def normalize_fen(board: chess.Board) -> str:
    """Normalize FEN for deduplication (board + turn + castling + ep)."""
    return board.board_fen() + (" w " if board.turn else " b ") + board.castling_xfen()


def generate_positions(n: int, seed: int = 42) -> list[chess.Board]:
    """Generate diverse, deduplicated positions via random play with stratified ply depths."""
    random.seed(seed)
    seen = set()
    positions = []

    ply_ranges = [
        (4, 15, 0.25),    # opening
        (16, 40, 0.45),   # middlegame
        (41, 80, 0.20),   # endgame
        (10, 30, 0.10),   # extra middlegame diversity
    ]

    for min_ply, max_ply, fraction in ply_ranges:
        target = int(n * fraction)
        collected = 0
        attempts = 0
        while collected < target and attempts < target * 20:
            board = chess.Board()
            ply = random.randint(min_ply, max_ply)
            for _ in range(ply):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))
            if not board.is_game_over() and list(board.legal_moves):
                key = normalize_fen(board)
                if key not in seen:
                    seen.add(key)
                    positions.append(board.copy())
                    collected += 1
            attempts += 1

    random.shuffle(positions)
    return positions[:n]


def cp_to_wdl(cp: int, eval_type: str = "cp") -> list[float]:
    """Convert centipawn eval to [win, draw, loss] probabilities.

    Uses sigmoid scaling consistent with LC0/Lichess WDL model.
    """
    if eval_type == "mate":
        if cp > 0:
            return [1.0, 0.0, 0.0]
        elif cp < 0:
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]

    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    # Draw model: peaks near 0 cp
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return [round(win / total, 4), round(draw / total, 4), round(loss / total, 4)]


def label_position(sf, board: chess.Board) -> dict | None:
    """Label one position with Stockfish evals for all legal moves.

    Returns dict with fen, phase, best move, all move values, WDL.
    """
    fen = board.fen()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    move_values = []
    best_cp = None
    best_uci = None
    best_sort = -float("inf")

    for move in legal_moves:
        board.push(move)
        child_fen = board.fen()
        board.pop()

        try:
            sf.set_fen_position(child_fen)
            ev = sf.get_evaluation()
        except Exception:
            continue

        eval_type = ev.get("type", "cp")
        eval_value = ev.get("value", 0)

        # Negate: child position is from opponent's perspective
        cp_from_mover = -eval_value

        move_values.append({
            "uci": move.uci(),
            "cp": cp_from_mover,
            "type": eval_type,
        })

        if eval_type == "mate":
            sort_val = (100000 - abs(cp_from_mover)) if cp_from_mover > 0 else (-100000 + abs(cp_from_mover))
        else:
            sort_val = cp_from_mover

        if sort_val > best_sort:
            best_sort = sort_val
            best_uci = move.uci()
            best_cp = cp_from_mover

    if not move_values or best_uci is None:
        return None

    # Sort descending by eval (best first)
    def sort_key(mv):
        if mv["type"] == "mate":
            return (100000 - abs(mv["cp"])) if mv["cp"] > 0 else (-100000 + abs(mv["cp"]))
        return mv["cp"]
    move_values.sort(key=sort_key, reverse=True)

    return {
        "fen": fen,
        "phase": classify_phase(board),
        "best_uci": best_uci,
        "best_cp": best_cp,
        "move_values": move_values,
        "wdl": cp_to_wdl(best_cp, move_values[0]["type"]),
        "num_legal": len(legal_moves),
    }


def label_all(positions: list[chess.Board], depth: int, threads: int,
              output_path: Path, resume: bool = True) -> list[dict]:
    """Label all positions and write incrementally to JSONL."""
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=depth,
        parameters={"Threads": threads, "Hash": 256},
    )

    # Resume support: count existing lines
    start_idx = 0
    if resume and output_path.exists():
        with open(output_path) as f:
            start_idx = sum(1 for _ in f)
        print(f"  Resuming from position {start_idx}")

    labeled = []
    t0 = time.time()
    mode = "a" if (resume and start_idx > 0) else "w"

    with open(output_path, mode) as f:
        for i in range(start_idx, len(positions)):
            entry = label_position(sf, positions[i])
            if entry:
                f.write(json.dumps(entry) + "\n")
                labeled.append(entry)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                done = i + 1 - start_idx
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(positions) - i - 1) / rate if rate > 0 else 0
                phases = {}
                for e in labeled[-100:]:
                    phases[e["phase"]] = phases.get(e["phase"], 0) + 1
                print(f"  {i+1}/{len(positions)} | {rate:.1f} pos/s | "
                      f"ETA {eta/60:.1f}m | phases: {phases}")

    return labeled


def compute_stats(output_path: Path) -> dict:
    """Compute summary statistics for the labeled dataset."""
    phases = {}
    total_moves = 0
    cp_values = []
    n = 0

    with open(output_path) as f:
        for line in f:
            entry = json.loads(line)
            n += 1
            phases[entry["phase"]] = phases.get(entry["phase"], 0) + 1
            total_moves += entry["num_legal"]
            if entry["move_values"][0]["type"] == "cp":
                cp_values.append(entry["best_cp"])

    avg_moves = total_moves / max(n, 1)
    cp_mean = sum(cp_values) / max(len(cp_values), 1) if cp_values else 0
    cp_std = (sum((c - cp_mean)**2 for c in cp_values) / max(len(cp_values), 1)) ** 0.5 if cp_values else 0

    return {
        "total_positions": n,
        "total_move_evals": total_moves,
        "avg_legal_moves": round(avg_moves, 1),
        "phase_distribution": phases,
        "cp_mean": round(cp_mean, 1),
        "cp_std": round(cp_std, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Stockfish labeling pipeline")
    parser.add_argument("--num", type=int, default=5000, help="Number of positions to generate")
    parser.add_argument("--depth", type=int, default=8, help="Stockfish search depth per move")
    parser.add_argument("--threads", type=int, default=2, help="Stockfish threads")
    parser.add_argument("--output", type=str, default="data/sf_labels.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Stockfish Labeling Pipeline ===")
    print(f"  Positions: {args.num}")
    print(f"  Depth: {args.depth}")
    print(f"  Threads: {args.threads}")
    print(f"  Output: {output_path}")
    print(f"  Seed: {args.seed}")

    # Generate positions
    print(f"\n[1/3] Generating {args.num} diverse positions...")
    t0 = time.time()
    positions = generate_positions(args.num, seed=args.seed)
    print(f"  Generated {len(positions)} unique positions in {time.time()-t0:.1f}s")

    # Label
    print(f"\n[2/3] Labeling with Stockfish depth={args.depth}...")
    labeled = label_all(positions, depth=args.depth, threads=args.threads,
                        output_path=output_path, resume=not args.no_resume)

    # Stats
    print(f"\n[3/3] Summary:")
    stats = compute_stats(output_path)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save stats alongside
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
