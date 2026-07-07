"""lichess_data.py — Stream Lichess evaluation data from HuggingFace.

Streams from Lichess/chess-position-evaluations (845M positions with SF evals).
Converts to our training format on the fly.

Usage:
    from lichess_data import stream_lichess_training, load_lichess_eval_set

    for batch in stream_lichess_training(batch_size=256):
        # batch = list of dicts with board, move, wdl, phase, etc.
        ...
"""

import math
import os
from pathlib import Path

import chess
from datasets import load_dataset


def _load_hf_token():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


HF_TOKEN = _load_hf_token()
LICHESS_REPO = "Lichess/chess-position-evaluations"


def cp_to_wdl(cp, mate=None):
    """Convert centipawn/mate eval to (win, draw, loss) probabilities."""
    if mate is not None:
        if mate > 0:
            return (1.0, 0.0, 0.0)
        elif mate < 0:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)
    if cp is None:
        return (0.33, 0.34, 0.33)
    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


def classify_phase(board):
    """Classify position as opening/middlegame/endgame."""
    material = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            material += vals.get(p.piece_type, 0)
    if material >= 50 and board.fullmove_number <= 12:
        return "opening"
    elif material <= 26:
        return "endgame"
    return "middlegame"


def stream_lichess_training(batch_size=256, min_depth=10,
                            shuffle_seed=42, buffer_size=50000,
                            max_positions=None,
                            min_cp_abs=None, max_cp_abs=None):
    """Stream training batches from Lichess evaluation dataset.

    Each batch is a list of dicts:
      - board: chess.Board
      - move: chess.Move (best move from PV line)
      - eval_type: "cp" or "mate"
      - eval_value: int
      - wdl: (float, float, float)
      - phase: str
      - top_moves: list with single best move entry
      - ply: int

    Args:
        batch_size: Positions per batch
        min_depth: Minimum SF depth to include (quality filter)
        shuffle_seed: Seed for streaming shuffle
        buffer_size: Shuffle buffer size (larger = better shuffle, more RAM)
        max_positions: Stop after this many positions (None = infinite)
        min_cp_abs: Only include positions with |cp| >= this (tactical filter)
        max_cp_abs: Only include positions with |cp| <= this (balanced filter)
    """
    from move_vocab import UCI_TO_IDX

    ds = load_dataset(LICHESS_REPO, split="train", streaming=True, token=HF_TOKEN)
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=buffer_size)

    batch = []
    total = 0

    for row in ds:
        try:
            # Quality filter: minimum depth
            depth = row.get("depth")
            if depth is not None and int(depth) < min_depth:
                continue

            # Extract best move from PV line
            line = row.get("line", "")
            if not line:
                continue
            best_move_uci = line.split()[0]

            if best_move_uci not in UCI_TO_IDX:
                continue

            fen = row["fen"]
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move_uci)

            if move not in board.legal_moves:
                continue

            # Get evaluation
            cp = row.get("cp")
            mate = row.get("mate")

            # Optional eval filters
            if mate is None and cp is not None:
                abs_cp = abs(cp)
                if min_cp_abs is not None and abs_cp < min_cp_abs:
                    continue
                if max_cp_abs is not None and abs_cp > max_cp_abs:
                    continue

            eval_type = "mate" if mate is not None else "cp"
            eval_value = mate if mate is not None else (cp if cp is not None else 0)
            wdl = cp_to_wdl(cp, mate)

            # Build top_moves list (single entry from PV)
            top_moves = [{"uci": best_move_uci}]
            if mate is not None:
                top_moves[0]["mate"] = mate
            elif cp is not None:
                top_moves[0]["cp"] = cp

            batch.append({
                "board": board,
                "move": move,
                "eval_type": eval_type,
                "eval_value": eval_value,
                "wdl": wdl,
                "phase": classify_phase(board),
                "top_moves": top_moves,
                "ply": board.ply(),
                "depth": depth,
            })

            if len(batch) >= batch_size:
                yield batch
                total += len(batch)
                batch = []

                if max_positions and total >= max_positions:
                    return

        except Exception:
            continue

    if batch:
        yield batch


def load_lichess_eval_set(n=2000, min_depth=15):
    """Load a fixed eval set from Lichess data.

    Uses higher min_depth for eval to ensure quality.
    Streams n positions (no full download needed).
    """
    from move_vocab import UCI_TO_IDX

    ds = load_dataset(LICHESS_REPO, split="train", streaming=True, token=HF_TOKEN)

    data = []
    for row in ds:
        if len(data) >= n:
            break
        try:
            depth = row.get("depth")
            if depth is not None and int(depth) < min_depth:
                continue

            line = row.get("line", "")
            if not line:
                continue
            best_move_uci = line.split()[0]
            if best_move_uci not in UCI_TO_IDX:
                continue

            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(best_move_uci)
            if move not in board.legal_moves:
                continue

            cp = row.get("cp")
            mate = row.get("mate")
            wdl = cp_to_wdl(cp, mate)

            data.append({
                "board": board,
                "move": move,
                "eval_type": "mate" if mate is not None else "cp",
                "eval_value": mate if mate is not None else (cp or 0),
                "wdl": wdl,
                "phase": classify_phase(board),
                "top_moves": [{"uci": best_move_uci,
                               **({"mate": mate} if mate is not None else {"cp": cp or 0})}],
                "ply": board.ply(),
            })
        except Exception:
            continue

    return data


def download_lichess_batch(n=500000, min_depth=10, output_path=None,
                           seed=42):
    """Download a batch of Lichess positions to a local JSONL file.

    Useful for pre-downloading data to avoid streaming overhead during training.
    """
    import json

    if output_path is None:
        output_path = Path("outputs") / f"lichess_{n // 1000}k_d{min_depth}.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {n:,} positions (min_depth={min_depth}) → {output_path}")

    count = 0
    with open(output_path, "w") as f:
        for batch in stream_lichess_training(
            batch_size=1000, min_depth=min_depth,
            shuffle_seed=seed, buffer_size=100000,
            max_positions=n
        ):
            for item in batch:
                record = {
                    "fen": item["board"].fen(),
                    "best_move": item["move"].uci(),
                    "eval_type": item["eval_type"],
                    "eval_value": item["eval_value"],
                    "wdl": list(item["wdl"]),
                    "phase": item["phase"],
                    "top_moves": item["top_moves"],
                    "ply": item["ply"],
                }
                f.write(json.dumps(record) + "\n")
                count += 1

            if count % 10000 == 0:
                print(f"  {count:,}/{n:,} downloaded...", flush=True)

    print(f"  Done: {count:,} positions → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500000)
    parser.add_argument("--min-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    download_lichess_batch(args.n, args.min_depth, args.output, args.seed)
