"""hf_data.py — Streaming data loader for HuggingFace chess dataset.

Provides a clean interface to stream training data from avewright/chess-positions
without downloading the full dataset. Handles conversion to chess.Board objects
and move vocabulary indices on the fly.

Usage:
    from hf_data import stream_training_data, load_eval_set

    # Stream training batches (lazy, never OOMs):
    for batch in stream_training_data(batch_size=128, split="train"):
        boards, targets, metadata = batch
        ...

    # Load a fixed eval set into memory:
    eval_data = load_eval_set(n=2000, split="test")
"""

import json
import os
from pathlib import Path

import chess
import torch
from datasets import load_dataset


def _load_hf_token():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


REPO_ID = "avewright/chess-positions"
HF_TOKEN = _load_hf_token()


def stream_training_data(batch_size: int = 128, split: str = "train",
                         repo_id: str = REPO_ID, shuffle_seed: int = 42,
                         buffer_size: int = 10000):
    """Yield training batches by streaming from HuggingFace.

    Each batch is a list of dicts with keys:
      - board: chess.Board
      - move: chess.Move (best move)
      - eval_type: str
      - eval_value: int
      - wdl: (float, float, float)
      - phase: str
      - top_moves: list[dict] (parsed from JSON)

    Skips positions where the best move isn't in our move vocabulary.
    """
    from move_vocab import UCI_TO_IDX

    ds = load_dataset(repo_id, split=split, streaming=True, token=HF_TOKEN)
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=buffer_size)

    batch = []
    for row in ds:
        try:
            board = chess.Board(row["fen"])
            move_uci = row["best_move"]
            move = chess.Move.from_uci(move_uci)

            if move_uci not in UCI_TO_IDX:
                continue
            if move not in board.legal_moves:
                continue

            top_moves = json.loads(row["top_moves"]) if row.get("top_moves") else []

            batch.append({
                "board": board,
                "move": move,
                "eval_type": row["eval_type"],
                "eval_value": row["eval_value"],
                "wdl": (row["wdl_win"], row["wdl_draw"], row["wdl_loss"]),
                "phase": row["phase"],
                "top_moves": top_moves,
                "ply": row.get("ply", 0),
            })

            if len(batch) >= batch_size:
                yield batch
                batch = []

        except Exception:
            continue

    if batch:
        yield batch


def load_eval_set(n: int = 2000, split: str = "test",
                  repo_id: str = REPO_ID) -> list[dict]:
    """Load a fixed eval set into memory.

    Returns list of dicts with board + move ready for evaluation.
    """
    from move_vocab import UCI_TO_IDX

    ds = load_dataset(repo_id, split=split, token=HF_TOKEN)

    data = []
    for row in ds:
        try:
            board = chess.Board(row["fen"])
            move_uci = row["best_move"]
            move = chess.Move.from_uci(move_uci)

            if move_uci not in UCI_TO_IDX:
                continue
            if move not in board.legal_moves:
                continue

            data.append({
                "board": board,
                "move": move,
                "eval_type": row["eval_type"],
                "eval_value": row["eval_value"],
                "wdl": (row["wdl_win"], row["wdl_draw"], row["wdl_loss"]),
                "phase": row["phase"],
            })

            if len(data) >= n:
                break
        except Exception:
            continue

    return data


def load_training_set(n: int = None, split: str = "train",
                      repo_id: str = REPO_ID) -> list[dict]:
    """Load training data into memory (for small datasets or experiments).

    For large datasets, prefer stream_training_data() instead.
    """
    from move_vocab import UCI_TO_IDX

    ds = load_dataset(repo_id, split=split, token=HF_TOKEN)

    data = []
    for row in ds:
        try:
            board = chess.Board(row["fen"])
            move_uci = row["best_move"]
            move = chess.Move.from_uci(move_uci)

            if move_uci not in UCI_TO_IDX:
                continue
            if move not in board.legal_moves:
                continue

            data.append({
                "board": board,
                "move": move,
                "eval_type": row["eval_type"],
                "eval_value": row["eval_value"],
                "wdl": (row["wdl_win"], row["wdl_draw"], row["wdl_loss"]),
                "phase": row["phase"],
            })

            if n and len(data) >= n:
                break
        except Exception:
            continue

    return data


def dataset_info(repo_id: str = REPO_ID) -> dict:
    """Get basic info about the HF dataset."""
    try:
        ds = load_dataset(repo_id, token=HF_TOKEN)
        info = {}
        for split_name in ds:
            info[split_name] = {
                "num_rows": len(ds[split_name]),
                "columns": ds[split_name].column_names,
            }
        return info
    except Exception as e:
        return {"error": str(e)}
