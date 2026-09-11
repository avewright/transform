#!/usr/bin/env python3
"""Pack Lichess puzzles as play-through solver plies for exp270.

Official FEN is before the opponent setup move. We push that, then label
every solver ply along the published line (one-hot). Values stay masked.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]
from exp193_puzzle_soft_harvest import board_to_arr, castling_byte, phase_id, phase_name
from move_vocab import UCI_TO_IDX

REPO = "Lichess/chess-puzzles"
OUT = ROOT / "outputs" / "exp270_mix_v1" / "puzzle_cache.pt"
SOURCE_ID = 6
MIN_RATING = 1400
MAX_RATING = 3200
TARGET = 400_000


def _row(board: chess.Board, mid: int) -> dict:
    ep = -1 if board.ep_square is None else int(board.ep_square)
    si = torch.full((8,), -1, dtype=torch.int64)
    sp = torch.zeros(8, dtype=torch.float32)
    si[0] = mid
    sp[0] = 1.0
    return {
        "board_array": torch.tensor(board_to_arr(board), dtype=torch.int8),
        "turn": torch.tensor(0 if board.turn else 1, dtype=torch.int8),
        "castling": torch.tensor(castling_byte(board), dtype=torch.int8),
        "ep_square": torch.tensor(ep, dtype=torch.int8),
        "move_idx": torch.tensor(mid, dtype=torch.int64),
        "cp": torch.tensor(0, dtype=torch.int32),
        "mate": torch.tensor(0, dtype=torch.int32),
        "soft_indices": si,
        "soft_probs": sp,
        "source": torch.tensor(SOURCE_ID, dtype=torch.int8),
        "value_valid": torch.tensor(0, dtype=torch.int8),
        "label_depth": torch.tensor(0, dtype=torch.int16),
        "phase": torch.tensor(phase_id(phase_name(board)), dtype=torch.int8),
    }


def play_puzzle(puzzle: dict) -> list[dict]:
    rating = int(puzzle.get("Rating") or 0)
    if rating < MIN_RATING or rating > MAX_RATING:
        return []
    fen = puzzle.get("FEN")
    moves = (puzzle.get("Moves") or "").split()
    if not fen or len(moves) < 2:
        return []
    try:
        board = chess.Board(fen)
    except Exception:
        return []
    rows = []
    for i, uci in enumerate(moves):
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            break
        if mv not in board.legal_moves:
            break
        if i % 2 == 1:
            mid = UCI_TO_IDX.get(uci)
            if mid is None:
                break
            rows.append(_row(board, mid))
        board.push(mv)
        if board.is_game_over(claim_draw=True):
            break
    return rows


def _pos_key(row: dict) -> bytes:
    return (
        row["board_array"].numpy().tobytes()
        + bytes((int(row["turn"]), int(row["castling"]), int(row["ep_square"]) & 0xFF))
    )


def main() -> None:
    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    rng = random.Random(270)
    rng.shuffle(files)
    print(f"files={len(files)} dest={OUT} target={TARGET:,} rating={MIN_RATING}-{MAX_RATING}", flush=True)
    kept: list[dict] = []
    seen: set[bytes] = set()
    scanned = skipped = 0
    for fname in files:
        if len(kept) >= TARGET:
            break
        local = Path(hf_hub_download(REPO, fname, repo_type="dataset"))
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=4096, columns=["PuzzleId", "FEN", "Moves", "Rating"]):
            cols = batch.to_pydict()
            n = len(cols["FEN"])
            for i in range(n):
                scanned += 1
                rows = play_puzzle({
                    "FEN": cols["FEN"][i],
                    "Moves": cols["Moves"][i],
                    "Rating": cols["Rating"][i],
                })
                if not rows:
                    skipped += 1
                    continue
                for row in rows:
                    key = _pos_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append(row)
                    if len(kept) >= TARGET:
                        break
                if len(kept) >= TARGET:
                    break
            if scanned % 50_000 < 4096:
                print(f"  scanned={scanned:,} kept={len(kept):,} skipped={skipped:,}", flush=True)
            if len(kept) >= TARGET:
                break
        print(f"  {fname} kept={len(kept):,}", flush=True)
    if not kept:
        raise SystemExit("no puzzle rows")
    data = {k: torch.stack([r[k] for r in kept], dim=0) for k in kept[0]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUT)
    report = {
        "status": "puzzle_play_complete",
        "repo": REPO,
        "n": int(data["board_array"].shape[0]),
        "scanned_puzzles": scanned,
        "skipped_puzzles": skipped,
        "min_rating": MIN_RATING,
        "max_rating": MAX_RATING,
        "path": str(OUT),
        "note": "every solver ply after opponent setup; value_valid=0",
    }
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
