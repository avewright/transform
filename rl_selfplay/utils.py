"""Shared helpers for self-play."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent


def resolve_stockfish() -> Path:
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "Stockfish not found. Set STOCKFISH_PATH or install stockfish."
    )


def q_to_wdl(q: float, sharpness: float = 4.0) -> list[float]:
    """Map MCTS Q in [-1, 1] (STM) to W/D/L probabilities."""
    w = 1.0 / (1.0 + math.exp(-sharpness * q))
    l = 1.0 - w
    d = max(0.0, 0.5 - abs(w - 0.5)) * 2.0
    w *= 1.0 - d * 0.5
    l *= 1.0 - d * 0.5
    total = w + d + l
    return [w / total, d / total, l / total]


def q_to_win_pct(q: float) -> float:
    return (q + 1.0) / 2.0


def game_result(board: chess.Board, color: chess.Color) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == color else 0.0


def material_balance(board: chess.Board) -> int:
    """Centipawn-style material from White's perspective."""
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900}
    score = 0
    for sq, piece in board.piece_map().items():
        v = values.get(piece.piece_type, 0)
        score += v if piece.color == chess.WHITE else -v
    return score


def should_adjudicate(board: chess.Board, ply: int, min_ply: int = 80) -> bool:
    """Early draw adjudication to reduce self-play draws."""
    if ply < min_ply:
        return False
    bal = abs(material_balance(board))
    if bal <= 100 and ply >= min_ply:
        return True
    if bal == 0 and ply >= 120:
        return True
    return False
