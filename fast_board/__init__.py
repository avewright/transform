"""Ultra-fast random legal chess position generator.

Positions come from random legal-move walks from the start position,
so every FEN is reachable and legal.

Example:
    from fast_board import generate_fens
    fens = generate_fens(100_000, min_ply=8, max_ply=60)
"""

from .generator import generate_fens, generate_fen, perft, version, benchmark

__all__ = [
    "generate_fens",
    "generate_fen",
    "perft",
    "version",
    "benchmark",
]
