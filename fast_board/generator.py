"""Python API for random legal board generation."""

from __future__ import annotations

import ctypes
import os
import time
from typing import Sequence

from . import _lib


def generate_fens(
    n: int,
    min_ply: int = 8,
    max_ply: int = 60,
    seed: int | None = None,
    skip_terminal: bool = True,
    max_retries: int = 64,
) -> list[str]:
    """Generate `n` random legal FENs via legal-move walks.

    Args:
        n: Number of positions to generate.
        min_ply / max_ply: Inclusive ply range for each walk.
        seed: RNG seed (None = nondeterministic from OS entropy).
        skip_terminal: Drop checkmates/stalemates (retry the walk).
        max_retries: Retries per position when a walk dies early.

    Returns:
        List of FEN strings (length may be < n only if retries exhausted).
    """
    if n <= 0:
        return []
    if min_ply < 0 or max_ply < min_ply:
        raise ValueError(f"bad ply range: [{min_ply}, {max_ply}]")

    lib = _lib.get_lib()
    fen_max = _lib.fen_max()
    buf = ctypes.create_string_buffer(n * fen_max)

    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") or 1

    written = lib.rbg_generate_fens_ex(
        buf,
        int(n),
        int(min_ply),
        int(max_ply),
        ctypes.c_uint64(seed),
        1 if skip_terminal else 0,
        int(max_retries),
    )
    if written < 0:
        raise RuntimeError("rbg_generate_fens_ex failed")

    out: list[str] = []
    raw = buf.raw
    for i in range(written):
        start = i * fen_max
        chunk = raw[start : start + fen_max]
        out.append(chunk.split(b"\x00", 1)[0].decode("ascii"))
    return out


def generate_fen(
    min_ply: int = 8,
    max_ply: int = 60,
    seed: int | None = None,
    skip_terminal: bool = True,
) -> str:
    """Generate a single random legal FEN."""
    fens = generate_fens(1, min_ply, max_ply, seed, skip_terminal)
    if not fens:
        raise RuntimeError("failed to generate a position")
    return fens[0]


def perft(depth: int) -> int:
    """Node count from startpos (correctness check)."""
    return _lib.perft(depth)


def version() -> str:
    return _lib.version()


def benchmark(
    n: int = 50_000,
    min_ply: int = 8,
    max_ply: int = 40,
    seed: int = 42,
    compare_python_chess: bool = True,
) -> dict:
    """Throughput benchmark. Returns positions/sec stats."""
    t0 = time.perf_counter()
    fens = generate_fens(n, min_ply=min_ply, max_ply=max_ply, seed=seed)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    stats = {
        "n": len(fens),
        "seconds": elapsed,
        "positions_per_sec": len(fens) / elapsed if elapsed > 0 else float("inf"),
        "backend": version(),
        "min_ply": min_ply,
        "max_ply": max_ply,
    }

    if compare_python_chess:
        import random

        import chess

        rng = random.Random(seed)
        t0 = time.perf_counter()
        got = 0
        target = min(n, 2_000)  # python-chess is slow; sample
        while got < target:
            board = chess.Board()
            plies = rng.randint(min_ply, max_ply)
            dead = False
            for _ in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    dead = True
                    break
                board.push(rng.choice(moves))
            if dead or board.is_game_over(claim_draw=True):
                continue
            got += 1
        t1 = time.perf_counter()
        pc_elapsed = t1 - t0
        stats["python_chess_n"] = got
        stats["python_chess_seconds"] = pc_elapsed
        stats["python_chess_positions_per_sec"] = got / pc_elapsed if pc_elapsed > 0 else 0.0
        if stats["python_chess_positions_per_sec"] > 0:
            stats["speedup"] = stats["positions_per_sec"] / stats["python_chess_positions_per_sec"]

    return stats


def validate_fens(fens: Sequence[str], sample: int = 200) -> None:
    """Raise if sampled FENs are illegal under python-chess."""
    import chess

    step = max(1, len(fens) // sample) if fens else 1
    for fen in list(fens)[::step][:sample]:
        board = chess.Board(fen)
        if not board.is_valid():
            raise AssertionError(f"invalid FEN: {fen}")
        # Reachable walks should never leave opponent already in check incorrectly:
        # python-chess Board(fen) accepts; side to move must not have king capturable
        # by the side that just moved — is_valid covers too_many_checkers etc.
