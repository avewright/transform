"""Move vocabulary: maps between UCI move strings and integer indices.

Enumerates all possible chess moves (from_sq, to_sq, promotion) as a
fixed vocabulary. This lets us use a simple classification head instead
of autoregressive text generation.

Two vocab versions:
  - LEGACY (5504): all 64×63 square pairs + promotions. Used by all
    checkpoints trained before April 2026.
  - COMPACT (1968): only geometrically reachable moves (ray, knight,
    pawn promotions). ~2.8× smaller output head.  Use for new training
    runs via MOVE_VOCAB_VERSION=legacy env var or by importing
    LEGACY_* symbols directly.

Default is COMPACT (1968 moves) for all new training runs.
"""

import os
import chess
import torch

# ── Legacy vocab (5504) — all sq pairs, used by existing checkpoints ──

def _build_legacy_vocab() -> tuple[list[str], dict[str, int]]:
    moves = set()
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            uci = chess.square_name(from_sq) + chess.square_name(to_sq)
            moves.add(uci)
            to_rank = chess.square_rank(to_sq)
            from_rank = chess.square_rank(from_sq)
            if to_rank in (0, 7) and abs(from_rank - to_rank) <= 2:
                for promo in "qrbn":
                    moves.add(uci + promo)
    move_list = sorted(moves)
    return move_list, {m: i for i, m in enumerate(move_list)}


# ── Compact vocab (1968) — only geometrically reachable moves ──

def _build_compact_vocab() -> tuple[list[str], dict[str, int]]:
    _KNIGHT_DELTAS = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
    _RAY_DELTAS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    moves: set[str] = set()
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        name = chess.square_name(sq)
        # Knight
        for df, dr in _KNIGHT_DELTAS:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                moves.add(name + chess.square_name(chess.square(nf, nr)))
        # Rays (covers rook, bishop, queen, king, and non-promo pawn moves)
        for df, dr in _RAY_DELTAS:
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                moves.add(name + chess.square_name(chess.square(nf, nr)))
                nf += df
                nr += dr
        # Pawn promotions (white rank 6→7, black rank 1→0)
        for promo_from, promo_to in [(6, 7), (1, 0)]:
            if r == promo_from:
                for df in (-1, 0, 1):
                    nf = f + df
                    if 0 <= nf < 8:
                        to_name = chess.square_name(chess.square(nf, promo_to))
                        for p in "qrbn":
                            moves.add(name + to_name + p)
    move_list = sorted(moves)
    return move_list, {m: i for i, m in enumerate(move_list)}


# ── Select active vocab based on env var (default: compact for new runs) ──

_VOCAB_VERSION = os.environ.get("MOVE_VOCAB_VERSION", "compact")

LEGACY_IDX_TO_UCI, LEGACY_UCI_TO_IDX = _build_legacy_vocab()
LEGACY_VOCAB_SIZE = len(LEGACY_IDX_TO_UCI)

COMPACT_IDX_TO_UCI, COMPACT_UCI_TO_IDX = _build_compact_vocab()
COMPACT_VOCAB_SIZE = len(COMPACT_IDX_TO_UCI)

if _VOCAB_VERSION == "compact":
    IDX_TO_UCI, UCI_TO_IDX = COMPACT_IDX_TO_UCI, COMPACT_UCI_TO_IDX
else:
    IDX_TO_UCI, UCI_TO_IDX = LEGACY_IDX_TO_UCI, LEGACY_UCI_TO_IDX

VOCAB_SIZE = len(IDX_TO_UCI)


def legacy_to_compact_map() -> dict[int, int]:
    """Return {legacy_idx: compact_idx} for moves that exist in both vocabs."""
    return {LEGACY_UCI_TO_IDX[m]: COMPACT_UCI_TO_IDX[m]
            for m in COMPACT_UCI_TO_IDX if m in LEGACY_UCI_TO_IDX}


def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move to vocabulary index."""
    uci = move.uci()
    if uci in UCI_TO_IDX:
        return UCI_TO_IDX[uci]
    # Castling: python-chess may use e1g1, data uses e1h1
    if uci in _CASTLE_STD_TO_960:
        return UCI_TO_IDX[_CASTLE_STD_TO_960[uci]]
    raise KeyError(f"Move {uci} not in vocabulary")


def index_to_move(idx: int) -> chess.Move:
    """Convert a vocabulary index to a chess.Move.

    Handles king-to-rook castling indices by converting to standard UCI.
    """
    uci = IDX_TO_UCI[idx]
    if uci in _CASTLE_960_TO_STD:
        uci = _CASTLE_960_TO_STD[uci]
    return chess.Move.from_uci(uci)


# Castling: python-chess uses king-to-target (e1g1), but Stockfish/data uses
# king-to-rook (e1h1). Map both directions so we handle either format.
_CASTLE_STD_TO_960 = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}
_CASTLE_960_TO_STD = {v: k for k, v in _CASTLE_STD_TO_960.items()}


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Create a boolean mask over the move vocabulary for legal moves.

    Handles castling in both UCI formats (king-to-target e1g1 AND
    king-to-rook e1h1) since training data uses king-to-rook style.
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in UCI_TO_IDX:
            mask[UCI_TO_IDX[uci]] = True
        # Also enable the king-to-rook variant for castling moves
        if uci in _CASTLE_STD_TO_960:
            alt = _CASTLE_STD_TO_960[uci]
            if alt in UCI_TO_IDX:
                mask[UCI_TO_IDX[alt]] = True
    return mask
