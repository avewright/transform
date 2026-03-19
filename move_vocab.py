"""Move vocabulary: maps between UCI move strings and integer indices.

Enumerates all possible chess moves (from_sq, to_sq, promotion) as a
fixed vocabulary. This lets us use a simple classification head instead
of autoregressive text generation.
"""

import chess
import torch

# Build the full move vocabulary once at import time.
# We enumerate all (from_sq, to_sq) pairs that are geometrically reachable,
# plus promotion variants. This is a superset of legal moves in any position.

def _build_move_vocab() -> tuple[list[str], dict[str, int]]:
    """Build the complete UCI move vocabulary.

    Returns (idx_to_uci, uci_to_idx).
    """
    moves = set()

    # Generate all moves from every possible position configuration.
    # Easier: enumerate geometrically. A piece on any square can move to
    # any other square. Promotions add 4 variants per pawn push to back rank.
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            uci = chess.square_name(from_sq) + chess.square_name(to_sq)
            moves.add(uci)

            # Promotion moves: pawn reaching rank 0 or rank 7
            to_rank = chess.square_rank(to_sq)
            from_rank = chess.square_rank(from_sq)
            if to_rank in (0, 7) and abs(from_rank - to_rank) <= 2:
                for promo in "qrbn":
                    moves.add(uci + promo)

    move_list = sorted(moves)
    move_to_idx = {m: i for i, m in enumerate(move_list)}
    return move_list, move_to_idx


IDX_TO_UCI, UCI_TO_IDX = _build_move_vocab()
VOCAB_SIZE = len(IDX_TO_UCI)


def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move to vocabulary index."""
    return UCI_TO_IDX[move.uci()]


def index_to_move(idx: int) -> chess.Move:
    """Convert a vocabulary index to a chess.Move."""
    return chess.Move.from_uci(IDX_TO_UCI[idx])


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Create a boolean mask over the move vocabulary for legal moves."""
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in UCI_TO_IDX:
            mask[UCI_TO_IDX[uci]] = True
    return mask
