"""Board feature extraction: chess.Board -> tensor feature planes.

Converts a chess position into an 8x8xF tensor suitable for a CNN encoder.
Feature planes follow the standard AlphaZero-style representation.

Planes (18 total):
  0-5:   White pieces (P, N, B, R, Q, K)
  6-11:  Black pieces (p, n, b, r, q, k)
  12:    Side to move (all 1s = White, all 0s = Black)
  13:    White kingside castling
  14:    White queenside castling
  15:    Black kingside castling
  16:    Black queenside castling
  17:    En passant (1 on the target square, 0 elsewhere)
"""

import chess
import torch

NUM_PLANES = 18

# Piece type -> plane index offset (0 for white, 6 for black)
_PIECE_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}


def board_to_planes(board: chess.Board) -> torch.Tensor:
    """Convert a chess.Board to an 18x8x8 float tensor.

    The board is oriented with rank 1 at row 0 (White's perspective).
    """
    planes = torch.zeros(NUM_PLANES, 8, 8)

    # Piece planes (0-11)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        plane_idx = _PIECE_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane_idx += 6
        planes[plane_idx, rank, file] = 1.0

    # Side to move (plane 12)
    if board.turn == chess.WHITE:
        planes[12] = 1.0

    # Castling rights (planes 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16] = 1.0

    # En passant (plane 17)
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        planes[17, rank, file] = 1.0

    return planes


def batch_boards_to_planes(boards: list[chess.Board]) -> torch.Tensor:
    """Convert a list of boards to a batch tensor (B, 18, 8, 8)."""
    return torch.stack([board_to_planes(b) for b in boards])


# --- Learned-embedding tokenization ---

# Piece type -> integer ID for embedding lookup
PIECE_IDX = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}
NUM_PIECE_TYPES = 7   # 0=empty + 6 piece types
NUM_COLORS = 3         # 0=none(empty), 1=white, 2=black
NUM_CASTLING_STATES = 16  # 4-bit encoding
NUM_EP_STATES = 9      # 0=none, 1-8=file a-h


def board_to_token_ids(board: chess.Board) -> dict[str, torch.Tensor]:
    """Convert a chess.Board to integer token IDs for LearnedBoardEncoder.

    Returns dict with:
        piece_ids:  (64,) long — 0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K
        color_ids:  (64,) long — 0=none, 1=white, 2=black
        turn:       (1,)  long — 0=white, 1=black
        castling:   (1,)  long — 0-15 (4-bit)
        ep_file:    (1,)  long — 0=none, 1-8=file a-h
    """
    piece_ids = torch.zeros(64, dtype=torch.long)
    color_ids = torch.zeros(64, dtype=torch.long)

    for sq, piece in board.piece_map().items():
        piece_ids[sq] = PIECE_IDX[piece.piece_type]
        color_ids[sq] = 1 if piece.color == chess.WHITE else 2

    turn = 0 if board.turn == chess.WHITE else 1

    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8

    ep_file = 0
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square) + 1  # 1-8

    return {
        "piece_ids": piece_ids,
        "color_ids": color_ids,
        "turn": torch.tensor([turn], dtype=torch.long),
        "castling": torch.tensor([castling], dtype=torch.long),
        "ep_file": torch.tensor([ep_file], dtype=torch.long),
    }


def batch_boards_to_token_ids(
    boards: list[chess.Board],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a list of boards to batched token IDs for LearnedBoardEncoder."""
    batch = [board_to_token_ids(b) for b in boards]
    result = {
        "piece_ids": torch.stack([b["piece_ids"] for b in batch]),
        "color_ids": torch.stack([b["color_ids"] for b in batch]),
        "turn": torch.cat([b["turn"] for b in batch]),
        "castling": torch.cat([b["castling"] for b in batch]),
        "ep_file": torch.cat([b["ep_file"] for b in batch]),
    }
    if device is not None:
        result = {k: v.to(device) for k, v in result.items()}
    return result
