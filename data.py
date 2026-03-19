"""Chess data pipeline: FEN positions → (input_text, target_move) pairs.

Supports:
  - Generating positions from PGN files
  - Labeling with Stockfish best moves
  - Formatting for LLM consumption
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from config import DataConfig


@dataclass
class ChessPosition:
    """A single chess position with its best move."""
    fen: str
    best_move_uci: str  # e.g. "e2e4"
    best_move_san: str  # e.g. "e4"
    eval_cp: int | None = None  # centipawn evaluation if available


# ---------------------------------------------------------------------------
# FEN → text formatting (multiple encoding strategies)
# ---------------------------------------------------------------------------

def fen_to_prompt(fen: str, encoding: str = "fen") -> str:
    """Convert a FEN string to a prompt for the model.

    Encoding strategies:
      "fen"          — raw FEN string (original, compact but hard for tokenizer)
      "grid"         — 8x8 ASCII board with rank/file labels + metadata
      "grid_compact" — 8x8 grid without labels, minimal metadata
      "squares"      — explicit square-by-square piece listing
    """
    if encoding == "fen":
        return _encode_fen(fen)
    elif encoding == "grid":
        return _encode_grid(fen)
    elif encoding == "grid_compact":
        return _encode_grid_compact(fen)
    elif encoding == "squares":
        return _encode_squares(fen)
    else:
        return _encode_fen(fen)


def _encode_fen(fen: str) -> str:
    """Original FEN encoding."""
    board = chess.Board(fen)
    turn = "White" if board.turn == chess.WHITE else "Black"
    move_num = board.fullmove_number
    return (
        f"[FEN] {fen}\n"
        f"[Turn] {turn} to move (move {move_num})\n"
        f"[Best Move]"
    )


# Piece display: uppercase=White, lowercase=Black, .=empty
_PIECE_SYMBOLS = {
    (chess.PAWN, chess.WHITE): "P", (chess.KNIGHT, chess.WHITE): "N",
    (chess.BISHOP, chess.WHITE): "B", (chess.ROOK, chess.WHITE): "R",
    (chess.QUEEN, chess.WHITE): "Q", (chess.KING, chess.WHITE): "K",
    (chess.PAWN, chess.BLACK): "p", (chess.KNIGHT, chess.BLACK): "n",
    (chess.BISHOP, chess.BLACK): "b", (chess.ROOK, chess.BLACK): "r",
    (chess.QUEEN, chess.BLACK): "q", (chess.KING, chess.BLACK): "k",
}


def _board_to_grid_rows(board: chess.Board) -> list[str]:
    """Return 8 strings, one per rank (8 down to 1), each with 8 piece chars."""
    rows = []
    for rank in range(7, -1, -1):  # 8 down to 1
        row = []
        for file in range(8):  # a to h
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row.append(".")
            else:
                row.append(_PIECE_SYMBOLS[(piece.piece_type, piece.color)])
        rows.append(" ".join(row))
    return rows


def _castling_str(board: chess.Board) -> str:
    """Human-readable castling rights."""
    rights = []
    if board.has_kingside_castling_rights(chess.WHITE):
        rights.append("W-OO")
    if board.has_queenside_castling_rights(chess.WHITE):
        rights.append("W-OOO")
    if board.has_kingside_castling_rights(chess.BLACK):
        rights.append("B-oo")
    if board.has_queenside_castling_rights(chess.BLACK):
        rights.append("B-ooo")
    return " ".join(rights) if rights else "none"


def _ep_str(board: chess.Board) -> str:
    """En passant target square or 'none'."""
    if board.ep_square is not None:
        return chess.square_name(board.ep_square)
    return "none"


def _encode_grid(fen: str) -> str:
    """Full ASCII grid with file/rank labels and all metadata.

    Example:
      [Board]
        a b c d e f g h
      8 r n b q k b n r
      7 p p p p p p p p
      6 . . . . . . . .
      5 . . . . . . . .
      4 . . . . . . . .
      3 . . . . . . . .
      2 P P P P P P P P
      1 R N B Q K B N R
      [Turn] White
      [Castling] W-OO W-OOO B-oo B-ooo
      [En passant] none
      [Move] 1
      [Best Move]
    """
    board = chess.Board(fen)
    rows = _board_to_grid_rows(board)
    turn = "White" if board.turn == chess.WHITE else "Black"

    lines = ["[Board]", "  a b c d e f g h"]
    for i, row in enumerate(rows):
        rank_num = 8 - i
        lines.append(f"{rank_num} {row}")
    lines.append(f"[Turn] {turn}")
    lines.append(f"[Castling] {_castling_str(board)}")
    lines.append(f"[En passant] {_ep_str(board)}")
    lines.append(f"[Move] {board.fullmove_number}")
    lines.append("[Best Move]")
    return "\n".join(lines)


def _encode_grid_compact(fen: str) -> str:
    """Compact grid: no labels, minimal metadata. Fewer tokens.

    Example:
      rnbqkbnr
      pppppppp
      ........
      ........
      ........
      ........
      PPPPPPPP
      RNBQKBNR
      W KQkq - 1
      [Best Move]
    """
    board = chess.Board(fen)
    lines = []
    for rank in range(7, -1, -1):
        row = ""
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += "."
            else:
                row += _PIECE_SYMBOLS[(piece.piece_type, piece.color)]
        lines.append(row)

    # Compact metadata line: turn castling ep_square move_number
    turn_ch = "W" if board.turn == chess.WHITE else "B"
    castling = board.castling_xfen() if board.castling_rights else "-"
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    lines.append(f"{turn_ch} {castling} {ep} {board.fullmove_number}")
    lines.append("[Best Move]")
    return "\n".join(lines)


def _encode_squares(fen: str) -> str:
    """Explicit square-by-square piece listing with metadata.

    Only lists occupied squares, making positions with few pieces very compact.
    Example:
      [White] Ke1 Qd1 Ra1 Rh1 Bc1 Bf1 Nb1 Ng1 Pa2 Pb2 ...
      [Black] ke8 qd8 ra8 rh8 bc8 bf8 nb8 ng8 pa7 pb7 ...
      [Turn] White
      [Castling] W-OO W-OOO B-oo B-ooo
      [En passant] none
      [Best Move]
    """
    board = chess.Board(fen)
    white_pieces = []
    black_pieces = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        sq_name = chess.square_name(sq)
        symbol = _PIECE_SYMBOLS[(piece.piece_type, piece.color)]
        entry = f"{symbol}{sq_name}"
        if piece.color == chess.WHITE:
            white_pieces.append(entry)
        else:
            black_pieces.append(entry)

    turn = "White" if board.turn == chess.WHITE else "Black"
    lines = [
        f"[White] {' '.join(white_pieces)}",
        f"[Black] {' '.join(black_pieces)}",
        f"[Turn] {turn}",
        f"[Castling] {_castling_str(board)}",
        f"[En passant] {_ep_str(board)}",
        "[Best Move]",
    ]
    return "\n".join(lines)


def position_to_training_text(pos: ChessPosition, encoding: str = "fen") -> str:
    """Full training example: prompt + target."""
    prompt = fen_to_prompt(pos.fen, encoding=encoding)
    return f"{prompt} {pos.best_move_uci}"


# ---------------------------------------------------------------------------
# PGN → positions
# ---------------------------------------------------------------------------

def positions_from_pgn(
    pgn_path: str | Path,
    max_positions: int = 100_000,
    min_ply: int = 6,       # skip very early opening moves
    sample_rate: float = 0.3,  # don't take every position in every game
) -> list[str]:
    """Extract FEN positions from a PGN file.

    Returns a list of FEN strings (without best-move labels yet).
    """
    fens: list[str] = []
    pgn_path = Path(pgn_path)

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while len(fens) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply < min_ply:
                    continue
                if random.random() > sample_rate:
                    continue
                # skip positions where game is over
                if board.is_game_over():
                    continue
                fens.append(board.fen())
                if len(fens) >= max_positions:
                    break

    return fens


# ---------------------------------------------------------------------------
# Stockfish labeling
# ---------------------------------------------------------------------------

def label_with_stockfish(
    fens: list[str],
    stockfish_path: str,
    depth: int = 12,
    threads: int = 4,
) -> list[ChessPosition]:
    """Label FEN positions with Stockfish best moves.

    Requires the `stockfish` Python package and the Stockfish binary.
    """
    try:
        from stockfish import Stockfish
    except ImportError:
        raise ImportError(
            "Install the stockfish package: pip install stockfish\n"
            "And ensure the Stockfish binary is available."
        )

    sf = Stockfish(path=stockfish_path, depth=depth)
    sf.update_engine_parameters({"Threads": threads, "Hash": 128})

    positions = []
    for fen in fens:
        sf.set_fen_position(fen)
        best_move_uci = sf.get_best_move()
        if best_move_uci is None:
            continue

        board = chess.Board(fen)
        move = chess.Move.from_uci(best_move_uci)
        best_move_san = board.san(move)

        eval_info = sf.get_evaluation()
        eval_cp = eval_info.get("value") if eval_info.get("type") == "cp" else None

        positions.append(ChessPosition(
            fen=fen,
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            eval_cp=eval_cp,
        ))

    return positions


# ---------------------------------------------------------------------------
# Synthetic positions (for quick testing without PGN/Stockfish)
# ---------------------------------------------------------------------------

def generate_random_positions(n: int = 10_000, max_ply: int = 80) -> list[ChessPosition]:
    """Generate chess positions by playing random legal moves.

    Each position is labeled with a random legal move (NOT the best move).
    Useful only for testing the pipeline — replace with Stockfish labels for real training.
    """
    positions = []
    for _ in range(n):
        board = chess.Board()
        ply = random.randint(6, max_ply)
        for _ in range(ply):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))

        if board.is_game_over():
            continue

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            continue

        move = random.choice(legal_moves)
        positions.append(ChessPosition(
            fen=board.fen(),
            best_move_uci=move.uci(),
            best_move_san=board.san(move),
        ))

    return positions


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_positions(positions: list[ChessPosition], tokenizer, max_length: int = 256):
    """Tokenize chess positions for model consumption.

    Returns a list of dicts with input_ids, attention_mask, and labels.
    """
    examples = []
    for pos in positions:
        text = position_to_training_text(pos)
        prompt = fen_to_prompt(pos.fen)

        # Tokenize full text
        full = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        # Tokenize prompt only to compute label masking
        prompt_tok = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
        prompt_len = prompt_tok["input_ids"].shape[1]

        labels = full["input_ids"].clone()
        # Mask prompt tokens so loss is only on the move prediction
        labels[0, :prompt_len] = -100

        examples.append({
            "input_ids": full["input_ids"].squeeze(0),
            "attention_mask": full["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "fen": pos.fen,
            "target_uci": pos.best_move_uci,
        })

    return examples


# ---------------------------------------------------------------------------
# High-level data loading
# ---------------------------------------------------------------------------

def load_chess_data(cfg: DataConfig) -> tuple[list[ChessPosition], list[ChessPosition]]:
    """Load or generate chess position data, split into train/val.

    For real training, provide a PGN and Stockfish path.
    For quick testing, uses random positions.
    """
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.pgn_path and cfg.stockfish_path:
        # Real data pipeline
        fens = positions_from_pgn(cfg.pgn_path, max_positions=cfg.max_positions)
        positions = label_with_stockfish(
            fens,
            stockfish_path=cfg.stockfish_path,
            depth=cfg.stockfish_depth,
            threads=cfg.stockfish_threads,
        )
    else:
        # Synthetic data for testing
        print("⚠ No PGN/Stockfish configured — generating random positions for testing")
        positions = generate_random_positions(n=min(cfg.max_positions, 10_000))

    random.shuffle(positions)
    split = int(len(positions) * cfg.train_split)
    return positions[:split], positions[split:]
