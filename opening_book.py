"""opening_book.py — Simple opening book for the chess model.

Provides principled opening moves for the first ~10 moves.
Uses hardcoded book lines from strong theoretical mainlines.
If the position is in the book, returns the book move.
Otherwise returns None (let the model play).
"""

import chess


# Book format: dict mapping FEN (board only, excluding clocks) -> UCI move
# Using well-known mainlines that are theoretically sound.
# Prioritizing SOLID play (avoiding sharp/tactical lines where the model is out of depth).

def _fen_key(board: chess.Board) -> str:
    """Normalized position key: 'pieces turn castling ep'"""
    parts = board.fen().split()
    return " ".join(parts[:4])  # pieces, turn, castling, en-passant


# Build book from move sequences
_BOOK_LINES: list[list[str]] = [
    # === As White (model plays White, odd-indexed moves are opponent's) ===
    
    # 1.e4 repertoire
    # e4 e5: Italian Game / Giuoco Piano
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6", "b1d2"],
    # e4 e5: Ruy Lopez mainline
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6"],
    # e4 c5: Sicilian - Open (strong for White)
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "f1e2"],
    # e4 c5: Sicilian - Anti-Sicilian Alapin
    ["e2e4", "c7c5", "c2c3", "d7d5", "e4d5", "d8d5", "d2d4", "g8f6", "g1f3"],
    # e4 e6: French Defense - Advance
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6", "g1f3"],
    # e4 e6: French Defense - Tarrasch
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5", "f6d7", "f1d3", "c7c5", "c2c3"],
    # e4 d5: Scandinavian
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6", "g1f3", "c7c6", "f1c4"],
    # e4 c6: Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7", "g1f3", "g8f6", "e4f6"],
    # e4 d6: Pirc
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3", "f8g7", "f1e2", "e8g8", "e1g1"],
    
    # 1.d4 repertoire
    # d4 d5: Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3"],
    # d4 d5: Slav
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4", "a2a4", "c8f5", "e2e3"],
    # d4 Nf6: Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2", "e8g8", "a2a3", "b4c3", "c2c3"],
    # d4 Nf6: King's Indian
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2"],
    
    # 1.Nf3 repertoire
    ["g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8e7", "c1g5"],
    ["g1f3", "g8f6", "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "f8e7"],
    
    # 1.c4 repertoire  
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4", "f1g2", "e8g8"],
    ["c2c4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
    
    # === As Black (model plays Black, even-indexed moves are opponent's) ===
    
    # vs 1.e4 — Sicilian Defense (solid Najdorf/Scheveningen)
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    ["e2e4", "c7c5", "c2c3", "g8f6", "e4e5", "f6d5", "d2d4", "c5d4", "c3d4", "d7d6"],
    
    # vs 1.e4 — e5 systems  
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "g8f6"],
    
    # vs 1.e4 — French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5", "f8e7"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6", "g1f3", "d8b6"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "c7c5", "g1f3", "c5d4", "e4d5", "d8d5"],
    
    # vs 1.d4 — Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8"],
    ["d2d4", "d7d5", "c2c4", "e7e6", "g1f3", "g8f6", "b1c3", "f8e7", "c1f4", "e8g8"],
    
    # vs 1.d4 — Slav
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "e7e6"],
    
    # vs 1.d4 Nf6 — Nimzo/QID
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2"],
    
    # vs 1.c4 — English
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"],
    
    # vs 1.Nf3
    ["g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6"],
    ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "e7e6", "e1g1", "f8e7"],
]


def _build_book() -> dict[str, str]:
    """Build lookup table: fen_key -> next_move_uci."""
    book = {}
    for line in _BOOK_LINES:
        board = chess.Board()
        for i, uci in enumerate(line[:-1]):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
            # After this move, the next move in the line is the book response
            key = _fen_key(board)
            next_uci = line[i + 1]
            next_move = chess.Move.from_uci(next_uci)
            # Only add if the next move is legal
            if next_move in board.legal_moves:
                book[key] = next_uci
        # Also process the last move: only needed if we want the full line
        # including first move from starting position
    
    # Add first moves from starting position
    start_board = chess.Board()
    start_key = _fen_key(start_board)
    # Model as White: prefer 1.e4 (most common, well-explored)
    book[start_key] = "e2e4"
    
    return book


_BOOK: dict[str, str] | None = None


def get_book_move(board: chess.Board) -> chess.Move | None:
    """Look up the position in the opening book. Returns a move or None."""
    global _BOOK
    if _BOOK is None:
        _BOOK = _build_book()
    
    key = _fen_key(board)
    uci = _BOOK.get(key)
    if uci is None:
        return None
    
    move = chess.Move.from_uci(uci)
    if move in board.legal_moves:
        return move
    return None


def book_coverage() -> int:
    """Return the number of positions covered by the book."""
    global _BOOK
    if _BOOK is None:
        _BOOK = _build_book()
    return len(_BOOK)


if __name__ == "__main__":
    # Test the book
    n = book_coverage()
    print(f"Book covers {n} positions")
    
    # Test a few lines
    board = chess.Board()
    moves_played = []
    for _ in range(20):
        move = get_book_move(board)
        if move is None:
            break
        moves_played.append(move.uci())
        board.push(move)
    print(f"Main line: {' '.join(moves_played)}")
    print(f"Position after book: {board.fen()}")
