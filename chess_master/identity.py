"""Position identity, equivalence keys, and board reconstruction."""
from __future__ import annotations

import hashlib
from typing import Any

import chess
import numpy as np

from chess_master.schema import PHASE_METHOD, UNKNOWN

_ID_TO_SYMBOL = {
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}
_CASTLE_BITS = ((8, "K"), (4, "Q"), (2, "k"), (1, "q"))
_MATERIAL_KEYS = (
    ("mat_wp", 1), ("mat_wn", 2), ("mat_wb", 3), ("mat_wr", 4), ("mat_wq", 5),
    ("mat_bp", 7), ("mat_bn", 8), ("mat_bb", 9), ("mat_br", 10), ("mat_bq", 11),
)


def castle_fen(castling: int) -> str:
    return "".join(ch for bit, ch in _CASTLE_BITS if int(castling) & bit) or "-"


def ep_name(ep_square: int) -> str:
    ep = int(ep_square)
    if 0 <= ep <= 63:
        return chess.square_name(ep)
    return "-"


def fen_4(board_array, turn, castling, ep_square) -> str:
    ranks = []
    ba = np.asarray(board_array, dtype=np.int8).reshape(64)
    for rank in range(7, -1, -1):
        empty = 0
        cells: list[str] = []
        for file_idx in range(8):
            pid = int(ba[rank * 8 + file_idx])
            if pid <= 0:
                empty += 1
                continue
            if empty:
                cells.append(str(empty))
                empty = 0
            cells.append(_ID_TO_SYMBOL.get(pid, "1"))
        if empty:
            cells.append(str(empty))
        ranks.append("".join(cells))
    stm = "b" if int(turn) else "w"
    return f"{'/'.join(ranks)} {stm} {castle_fen(castling)} {ep_name(ep_square)}"


def fen_6(board_array, turn, castling, ep_square, halfmove=None, fullmove=None) -> str | None:
    if halfmove is None or fullmove is None:
        return None
    return f"{fen_4(board_array, turn, castling, ep_square)} {int(halfmove)} {int(fullmove)}"


def position_id(board_array, turn, castling, ep_square) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(board_array, dtype=np.int8).reshape(64)).tobytes())
    h.update(bytes([int(turn) & 0xFF, int(castling) & 0xFF]))
    ep = int(ep_square)
    h.update(bytes([255 if ep < 0 else ep & 0xFF]))
    return h.hexdigest()[:32]


def legacy_hash(board_array, turn, castling, ep_square) -> int:
    ba = np.ascontiguousarray(np.asarray(board_array, dtype=np.int8).reshape(1, 64))
    view = ba.view(np.uint8)
    h = np.uint64(0)
    mul = np.uint64(1315423911)
    with np.errstate(over="ignore"):
        for i in range(view.shape[1]):
            h = h * mul + np.uint64(view[0, i])
    h ^= np.uint64(int(turn) + 1)
    h ^= (np.uint64(int(castling) + 1)) << np.uint64(8)
    h ^= (np.uint64(int(ep_square) + 1)) << np.uint64(16)
    return int(h)


def hflip_board_array(board_array) -> np.ndarray:
    ba = np.asarray(board_array, dtype=np.int8).reshape(8, 8)
    return np.ascontiguousarray(ba[:, ::-1]).reshape(64)


def hflip_ep(ep_square: int) -> int:
    ep = int(ep_square)
    if ep < 0 or ep > 63:
        return -1
    rank, file_idx = divmod(ep, 8)
    return rank * 8 + (7 - file_idx)


def equivalence_key(board_array, turn, castling, ep_square) -> tuple[str, str]:
    """Only the validated no-castling horizontal flip is an equivalence."""
    pos = position_id(board_array, turn, castling, ep_square)
    if int(castling) != 0:
        return f"exact:{pos}", "identity"
    flipped = hflip_board_array(board_array)
    a = legacy_hash(board_array, turn, castling, ep_square)
    b = legacy_hash(flipped, turn, 0, hflip_ep(ep_square))
    return f"hflip64:{min(a, b)}", "hflip_no_castling"


def reconstruct_board(board_array, turn, castling, ep_square) -> chess.Board | None:
    b = chess.Board(None)
    ba = np.asarray(board_array, dtype=np.int8).reshape(64)
    for sq, v in enumerate(ba.tolist()):
        if v:
            b.set_piece_at(sq, chess.Piece((int(v) - 1) % 6 + 1, int(v) <= 6))
    b.turn = int(turn) == 0
    b.set_castling_fen(castle_fen(castling))
    ep = int(ep_square)
    b.ep_square = ep if 0 <= ep <= 63 else None
    try:
        if not b.is_valid():
            return None
    except Exception:
        return None
    return b


def material_counts(board_array) -> dict[str, int]:
    ba = np.asarray(board_array, dtype=np.int8).reshape(64)
    out = {}
    for key, pid in _MATERIAL_KEYS:
        out[key] = int((ba == pid).sum())
    return out


def phase_from_non_king(non_king: int) -> int:
    if non_king >= 20:
        return 0
    if non_king >= 10:
        return 1
    return 2


def position_record(board_array, turn, castling, ep_square, *, halfmove=None, fullmove=None,
                    opening_eco=None, opening_name=None) -> dict[str, Any]:
    ba = np.ascontiguousarray(np.asarray(board_array, dtype=np.int8).reshape(64))
    board = reconstruct_board(ba, turn, castling, ep_square)
    mats = material_counts(ba)
    piece_count = int((ba > 0).sum())
    non_king = piece_count - int((ba == 6).sum()) - int((ba == 12).sum())
    eq, eq_method = equivalence_key(ba, turn, castling, ep_square)
    clocks = halfmove is not None and fullmove is not None
    if board is None:
        legality = "invalid"
        in_check = None
    elif board.is_game_over(claim_draw=False):
        legality = "terminal"
        in_check = bool(board.is_check())
    else:
        legality = "legal"
        in_check = bool(board.is_check())
    return {
        "position_id": position_id(ba, turn, castling, ep_square),
        "legacy_hash": np.uint64(legacy_hash(ba, turn, castling, ep_square)),
        "equivalence_key": eq,
        "equivalence_method": eq_method,
        "fen_4": fen_4(ba, turn, castling, ep_square),
        "fen_6": fen_6(ba, turn, castling, ep_square, halfmove, fullmove),
        "board_array": ba.tolist(),
        "turn": np.int8(turn),
        "castling": np.int8(castling),
        "ep_square": np.int8(ep_square if int(ep_square) > 0 or int(ep_square) == 0 else -1),
        "halfmove": None if halfmove is None else int(halfmove),
        "fullmove": None if fullmove is None else int(fullmove),
        "rule_state_available": bool(clocks),
        "piece_count": np.int8(piece_count),
        "non_king_count": np.int8(non_king),
        **{k: np.int8(v) for k, v in mats.items()},
        "phase": np.int8(phase_from_non_king(non_king)),
        "phase_method": PHASE_METHOD,
        "in_check": in_check,
        "legality_status": legality,
        "opening_eco": opening_eco,
        "opening_name": opening_name,
    }


def namespace_game_id(source_name: str, raw) -> tuple[str | None, str]:
    if raw is None or raw == "" or raw == -1:
        return None, f"{source_name}:unknown"
    text = str(raw)
    if source_name == "sf19":
        return text, f"sf19-game:{text}"
    if source_name == "puzzles":
        return text, f"lichess-game:{text}"
    if source_name == "lichess":
        return text, f"lichess-eval:{text}"
    return text, f"{source_name}:{text}"
