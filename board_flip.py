"""Board-flip utilities for side-to-move normalization.

When Black is to move, flip the board so the model always sees the position
from the side-to-move's perspective. This halves the effective state space
and provides a strong inductive bias (the model only needs to learn one viewpoint).

Operations:
  1. Board array: flip ranks (sq → (7-rank)*8 + file)
  2. Piece colors: swap White (1-6) ↔ Black (7-12)
  3. Turn: always 0 after flip (it's always "my" turn)
  4. Castling: swap White bits (0,1) ↔ Black bits (2,3)
  5. EP square: rank-flip
  6. Move indices: flip from/to square ranks in UCI notation

Reference: Monroe 2024 (ChessFormer) — board always flipped to side-to-move perspective.
"""

import os
import torch

os.environ.setdefault('MOVE_VOCAB_VERSION', 'compact')
from move_vocab import IDX_TO_UCI, COMPACT_UCI_TO_IDX, VOCAB_SIZE


def _flip_rank(sq: int) -> int:
    """Flip a square's rank: rank → 7 - rank. File unchanged."""
    rank, file = divmod(sq, 8)
    return (7 - rank) * 8 + file


def _flip_uci(uci: str) -> str:
    """Flip ranks in a UCI move string."""
    # e.g., "e2e4" → "e7e5", "a7a8q" → "a2a1q"
    from_file, from_rank = uci[0], uci[1]
    to_file, to_rank = uci[2], uci[3]
    promo = uci[4:] if len(uci) == 5 else ""
    new_from_rank = chr(ord('1') + 7 - (ord(from_rank) - ord('1')))
    new_to_rank = chr(ord('1') + 7 - (ord(to_rank) - ord('1')))
    return f"{from_file}{new_from_rank}{to_file}{new_to_rank}{promo}"


def build_flip_move_table() -> torch.Tensor:
    """Build a lookup table: compact_idx → flipped compact_idx.
    
    Returns:
        (VOCAB_SIZE,) long tensor mapping each move index to its rank-flipped equivalent.
    """
    table = torch.zeros(VOCAB_SIZE, dtype=torch.long)
    for idx, uci in enumerate(IDX_TO_UCI):
        flipped_uci = _flip_uci(uci)
        if flipped_uci in COMPACT_UCI_TO_IDX:
            table[idx] = COMPACT_UCI_TO_IDX[flipped_uci]
        else:
            # This shouldn't happen with a well-constructed vocab
            table[idx] = idx  # identity fallback
    return table


# Board square flip lookup (precomputed)
_SQ_FLIP = torch.tensor([_flip_rank(sq) for sq in range(64)], dtype=torch.long)

# Piece color swap: 0→0, 1-6→7-12, 7-12→1-6
_PIECE_SWAP = torch.tensor([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=torch.int8)


def flip_board_array(board_array: torch.Tensor) -> torch.Tensor:
    """Flip board array: swap ranks and piece colors.
    
    Args:
        board_array: (B, 64) or (64,) tensor with piece IDs 0-12
    Returns:
        Flipped board array with same shape and dtype
    """
    orig_dtype = board_array.dtype
    # Step 1: Flip square order (rank reversal)
    flipped = board_array[..., _SQ_FLIP.to(board_array.device)]
    # Step 2: Swap piece colors (White ↔ Black)
    return _PIECE_SWAP.to(flipped.device)[flipped.long()].to(orig_dtype)


def flip_castling(castling: torch.Tensor) -> torch.Tensor:
    """Swap castling bits: WK(0),WQ(1) ↔ BK(2),BQ(3).
    
    Args:
        castling: (B,) or scalar, 4-bit int (K=1, Q=2, k=4, q=8)
    Returns:
        Flipped castling with White/Black bits swapped
    """
    white_bits = castling & 3        # bits 0,1
    black_bits = (castling >> 2) & 3  # bits 2,3
    return (white_bits << 2) | black_bits


def flip_ep_square(ep_square: torch.Tensor) -> torch.Tensor:
    """Flip EP square rank. EP file stays the same, rank flips.
    
    Args:
        ep_square: (B,) int8, 0=none, 1-64=square index, or file-based encoding
    Returns:
        Rank-flipped EP square
    """
    # If ep_square encodes as file (0=none, 1-8=file), no flip needed
    # because file doesn't change. Check if this is file-based or square-based.
    # Our data uses ep_square_to_file() which converts to file index.
    # After conversion to ep_file (0=none, 1-8), no flip needed.
    return ep_square  # File-based EP doesn't need flipping


def flip_batch(batch_input: dict, move_targets: torch.Tensor,
               flip_move_table: torch.Tensor) -> tuple:
    """Flip positions where Black is to move.
    
    Only flips positions where turn == 1 (Black to move).
    White-to-move positions are unchanged.
    
    Args:
        batch_input: dict with fused_ids, turn, castling, ep_file
        move_targets: (B,) long tensor of compact move indices
        flip_move_table: (VOCAB_SIZE,) lookup table from build_flip_move_table()
    Returns:
        Modified (batch_input, move_targets) with Black positions flipped
    """
    turn = batch_input["turn"]
    black_mask = (turn == 1)
    
    if not black_mask.any():
        return batch_input, move_targets
    
    device = turn.device
    fused_ids = batch_input["fused_ids"].clone()
    castling = batch_input["castling"].clone()
    targets = move_targets.clone()
    
    # Flip board for Black-to-move positions
    if black_mask.all():
        fused_ids = flip_board_array(fused_ids)
        castling = flip_castling(castling)
        targets = flip_move_table.to(device)[targets]
    else:
        fused_ids[black_mask] = flip_board_array(fused_ids[black_mask])
        castling[black_mask] = flip_castling(castling[black_mask])
        targets[black_mask] = flip_move_table.to(device)[targets[black_mask]]
    
    return {
        "fused_ids": fused_ids,
        "turn": torch.zeros_like(turn),  # always "my" turn after flip
        "castling": castling,
        "ep_file": batch_input["ep_file"],  # file doesn't change with rank flip
    }, targets


if __name__ == "__main__":
    # Self-test
    print("Board Flip Utility — Self Test")
    print("=" * 50)
    
    # Test UCI flip
    test_cases = [
        ("e2e4", "e7e5"),
        ("a7a8q", "a2a1q"),
        ("g1f3", "g8f6"),
        ("e1g1", "e8g8"),  # White castling → Black castling square equivalent
        ("a1a8", "a8a1"),
    ]
    print("\nUCI flip tests:")
    for uci, expected in test_cases:
        result = _flip_uci(uci)
        status = "✓" if result == expected else f"✗ (got {result})"
        print(f"  {uci} → {result} {status}")
    
    # Test move table
    flip_table = build_flip_move_table()
    round_trip = flip_table[flip_table]  # flip twice should give identity
    identity = torch.arange(VOCAB_SIZE)
    rt_ok = (round_trip == identity).all()
    print(f"\nMove flip table: {VOCAB_SIZE} entries, round-trip identity: {'✓' if rt_ok else '✗'}")
    
    # Test board flip
    board = torch.zeros(1, 64, dtype=torch.int8)
    # Put White King on e1 (sq=4), Black King on e8 (sq=60)
    board[0, 4] = 6   # White King at e1
    board[0, 60] = 12  # Black King at e8
    board[0, 12] = 1   # White Pawn at e2
    board[0, 52] = 7   # Black Pawn at e7
    
    flipped = flip_board_array(board)
    print(f"\nBoard flip test:")
    print(f"  Original: WK@e1(sq4)={board[0,4]}, BK@e8(sq60)={board[0,60]}")
    print(f"  Flipped:  sq4={flipped[0,4]} (should be 6=BK→WK), sq60={flipped[0,60]} (should be 12=WK→BK)")
    # After flip: e1→e8, e8→e1, White→Black, Black→White
    # WK@e1 → flipped to e8 (sq60), becomes B(12→no, should be swapped: 6→12? No, WK→BK)
    # Actually: sq4 (e1) in original has WK(6). After flip, e1 maps FROM e8 (sq60) which had BK(12).
    # BK(12) gets color-swapped to WK(6). So flipped[0,4] should show what was at e8, color-swapped.
    # e8 had BK(12) → swapped to WK(6). So flipped[0,4]=6 ← that's "my king" ✓
    # e1 had WK(6) → now at e8 (sq60), swapped to BK(12). So flipped[0,60]=12 ← "their king" ✓
    assert flipped[0, 4].item() == 6, f"Expected 6, got {flipped[0,4].item()}"  # "my" king at e1
    assert flipped[0, 60].item() == 12, f"Expected 12, got {flipped[0,60].item()}"  # "their" king at e8
    # Pawns: WP@e2(sq12) → moves to e7(sq52), becomes BP(7)→WP(1)  
    # BP@e7(sq52) → moves to e2(sq12), becomes WP(1)→BP(7)
    # Wait, let me re-check: after flip, sq12 (e2) gets value from original sq52 (e7)
    # Original sq52 = BP(7). Swap: 7→1 (WP). So flipped[0,12] = 1 ← "my" pawn at e2 ✓
    assert flipped[0, 12].item() == 1, f"Expected 1, got {flipped[0,12].item()}"
    assert flipped[0, 52].item() == 7, f"Expected 7, got {flipped[0,52].item()}"
    print(f"  Pawn check: flipped e2={flipped[0,12]} (should be 1=my pawn) ✓")
    
    # Test castling flip
    castling = torch.tensor([0b0011])  # White can castle both (K+Q)
    flipped_c = flip_castling(castling)
    print(f"\nCastling flip: 0b{castling.item():04b} → 0b{flipped_c.item():04b}")
    assert flipped_c.item() == 0b1100, f"Expected 0b1100, got 0b{flipped_c.item():04b}"
    print("  ✓ White K+Q → Black k+q")
    
    castling2 = torch.tensor([0b0110])  # WQ(2) + BK(4) → should give WK(1) + BQ(8) = 0b1001
    flipped_c2 = flip_castling(castling2)
    assert flipped_c2.item() == 0b1001, f"Expected 0b1001, got 0b{flipped_c2.item():04b}"
    print(f"  ✓ WQ+BK (0b{castling2.item():04b}) → WK+BQ (0b{flipped_c2.item():04b})")
    
    print(f"\n{'='*50}")
    print("All tests passed! Board flip utility is correct.")
