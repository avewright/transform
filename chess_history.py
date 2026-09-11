"""History features in absolute square orientation, using retained Board stacks.

Occurrences only cover available history. A FEN cannot reconstruct repetitions.
The third rule feature marks history known from the standard initial position.
"""
from collections import Counter
import chess
import torch
from chess_features import board_to_fused_token_ids
from move_vocab import UCI_TO_IDX, VOCAB_SIZE


def position_key(board):
    # Legal EP semantics: an uncapturable EP square does not alter repetition.
    return (board.board_fen(), board.turn, board.clean_castling_rights(),
            board.ep_square if board.has_legal_en_passant() else None)


def batch_history_features(boards, device=None):
    histories, masks, rules, moves = [], [], [], []
    for board in boards:
        cursor = board.copy(stack=True)
        counts = Counter([position_key(cursor)])
        previous = []
        while cursor.move_stack:
            cursor.pop()
            counts[position_key(cursor)] += 1
            if len(previous) < 2:
                previous.append(board_to_fused_token_ids(cursor)["fused_ids"])
        complete = cursor.fen() == chess.STARTING_FEN
        history = torch.zeros(2, 64, dtype=torch.long)
        mask = torch.zeros(2)
        for i, ids in enumerate(previous):
            history[i] = ids
            mask[i] = 1
        repetition = torch.zeros(VOCAB_SIZE)
        child = board.copy(stack=False)
        for move in list(child.legal_moves):
            child.push(move)
            repetition[UCI_TO_IDX[move.uci()]] = float(counts[position_key(child)] >= 2)
            child.pop()
        histories.append(history)
        masks.append(mask)
        rules.append([min(board.halfmove_clock, 150) / 150,
                      min(counts[position_key(board)], 3) / 3, float(complete)])
        moves.append(repetition)
    return {"history_fused_ids": torch.stack(histories).to(device),
            "history_mask": torch.stack(masks).to(device),
            "rule_features": torch.tensor(rules, dtype=torch.float32, device=device),
            "move_repetition": torch.stack(moves).to(device)}
