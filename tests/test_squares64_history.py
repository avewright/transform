import chess
import torch
from chess_history import batch_history_features, position_key
from chess_squares64 import Squares64RecurrentConfig, build_squares64, upgrade_with_history
from move_vocab import UCI_TO_IDX


def test_repetition_and_no_mutation():
    board = chess.Board()
    for move in 'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1'.split():
        board.push_uci(move)
    before = board.fen(), list(board.move_stack)
    f = batch_history_features([board])
    assert f['move_repetition'][0, UCI_TO_IDX['f6g8']] == 1
    assert f['rule_features'][0, 2] == 1
    assert f['history_mask'].tolist() == [[1, 1]]
    assert (board.fen(), board.move_stack) == before
    isolated = batch_history_features([chess.Board(board.fen())])
    assert isolated['rule_features'][0, 2] == 0
    assert isolated['history_mask'].sum() == 0
    assert isolated['move_repetition'].sum() == 0


def test_position_identity():
    a = chess.Board(); a.push_uci('e2e4')
    b = a.copy(); b.ep_square = None
    assert position_key(a) == position_key(b)
    b.turn = not b.turn
    assert position_key(a) != position_key(b)
    b = chess.Board(); b.castling_rights = 0
    assert position_key(b) != position_key(chess.Board())


def test_upgrade_equivalence_and_gradients():
    torch.set_num_threads(1)
    cfg = Squares64RecurrentConfig(encoder_dim=16, hidden_dim=32, num_heads=4,
        prefix_layers=1, recurrent_layers=1, recurrent_unrolls=1, suffix_layers=1,
        policy_head_dim=16, value_hidden=16, dropout=0)
    old = build_squares64(cfg).eval()
    new = upgrade_with_history(old, threat_head=True)
    board = chess.Board()
    for move in 'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1'.split():
        board.push_uci(move)
    x = new.encoder.prepare_input(board, torch.device('cpu'))
    a, b = old(x), new(x)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert b['threat_logits'].shape[-1] == 3
    loss = b['policy_logits'][0, UCI_TO_IDX['f6g8']] + b['threat_logits'].sum()
    loss.backward()
    assert new.encoder.history_proj.weight.grad.abs().sum() > 0
    assert new.repetition_gate.weight.grad.abs().sum() > 0
    assert new.threat_head.weight.grad.abs().sum() > 0
    restored = build_squares64(new.config.to_dict())
    restored.load_state_dict(new.state_dict(), strict=True)
