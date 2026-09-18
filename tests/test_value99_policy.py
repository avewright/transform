import chess
import torch

from chess_chessbot import CHESSBOT_UCI_TO_IDX, CHESSBOT_VOCAB_SIZE, boards_to_planes
from chess_value99 import ValueConfig, ValueTransformer
from chess_value99_policy import PolicyConfig, build_value99_policy, chessbot_planes_to_a1, load_value99_trunk
from experiments.exp287_chessbot_99m import policy_losses, synthetic_rows, collate_rows


def test_policy_logits_are_chessbot_shape():
    torch.set_num_threads(1)
    model = build_value99_policy(PolicyConfig(width=32, layers=2, heads=4, ffn_inner=64,
                                              value_square=8, value_hidden=16, dropout=0))
    planes = boards_to_planes([chess.Board()])
    out = model(planes)
    assert out['policy_logits'].shape == (1, CHESSBOT_VOCAB_SIZE)
    assert out['value'].shape == (1,)
    assert (out['value'].abs() <= 1).all()


def test_a1_align_puts_white_pawn_on_rank2():
    planes = boards_to_planes([chess.Board()])
    aligned = chessbot_planes_to_a1(planes)
    # After a1=0, square 8 is a2. ChessBot plane 0 is White pawn.
    assert float(aligned[0, 8, 0]) == 1.0
    assert float(planes[0, 8, 0]) == 0.0


def test_load_value99_trunk_copies_blocks_not_policy():
    torch.set_num_threads(1)
    cfg = dict(width=32, layers=2, heads=4, ffn_inner=64, value_square=8, value_hidden=16, dropout=0)
    src = ValueTransformer(ValueConfig(**cfg))
    dst = build_value99_policy(PolicyConfig(**cfg))
    before = dst.policy_head.weight.detach().clone()
    report = load_value99_trunk(dst, {'model': src.state_dict()})
    assert any(k.startswith('blocks.0.') for k in report['loaded'])
    assert 'policy_head.weight' not in report['loaded']
    torch.testing.assert_close(dst.blocks[0].qkv.weight, src.blocks[0].qkv.weight)
    torch.testing.assert_close(dst.policy_head.weight, before)
    assert 'policy_head.weight' not in src.state_dict()


def test_policy_loss_on_synthetic_startpos():
    torch.set_num_threads(1)
    model = build_value99_policy(PolicyConfig(width=32, layers=2, heads=4, ffn_inner=64,
                                              value_square=8, value_hidden=16, dropout=0))
    planes, policy, *_ = collate_rows(synthetic_rows(2), 0.0)
    loss, hard, valid = policy_losses(model(planes)['policy_logits'], policy, 0.85)
    assert torch.isfinite(loss) and valid.all()
    loss.backward()
    assert model.policy_head.weight.grad is not None
    idx = CHESSBOT_UCI_TO_IDX['e2e4']
    assert idx < 1858
