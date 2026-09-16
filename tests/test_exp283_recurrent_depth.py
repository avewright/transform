import os
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import pytest
import torch

from chess_squares64 import Squares64RecurrentConfig, build_squares64
from experiments.exp283_recurrent_depth import (
    depth_schedule, save_model, train_arm, exclude_overlap, sweep,
)
from autoresearch_8gb.pipeline import board_to_cache_row, stack_rows, attach_static_targets
from types import SimpleNamespace


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(283)
    return build_squares64(Squares64RecurrentConfig(
        encoder_dim=16, hidden_dim=32, num_heads=4, prefix_layers=1,
        recurrent_layers=1, recurrent_unrolls=3, suffix_layers=1,
        policy_head_dim=16, value_hidden=16, dropout=0, zero_init_out_proj=False,
    ))


def test_override_preserves_default_and_checkpoint_keys():
    model = tiny().eval()
    x = model.encoder.prepare_input(chess.Board(), torch.device("cpu"))
    keys = set(model.state_dict())
    with torch.no_grad():
        default = model(x)
        explicit = model(x, recurrent_unrolls=3)
        extra = model(x, recurrent_unrolls=5)
        after = model(x)
    for k in default:
        torch.testing.assert_close(default[k], explicit[k], rtol=0, atol=0)
        torch.testing.assert_close(default[k], after[k], rtol=0, atol=0)
    assert not torch.allclose(default["policy_logits"], extra["policy_logits"])
    assert model.config.recurrent_unrolls == 3
    assert set(model.state_dict()) == keys


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_bad_depth(depth):
    with pytest.raises(ValueError):
        tiny()({}, recurrent_unrolls=depth)


def test_all_passes_receive_gradients():
    model = tiny().train()
    x = model.encoder.prepare_input(chess.Board(), torch.device("cpu"))
    visits = []
    hook = model.bank[0].register_forward_hook(lambda *args: visits.append(1))
    model(x, recurrent_unrolls=5)["policy_logits"].square().mean().backward()
    hook.remove()
    assert len(visits) == 5
    assert model.bank[0].attn.out_proj.weight.grad.abs().sum() > 0


def test_compute_matched_deterministic_schedule():
    a = depth_schedule(120, [2, 3, 4], 283)
    assert a == depth_schedule(120, [2, 3, 4], 283)
    assert sum(a) == 3 * len(a)
    assert {i: a.count(i) for i in set(a)} == {2: 40, 3: 40, 4: 40}
    with pytest.raises(ValueError):
        depth_schedule(120, [2, 3, 6], 283)
    with pytest.raises(ValueError):
        depth_schedule(121, [2, 3, 4], 283)


def test_export_depth_loads_strictly(tmp_path):
    model = tiny().eval()
    path = tmp_path / "loops_5.pt"
    save_model(model, path, {"test": True}, depth=5)
    payload = torch.load(path, weights_only=False)
    other = build_squares64(payload["config"]).eval()
    other.load_state_dict(payload["model_state_dict"], strict=True)
    x = model.encoder.prepare_input(chess.Board(), torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(other(x)["policy_logits"],
                                   model(x, recurrent_unrolls=5)["policy_logits"], rtol=0, atol=0)
    assert model.config.recurrent_unrolls == 3


def test_training_and_eval_end_to_end(tmp_path):
    board = chess.Board()
    rows = []
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        move = chess.Move.from_uci(uci)
        rows.append(board_to_cache_row(board, move))
        board.push(move)
    data = attach_static_targets(stack_rows(rows))
    ev = {k: v[:1] for k, v in data.items()}
    train, removed = exclude_overlap(data, ev)
    assert removed == 1 and len(train["turn"]) == 3
    model = tiny()
    before = model.bank[0].attn.out_proj.weight.detach().clone()
    args = SimpleNamespace(seed=283, lr=1e-4, batch_size=2, save_every=3, arm="variable")
    train_arm(model, train, [2, 3, 4], args, tmp_path)
    assert not torch.equal(before, model.bank[0].attn.out_proj.weight)
    assert (tmp_path / "latest.pt").exists()
    report = sweep(model, ev, [3, 4], torch.device("cpu"), 1)
    assert len(report) == 2 and report[0]["rescued_vs_3"] == 0
    assert all(0 <= r["legal_top1"] <= 1 for r in report)
