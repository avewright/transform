import os
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import json
from pathlib import Path
from types import SimpleNamespace
import chess
import pytest
import torch
from chess_squares64 import Squares64RecurrentConfig, build_squares64, upgrade_with_geometry
from chess_inference import load_checkpoint
from experiments.exp283_recurrent_depth import save_model, train_arm
from autoresearch_8gb.pipeline import board_to_cache_row, stack_rows, attach_static_targets


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(284)
    return build_squares64(Squares64RecurrentConfig(encoder_dim=16, hidden_dim=32,
        num_heads=4, prefix_layers=1, recurrent_layers=1, suffix_layers=1,
        policy_head_dim=16, value_hidden=16, dropout=0, zero_init_out_proj=False)).eval()


@pytest.mark.parametrize("mode", ["gab", "shaw", "both"])
@pytest.mark.parametrize("scope", ["bank", "all"])
def test_upgrade_gradients_and_inference_reload(mode, scope, tmp_path):
    original = tiny()
    model = upgrade_with_geometry(original, geometry_attention=mode, geometry_scope=scope,
                                  gab_d1=4, gab_d2=8, gab_d3=4)
    x = model.encoder.prepare_input(chess.Board(), torch.device("cpu"))
    with torch.no_grad():
        for depth in (2, 3, 4):
            a = original(x, recurrent_unrolls=depth)
            b = model(x, recurrent_unrolls=depth)
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6)
    for key, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    model.train()
    model(x, recurrent_unrolls=4)["policy_logits"].square().mean().backward()
    attn = model.bank[0].attn
    if mode in {"gab", "both"}:
        assert attn.gab.templates.weight.grad.abs().sum() > 0
    if mode in {"shaw", "both"}:
        for name in ("shaw_q", "shaw_k", "shaw_v"):
            assert getattr(attn, name).weight.grad.abs().sum() > 0
        ids = attn.relative_ids
        assert ids.min() == 0 and ids.max() == 224
        assert ids[0, 1] == ids[8, 9] and ids[0, 1] != ids[1, 0]
    save_model(model, tmp_path / "model.pt", {})
    restored = load_checkpoint(tmp_path / "model.pt", device="cpu")
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x)["policy_logits"], restored(x)["policy_logits"])


def test_geometry_training_updates_and_checkpointing(tmp_path):
    from dataclasses import replace
    base = tiny()
    base.config = replace(base.config, gradient_checkpointing=True)
    model = upgrade_with_geometry(base, geometry_attention="both", gab_d1=4, gab_d2=8, gab_d3=4)
    board = chess.Board()
    data = attach_static_targets(stack_rows([board_to_cache_row(board, chess.Move.from_uci("e2e4"))]))
    args = SimpleNamespace(seed=284, lr=1e-5, geometry_lr=1e-3, batch_size=1,
                          save_every=3, arm="variable", warmup=1, cosine_decay=True)
    train_arm(model, data, [2, 3, 4], args, tmp_path)
    assert model.bank[0].attn.gab.templates.weight.abs().sum() > 0
    assert model.bank[0].attn.shaw_v.weight.abs().sum() > 0
    assert (tmp_path / "latest.pt").exists()


def test_config_valid_and_unknown_mode_rejected():
    from experiments.exp284_recurrent_geometry import validate
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/exp284_recurrent_geometry_99m.json").read_text())
    validate(cfg)
    with pytest.raises(ValueError):
        upgrade_with_geometry(tiny(), geometry_attention="typo")
