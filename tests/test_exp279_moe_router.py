"""CPU tests for the frozen-expert MoE router."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import torch

from chess_moe import (
    EXPERT_NAMES,
    N_EXPERTS,
    ExpertRouter,
    FrozenExpertMoE,
    best_expert_from_ce,
    expected_ce_loss,
    freeze_expert,
    is_moe_router_ckpt,
    phase_label_to_expert,
    phase_prior,
    robust_best_expert,
    router_param_count,
    soft_expert_targets,
    soft_target_loss,
)
import torch.nn as nn


def test_five_experts_named():
    assert EXPERT_NAMES == ["incumbent", "puzzle", "endgame", "opening", "middlegame"]
    assert N_EXPERTS == 5


def test_phase_prior_buckets():
    n = torch.tensor([32, 26, 20, 14, 10, 6], dtype=torch.int16)
    p = phase_prior(n)
    assert p.tolist() == [3, 3, 4, 4, 2, 2]
    assert phase_label_to_expert(torch.tensor([0, 1, 2])).tolist() == [3, 4, 2]


def test_router_forward_and_size():
    r = ExpertRouter()
    assert router_param_count() < 1_000_000
    bsz = 4
    logits = r(torch.randn(bsz, r.hidden_dim))
    assert logits.shape == (bsz, 5)
    for p in freeze_expert(r).parameters():
        assert not p.requires_grad


def test_best_expert_and_soft():
    ce = torch.tensor([[2.0, 1.0, 3.0, 4.0, 5.0], [0.1, 2.0, 2.0, 2.0, 2.0]])
    assert best_expert_from_ce(ce).tolist() == [1, 0]
    soft = soft_expert_targets(ce, tau=0.4)
    assert soft.shape == (2, 5)
    assert torch.allclose(soft.sum(-1), torch.ones(2), atol=1e-5)
    assert int(soft[0].argmax()) == 1


def test_soft_target_loss_prefers_low_ce_expert():
    ce = torch.tensor([[5.0, 0.2, 5.0, 5.0, 5.0]])
    good = torch.tensor([[0.0, 4.0, 0.0, 0.0, 0.0]])
    bad = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0]])
    assert float(soft_target_loss(good, ce, tau=0.4)) < float(soft_target_loss(bad, ce, tau=0.4))


def test_label_source_slices_general():
    sys.path[:0] = [str(ROOT / "experiments")]
    from exp279_moe_router import (
        attach_source_route_targets,
        label_source_slices,
        source_row_labels,
        subset_label_rows,
    )

    sl = label_source_slices(61504)
    tr, va = sl["general"]
    assert (tr.start, tr.stop) == (43456, 51456)
    assert (va.start, va.stop) == (51456, 52480)
    data = {
        "move_idx": torch.arange(61504),
        "board_array": torch.zeros(61504, 64, dtype=torch.int8),
    }
    sub = subset_label_rows(data, "general")
    assert int(sub["move_idx"].shape[0]) == 9024
    assert int(sub["move_idx"][0]) == 43456
    spec = subset_label_rows(data, "specialists")
    assert int(spec["move_idx"].shape[0]) == 61504 - 9024
    src, expert = source_row_labels(61504)
    assert src[43456].item() == 3  # general
    assert expert[52480].item() == 1  # puzzles → puzzle
    tagged = attach_source_route_targets(data)
    no_gen = subset_label_rows(tagged, "specialists")
    assert int((no_gen["source_id"] == 3).sum()) == 0


def test_robust_best_requires_puzzle_margin():
    thin = torch.tensor([[2.0, 1.0, 1.2, 3.0, 3.0]])
    assert robust_best_expert(thin, puzzle_margin=0.5).tolist() == [2]
    clear = torch.tensor([[2.0, 1.0, 1.9, 3.0, 3.0]])
    assert robust_best_expert(clear, puzzle_margin=0.5).tolist() == [1]
    incumbent = torch.tensor([[1.0, 1.05, 2.0, 2.0, 2.0]])
    assert robust_best_expert(incumbent, puzzle_margin=0.5).tolist() == [0]


def test_expected_ce_is_weighted_mean():
    gates = torch.tensor([[40.0, 0.0, 0.0, 0.0, 0.0]])
    ce = torch.tensor([[1.5, 3.0, 4.0, 5.0, 6.0]])
    loss = expected_ce_loss(gates, ce)
    assert abs(float(loss) - 1.5) < 1e-5
    mix = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    assert abs(float(expected_ce_loss(mix, ce)) - float(ce.mean())) < 1e-5


class _StubExpert(nn.Module):
    def __init__(self, eid: int):
        super().__init__()
        self.eid = eid
        self.bias = nn.Parameter(torch.tensor(float(eid)))

    def forward(self, board_input):
        b = board_input["fused_ids"].shape[0]
        hidden = torch.zeros(b, 736)
        hidden[:, 0] = board_input["fused_ids"][:, 0].float()
        return {
            "policy_logits": torch.zeros(b, 8) + self.bias,
            "value_logits": torch.zeros(b, 3) + self.bias,
            "global_hidden": hidden,
        }


def test_moe_dispatches_to_chosen_expert():
    class _FixedRouter(nn.Module):
        def forward(self, global_hidden):
            b = global_hidden.shape[0]
            logits = torch.full((b, 5), -10.0)
            pick = (global_hidden[:, 0] > 0).long()  # 0 or 1
            logits[torch.arange(b), pick] = 10.0
            return logits

    experts = [_StubExpert(i) for i in range(5)]
    moe = FrozenExpertMoE(_FixedRouter(), experts)
    bi = {
        "fused_ids": torch.tensor([[0] * 64, [2] * 64]),
        "turn": torch.zeros(2, dtype=torch.long),
        "castling": torch.zeros(2, dtype=torch.long),
        "ep_file": torch.zeros(2, dtype=torch.long),
    }
    out = moe(bi)
    assert out.expert_id.tolist() == [0, 1]
    assert abs(float(out.policy_logits[0, 0]) - 0.0) < 1e-5
    assert abs(float(out.policy_logits[1, 0]) - 1.0) < 1e-5
    assert out.expert_name == "mixed"
    assert out["policy_logits"].shape == out.policy_logits.shape
    assert out["expert_name"] == "mixed"
    assert out.stem_policy_logits is not None


def test_unfrozen_moe_trains_routed_expert():
    class _FixedRouter(nn.Module):
        def forward(self, global_hidden):
            b = global_hidden.shape[0]
            logits = torch.full((b, 5), -10.0)
            logits[:, 1] = 10.0
            return logits

    experts = [_StubExpert(i) for i in range(5)]
    moe = FrozenExpertMoE(_FixedRouter(), experts, freeze=False)
    bi = {
        "fused_ids": torch.ones(2, 64),
        "turn": torch.zeros(2, dtype=torch.long),
        "castling": torch.zeros(2, dtype=torch.long),
        "ep_file": torch.zeros(2, dtype=torch.long),
    }
    out = moe(bi)
    out.policy_logits.sum().backward()
    assert experts[1].bias.grad is not None
    assert float(experts[1].bias.grad) != 0.0


def test_is_moe_router_ckpt():
    assert is_moe_router_ckpt({"arch": "frozen_moe_router", "model_state_dict": {}})
    assert is_moe_router_ckpt({"experts": [], "model_state_dict": {"mlp.1.weight": 0}})
    assert not is_moe_router_ckpt({"model_state_dict": {"encoder.weight": 0}})
    assert not is_moe_router_ckpt({"experts": []})


def test_expected_ce_pulls_mass_to_best_expert():
    logits = torch.zeros(1, 5, requires_grad=True)
    ce = torch.tensor([[5.0, 0.2, 5.0, 5.0, 5.0]])
    expected_ce_loss(logits, ce).backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 1]) < 0
    assert float(logits.grad[0, 1]) == min(logits.grad[0].tolist())


def test_attach_onehot_is_hard_label():
    sys.path[:0] = [str(ROOT / "experiments")]
    from exp279_moe_router import attach_onehot

    n = 3
    data = attach_onehot(
        {
            "board_array": torch.zeros(n, 64, dtype=torch.int8),
            "turn": torch.zeros(n, dtype=torch.int8),
            "castling": torch.zeros(n, dtype=torch.int8),
            "ep_square": torch.full((n,), -1, dtype=torch.int8),
            "move_idx": torch.tensor([4, 7, 11], dtype=torch.int64),
        }
    )
    assert data["soft_indices"].shape == (n, 8)
    assert data["soft_probs"][:, 0].tolist() == [1.0, 1.0, 1.0]
    assert data["soft_indices"][:, 0].tolist() == [4, 7, 11]


def test_parquet_roundtrip(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path[:0] = [str(ROOT / "experiments")]
    from exp279_moe_router import parquet_to_rows, _split_holdout

    n = 6
    table = pa.table(
        {
            "board_array": [[i] * 64 for i in range(n)],
            "turn": np.zeros(n, dtype=np.int8),
            "castling": np.zeros(n, dtype=np.int8),
            "ep_square": np.full(n, -1, dtype=np.int8),
            "move_idx": np.arange(n, dtype=np.int64),
        }
    )
    dest = tmp_path / "rows.parquet"
    pq.write_table(table, dest)
    rows = parquet_to_rows(dest)
    assert rows["board_array"].shape == (n, 64)
    assert rows["move_idx"].tolist() == list(range(n))
    train, val = _split_holdout(rows, 2, torch.Generator().manual_seed(0))
    assert int(val["move_idx"].shape[0]) == 2
    assert int(train["move_idx"].shape[0]) == 4


def test_sample_lichess_inbox(tmp_path):
    sys.path[:0] = [str(ROOT / "experiments")]
    from exp279_moe_router import sample_lichess_inbox

    inbox = tmp_path / "inbox"
    for i in range(3):
        sh = inbox / f"shard_{i:06d}"
        sh.mkdir(parents=True)
        n = 8
        torch.save(
            {
                "board_array": torch.zeros(n, 64, dtype=torch.int8),
                "turn": torch.zeros(n, dtype=torch.int8),
                "castling": torch.zeros(n, dtype=torch.int8),
                "ep_square": torch.full((n,), -1, dtype=torch.int8),
                "move_idx": torch.full((n,), i, dtype=torch.int64),
            },
            sh / "soft_cache.pt",
        )
        (sh / "READY").write_text("ok\n")
    rng = torch.Generator().manual_seed(0)
    out = sample_lichess_inbox(inbox, 10, rng)
    assert out is not None
    assert int(out["move_idx"].shape[0]) == 10
    assert sample_lichess_inbox(tmp_path / "empty", 10, rng) is None
