#!/usr/bin/env python3
"""Shape, backward, and init-policy tests for the ~270M squares64 width scale."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import torch

from chess_squares64 import (
    DEFAULT_100M_SQUARES64_CONFIG,
    DEFAULT_270M_SQUARES64_CONFIG,
    EXPECTED_270M_PARAMS,
    Squares64RecurrentConfig,
    average_recurrent_grads,
    build_squares64,
    count_parameters,
)
from exp270_squares64_pretrain import refuse_incumbent_resume
from move_vocab import VOCAB_SIZE


def test_270m_width_compat_and_param_count():
    cfg = DEFAULT_270M_SQUARES64_CONFIG
    assert cfg.hidden_dim == 1216
    assert cfg.num_heads == 16
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert cfg.hidden_dim // cfg.num_heads == 76
    assert cfg.prefix_layers == DEFAULT_100M_SQUARES64_CONFIG.prefix_layers == 4
    assert cfg.recurrent_layers == DEFAULT_100M_SQUARES64_CONFIG.recurrent_layers == 7
    assert cfg.recurrent_unrolls == DEFAULT_100M_SQUARES64_CONFIG.recurrent_unrolls == 3
    assert cfg.suffix_layers == DEFAULT_100M_SQUARES64_CONFIG.suffix_layers == 4
    assert cfg.effective_depth == 29
    cfg.validate()
    model = build_squares64(cfg)
    n = count_parameters(model)
    assert n == EXPECTED_270M_PARAMS, (n, EXPECTED_270M_PARAMS)
    assert VOCAB_SIZE == 1968


def test_invalid_head_count_rejected():
    try:
        Squares64RecurrentConfig(hidden_dim=1216, num_heads=15).validate()
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_270m_forward_backward_and_recurrent_grad_avg():
    torch.manual_seed(0)
    cfg = DEFAULT_270M_SQUARES64_CONFIG
    model = build_squares64(cfg)
    model.train()
    x = {
        "fused_ids": torch.randint(0, 13, (1, 64)),
        "turn": torch.zeros(1, dtype=torch.long),
        "castling": torch.zeros(1, dtype=torch.long),
        "ep_file": torch.zeros(1, dtype=torch.long),
    }
    out = model(x)
    assert out["square_hidden"].shape == (1, 64, 1216)
    assert out["policy_logits"].shape == (1, VOCAB_SIZE)
    assert out["value_logits"].shape == (1, 3)
    loss = out["policy_logits"].float().pow(2).mean()
    loss.backward()
    before = [p.grad.detach().clone() for p in model.recurrent_parameters() if p.grad is not None]
    assert before
    average_recurrent_grads(model)
    after = [p.grad.detach() for p in model.recurrent_parameters() if p.grad is not None]
    checked = 0
    for a, b in zip(after, before):
        if float(b.abs().mean()) < 1e-8:
            continue
        assert torch.allclose(a, b / 3, rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked >= 1


def test_fresh_init_refuses_incumbent_optimizer():
    try:
        refuse_incumbent_resume(ROOT / "outputs/sf19_ft/overnight_20260908/latest.pt")
    except SystemExit as exc:
        assert "fresh init" in str(exc).lower() or "refusing" in str(exc).lower()
    else:
        raise AssertionError("expected SystemExit")
    refuse_incumbent_resume(ROOT / "outputs/exp270_squares64_pretrain/latest.pt")
