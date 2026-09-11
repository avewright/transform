#!/usr/bin/env python3
"""CPU tests for 99M-logit soft targets and Hinton KD."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import torch.nn.functional as F

from autoresearch_8gb.pipeline import (
    logits_to_soft_targets,
    teacher_logits_kd_loss,
    teacher_wdl_kl_loss,
)


def test_logits_to_soft_targets_renorm_and_hard():
    logits = torch.tensor([
        [1.0, 3.0, 0.0, -2.0],
        [0.2, 0.1, 0.3, 0.0],
    ])
    hard, idx, prob = logits_to_soft_targets(logits, k=2, temperature=1.0)
    assert hard.tolist() == [1, 2]
    assert idx.shape == (2, 2)
    assert torch.allclose(prob.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert idx[0, 0].item() == 1
    assert idx[1, 0].item() == 2
    assert (prob > 0).all()


def test_teacher_logits_kd_self_is_zero():
    torch.manual_seed(0)
    z = torch.randn(4, 16)
    loss = teacher_logits_kd_loss(z, z, temperature=2.0)
    assert float(loss) < 1e-6


def test_teacher_logits_kd_pulls_toward_teacher():
    torch.manual_seed(1)
    teacher = torch.zeros(2, 8)
    teacher[:, 3] = 5.0
    student = torch.randn(2, 8, requires_grad=True)
    loss = teacher_logits_kd_loss(student, teacher.detach(), temperature=2.0)
    loss.backward()
    assert student.grad is not None
    # Student mass on the teacher mode should want to increase (grad negative).
    assert float(student.grad[:, 3].mean()) < 0


def test_teacher_wdl_kl_self_is_zero():
    wdl = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert float(teacher_wdl_kl_loss(wdl, wdl)) < 1e-6
    p = F.softmax(wdl.float(), dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-6)
