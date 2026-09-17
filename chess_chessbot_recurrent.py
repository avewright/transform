"""Recurrent wrap of published Maxlegrec/ChessBot (10×512).

Split: prefix 0–1 / bank 2–7 × N / suffix 8–9.
N=1 is the published net. Extra unrolls reuse the bank; parameter count stays
~34.7M. Do not add SwiGLU, QK-norm, Polar, or new randomly initialized layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_chessbot import (
    CHESSBOT_POLICY,
    CHESSBOT_VOCAB_SIZE,
    count_parameters,
    fens_to_planes,
    legal_policy_mask,
    policy_index_to_move,
)


REPO = "Maxlegrec/ChessBot"
ARCH = "chessbot34_recurrent"
PREFIX, BANK, SUFFIX = 2, 6, 2
PUBLISHED_LAYERS = PREFIX + BANK + SUFFIX


@dataclass(frozen=True)
class RecurrentSplit:
    prefix: int = PREFIX
    bank: int = BANK
    suffix: int = SUFFIX

    def validate(self, num_layers: int) -> None:
        if min(self.prefix, self.bank, self.suffix) < 1:
            raise ValueError("prefix, bank, and suffix must be >= 1")
        if self.prefix + self.bank + self.suffix != num_layers:
            raise ValueError(
                f"split {self.prefix}+{self.bank}+{self.suffix} != {num_layers} layers"
            )


def load_published_chessbot(device: torch.device, repo: str = REPO):
    """Card load: config + safetensors, not broken from_pretrained finalize."""
    import sys
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    local = snapshot_download(repo)
    AutoConfig.from_pretrained(local, trust_remote_code=True)
    mod = next(
        m for n, m in sys.modules.items()
        if n.endswith("modeling_chessbot") and hasattr(m, "ChessBotModel")
    )
    if not hasattr(mod.ChessBotModel, "all_tied_weights_keys"):
        mod.ChessBotModel.all_tied_weights_keys = {}
    model = mod.ChessBotModel(AutoConfig.from_pretrained(local, trust_remote_code=True))
    missing, unexpected = model.load_state_dict(
        load_file(str(Path(local) / "model.safetensors")), strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(f"ChessBot weight mismatch missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


def build_empty_chessbot(num_layers: int = 10, d_model: int = 512, d_ff: int = 1024, num_heads: int = 8):
    import sys
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    local = snapshot_download(REPO)
    AutoConfig.from_pretrained(local, trust_remote_code=True)
    mod = next(
        m for n, m in sys.modules.items()
        if n.endswith("modeling_chessbot") and hasattr(m, "ChessBotModel")
    )
    if not hasattr(mod.ChessBotModel, "all_tied_weights_keys"):
        mod.ChessBotModel.all_tied_weights_keys = {}
    cfg = mod.ChessBotConfig(
        num_layers=num_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads,
        vocab_size=CHESSBOT_VOCAB_SIZE,
    )
    return mod.ChessBotModel(cfg)


def planes_to_chessbot_input(planes: torch.Tensor) -> torch.Tensor:
    """(B, 64, 19) or (B, 8, 8, 19) → (B, 1, 8, 8, 19)."""
    if planes.ndim == 5:
        return planes
    if planes.ndim == 4:
        return planes.unsqueeze(1)
    if planes.ndim != 3 or planes.shape[-2:] != (64, 19):
        raise ValueError(f"expected (B, 64, 19), got {tuple(planes.shape)}")
    b = planes.size(0)
    return planes.reshape(b, 8, 8, 19).unsqueeze(1)


class RecurrentChessBot(nn.Module):
    """Published ChessBot with a runtime unroll count on the middle bank."""

    def __init__(self, base, default_unrolls: int = 2, split: RecurrentSplit | None = None):
        super().__init__()
        self.base = base
        self.default_unrolls = int(default_unrolls)
        self.split = split or RecurrentSplit()
        self.split.validate(int(base.num_layers))
        if self.default_unrolls < 1:
            raise ValueError("default_unrolls must be >= 1")

    @property
    def config(self):
        return self.base.config

    def effective_depth(self, unrolls: int | None = None) -> int:
        n = self.default_unrolls if unrolls is None else int(unrolls)
        return self.split.prefix + self.split.bank * n + self.split.suffix

    def recurrent_parameters(self):
        start = self.split.prefix
        end = start + self.split.bank
        for layer in self.base.layers[start:end]:
            yield from layer.parameters()

    def _encode(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        x = planes_to_chessbot_input(planes)
        b, seq_len, _, _, emb = x.size()
        x = x.reshape(b * seq_len, 64, emb)
        x = self.base.linear1(x)
        x = F.gelu(x)
        x = self.base.layernorm1(x)
        x = self.base.ma_gating(x)
        return x, self.base.positional(x), b, seq_len

    def _heads(self, x: torch.Tensor, pos_enc: torch.Tensor, b: int, seq_len: int) -> dict[str, torch.Tensor]:
        value_h = self.base.value_head(x).view(b, seq_len, 3)
        value_h_q = self.base.value_head_q(x).view(b, seq_len, 3)
        tokens = F.gelu(self.base.policy_tokens_lin(x)) + pos_enc
        queries = self.base.queries_pol(tokens)
        keys = self.base.keys_pol(tokens)
        attn = torch.matmul(queries, keys.transpose(-2, -1))
        attn = attn / torch.sqrt(torch.tensor(self.base.d_model, dtype=torch.float32, device=x.device))
        policy = self.base.policy_head(attn.view(b, seq_len, 64 * 64))
        if seq_len == 1:
            policy = policy[:, 0]
            value_h = value_h[:, 0]
            value_h_q = value_h_q[:, 0]
        return {
            "policy_logits": policy,
            "value_logits": value_h,
            "value_logits_q": value_h_q,
        }

    def forward(self, planes: torch.Tensor, recurrent_unrolls: int | None = None) -> dict[str, torch.Tensor]:
        n = self.default_unrolls if recurrent_unrolls is None else recurrent_unrolls
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("recurrent_unrolls must be a positive integer")
        x, pos_enc, b, seq_len = self._encode(planes)
        p, k, s = self.split.prefix, self.split.bank, self.split.suffix
        for layer in self.base.layers[:p]:
            x = layer(x, pos_enc)
        for _ in range(n):
            for layer in self.base.layers[p:p + k]:
                x = layer(x, pos_enc)
        for layer in self.base.layers[p + k:p + k + s]:
            x = layer(x, pos_enc)
        return self._heads(x, pos_enc, b, seq_len)

    def published_forward(self, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Untouched published loop (all 10 layers once)."""
        x, pos_enc, b, seq_len = self._encode(planes)
        for layer in self.base.layers:
            x = layer(x, pos_enc)
        return self._heads(x, pos_enc, b, seq_len)

    def select_move(self, board: chess.Board, device: torch.device, temperature: float = 0.0, unrolls: int | None = None):
        planes = fens_to_planes([board.fen()], device)
        out = self(planes, recurrent_unrolls=unrolls)
        logits = out["policy_logits"][0].float()
        mask = legal_policy_mask(board, device)
        if not bool(mask.any()):
            raise ValueError(f"No ChessBot-vocab legal moves on {board.fen()}")
        logits = logits.masked_fill(~mask, -1e9)
        if temperature <= 0:
            idx = int(logits.argmax())
        else:
            idx = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        move = policy_index_to_move(idx, board)
        wdl = F.softmax(out["value_logits_q"][0].float(), dim=-1)
        return move, {
            "source": "chessbot34_recurrent",
            "unrolls": self.default_unrolls if unrolls is None else unrolls,
            "temperature": float(temperature),
            "wdl": {"black": wdl[0].item(), "draw": wdl[1].item(), "white": wdl[2].item()},
        }


def wrap_published(device: torch.device, default_unrolls: int = 2, repo: str = REPO) -> RecurrentChessBot:
    return RecurrentChessBot(load_published_chessbot(device, repo), default_unrolls=default_unrolls)


def average_recurrent_grads(model: RecurrentChessBot, unrolls: int) -> None:
    if unrolls <= 1:
        return
    scale = 1.0 / float(unrolls)
    for p in model.recurrent_parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def identity_errors(student: RecurrentChessBot, planes: torch.Tensor) -> dict[str, float]:
    student.eval()
    with torch.no_grad():
        a = student(planes, recurrent_unrolls=1)
        b = student.published_forward(planes)
    return {
        name: float((a[name] - b[name]).abs().max())
        for name in ("policy_logits", "value_logits", "value_logits_q")
    }


def is_recurrent_chessbot_ckpt(ckpt: dict) -> bool:
    return ckpt.get("arch") == ARCH
