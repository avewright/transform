"""Value99 trunk + ChessBot encoder / QK policy head.

Input is ChessBot's 19-plane board (token 0 = a8). Tokens are rank-aligned to
Value99's a1=0 relative geometry before the 24-layer trunk. Policy logits are
the ChessBot 1929-way map (1858 LC0 ChessFENS slots + suffix).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import chess
import torch
from torch import nn
from torch.nn import functional as F

from chess_chessbot import (
    CHESSBOT_POLICY,
    CHESSBOT_VOCAB_SIZE,
    AbsolutePositionalEncoder,
    MaGating,
    boards_to_planes,
    legal_policy_mask,
    policy_index_to_move,
)
from chess_value99 import Block, ValueConfig


@dataclass
class PolicyConfig(ValueConfig):
    vocab_size: int = CHESSBOT_VOCAB_SIZE


def chessbot_planes_to_a1(planes: torch.Tensor) -> torch.Tensor:
    """ChessBot (B, 64, 19) a8=0 → Value99 square order a1=0."""
    return planes.reshape(planes.size(0), 8, 8, planes.size(-1)).flip(-3).reshape(planes.size(0), 64, planes.size(-1))


class Value99Policy(nn.Module):
    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        c = config or PolicyConfig()
        if c.width % c.heads:
            raise ValueError('Width must divide head count')
        if c.vocab_size != CHESSBOT_VOCAB_SIZE:
            raise ValueError(f'vocab_size must be {CHESSBOT_VOCAB_SIZE}')
        self.config = c
        self.linear1 = nn.Linear(19, c.width)
        self.layernorm1 = nn.LayerNorm(c.width)
        self.ma_gating = MaGating(c.width)
        self.square = nn.Embedding(64, c.width)
        self.input_norm = nn.LayerNorm(c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.square_value = nn.Linear(c.width, c.value_square)
        self.decoder = nn.Sequential(
            nn.Linear(64 * c.value_square, c.value_hidden), nn.GELU(), nn.Linear(c.value_hidden, 1),
        )
        self.positional = AbsolutePositionalEncoder(c.width)
        self.policy_tokens_lin = nn.Linear(c.width, c.width)
        self.queries_pol = nn.Linear(c.width, c.width)
        self.keys_pol = nn.Linear(c.width, c.width)
        self.policy_head = nn.Linear(64 * 64, c.vocab_size, bias=False)
        sq = torch.arange(64)
        rank, file = sq // 8, sq % 8
        index = (rank[:, None] - rank[None, :] + 7) * 15 + (file[:, None] - file[None, :] + 7)
        self.register_buffer('relative_index', index, persistent=False)
        for block in self.blocks:
            with torch.no_grad():
                block.out.weight.div_((2 * c.layers) ** 0.5)
                block.down.weight.div_((2 * c.layers) ** 0.5)
        nn.init.normal_(self.decoder[-1].weight, std=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        if planes.dim() == 4:
            planes = planes.view(planes.size(0), 64, 19)
        x = chessbot_planes_to_a1(planes)
        x = self.ma_gating(self.layernorm1(F.gelu(self.linear1(x))))
        x = self.input_norm(x + self.square.weight[None])
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, self.relative_index, use_reentrant=False)
            else:
                x = block(x, self.relative_index)
        return self.norm(x)

    def policy_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        tokens = F.gelu(self.policy_tokens_lin(hidden)) + self.positional(hidden)
        scores = torch.matmul(self.queries_pol(tokens), self.keys_pol(tokens).transpose(-2, -1))
        scores = scores / math.sqrt(self.config.width)
        return self.policy_head(scores.reshape(hidden.size(0), 64 * 64))

    def forward(self, planes: torch.Tensor, return_logit: bool = False) -> dict[str, torch.Tensor]:
        hidden = self.encode(planes)
        value_logit = self.decoder(F.gelu(self.square_value(hidden)).flatten(1))[:, 0].float()
        value = value_logit if return_logit else 2 * torch.sigmoid(value_logit) - 1
        return {
            'policy_logits': self.policy_logits(hidden),
            'value': value,
            'value_logit': value_logit,
            'square_hidden': hidden,
        }

    def prepare_input(self, board: chess.Board, device: torch.device) -> torch.Tensor:
        return boards_to_planes([board], device)

    @torch.no_grad()
    def select_move(self, board: chess.Board, device: torch.device, temperature: float = 0.0):
        out = self(self.prepare_input(board, device))
        logits = out['policy_logits'][0].float()
        mask = legal_policy_mask(board, device)
        if not bool(mask.any()):
            raise ValueError(f'No ChessBot-vocab legal moves on {board.fen()}')
        logits = logits.masked_fill(~mask, -1e9)
        if temperature <= 0:
            idx = int(logits.argmax())
        else:
            idx = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        return policy_index_to_move(idx, board), {
            'source': 'value99_policy',
            'index': idx,
            'uci': CHESSBOT_POLICY[idx],
        }


def trunk_key(name: str) -> bool:
    return name.startswith('blocks.') or name.startswith('decoder.') or name in {
        'square.weight', 'norm.weight', 'norm.bias',
        'square_value.weight', 'square_value.bias',
        'input_norm.weight', 'input_norm.bias',
    }


def load_value99_trunk(model: Value99Policy, ckpt: dict) -> dict:
    """Copy matching Value99 trunk tensors. Encoder and policy stay fresh."""
    src = ckpt.get('model') or ckpt
    own = model.state_dict()
    mapped = {}
    skipped = []
    for key, tensor in src.items():
        name = key[10:] if key.startswith('_orig_mod.') else key
        if not trunk_key(name):
            skipped.append(name)
            continue
        if name not in own or own[name].shape != tensor.shape:
            skipped.append(name)
            continue
        mapped[name] = tensor
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return {
        'loaded': sorted(mapped),
        'missing': list(missing),
        'unexpected': list(unexpected),
        'skipped': skipped,
    }


def build_value99_policy(config: PolicyConfig | dict | None = None) -> Value99Policy:
    if config is None:
        cfg = PolicyConfig()
    elif isinstance(config, PolicyConfig):
        cfg = config
    else:
        known = {f.name for f in PolicyConfig.__dataclass_fields__.values()}
        cfg = PolicyConfig(**{k: v for k, v in config.items() if k in known})
    return Value99Policy(cfg)
