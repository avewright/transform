"""64-square recurrent chess transformer.

Design
------
- Attention is strictly over the 64 board squares (64×64). No castling / STM /
  EP tokens in the sequence.
- Side information (turn, castling rights, en passant) is applied as FiLM
  affine transforms on the square stream.
- Piece identity is a fused embedding table: empty + (color × piece) so
  white-queen and black-queen are distinct learned vectors.
- Trunk is recurrent:
    prefix  (layers 1–4): unique, run once
    bank    (blocks 5–11): 7 weight-tied blocks, unrolled ``recurrent_unrolls``
             times (default 3) → 21 effective layers
    suffix  (last 4): unique, run once
  Effective depth = 4 + 7*3 + 4 = 29. Unique layer modules = 15.

Recurrent gradient averaging
----------------------------
Autograd already *sums* grads across the 3 unrolls of each bank block. Call
``average_recurrent_grads(model)`` after ``loss.backward()`` and before
``optimizer.step()`` to divide bank-parameter *gradients* by
``recurrent_unrolls``. This does **not** divide Polar-NorMuon (or RMS-Adam)
updates by the same factor; those updates are approximately scale-invariant.
It does shrink the pre-clip global grad norm. Use param-group LRs if you
want different update sizes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_features import (
    NUM_CASTLING_STATES,
    NUM_EP_STATES,
    NUM_FUSED_TOKENS,
    batch_boards_to_fused_token_ids,
)
from chess_transformer_factory import (
    ChessTransformerEncoderLayer,
    SpatialPolicyHead,
    SwiGLUFFN,
)
from move_vocab import VOCAB_SIZE


@dataclass(frozen=True)
class Squares64RecurrentConfig:
    encoder_dim: int = 256
    hidden_dim: int = 736
    num_heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.05
    use_swiglu: bool = True
    use_qk_norm: bool = True
    zero_init_out_proj: bool = True
    gradient_checkpointing: bool = False

    # 4 unique + 7 recurrent × 3 unrolls + 4 unique = 29 effective depth
    prefix_layers: int = 4
    recurrent_layers: int = 7  # "blocks 5–11"
    recurrent_unrolls: int = 3
    suffix_layers: int = 4

    policy_head_dim: int = 384
    value_hidden: int = 384
    n_value_classes: int = 3  # WDL aux

    @property
    def unique_layers(self) -> int:
        return self.prefix_layers + self.recurrent_layers + self.suffix_layers

    @property
    def effective_depth(self) -> int:
        return (
            self.prefix_layers
            + self.recurrent_layers * self.recurrent_unrolls
            + self.suffix_layers
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "Squares64RecurrentConfig":
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", data)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in model.items() if k in known})


# ~100M params @ compact vocab (verify with count_parameters).
# 15 unique layers @ 736d → ~99M including embeds/heads.
DEFAULT_100M_SQUARES64_CONFIG = Squares64RecurrentConfig(
    hidden_dim=736,
    num_heads=8,
)


class Squares64Encoder(nn.Module):
    """Fused piece-color + square embeds; turn/castling/EP as FiLM transforms.

    Output: (B, 64, embed_dim) — squares only.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.piece_color_embed = nn.Embedding(NUM_FUSED_TOKENS, embed_dim)
        self.square_embed = nn.Embedding(64, embed_dim)

        # Each context factor → (γ, β) broadcast over squares.
        self.turn_film = nn.Embedding(2, 2 * embed_dim)
        self.castling_film = nn.Embedding(NUM_CASTLING_STATES, 2 * embed_dim)
        self.ep_film = nn.Embedding(NUM_EP_STATES, 2 * embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.zeros_(self.turn_film.weight)
        nn.init.zeros_(self.castling_film.weight)
        nn.init.zeros_(self.ep_film.weight)

    def _apply_film(self, h: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        # h: (B, 64, D), film: (B, 2D) → γ,β with residual scale (1+γ)
        d = h.shape[-1]
        gamma, beta = film[:, :d], film[:, d:]
        return h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_ids = token_ids["fused_ids"]  # (B, 64)
        bsz = fused_ids.shape[0]
        sq_idx = torch.arange(64, device=fused_ids.device)
        h = self.piece_color_embed(fused_ids) + self.square_embed(sq_idx)

        h = self._apply_film(h, self.turn_film(token_ids["turn"]))
        h = self._apply_film(h, self.castling_film(token_ids["castling"]))
        h = self._apply_film(h, self.ep_film(token_ids["ep_file"]))
        return self.norm(h)

    def prepare_input(self, board: chess.Board, device: torch.device):
        return batch_boards_to_fused_token_ids([board], device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        return batch_boards_to_fused_token_ids(boards, device)


class Squares64RecurrentTransformer(nn.Module):
    """Prefix → recurrent bank (×N) → suffix; attention only on 64 squares."""

    def __init__(self, config: Squares64RecurrentConfig = DEFAULT_100M_SQUARES64_CONFIG):
        super().__init__()
        self.config = config
        self.encoder = Squares64Encoder(config.encoder_dim)
        self.input_proj = nn.Linear(config.encoder_dim, config.hidden_dim)

        ffn_dim = config.hidden_dim * config.ffn_ratio
        def _layer() -> ChessTransformerEncoderLayer:
            return ChessTransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                ffn_dim=ffn_dim,
                dropout=config.dropout,
                use_swiglu=config.use_swiglu,
                full_dim_attention=False,
                use_qk_norm=config.use_qk_norm,
                zero_init_out_proj=config.zero_init_out_proj,
            )

        self.prefix = nn.ModuleList([_layer() for _ in range(config.prefix_layers)])
        self.bank = nn.ModuleList([_layer() for _ in range(config.recurrent_layers)])
        self.suffix = nn.ModuleList([_layer() for _ in range(config.suffix_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)

        # n_ctx=0: policy reads all 64 tokens as squares.
        self.policy_head = SpatialPolicyHead(
            config.hidden_dim,
            n_ctx_tokens=0,
            head_dim=config.policy_head_dim,
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.value_hidden),
            nn.GELU(),
            nn.Linear(config.value_hidden, config.n_value_classes),
        )

    def _run_block(self, layer: nn.Module, h: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(layer, h, None, use_reentrant=False)
        return layer(h, attn_bias=None)

    def forward(self, board_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.input_proj(self.encoder(board_input))  # (B, 64, D)

        for layer in self.prefix:
            h = self._run_block(layer, h)

        for _ in range(self.config.recurrent_unrolls):
            for layer in self.bank:
                h = self._run_block(layer, h)

        for layer in self.suffix:
            h = self._run_block(layer, h)

        h = self.norm(h)
        global_h = h.mean(dim=1)  # no CLS token — mean pool over 64 squares
        return {
            "policy_logits": self.policy_head(h, global_h),
            "value_logits": self.value_head(global_h),
            "square_hidden": h,
            "global_hidden": global_h,
        }

    def recurrent_parameters(self):
        """Parameters belonging to the weight-tied recurrent bank."""
        return self.bank.parameters()


def average_recurrent_grads(
    model: Squares64RecurrentTransformer,
    unrolls: int | None = None,
) -> None:
    """Divide recurrent-bank grads by unroll count (call after backward).

    Autograd sums the three bank passes. Averaging shrinks ``.grad`` and the
    global clip norm; Polar-NorMuon / RMS-Adam update *sizes* stay about the
    same because those updates are scale-normalized.
    """
    n = unrolls if unrolls is not None else model.config.recurrent_unrolls
    if n <= 1:
        return
    scale = 1.0 / float(n)
    for p in model.recurrent_parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_squares64(
    config: Squares64RecurrentConfig | dict | None = None,
) -> Squares64RecurrentTransformer:
    if config is None:
        cfg = DEFAULT_100M_SQUARES64_CONFIG
    elif isinstance(config, Squares64RecurrentConfig):
        cfg = config
    else:
        known = {f.name for f in Squares64RecurrentConfig.__dataclass_fields__.values()}
        cfg = Squares64RecurrentConfig(**{k: v for k, v in config.items() if k in known})
    return Squares64RecurrentTransformer(cfg)
