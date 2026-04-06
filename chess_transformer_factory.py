from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_model import FusedBoardEncoder
from move_vocab import IDX_TO_UCI, VOCAB_SIZE


@dataclass(frozen=True)
class ChessTransformerConfig:
    encoder_dim: int = 256
    hidden_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    ffn_ratio: int = 4
    dropout: float = 0.1
    policy_head_dim: int = 512
    value_hidden: int = 512
    use_pos_embed: bool = True
    n_ctx_tokens: int = 4

    @classmethod
    def from_json(cls, path: str | Path) -> "ChessTransformerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_data = data.get("model", data)
        return cls(**model_data)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_200M_CONFIG = ChessTransformerConfig()
CONFIG_16L_P256_NO_POS = ChessTransformerConfig(
    num_layers=16,
    policy_head_dim=256,
    use_pos_embed=False,
)


def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for i in range(VOCAB_SIZE):
        uci = IDX_TO_UCI[i]
        from_sqs.append(chess.parse_square(uci[:2]))
        to_sqs.append(chess.parse_square(uci[2:4]))
        promo_types.append(promo_map.get(uci[4:5], 0))
    return (
        torch.tensor(from_sqs, dtype=torch.long),
        torch.tensor(to_sqs, dtype=torch.long),
        torch.tensor(promo_types, dtype=torch.long),
    )


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size: int, n_ctx_tokens: int = 4, head_dim: int = 512):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

    def forward(self, hidden_states: torch.Tensor, cls_hidden: torch.Tensor) -> torch.Tensor:
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        # Project all 64 squares first (cheap), then gather per-move (smaller dim)
        all_from = self.from_proj(sq_hidden)              # (B, 64, head_dim)
        all_to = self.to_proj(sq_hidden)                  # (B, 64, head_dim)
        from_feats = all_from[:, self.from_sqs, :]        # (B, V, head_dim)
        to_feats = all_to[:, self.to_sqs, :]              # (B, V, head_dim)
        combined = (
            from_feats * to_feats
            + self.global_proj(cls_hidden).unsqueeze(1)
            + self.promo_embed(self.promo_types).unsqueeze(0)
        )
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformer(nn.Module):
    def __init__(self, config: ChessTransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = FusedBoardEncoder(embed_dim=config.encoder_dim)
        self.input_proj = nn.Linear(config.encoder_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.pos_embed = (
            nn.Parameter(torch.randn(1, 68, config.hidden_dim) * 0.02)
            if config.use_pos_embed
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * config.ffn_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.policy_head = SpatialPolicyHead(
            config.hidden_dim,
            n_ctx_tokens=config.n_ctx_tokens,
            head_dim=config.policy_head_dim,
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 3),
        )

    def forward(self, board_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.input_proj(self.encoder(board_input))
        batch_size = hidden.shape[0]
        hidden = torch.cat([self.cls_token.expand(batch_size, -1, -1), hidden], dim=1)
        if self.pos_embed is not None:
            hidden = hidden + self.pos_embed
        hidden = self.norm(self.transformer(hidden))
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
            "cls_hidden": cls_hidden,
        }


def build_model(config: ChessTransformerConfig | dict | str | Path | None = None) -> ChessTransformer:
    if config is None:
        resolved = DEFAULT_200M_CONFIG
    elif isinstance(config, ChessTransformerConfig):
        resolved = config
    elif isinstance(config, (str, Path)):
        resolved = ChessTransformerConfig.from_json(config)
    else:
        resolved = ChessTransformerConfig(**config)
    return ChessTransformer(resolved)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
