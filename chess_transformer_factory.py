from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chess_model import FusedBoardEncoder, ChessRelativeBias, StrengthenedBoardEncoder
from move_vocab import IDX_TO_UCI, VOCAB_SIZE


@dataclass(frozen=True)
class ChessTransformerConfig:
    encoder_dim: int = 256
    encoder_type: str = "fused"  # "fused" | "strengthened"
    encoder_conv_blocks: int = 2
    normalize_stm: bool = False
    hidden_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    ffn_ratio: int = 4
    dropout: float = 0.1
    policy_head_dim: int = 512
    value_hidden: int = 512
    use_pos_embed: bool = True
    n_ctx_tokens: int = 4
    value_head_type: str = "cls"  # "cls" (CLS only) or "pool" (CLS + mean pool)
    n_value_classes: int = 3     # 3 for WDL, 128 for distributional HL-Gauss
    use_swiglu: bool = False       # SwiGLU gated FFN (Llama/Ruoss-style)
    use_rel_bias: bool = False     # Chess-aware relative geometry attention bias
    gradient_checkpointing: bool = False
    # If True, each attention head keeps full hidden_dim (no d/H shrink).
    # Cheap here because the board sequence is only ~68 tokens.
    full_dim_attention: bool = False

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
# Deep-narrow ~700M for cloud GPUs (A100+). Do not train on 8GB laptops.
DEFAULT_700M_CONFIG = ChessTransformerConfig(
    encoder_dim=384,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=960,
    num_layers=63,
    num_heads=12,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=512,
    value_hidden=512,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",
    n_value_classes=128,
    use_swiglu=True,
    use_rel_bias=True,
    gradient_checkpointing=True,
)

# Deep-narrow ~309M for 8GB laptop: 96 thin layers, strengthened encoder, Muon-friendly.
DEFAULT_8GB_CONFIG = ChessTransformerConfig(
    encoder_dim=384,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=512,
    num_layers=96,
    num_heads=8,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=384,
    value_hidden=384,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",
    n_value_classes=128,
    use_swiglu=True,
    use_rel_bias=True,
    gradient_checkpointing=True,
)

# A100 80GB: 705M with grad ckpt (fits bs=64); 309M without ckpt (fast path).
DEFAULT_A100_700M_CONFIG = DEFAULT_700M_CONFIG
DEFAULT_A100_309M_CONFIG = replace(DEFAULT_8GB_CONFIG, gradient_checkpointing=False)

# A40 48GB: shallower/wider than deep-narrow 16L/96L stacks.
# Big piece embeds (768), full-dim multi-head attn (no d/H shrink), no grad ckpt.
# ~8L/1152d/8H-full ≈ 380M — leaves headroom for large batches on 45GB.
DEFAULT_A40_WIDE_CONFIG = ChessTransformerConfig(
    encoder_dim=768,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=1152,
    num_layers=8,
    num_heads=8,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=576,
    value_hidden=256,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",
    n_value_classes=3,
    use_swiglu=True,
    use_rel_bias=True,
    gradient_checkpointing=False,
    full_dim_attention=True,
)

# A40 fast path: deep residual stack that finishes ~10-15M positions in 3-4h.
# Many layers + standard multi-head attn + SwiGLU + chess rel-bias; no grad ckpt.
# ~28L/256d/8H ≈ 25M — large batches (~1024) → ~1k pos/s on A40.
DEFAULT_A40_DEEP_SMALL_CONFIG = ChessTransformerConfig(
    encoder_dim=256,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=256,
    num_layers=28,
    num_heads=8,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=192,
    value_hidden=128,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",
    n_value_classes=3,
    use_swiglu=True,
    use_rel_bias=True,
    gradient_checkpointing=False,
    full_dim_attention=False,
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


# ── SwiGLU FFN (validated in exp169/exp170) ──
class SwiGLUFFN(nn.Module):
    """Gated FFN: SiLU(Wg·x) * Wu·x → Wd.
    Inner dim is 2/3 of ffn_dim to match parameter count with standard FFN."""
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        inner = int(ffn_dim * 2 / 3)
        self.w_gate = nn.Linear(d_model, inner)
        self.w_up = nn.Linear(d_model, inner)
        self.w_down = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class FullDimMultiheadAttention(nn.Module):
    """Multi-head attention where each head keeps the full model dimension.

    Standard MHA shrinks to d_model // n_heads per head. With only ~68 board
    tokens that shrink wastes capacity; here each head projects Q/K/V at
    d_model and outputs are concatenated then projected back to d_model.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if nhead < 1:
            raise ValueError("nhead must be >= 1")
        self.d_model = d_model
        self.nhead = nhead
        self.scale = d_model ** -0.5
        self.q_proj = nn.Linear(d_model, d_model * nhead, bias=False)
        self.k_proj = nn.Linear(d_model, d_model * nhead, bias=False)
        self.v_proj = nn.Linear(d_model, d_model * nhead, bias=False)
        self.out_proj = nn.Linear(d_model * nhead, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        del need_weights  # API compat with nn.MultiheadAttention
        B, T, _ = query.shape
        H, D = self.nhead, self.d_model

        q = self.q_proj(query).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(key).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(value).view(B, T, H, D).transpose(1, 2)

        # Prefer fused SDPA when no additive bias is present.
        if attn_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # attn_mask: (B*H, T, T) float additive bias from ChessRelativeBias
            if attn_mask.dim() == 3:
                bias = attn_mask.view(B, H, T, T)
            else:
                bias = attn_mask
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores + bias
            attn = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.dropout(self.out_proj(out)), None


# ── Custom encoder layer supporting SwiGLU + attention bias ──
class ChessTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with optional SwiGLU FFN and attention bias."""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        full_dim_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if full_dim_attention:
            self.attn = FullDimMultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = (
            SwiGLUFFN(d_model, ffn_dim, dropout) if use_swiglu
            else nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(dropout),
            )
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_bias)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


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


class PooledValueHead(nn.Module):
    """Value head that reads CLS + mean-pooled square tokens for richer input."""
    def __init__(self, hidden_dim: int, value_hidden: int = 512, n_ctx_tokens: int = 4):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        # Input is CLS (hidden_dim) + mean pool of 64 squares (hidden_dim) = 2*hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, value_hidden // 2),
            nn.ReLU(),
            nn.Linear(value_hidden // 2, 3),
        )

    def forward(self, hidden: torch.Tensor, cls_hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, 1+ctx+64, hidden_dim), cls_hidden: (B, hidden_dim)
        sq_hidden = hidden[:, self.n_ctx:self.n_ctx + 64, :]  # (B, 64, hidden_dim)
        pool = sq_hidden.mean(dim=1)  # (B, hidden_dim)
        combined = torch.cat([cls_hidden, pool], dim=-1)  # (B, 2*hidden_dim)
        return self.mlp(combined)


class ChessTransformer(nn.Module):
    def __init__(self, config: ChessTransformerConfig):
        super().__init__()
        self.config = config
        if config.encoder_type == "strengthened":
            self.encoder = StrengthenedBoardEncoder(
                embed_dim=config.encoder_dim,
                conv_blocks=config.encoder_conv_blocks,
                normalize_stm=config.normalize_stm,
            )
        else:
            self.encoder = FusedBoardEncoder(embed_dim=config.encoder_dim)
        self.input_proj = nn.Linear(config.encoder_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.pos_embed = (
            nn.Parameter(torch.randn(1, 68, config.hidden_dim) * 0.02)
            if config.use_pos_embed
            else None
        )

        # Relative bias (chess geometry)
        self.rel_bias = (
            ChessRelativeBias(config.num_heads, n_ctx=config.n_ctx_tokens)
            if config.use_rel_bias else None
        )

        # Encoder layers: custom (SwiGLU/bias/full-dim) or standard
        ffn_dim = config.hidden_dim * config.ffn_ratio
        use_custom = config.use_swiglu or config.use_rel_bias or config.full_dim_attention
        if use_custom:
            self.layers = nn.ModuleList([
                ChessTransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    ffn_dim=ffn_dim,
                    dropout=config.dropout,
                    use_swiglu=config.use_swiglu,
                    full_dim_attention=config.full_dim_attention,
                )
                for _ in range(config.num_layers)
            ])
            self.transformer = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=ffn_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.layers = None

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.policy_head = SpatialPolicyHead(
            config.hidden_dim,
            n_ctx_tokens=config.n_ctx_tokens,
            head_dim=config.policy_head_dim,
        )
        if config.value_head_type == "pool":
            self.value_head = PooledValueHead(
                config.hidden_dim, config.value_hidden, config.n_ctx_tokens
            )
            self._pool_value = True
        else:
            self.value_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.value_hidden),
                nn.ReLU(),
                nn.Linear(config.value_hidden, config.n_value_classes),
            )
            self._pool_value = False

    def forward(self, board_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.input_proj(self.encoder(board_input))
        batch_size = hidden.shape[0]
        hidden = torch.cat([self.cls_token.expand(batch_size, -1, -1), hidden], dim=1)
        if self.pos_embed is not None:
            hidden = hidden + self.pos_embed

        if self.layers is not None:
            # Custom layers path (SwiGLU and/or RelBias)
            attn_bias = None
            if self.rel_bias is not None:
                bias = self.rel_bias()  # (H, seq, seq)
                # Expand to (B*H, seq, seq) for nn.MultiheadAttention
                nhead = self.config.num_heads
                attn_bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
                    batch_size * nhead, hidden.shape[1], hidden.shape[1]
                )
            for layer in self.layers:
                if self.config.gradient_checkpointing and self.training:
                    hidden = checkpoint(
                        layer, hidden, attn_bias,
                        use_reentrant=False,
                    )
                else:
                    hidden = layer(hidden, attn_bias=attn_bias)
        else:
            hidden = self.transformer(hidden)

        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        value_logits = (
            self.value_head(hidden, cls_hidden) if self._pool_value
            else self.value_head(cls_hidden)
        )
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": value_logits,
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
