"""Minimal encoder-only transformer blocks with RoPE self-attention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the standard RoPE quarter-turn on the last dimension."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Caches sin/cos tables for rotary position embedding."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dimension must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype=dtype)
            sin = emb.sin().to(dtype=dtype)
            self._cos_cached = cos.unsqueeze(0).unsqueeze(0)
            self._sin_cached = sin.unsqueeze(0).unsqueeze(0)
            self._seq_len_cached = seq_len
        return (
            self._cos_cached[:, :, :seq_len, :],
            self._sin_cached[:, :, :seq_len, :],
        )


def apply_rope(x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
    """Apply rotary embedding to (B, heads, seq, head_dim)."""
    _, _, seq_len, _ = x.shape
    cos, sin = rope.get_cos_sin(seq_len, x.device, x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class RopeSelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding on q/k."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope)
        k = apply_rope(k, self.rope)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.out_dropout(self.out_proj(attn))


class RopeEncoderLayer(nn.Module):
    """Pre-norm encoder block with RoPE attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = RopeSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RopeEncoder(nn.Module):
    """Stack of encoder blocks with rotary self-attention."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [RopeEncoderLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
