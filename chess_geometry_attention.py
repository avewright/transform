"""64-square QK-normalized attention with optional dynamic GAB and 2D Shaw.

Base projection names match QKNormMultiheadAttention for strict warm starts.
Shaw uses displacement-tied Q/K/V vectors (225 file/rank offsets), not a
scalar positional bias. All added output contributions initialize to zero.
"""
import torch
from torch import nn
from torch.nn import functional as F
from chess_transformer_factory import QKNormMultiheadAttention, GeometricAttentionBias


class GeometryAttention(QKNormMultiheadAttention):
    def __init__(self, d_model, nhead, dropout=0., *, mode="gab", d1=16, d2=64, d3=32):
        super().__init__(d_model, nhead, dropout)
        if mode not in {"gab", "shaw", "both"}:
            raise ValueError(f"Unknown geometry mode: {mode}")
        self.gab = (GeometricAttentionBias(d_model, nhead, d1, d2, d3, n_ctx=0)
                    if mode in {"gab", "both"} else None)
        self.use_shaw = mode in {"shaw", "both"}
        if self.use_shaw:
            sq = torch.arange(64)
            rank, file = sq // 8, sq % 8
            rel = (rank[:, None] - rank[None, :] + 7) * 15 + file[:, None] - file[None, :] + 7
            self.register_buffer("relative_ids", rel, persistent=False)
            for name in ("shaw_q", "shaw_k", "shaw_v"):
                emb = nn.Embedding(225, d_model)
                nn.init.zeros_(emb.weight)
                setattr(self, name, emb)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if query.shape[1] != 64 or key.shape[1] != 64 or value.shape[1] != 64:
            raise ValueError("GeometryAttention requires 64 square tokens")
        b = query.shape[0]
        q = self.q_norm(self._shape(self.q_proj(query)))
        k = self.k_norm(self._shape(self.k_proj(key)))
        v = self._shape(self.v_proj(value))
        bias = self.gab(query) if self.gab is not None else None
        if attn_mask is not None:
            extra = attn_mask.reshape(b, self.nhead, 64, 64)
            bias = extra if bias is None else bias + extra
        if not self.use_shaw:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                dropout_p=self.attn_dropout.p if self.training else 0., scale=self.scale)
        else:
            def relative(embedding):
                return embedding(self.relative_ids).view(64, 64, self.nhead, self.head_dim).permute(2, 0, 1, 3)
            aq, ak, av = (relative(x) for x in (self.shaw_q, self.shaw_k, self.shaw_v))
            scores = q @ k.transpose(-2, -1)
            scores = scores + torch.einsum("bhid,hijd->bhij", q, ak)
            scores = scores + torch.einsum("hijd,bhjd->bhij", aq, k)
            scores = (scores + (aq * ak).sum(-1).unsqueeze(0)) * self.scale
            if bias is not None:
                scores = scores + bias
            weights = self.attn_dropout(scores.float().softmax(-1).to(v.dtype))
            out = weights @ v + torch.einsum("bhij,hijd->bhid", weights, av)
        out = out.transpose(1, 2).contiguous().view(b, 64, self.d_model)
        return self.proj_dropout(self.out_proj(out)), None
