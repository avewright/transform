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
    # DeBERTa-style factored attention: content×position streams (4 logit terms).
    use_meta_attention: bool = False
    # Original piece×square dual: Q/K/V on BOTH streams, cross + self, update both.
    # (DeBERTa meta only puts V on content and never residual-updates position.)
    use_piece_square_dual: bool = False
    # Shaw relative vectors on the position↔position term only (Δfile, Δrank buckets).
    use_shaw_on_pos: bool = True
    # Modded-NanoGPT: RMSNorm on Q and K before attention scores.
    use_qk_norm: bool = False
    # Modded-NanoGPT / muP-like: zero-init attention out_proj and FFN down-projection.
    zero_init_out_proj: bool = False
    # Chessformer-2026 Geometric Attention Bias: board-conditioned additive attn bias.
    use_gab: bool = False
    gab_d1: int = 16
    gab_d2: int = 64
    gab_d3: int = 8
    # O(N) kernelized linear attention over the ~68 board tokens for cheap
    # piece-interaction (Max Elo/FLOP with a small model).
    use_linear_attention: bool = False
    la_use_qk_norm: bool = True
    la_n_heads: int = 4
    la_feature_dim: int | None = None
    # O(N) linear meta attention over the content/position factored streams.
    # Performer-style kernel on each of the 4 terms (cc/ss/cs/sc); values from
    # content. Keeps the meta geometry inductive bias at linear cost.
    use_linear_meta_attention: bool = False
    meta_kernel_dim: int | None = None
    # Value head that runs a lightweight "search transformer": refine top-k
    # latent children with cross+joint attention, then back up the best-child
    # scalar into the value output (Stockfish-style search-informed eval).
    use_search_value_head: bool = False
    value_topk: int = 8
    value_search_steps: int = 3
    # Neural one-ply search POLICY head: refine top-k latent children and
    # scatter step deltas into the base logits. Elo picks the argmax of the
    # refined policy_logits, so this directly sharpens the move (the
    # "Stockfish-style internal search" policy).
    use_search_policy_head: bool = False
    policy_topk: int = 16
    policy_search_steps: int = 3

    @classmethod
    def from_json(cls, path: str | Path) -> "ChessTransformerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_data = data.get("model", data)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in model_data.items() if k in known})

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

# ≥400M meta-factored attention: content×position streams + Shaw on ss only.
# No handcrafted rel-bias, no absolute seq PE, no full-dim attention.
# Train from scratch (incompatible with 200M checkpoints).
# A40 default: grad checkpoint OFF for throughput; enable via --grad-checkpoint if OOM.
DEFAULT_400M_META_CONFIG = ChessTransformerConfig(
    encoder_dim=512,
    encoder_type="strengthened",
    encoder_conv_blocks=2,
    normalize_stm=True,
    hidden_dim=1280,
    num_layers=18,
    num_heads=20,
    ffn_ratio=4,
    dropout=0.05,
    policy_head_dim=512,
    value_hidden=512,
    use_pos_embed=False,
    n_ctx_tokens=4,
    value_head_type="cls",
    n_value_classes=3,
    use_swiglu=True,
    use_rel_bias=False,
    gradient_checkpointing=False,
    full_dim_attention=False,
    use_meta_attention=True,
    use_shaw_on_pos=True,
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


class LinearAttentionMultihead(nn.Module):
    """O(N) kernelized (Performer-style) multi-head attention.

    Rewrites the N^2 softmax as q @ sum(phi(k)^T v) / sum(phi(k)) using a
    positive feature map (relu+1). Gives piece-interaction over the ~68 board
    tokens at linear cost — good Elo/FLOP for a small model.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0,
                 use_qk_norm: bool = True, feature_dim: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = max(1, d_model // nhead)
        self.feature_dim = feature_dim or self.head_dim
        self.use_qk_norm = use_qk_norm
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        self.scale = self.feature_dim ** -0.25

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + 1.0

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = self._feature_map(q) * self.scale
        k = self._feature_map(k) * self.scale
        kv = torch.einsum('bhtd,bhse->bhde', k, v)  # sum_t phi(k)^T v
        denom = k.sum(dim=2)  # (B,H,Dh)
        out = torch.einsum('bhqd,bhde->bhqe', q, kv) / (denom[:, :, None, :] + 1e-6)
        out = out.transpose(1, 2).reshape(B, T, self.nhead * self.head_dim)
        return self.dropout(self.out_proj(out))


class LinearMetaFactoredAttention(nn.Module):
    """O(N) linear version of the 4-term content×position meta attention.

    The quadratic softmax meta score ``score_ij = cc + ss + cs + sc`` is
    approximated with a positive (Performer-style) feature map ``phi`` so each
    term becomes a rank-structured bilinear form:

        cc = phi(qc_i) · phi(kc_j)          # content ↔ content
        ss = phi(qp_i) · phi(kp_j)          # position ↔ position
        cs = phi(qc_i) · phi(kp_j)          # content query, position key
        sc = phi(qp_i) · phi(kc_j)          # position query, content key

    Outputs stay on the **content** stream (values come from content), matching
    MetaFactoredAttention. Every term is a pure dot product of a query feature
    with a key feature, so we only need two per-head KV moment matrices (one
    keyed by content, one by position — both against content values).

        out_i = phi(qc_i)·KV_c + phi(qp_i)·KV_p + phi(qc_i)·KV_p + phi(qp_i)·KV_c

    Scaling each of the four terms by its own row-sum denominator keeps the
    q/k gradient healthy: sharing one denominator across both query streams
    makes the query-direction gradient cancel algebraically at init. Each term
    reduces to the classic Performer form ``(q·KV)/(q·Ks)``.

    A small learned per-head absolute square bias is added to the position keys
    to give the model cheap absolute geometry (Shaw deltas are inherently
    quadratic and thus omitted here).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        n_ctx: int = 4,
        kernel_dim: int | None = None,
        use_qk_norm: bool = True,
        use_square_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = max(1, d_model // nhead)
        self.kernel_dim = kernel_dim or self.head_dim
        self.n_ctx = n_ctx
        self.use_qk_norm = use_qk_norm
        self.use_square_bias = use_square_bias

        self.q_c = nn.Linear(d_model, d_model, bias=False)
        self.k_c = nn.Linear(d_model, d_model, bias=False)
        self.q_p = nn.Linear(d_model, d_model, bias=False)
        self.k_p = nn.Linear(d_model, d_model, bias=False)
        self.v_c = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        self.scale = self.kernel_dim ** -0.25
        if use_square_bias:
            # Per-head additive offset on the position keys of the 64 squares =>
            # cheap absolute geometry that keeps its inductive strength.
            self.square_bias = nn.Parameter(torch.zeros(nhead, 64))
        else:
            self.register_parameter("square_bias", None)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + 1.0

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, content: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        B, T, _ = content.shape
        H, Dh = self.nhead, self.head_dim
        c = self.n_ctx  # squares start after CLS + ctx tokens

        qc = self._split(self.q_c(content))
        kc = self._split(self.k_c(content))
        qp = self._split(self.q_p(position))
        kp = self._split(self.k_p(position))
        vc = self._split(self.v_c(content))
        if self.use_qk_norm:
            qc, kc = self.q_norm(qc), self.k_norm(kc)
            qp, kp = self.q_norm(qp), self.k_norm(kp)

        qc = self._feature_map(qc) * self.scale
        kc = self._feature_map(kc) * self.scale
        qp = self._feature_map(qp) * self.scale
        kp = self._feature_map(kp) * self.scale

        if self.square_bias is not None:
            # Add per-head absolute square bias to the position keys of squares.
            kp[:, :, c:c + 64] = kp[:, :, c:c + 64] + self.square_bias[None, :, :, None]

        # Two KV moment matrices against content values:
        # KV_c = sum_j outer(phi(kc_j), v_j),  KV_p = sum_j outer(phi(kp_j), v_j)
        kv_c = torch.einsum("bhtd,bhse->bhde", kc, vc)
        kv_p = torch.einsum("bhtd,bhse->bhde", kp, vc)
        kc_sum = kc.sum(dim=2)  # (B,H,Dh)
        kp_sum = kp.sum(dim=2)

        # Each factored term is normalized by its own row-sum (Performer-style
        # generalized-Katharopoulos) so the four meta terms keep independent,
        # well-conditioned q/k gradients. ``(q·KV)/(q·Ks)`` is the healthy linear
        # attention form: sharing ONE denominator across the two query streams
        # makes the q-direction gradient cancel algebraically (only the value
        # moves), so we normalize per term and average instead.
        cc = torch.einsum("bhqd,bhde->bhqe", qc, kv_c) / (
            torch.einsum("bhqd,bhd->bhq", qc, kc_sum).unsqueeze(-1) + 1e-6
        )
        ss = torch.einsum("bhqd,bhde->bhqe", qp, kv_p) / (
            torch.einsum("bhqd,bhd->bhq", qp, kp_sum).unsqueeze(-1) + 1e-6
        )
        cs = torch.einsum("bhqd,bhde->bhqe", qc, kv_p) / (
            torch.einsum("bhqd,bhd->bhq", qc, kp_sum).unsqueeze(-1) + 1e-6
        )
        sc = torch.einsum("bhqd,bhde->bhqe", qp, kv_c) / (
            torch.einsum("bhqd,bhd->bhq", qp, kc_sum).unsqueeze(-1) + 1e-6
        )

        out = (cc + ss + cs + sc) / 4.0
        out = out.transpose(1, 2).reshape(B, T, H * Dh)
        return self.dropout(self.out_proj(out))


class LinearMetaEncoderLayer(nn.Module):
    """Pre-norm block: linear meta attention over content/position + SwiGLU FFN.

    Mirrors MetaFactoredEncoderLayer but uses the O(N) LinearMetaFactoredAttention
    so the meta geometry carries through a deeper/cheaper trunk. Only the content
    stream is carried forward residually; position stays an absolute anchor.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        n_ctx: int = 4,
        kernel_dim: int | None = None,
        use_qk_norm: bool = True,
        use_square_bias: bool = True,
    ):
        super().__init__()
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        self.attn = LinearMetaFactoredAttention(
            d_model, nhead, dropout=dropout, n_ctx=n_ctx,
            kernel_dim=kernel_dim, use_qk_norm=use_qk_norm,
            use_square_bias=use_square_bias,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = (
            SwiGLUFFN(d_model, ffn_dim, dropout)
            if use_swiglu
            else nn.Sequential(
                nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
            )
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, content: torch.Tensor, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nc, np_ = self.norm_c(content), self.norm_p(position)
        content = content + self.dropout(self.attn(nc, np_))
        content = content + self.ffn(self.norm_ff(content))
        return content, position


class QKNormMultiheadAttention(nn.Module):
    """Standard MHA (d_model // nhead per head) with RMSNorm on Q and K.

    Ported idea from modded-nanogpt: stabilize attention logits without changing
    board topology inductive bias.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        del need_weights
        B, T, _ = query.shape
        q = self.q_norm(self._shape(self.q_proj(query)))
        k = self.k_norm(self._shape(self.k_proj(key)))
        v = self._shape(self.v_proj(value))

        if attn_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            if attn_mask.dim() == 3:
                bias = attn_mask.view(B, self.nhead, T, T)
            else:
                bias = attn_mask
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias
            attn = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.proj_dropout(self.out_proj(out)), None


class PieceSquareDualAttention(nn.Module):
    """Piece x square dual attention (original meta idea).

    Streams stay separate: content = piece/what, position = square/where.

    Cross (one stream Q, the other K; V from the keyed stream):
      delta_c = softmax(Q_c @ K_p.T) @ V_p   # piece queries squares
      delta_p = softmax(Q_p @ K_c.T) @ V_c   # square queries pieces

    Then duplicate full QKV on each stream (self):
      delta_c += softmax(Q_c @ K_c.T) @ V_c
      delta_p += softmax(Q_p @ K_p.T) @ V_p

    Unlike DeBERTa meta, both streams have V and both get residual updates.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = False

        self.q_c = nn.Linear(d_model, d_model, bias=False)
        self.k_c = nn.Linear(d_model, d_model, bias=False)
        self.v_c = nn.Linear(d_model, d_model, bias=False)
        self.q_p = nn.Linear(d_model, d_model, bias=False)
        self.k_p = nn.Linear(d_model, d_model, bias=False)
        self.v_p = nn.Linear(d_model, d_model, bias=False)
        self.out_c = nn.Linear(d_model, d_model)
        self.out_p = nn.Linear(d_model, d_model)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        return torch.matmul(attn, v)

    def forward(
        self, content: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qc = self._shape(self.q_c(content))
        kc = self._shape(self.k_c(content))
        vc = self._shape(self.v_c(content))
        qp = self._shape(self.q_p(position))
        kp = self._shape(self.k_p(position))
        vp = self._shape(self.v_p(position))
        if self.use_qk_norm:
            qc, kc = self.q_norm(qc), self.k_norm(kc)
            qp, kp = self.q_norm(qp), self.k_norm(kp)

        # Cross: piece→square and square→piece
        out_c = self._attn(qc, kp, vp) + self._attn(qc, kc, vc)
        out_p = self._attn(qp, kc, vc) + self._attn(qp, kp, vp)

        out_c = self.proj_dropout(self.out_c(self._merge(out_c)))
        out_p = self.proj_dropout(self.out_p(self._merge(out_p)))
        return out_c, out_p


class MetaFactoredAttention(nn.Module):
    """4-term content×position attention + optional Shaw on position↔position.

    score = cc + ss + cs + sc  (add logits; softmax multiplies preferences)
    values come from the content stream only.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        use_shaw: bool = True,
        n_ctx: int = 4,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.n_ctx = n_ctx
        self.use_shaw = use_shaw

        self.use_qk_norm = use_qk_norm
        self.q_c = nn.Linear(d_model, d_model, bias=False)
        self.k_c = nn.Linear(d_model, d_model, bias=False)
        self.v_c = nn.Linear(d_model, d_model, bias=False)
        self.q_p = nn.Linear(d_model, d_model, bias=False)
        self.k_p = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Precompute Δfile/Δrank bucket ids for 64×64 square pairs (15×15).
        rel_ids = torch.zeros(64, 64, dtype=torch.long)
        for i in range(64):
            ri, fi = divmod(i, 8)
            for j in range(64):
                rj, fj = divmod(j, 8)
                rel_ids[i, j] = (ri - rj + 7) * 15 + (fi - fj + 7)
        self.register_buffer("rel_ids", rel_ids, persistent=False)

        if use_shaw:
            n_rel = 15 * 15
            # Per-head Shaw vectors on the position stream.
            self.shaw_aq = nn.Embedding(n_rel, nhead * self.head_dim)
            self.shaw_ak = nn.Embedding(n_rel, nhead * self.head_dim)
            nn.init.zeros_(self.shaw_aq.weight)
            nn.init.zeros_(self.shaw_ak.weight)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, content: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        B, T, _ = content.shape
        H, Dh = self.nhead, self.head_dim
        c = self.n_ctx  # square tokens start after CLS+ctx; encoder has 3 ctx, +CLS => 4

        qc = self._shape(self.q_c(content))
        kc = self._shape(self.k_c(content))
        vc = self._shape(self.v_c(content))
        qp = self._shape(self.q_p(position))
        kp = self._shape(self.k_p(position))
        if self.use_qk_norm:
            qc, kc = self.q_norm(qc), self.k_norm(kc)
            qp, kp = self.q_norm(qp), self.k_norm(kp)

        # Four factored terms (B, H, T, T)
        cc = torch.matmul(qc, kc.transpose(-2, -1))
        ss = torch.matmul(qp, kp.transpose(-2, -1))
        cs = torch.matmul(qc, kp.transpose(-2, -1))
        sc = torch.matmul(qp, kc.transpose(-2, -1))
        scores = cc + ss + cs + sc

        if self.use_shaw:
            # Shaw only on square↔square region (indices c .. c+64).
            # Expanded: (qp + aQ)·(kp + aK) = qp·kp + qp·aK + aQ·kp + aQ·aK
            # We already have qp·kp in ss; add the remaining three on the sq block.
            rid = self.rel_ids  # (64, 64)
            aq = self.shaw_aq(rid).view(64, 64, H, Dh).permute(2, 0, 1, 3)  # (H,64,64,Dh)
            ak = self.shaw_ak(rid).view(64, 64, H, Dh).permute(2, 0, 1, 3)

            qp_sq = qp[:, :, c:c + 64, :]  # (B,H,64,Dh)
            kp_sq = kp[:, :, c:c + 64, :]

            # qp · aK^T  and  aQ · kp^T  via einsum
            # aq/ak: (H,64,64,Dh); qp_sq: (B,H,64,Dh)
            qp_ak = torch.einsum("bhik,hijk->bhij", qp_sq, ak)
            aq_kp = torch.einsum("hijk,bhjk->bhij", aq, kp_sq)
            aq_ak = torch.einsum("hijk,hijk->hij", aq, ak).unsqueeze(0)  # (1,H,64,64)

            scores[:, :, c:c + 64, c:c + 64] = (
                scores[:, :, c:c + 64, c:c + 64] + qp_ak + aq_kp + aq_ak
            )

        scores = scores * self.scale
        attn = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        out = torch.matmul(attn, vc)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.proj_dropout(self.out_proj(out))


def _zero_init_out_projections(attn: nn.Module, ffn: nn.Module) -> None:
    """Zero-init residual branches (modded-nanogpt / muP-like)."""
    if hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
        nn.init.zeros_(attn.out_proj.weight)
        if attn.out_proj.bias is not None:
            nn.init.zeros_(attn.out_proj.bias)
    if isinstance(ffn, SwiGLUFFN):
        nn.init.zeros_(ffn.w_down.weight)
        if ffn.w_down.bias is not None:
            nn.init.zeros_(ffn.w_down.bias)
    elif isinstance(ffn, nn.Sequential):
        for mod in reversed(list(ffn.modules())):
            if isinstance(mod, nn.Linear):
                nn.init.zeros_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
                break


class MetaFactoredEncoderLayer(nn.Module):
    """Pre-norm encoder layer with meta-factored attention + SwiGLU/GELU FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_swiglu: bool = True,
        use_shaw: bool = True,
        n_ctx: int = 4,
        use_qk_norm: bool = False,
        zero_init_out_proj: bool = False,
        use_piece_square_dual: bool = False,
    ):
        super().__init__()
        self.use_piece_square_dual = use_piece_square_dual
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        if use_piece_square_dual:
            self.attn = PieceSquareDualAttention(d_model, nhead, dropout=dropout)
        else:
            self.attn = MetaFactoredAttention(
                d_model, nhead, dropout=dropout, use_shaw=use_shaw, n_ctx=n_ctx,
                use_qk_norm=use_qk_norm,
            )
        self.norm_ff = nn.LayerNorm(d_model)
        self.norm_ff_p = nn.LayerNorm(d_model) if use_piece_square_dual else None
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
        # Light FFN on position stream so geometry can evolve (dual mode only).
        self.ffn_p = (
            (
                SwiGLUFFN(d_model, ffn_dim, dropout) if use_swiglu
                else nn.Sequential(
                    nn.Linear(d_model, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, d_model),
                    nn.Dropout(dropout),
                )
            )
            if use_piece_square_dual else None
        )
        self.dropout = nn.Dropout(dropout)
        if zero_init_out_proj:
            if use_piece_square_dual:
                for proj in (self.attn.out_c, self.attn.out_p):
                    nn.init.zeros_(proj.weight)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)
                _zero_init_out_projections(self.attn, self.ffn)
                if self.ffn_p is not None:
                    _zero_init_out_projections(self.attn, self.ffn_p)
            else:
                _zero_init_out_projections(self.attn, self.ffn)

    def forward(
        self, content: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nc, np_ = self.norm_c(content), self.norm_p(position)
        if self.use_piece_square_dual:
            dc, dp = self.attn(nc, np_)
            content = content + self.dropout(dc)
            position = position + self.dropout(dp)
            content = content + self.ffn(self.norm_ff(content))
            position = position + self.ffn_p(self.norm_ff_p(position))
            return content, position
        attn_out = self.attn(nc, np_)
        content = content + self.dropout(attn_out)
        content = content + self.ffn(self.norm_ff(content))
        return content, position


# ── Custom encoder layer supporting SwiGLU + attention bias ──
class LinearAttentionEncoderLayer(nn.Module):
    """Pre-norm block with O(N) LinearAttention + SwiGLU FFN (small model)."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0,
                 use_swiglu: bool = True, use_qk_norm: bool = True,
                 feature_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttentionMultihead(
            d_model, nhead=nhead, dropout=dropout,
            use_qk_norm=use_qk_norm, feature_dim=feature_dim,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = (
            SwiGLUFFN(d_model, ffn_dim, dropout)
            if use_swiglu
            else nn.Sequential(
                nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
            )
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


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
        use_qk_norm: bool = False,
        zero_init_out_proj: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if full_dim_attention:
            self.attn = FullDimMultiheadAttention(d_model, nhead, dropout=dropout)
        elif use_qk_norm:
            self.attn = QKNormMultiheadAttention(d_model, nhead, dropout=dropout)
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
        if zero_init_out_proj:
            _zero_init_out_projections(self.attn, self.ffn)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_bias)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class GeometricAttentionBias(nn.Module):
    """Board-conditioned attention bias (Chessformer GAB, arXiv:2605.19091).

    Compresses the 64 square tokens into per-head coefficients that mix a shared
    bank of 64×64 bias templates. Applied only on the square↔square block;
    CLS/ctx rows/cols stay zero. Templates start at zero so early training
    matches the no-GAB baseline.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d1: int = 16,
        d2: int = 64,
        d3: int = 8,
        n_ctx: int = 4,
    ):
        super().__init__()
        self.nhead = nhead
        self.n_ctx = n_ctx
        self.d3 = d3
        self.compress = nn.Linear(d_model, d1)
        self.bottleneck = nn.Sequential(
            nn.Linear(64 * d1, d2),
            nn.GELU(),
            nn.LayerNorm(d2),
        )
        self.to_coeffs = nn.Linear(d2, nhead * d3)
        self.templates = nn.Linear(d3, 64 * 64, bias=False)
        nn.init.zeros_(self.templates.weight)

    def forward(self, sq_hidden: torch.Tensor) -> torch.Tensor:
        """Return additive bias (B, H, seq, seq); squares start at index n_ctx."""
        B = sq_hidden.shape[0]
        if sq_hidden.shape[1] != 64:
            raise ValueError(f"GAB expects 64 square tokens, got {sq_hidden.shape[1]}")
        h = self.compress(sq_hidden).reshape(B, -1)
        h = self.bottleneck(h)
        coeffs = self.to_coeffs(h).view(B, self.nhead, self.d3)
        bias64 = self.templates(coeffs).view(B, self.nhead, 64, 64)
        # n_ctx already counts CLS (+ board ctx tokens) before the 64 squares.
        seq = self.n_ctx + 64
        out = sq_hidden.new_zeros(B, self.nhead, seq, seq)
        s = self.n_ctx
        out[:, :, s:s + 64, s:s + 64] = bias64
        return out


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


class SearchPolicyHead(nn.Module):
    """Neural one-ply search policy head (Stockfish-style internal search).

    Elo uses policy **argmax**, so making the policy a refined-search output
    beats one-shot spatial scoring. This is exp064's latent-search policy head:

      Stage 1 — factorized spatial prior over all moves (base_logits).
      Stage 2 — latent child expansion: top-K candidates' child representations
                from the from/to square hidden states ("what the board looks
                like after each move", without re-running the encoder).
      Stage 3 — iterative refinement: candidates cross-attend to the parent
                board, [candidates; children] jointly self-attend, MLP refine,
                per-step delta logits scatter-added to the base.
      Stage 4 — final ``policy_logits`` = the last step's refined logits.

    The step deltas are gated (tanh(step_gate)) so early training matches the
    plain spatial head, then search can sharpen the prior as it learns.
    """

    def __init__(
        self, hidden_size, n_ctx_tokens=4, head_dim=256, topk=16,
        search_steps=3, num_heads=8, dropout=0.0,
    ):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.head_dim = head_dim
        self.topk = topk
        self.search_steps = search_steps
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

        # Stage 1: spatial prior
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.base_score = nn.Linear(head_dim, 1)

        # Stage 2: child expansion
        self.child_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, head_dim), nn.GELU(), nn.Linear(head_dim, head_dim),
        )

        # Stage 3: refinement search transformer
        self.board_k = nn.Linear(hidden_size, head_dim)
        self.board_v = nn.Linear(hidden_size, head_dim)
        self.cross_attn = nn.MultiheadAttention(
            head_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.joint_self_attn = nn.MultiheadAttention(
            head_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.refine_mlp = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim * 4),
            nn.GELU(), nn.Linear(head_dim * 4, head_dim),
        )
        self.delta_score = nn.Linear(head_dim, 1)
        self.step_gate = nn.Parameter(torch.zeros(search_steps))

    def _base_components(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = self.from_proj(sq_hidden)[:, self.from_sqs, :]
        to_feats = self.to_proj(sq_hidden)[:, self.to_sqs, :]
        move_states = (
            from_feats * to_feats
            + self.global_proj(cls_hidden).unsqueeze(1)
            + self.promo_embed(self.promo_types).unsqueeze(0)
        )
        base_logits = self.base_score(F.relu(move_states)).squeeze(-1)
        return move_states, base_logits

    def forward(self, hidden_states, cls_hidden):
        B = hidden_states.shape[0]
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        move_states, base_logits = self._base_components(hidden_states, cls_hidden)

        topk = min(self.topk, move_states.shape[-2])
        cand_idx = base_logits.topk(topk, dim=-1).indices  # (B, K)
        cand_states = torch.gather(
            move_states, 1, cand_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        )

        cand_from_sq = self.from_sqs[cand_idx]
        cand_to_sq = self.to_sqs[cand_idx]
        batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1)
        child_repr = self.child_proj(
            torch.cat([sq_hidden[batch_idx, cand_from_sq], sq_hidden[batch_idx, cand_to_sq]], dim=-1)
        )

        board_kv = (self.board_k(hidden_states), self.board_v(hidden_states))
        step_logits = [base_logits]
        for step in range(self.search_steps):
            cross_out, _ = self.cross_attn(
                cand_states, board_kv[0], board_kv[1], need_weights=False,
            )
            cand_states = cand_states + cross_out
            combined = torch.cat([cand_states, child_repr], dim=1)
            combined_out, _ = self.joint_self_attn(combined, combined, combined, need_weights=False)
            combined = combined + combined_out
            cand_states, child_repr = combined[:, :topk], combined[:, topk:]
            cand_states = cand_states + self.refine_mlp(cand_states)

            delta = self.delta_score(F.gelu(cand_states)).squeeze(-1)
            gated = torch.tanh(self.step_gate[step]) * delta
            step_logits.append(base_logits.scatter_add(1, cand_idx, gated))

        final = step_logits[-1]
        return {"policy_logits": final, "base_policy_logits": base_logits,
                "step_policy_logits": step_logits[1:],
                "candidate_indices": cand_idx}


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


class SearchValueHead(nn.Module):
    """Search-transformer value head (Stockfish-style neural one-ply + backup).

    Instead of reading only CLS → MLP, this head runs a lightweight search
    over the top-k legal-ish candidate children:

      Stage 1 — Coarse spatial prior over all moves (factorized from×to×promo)
                from the trunk squares, producing a per-move base score.
      Stage 2 — For the top-k candidates, build a child representation from the
                from/to square hidden states ("what the board looks like after").
      Stage 3 — Refinement "search": candidates cross-attend to the parent board
                tokens, [candidates; children] jointly self-attend, and an MLP
                refines states (iterate value_search_steps times).
      Stage 4 — A per-candidate child-value MLP scores each refined child; the
                backed-up scalar aggregates them (soft-max weighted) and is
                mixed with the CLS base value into the final WDL logits.

    Returns the usual ``value_logits`` plus ``searched_value`` (backed-up scalar)
    and ``base_value`` (CLS-only prior) for aux losses / analysis.
    """

    def __init__(
        self,
        hidden_dim: int,
        value_hidden: int = 512,
        n_ctx_tokens: int = 4,
        head_dim: int = 256,
        topk: int = 8,
        search_steps: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.topk = topk
        self.search_steps = search_steps
        self.head_dim = head_dim

        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

        # ── Stage 1: coarse spatial prior (shares the trunk square hidden states)
        self.from_proj = nn.Linear(hidden_dim, head_dim)
        self.to_proj = nn.Linear(hidden_dim, head_dim)
        self.global_proj = nn.Linear(hidden_dim, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.base_score = nn.Linear(head_dim, 1)

        # ── Stage 2: child representation from from/to square features
        self.child_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
        )

        # ── Stage 3: refinement search transformer
        self.board_k = nn.Linear(hidden_dim, head_dim)
        self.board_v = nn.Linear(hidden_dim, head_dim)
        # dropout=0 for both MHAs: MPS scaled_dot_product_attention rejects
        # dropout, and the step-gate + LayerNorm already regularize this head.
        self.cross_attn = nn.MultiheadAttention(
            head_dim, num_heads=num_heads, dropout=0.0, batch_first=True,
        )
        self.joint_self_attn = nn.MultiheadAttention(
            head_dim, num_heads=num_heads, dropout=0.0, batch_first=True,
        )
        self.refine_mlp = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim * 4),
            nn.GELU(),
            nn.Linear(head_dim * 4, head_dim),
        )
        self.step_gate = nn.Parameter(torch.zeros(search_steps))

        # ── Stage 4: child value + backup + final WDL
        self.child_value = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.backup_temp = nn.Parameter(torch.tensor(1.0))
        # CLS base value head (raw WDL prior)
        self.base_value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )
        # Fuse CLS prior + search-improved scalar into final WDL logits.
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + 2, 1),  # cls hidden + base scalar + searched scalar
        )

    def forward(
        self, hidden: torch.Tensor, cls_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        B = hidden.shape[0]
        sq_hidden = hidden[:, self.n_ctx:self.n_ctx + 64, :]  # (B, 64, H)

        # Stage 1: coarse spatial prior over all moves.
        all_from = self.from_proj(sq_hidden)              # (B, 64, D)
        all_to = self.to_proj(sq_hidden)                  # (B, 64, D)
        from_feats = all_from[:, self.from_sqs, :]        # (B, V, D)
        to_feats = all_to[:, self.to_sqs, :]              # (B, V, D)
        move_states = (
            from_feats * to_feats
            + self.global_proj(cls_hidden).unsqueeze(1)
            + self.promo_embed(self.promo_types).unsqueeze(0)
        )
        base_scores = self.base_score(F.relu(move_states)).squeeze(-1)  # (B, V)

        # Stage 2: select top-k candidates + build child representations.
        topk = min(self.topk, base_scores.shape[-1])
        cand_idx = base_scores.topk(topk, dim=-1).indices  # (B, K)
        gather_idx = cand_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        cand_states = torch.gather(move_states, 1, gather_idx)  # (B, K, D)

        cand_from_sq = self.from_sqs[cand_idx]  # (B, K)
        cand_to_sq = self.to_sqs[cand_idx]      # (B, K)
        batch_idx = torch.arange(B, device=hidden.device).unsqueeze(1)
        from_sq_feats = sq_hidden[batch_idx, cand_from_sq]  # (B, K, H)
        to_sq_feats = sq_hidden[batch_idx, cand_to_sq]      # (B, K, H)
        child_repr = self.child_proj(
            torch.cat([from_sq_feats, to_sq_feats], dim=-1)
        )  # (B, K, D)

        # Stage 3: refinement search transformer.
        board_kv = (self.board_k(hidden), self.board_v(hidden))
        for step in range(self.search_steps):
            cross_out, _ = self.cross_attn(
                cand_states, board_kv[0], board_kv[1], need_weights=False,
            )
            cand_states = cand_states + cross_out
            combined = torch.cat([cand_states, child_repr], dim=1)  # (B, 2K, D)
            combined_out, _ = self.joint_self_attn(
                combined, combined, combined, need_weights=False,
            )
            combined = combined + combined_out
            cand_states = combined[:, :topk]
            child_repr = combined[:, topk:]
            cand_states = cand_states + torch.tanh(self.step_gate[step]) * self.refine_mlp(
                cand_states
            )

        # Stage 4: per-child value + backup.
        child_val = self.child_value(
            torch.cat([cand_states, child_repr], dim=-1)
        ).squeeze(-1)  # (B, K)
        temp = self.backup_temp.abs() + 0.1
        weights = F.softmax(child_val / temp, dim=-1)
        backed_up = torch.tanh((weights * child_val).sum(dim=-1))  # (B,)

        # CLS base value prior.
        base_scalar = torch.tanh(self.base_value_mlp(cls_hidden).squeeze(-1))  # (B,)

        # Fuse into final WDL logits (3-way from search + prior scalars + CLS ctx).
        fuse_in = torch.cat([cls_hidden, base_scalar.unsqueeze(-1), backed_up.unsqueeze(-1)], dim=-1)
        prior = self.fuse(fuse_in)  # (B, 1)
        # Scale to a centered 3-class logit readout conditioned on the base value
        # direction, so search can flip WHITE/DRAW/BLACK when it sees a refutation.
        base_dir = torch.sign(base_scalar.detach() + 1e-6)
        value_logits = torch.stack([
            prior.squeeze(-1) * base_dir,
            torch.zeros_like(prior.squeeze(-1)),
            -prior.squeeze(-1) * base_dir,
        ], dim=-1)

        return {
            "value_logits": value_logits,
            "base_value": base_scalar,
            "searched_value": backed_up,
            "child_values": child_val,
            "candidate_indices": cand_idx,
        }


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
        self.pos_input_proj = (
            nn.Linear(config.encoder_dim, config.hidden_dim)
            if config.use_meta_attention or config.use_linear_meta_attention else None
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.cls_pos_token = (
            nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
            if config.use_meta_attention or config.use_linear_meta_attention else None
        )
        self.pos_embed = (
            nn.Parameter(torch.randn(1, 68, config.hidden_dim) * 0.02)
            if config.use_pos_embed
            else None
        )

        # Relative bias (chess geometry) — unused when meta attention is on
        self.rel_bias = (
            ChessRelativeBias(config.num_heads, n_ctx=config.n_ctx_tokens)
            if config.use_rel_bias and not config.use_meta_attention
            and not config.use_linear_meta_attention else None
        )
        self.gab = (
            GeometricAttentionBias(
                config.hidden_dim,
                config.num_heads,
                d1=config.gab_d1,
                d2=config.gab_d2,
                d3=config.gab_d3,
                n_ctx=config.n_ctx_tokens,
            )
            if config.use_gab and not config.use_meta_attention
            and not config.use_linear_meta_attention
            else None
        )

        ffn_dim = config.hidden_dim * config.ffn_ratio
        if config.use_linear_attention:
            self.layers = nn.ModuleList([
                LinearAttentionEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    ffn_dim=ffn_dim,
                    dropout=config.dropout,
                    use_swiglu=config.use_swiglu,
                    use_qk_norm=config.la_use_qk_norm,
                    feature_dim=config.la_feature_dim,
                )
                for _ in range(config.num_layers)
            ])
            self.transformer = None
        elif config.use_meta_attention:
            self.layers = nn.ModuleList([
                MetaFactoredEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    ffn_dim=ffn_dim,
                    dropout=config.dropout,
                    use_swiglu=config.use_swiglu,
                    use_shaw=config.use_shaw_on_pos,
                    n_ctx=config.n_ctx_tokens,
                    use_qk_norm=config.use_qk_norm,
                    zero_init_out_proj=config.zero_init_out_proj,
                    use_piece_square_dual=config.use_piece_square_dual,
                )
                for _ in range(config.num_layers)
            ])
            self.transformer = None
        elif config.use_linear_meta_attention:
            self.layers = nn.ModuleList([
                LinearMetaEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    ffn_dim=ffn_dim,
                    dropout=config.dropout,
                    use_swiglu=config.use_swiglu,
                    n_ctx=config.n_ctx_tokens,
                    kernel_dim=config.meta_kernel_dim,
                    use_qk_norm=config.use_qk_norm,
                    use_square_bias=True,
                )
                for _ in range(config.num_layers)
            ])
            self.transformer = None
        else:
            use_custom = (
                config.use_swiglu or config.use_rel_bias or config.full_dim_attention
                or config.use_qk_norm or config.zero_init_out_proj or config.use_gab
            )
            if use_custom:
                self.layers = nn.ModuleList([
                    ChessTransformerEncoderLayer(
                        d_model=config.hidden_dim,
                        nhead=config.num_heads,
                        ffn_dim=ffn_dim,
                        dropout=config.dropout,
                        use_swiglu=config.use_swiglu,
                        full_dim_attention=config.full_dim_attention,
                        use_qk_norm=config.use_qk_norm,
                        zero_init_out_proj=config.zero_init_out_proj,
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
        self.use_search_policy_head = config.use_search_policy_head
        if self.use_search_policy_head:
            self.policy_head = SearchPolicyHead(
                config.hidden_dim,
                n_ctx_tokens=config.n_ctx_tokens,
                head_dim=config.policy_head_dim,
                topk=config.policy_topk,
                search_steps=config.policy_search_steps,
                num_heads=config.num_heads,
                dropout=0.0 if config.dropout else config.dropout,
            )
        else:
            self.policy_head = SpatialPolicyHead(
                config.hidden_dim,
                n_ctx_tokens=config.n_ctx_tokens,
                head_dim=config.policy_head_dim,
            )
        self.use_search_value_head = config.use_search_value_head
        if self.use_search_value_head:
            self.value_head = SearchValueHead(
                config.hidden_dim,
                value_hidden=config.value_hidden,
                n_ctx_tokens=config.n_ctx_tokens,
                head_dim=config.policy_head_dim,
                topk=config.value_topk,
                search_steps=config.value_search_steps,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            self._pool_value = False
        elif config.value_head_type == "pool":
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
        batch_size = board_input["fused_ids"].shape[0]

        if self.config.use_meta_attention or self.config.use_linear_meta_attention:
            content, position = self.encoder.forward_streams(board_input)
            content = self.input_proj(content)
            position = self.pos_input_proj(position)
            content = torch.cat(
                [self.cls_token.expand(batch_size, -1, -1), content], dim=1
            )
            position = torch.cat(
                [self.cls_pos_token.expand(batch_size, -1, -1), position], dim=1
            )
            for layer in self.layers:
                if self.config.gradient_checkpointing and self.training:
                    content, position = checkpoint(
                        layer, content, position, use_reentrant=False,
                    )
                else:
                    content, position = layer(content, position)
            hidden = self.norm(content)
        else:
            hidden = self.input_proj(self.encoder(board_input))
            hidden = torch.cat([self.cls_token.expand(batch_size, -1, -1), hidden], dim=1)
            if self.pos_embed is not None:
                hidden = hidden + self.pos_embed

            if self.layers is not None:
                nhead = self.config.num_heads
                static_bias = None
                if self.rel_bias is not None:
                    bias = self.rel_bias()  # (H, seq, seq)
                    static_bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
                        batch_size * nhead, hidden.shape[1], hidden.shape[1]
                    )
                sq0 = self.config.n_ctx_tokens  # CLS+ctx count; squares follow
                for layer in self.layers:
                    attn_bias = static_bias
                    if self.gab is not None:
                        gab = self.gab(hidden[:, sq0:sq0 + 64, :])  # (B,H,T,T)
                        gab_flat = gab.reshape(
                            batch_size * nhead, hidden.shape[1], hidden.shape[1]
                        )
                        attn_bias = gab_flat if attn_bias is None else attn_bias + gab_flat
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
        if self.use_search_value_head:
            value_out = self.value_head(hidden, cls_hidden)
            value_logits = value_out["value_logits"]
        elif self._pool_value:
            value_logits = self.value_head(hidden, cls_hidden)
        else:
            value_logits = self.value_head(cls_hidden)

        result = {
            "value_logits": value_logits,
            "cls_hidden": cls_hidden,
        }
        if self.use_search_policy_head:
            policy_out = self.policy_head(hidden, cls_hidden)
            result.update(policy_out)
        else:
            result["policy_logits"] = self.policy_head(hidden, cls_hidden)
        if self.use_search_value_head:
            result.update(value_out)
        return result


def build_model(config: ChessTransformerConfig | dict | str | Path | None = None) -> ChessTransformer:
    if config is None:
        resolved = DEFAULT_200M_CONFIG
    elif isinstance(config, ChessTransformerConfig):
        resolved = config
    elif isinstance(config, (str, Path)):
        resolved = ChessTransformerConfig.from_json(config)
    else:
        known = {f.name for f in ChessTransformerConfig.__dataclass_fields__.values()}
        resolved = ChessTransformerConfig(**{k: v for k, v in config.items() if k in known})
    return ChessTransformer(resolved)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
