"""Attention Residuals (AttnRes) — Block variant.

Implements the Block AttnRes mechanism from:
  "Technical Report of Attention Residuals" (Kimi Team, 2026)
  https://arxiv.org/abs/2603.15031

Instead of fixed residual connections  h_l = h_{l-1} + f_{l-1}(h_{l-1}),
AttnRes uses  h_l = sum_i alpha_{i->l} * v_i  where alpha are softmax
attention weights computed from learned per-layer pseudo-queries.

Block AttnRes groups layers into blocks of size B, computes standard
residuals within each block, and applies cross-block attention only over
the N = L/B block-level representations. This reduces memory from O(Ld)
to O(Nd).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAttnRes(nn.Module):
    """Block Attention Residuals module.

    Maintains a buffer of block representations and applies softmax
    attention over them to compute the residual for new layers.

    Args:
        hidden_dim: Model hidden dimension (d).
        block_size: Number of layers per block (B).
        num_blocks: Total number of blocks (N = num_layers / block_size).
        head_dim: Dimension of the pseudo-query vectors. Defaults to hidden_dim.
    """

    def __init__(
        self,
        hidden_dim: int,
        block_size: int,
        num_blocks: int,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.head_dim = head_dim or hidden_dim

        # Per-block learned pseudo-query for computing attention over prior blocks
        # Shape: (num_blocks, head_dim)
        self.queries = nn.Parameter(torch.randn(num_blocks, self.head_dim) * 0.02)

        # Project block representations to key space if head_dim != hidden_dim
        if self.head_dim != hidden_dim:
            self.key_proj = nn.Linear(hidden_dim, self.head_dim, bias=False)
            self.val_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.key_proj = nn.Identity()
            self.val_proj = nn.Identity()

        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        block_reprs: torch.Tensor,
        current_block_idx: int,
    ) -> torch.Tensor:
        """Compute the AttnRes aggregation for a given block.

        Args:
            block_reprs: Tensor of shape (batch, num_available_blocks, hidden_dim)
                containing representations of all preceding blocks (including current).
            current_block_idx: Index of the current block (0-based).

        Returns:
            Aggregated representation of shape (batch, seq_len, hidden_dim).
            This replaces the standard residual sum.
        """
        n_available = block_reprs.shape[1]
        if n_available <= 1:
            # First block — just return it (no prior blocks to attend over)
            return block_reprs[:, 0]

        # query for current block: (1, head_dim) → (batch, 1, head_dim)
        q = self.queries[current_block_idx].unsqueeze(0).unsqueeze(0)
        q = q.expand(block_reprs.shape[0], -1, -1)

        # keys and values from all available blocks
        # block_reprs: (batch, n_available, hidden_dim)
        k = self.key_proj(block_reprs)  # (batch, n_available, head_dim)
        v = self.val_proj(block_reprs)  # (batch, n_available, hidden_dim)

        # Attention: softmax(q @ k^T / sqrt(d)) @ v
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self._scale  # (batch, 1, n_available)
        attn_weights = F.softmax(attn_logits, dim=-1)
        output = torch.bmm(attn_weights, v).squeeze(1)  # (batch, hidden_dim)

        return output


class AttnResWrapper(nn.Module):
    """Wraps a transformer model to use Block AttnRes instead of standard residuals.

    This is designed to be applied after loading a pretrained model.
    It intercepts the hidden states between transformer blocks and applies
    Block AttnRes aggregation.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        wrapper = AttnResWrapper(model, block_size=4)
        # Now use wrapper.forward() which routes through AttnRes
    """

    def __init__(self, model: nn.Module, block_size: int = 4, head_dim: int = 64):
        super().__init__()
        self.model = model

        # Detect model architecture and get layer list
        self.layers, self.layer_attr = self._find_layers(model)
        num_layers = len(self.layers)
        hidden_dim = self._get_hidden_dim(model)

        self.block_size = block_size
        self.num_blocks = math.ceil(num_layers / block_size)

        self.attnres = BlockAttnRes(
            hidden_dim=hidden_dim,
            block_size=block_size,
            num_blocks=self.num_blocks,
            head_dim=head_dim,
        )

        # Move attnres to same device/dtype as model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.attnres = self.attnres.to(device=device, dtype=dtype)

    @staticmethod
    def _find_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
        """Find the transformer layer list in common architectures."""
        # Qwen2, LLaMA, Mistral, etc.
        for attr_path in [
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "model.decoder.layers",
        ]:
            parts = attr_path.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                if isinstance(obj, nn.ModuleList):
                    return obj, attr_path
            except AttributeError:
                continue
        raise ValueError(
            "Could not find transformer layers. "
            "Supported architectures: Qwen2, LLaMA, Mistral, GPT-NeoX"
        )

    @staticmethod
    def _get_hidden_dim(model: nn.Module) -> int:
        """Extract hidden dimension from model config."""
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Model has no config attribute")
        for attr in ["hidden_size", "d_model", "n_embd"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError("Could not determine hidden dimension from model config")

    def forward_with_attnres(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with Block AttnRes replacing standard residuals.

        Runs each block of layers, collects block representations,
        and applies AttnRes cross-block aggregation.
        """
        model = self.model
        config = model.config

        # Get embeddings
        if hasattr(model, "model"):
            embed_tokens = model.model.embed_tokens
            norm = model.model.norm
            lm_head = model.lm_head
        elif hasattr(model, "transformer"):
            embed_tokens = model.transformer.wte
            norm = model.transformer.ln_f
            lm_head = model.lm_head
        else:
            raise ValueError("Unsupported model architecture for AttnRes")

        hidden = embed_tokens(input_ids)

        # Prepare causal attention mask if needed
        batch_size, seq_len = input_ids.shape

        # Process layers in blocks
        block_reprs = []  # list of (batch, hidden_dim) — mean-pooled block outputs
        block_hidden = hidden

        for block_idx in range(self.num_blocks):
            start_layer = block_idx * self.block_size
            end_layer = min(start_layer + self.block_size, len(self.layers))

            # Standard residual connections within the block
            for layer_idx in range(start_layer, end_layer):
                layer = self.layers[layer_idx]
                layer_out = layer(
                    block_hidden,
                    attention_mask=attention_mask,
                    position_ids=None,
                )
                # Most HF models return (hidden_states, ...) tuple
                if isinstance(layer_out, tuple):
                    block_hidden = layer_out[0]
                else:
                    block_hidden = layer_out

            # Mean-pool over sequence for the block representation
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                block_repr = (block_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                block_repr = block_hidden.mean(dim=1)  # (batch, hidden_dim)

            block_reprs.append(block_repr)

            # Apply AttnRes: cross-block aggregation
            if len(block_reprs) > 1:
                stacked = torch.stack(block_reprs, dim=1)  # (batch, n_blocks, hidden_dim)
                attnres_out = self.attnres(stacked, block_idx)  # (batch, hidden_dim)
                # Blend AttnRes output back into the sequence-level hidden states
                # Scale factor to not overwhelm the within-block residual
                block_hidden = block_hidden + 0.1 * attnres_out.unsqueeze(1)

        # Final norm + LM head
        hidden = norm(block_hidden)
        logits = lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def forward(self, **kwargs):
        """Standard forward — delegates to AttnRes-augmented forward."""
        return self.forward_with_attnres(**kwargs)
