"""ChessModel: learned board encoder -> Qwen backbone -> move policy head.

Architecture:
  1. BoardEncoder: small residual CNN over 8x8x18 feature planes
     -> 64 square embeddings + 1 global token = 65 tokens
  2. Projection: linear map to Qwen's hidden_size (1024)
  3. Qwen backbone: processes 65 tokens via inputs_embeds (no tokenizer)
  4. Policy head: linear map from Qwen hidden states -> move logits
  5. Value head (optional): scalar win/draw/loss prediction

At inference: mask illegal moves, take argmax or sample.
"""

import math

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_features import (
    board_to_planes, batch_boards_to_planes, NUM_PLANES,
    board_to_token_ids, batch_boards_to_token_ids,
    board_to_fused_token_ids, batch_boards_to_fused_token_ids,
    NUM_PIECE_TYPES, NUM_COLORS, NUM_CASTLING_STATES, NUM_EP_STATES,
    NUM_FUSED_TOKENS,
)
from move_vocab import VOCAB_SIZE, legal_move_mask, move_to_index, index_to_move


class ResidualBlock(nn.Module):
    """Simple residual conv block: conv -> BN -> ReLU -> conv -> BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class BoardEncoder(nn.Module):
    """CNN encoder: 8x8x18 planes -> 64 square embeddings + 1 global token.

    Output shape: (batch, 65, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, num_blocks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # Initial convolution: 18 input planes -> embed channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_PLANES, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(embed_dim) for _ in range(num_blocks)]
        )

        # Global token: pool spatial features into a single vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            planes: (B, 18, 8, 8) board feature planes
        Returns:
            (B, 65, embed_dim) — 64 square tokens + 1 global token
        """
        x = self.input_conv(planes)       # (B, C, 8, 8)
        x = self.res_blocks(x)            # (B, C, 8, 8)

        # Square tokens: reshape to (B, 64, C)
        B, C, H, W = x.shape
        square_tokens = x.view(B, C, 64).permute(0, 2, 1)  # (B, 64, C)

        # Global token: (B, C, 1, 1) -> (B, 1, C)
        global_feat = self.global_pool(x).view(B, C)
        global_token = self.global_proj(global_feat).unsqueeze(1)  # (B, 1, C)

        return torch.cat([global_token, square_tokens], dim=1)  # (B, 65, C)

    def prepare_input(self, board: chess.Board, device: torch.device):
        """Convert a single board to batched CNN input."""
        return board_to_planes(board).unsqueeze(0).to(device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        """Convert a list of boards to batched CNN input."""
        return batch_boards_to_planes(boards).to(device)


class LearnedBoardEncoder(nn.Module):
    """Learned-embedding encoder: per-square piece+color+position embeddings.

    Each square gets:  color_proj[color](piece_embed[piece]) + square_embed[sq]
    Context tokens appended: turn, castling, en passant

    Output shape: (batch, 67, embed_dim)
    """

    NUM_CONTEXT = 3  # turn + castling + ep tokens

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 64 + self.NUM_CONTEXT  # 67

        # Piece identity (7 types: empty + 6 piece types)
        self.piece_embed = nn.Embedding(NUM_PIECE_TYPES, embed_dim)

        # Color projection: a separate linear transform per color
        # none (empty squares), white, black
        self.color_proj = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(NUM_COLORS)]
        )

        # Square positional embeddings
        self.square_embed = nn.Embedding(64, embed_dim)

        # Context embeddings
        self.turn_embed = nn.Embedding(2, embed_dim)           # white / black
        self.castling_embed = nn.Embedding(NUM_CASTLING_STATES, embed_dim)  # 16 states
        self.ep_embed = nn.Embedding(NUM_EP_STATES, embed_dim) # none + 8 files

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_ids: dict with
                piece_ids (B,64), color_ids (B,64),
                turn (B,), castling (B,), ep_file (B,)
        Returns:
            (B, 67, embed_dim)
        """
        piece_ids = token_ids["piece_ids"]   # (B, 64)
        color_ids = token_ids["color_ids"]   # (B, 64)
        B = piece_ids.shape[0]

        # Look up base piece embeddings
        piece_emb = self.piece_embed(piece_ids)  # (B, 64, D)

        # Apply all three color projections, select per-square
        proj_stack = torch.stack(
            [proj(piece_emb) for proj in self.color_proj], dim=2
        )  # (B, 64, 3, D)
        color_sel = F.one_hot(color_ids, NUM_COLORS).float()  # (B, 64, 3)
        sq_emb = (proj_stack * color_sel.unsqueeze(-1)).sum(dim=2)  # (B, 64, D)

        # Add positional embeddings
        sq_idx = torch.arange(64, device=piece_ids.device)
        sq_emb = sq_emb + self.square_embed(sq_idx)  # broadcast over B

        # Context tokens
        turn_tok = self.turn_embed(token_ids["turn"]).unsqueeze(1)          # (B,1,D)
        castle_tok = self.castling_embed(token_ids["castling"]).unsqueeze(1) # (B,1,D)
        ep_tok = self.ep_embed(token_ids["ep_file"]).unsqueeze(1)           # (B,1,D)

        # [turn, castling, ep, sq0 ... sq63]
        tokens = torch.cat([turn_tok, castle_tok, ep_tok, sq_emb], dim=1)  # (B,67,D)
        return self.norm(tokens)

    def prepare_input(self, board: chess.Board, device: torch.device):
        """Convert a single board to batched token IDs."""
        return batch_boards_to_token_ids([board], device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        """Convert a list of boards to batched token IDs."""
        return batch_boards_to_token_ids(boards, device)


class FusedBoardEncoder(nn.Module):
    """Fused-token encoder: single embedding for 13 piece-color types.

    Instead of separate piece_embed + color_proj, uses one embedding table
    for all 13 tokens (empty + 6 white pieces + 6 black pieces).
    This is simpler, has fewer params, and lets the model learn piece-color
    interactions directly without the factored bottleneck.

    Output shape: (batch, 67, embed_dim) — same as LearnedBoardEncoder.
    """

    NUM_CONTEXT = 3  # turn + castling + ep tokens

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 64 + self.NUM_CONTEXT  # 67

        # Single fused embedding: 13 piece-color types
        self.piece_color_embed = nn.Embedding(NUM_FUSED_TOKENS, embed_dim)

        # Square positional embeddings
        self.square_embed = nn.Embedding(64, embed_dim)

        # Context embeddings (same as LearnedBoardEncoder)
        self.turn_embed = nn.Embedding(2, embed_dim)
        self.castling_embed = nn.Embedding(NUM_CASTLING_STATES, embed_dim)
        self.ep_embed = nn.Embedding(NUM_EP_STATES, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_ids: dict with
                fused_ids (B,64),
                turn (B,), castling (B,), ep_file (B,)
        Returns:
            (B, 67, embed_dim)
        """
        fused_ids = token_ids["fused_ids"]  # (B, 64)

        # Look up fused piece-color embeddings
        sq_emb = self.piece_color_embed(fused_ids)  # (B, 64, D)

        # Add positional embeddings
        sq_idx = torch.arange(64, device=fused_ids.device)
        sq_emb = sq_emb + self.square_embed(sq_idx)

        # Context tokens
        turn_tok = self.turn_embed(token_ids["turn"]).unsqueeze(1)
        castle_tok = self.castling_embed(token_ids["castling"]).unsqueeze(1)
        ep_tok = self.ep_embed(token_ids["ep_file"]).unsqueeze(1)

        tokens = torch.cat([turn_tok, castle_tok, ep_tok, sq_emb], dim=1)
        return self.norm(tokens)

    def prepare_input(self, board: chess.Board, device: torch.device):
        return batch_boards_to_fused_token_ids([board], device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        return batch_boards_to_fused_token_ids(boards, device)


class ChessRelativeBias(nn.Module):
    """Chess-aware relative geometry bias for attention.

    For each pair of squares (i, j), computes a learned bias based on:
      - rank distance (0..7, 8 buckets)
      - file distance (0..7, 8 buckets)
      - same diagonal (bool)
      - same anti-diagonal (bool)
      - knight-move relationship (bool)

    The bias is per-head and added to attention logits before softmax.
    Non-square tokens (CLS, turn, castling, ep) get separate learned biases.

    seq_len = n_ctx + 64 where n_ctx = number of context tokens (CLS + turn + castling + ep = 4).
    """

    def __init__(self, num_heads: int, n_ctx: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.n_ctx = n_ctx
        seq_len = n_ctx + 64

        # Precompute distance features for 64×64 square pairs
        rank_dist = torch.zeros(64, 64, dtype=torch.long)
        file_dist = torch.zeros(64, 64, dtype=torch.long)
        same_diag = torch.zeros(64, 64, dtype=torch.long)
        same_anti = torch.zeros(64, 64, dtype=torch.long)
        knight_rel = torch.zeros(64, 64, dtype=torch.long)

        for i in range(64):
            ri, fi = i // 8, i % 8
            for j in range(64):
                rj, fj = j // 8, j % 8
                dr, df = abs(ri - rj), abs(fi - fj)
                rank_dist[i, j] = dr
                file_dist[i, j] = df
                same_diag[i, j] = 1 if (ri - fi) == (rj - fj) else 0
                same_anti[i, j] = 1 if (ri + fi) == (rj + fj) else 0
                knight_rel[i, j] = 1 if (dr == 2 and df == 1) or (dr == 1 and df == 2) else 0

        self.register_buffer("rank_dist", rank_dist)
        self.register_buffer("file_dist", file_dist)
        self.register_buffer("same_diag", same_diag)
        self.register_buffer("same_anti", same_anti)
        self.register_buffer("knight_rel", knight_rel)

        # Learned bias tables (per head)
        self.rank_bias = nn.Embedding(8, num_heads)   # |rank_i - rank_j| in 0..7
        self.file_bias = nn.Embedding(8, num_heads)   # |file_i - file_j| in 0..7
        self.diag_bias = nn.Parameter(torch.zeros(num_heads))
        self.anti_bias = nn.Parameter(torch.zeros(num_heads))
        self.knight_bias = nn.Parameter(torch.zeros(num_heads))

        # Context token biases: ctx↔ctx and ctx↔sq interactions
        self.ctx_ctx_bias = nn.Parameter(torch.zeros(num_heads, n_ctx, n_ctx))
        self.ctx_sq_bias = nn.Parameter(torch.zeros(num_heads, n_ctx))  # broadcast over 64 sq
        self.sq_ctx_bias = nn.Parameter(torch.zeros(num_heads, n_ctx))

    def forward(self) -> torch.Tensor:
        """Returns (num_heads, seq_len, seq_len) attention bias."""
        seq_len = self.n_ctx + 64
        bias = torch.zeros(self.num_heads, seq_len, seq_len,
                           device=self.rank_dist.device)

        # Square-square region: n_ctx..n_ctx+64
        c = self.n_ctx
        rb = self.rank_bias(self.rank_dist)  # (64, 64, H)
        fb = self.file_bias(self.file_dist)  # (64, 64, H)
        sq_bias = rb + fb  # (64, 64, H)
        sq_bias = sq_bias + self.same_diag.unsqueeze(-1).float() * self.diag_bias
        sq_bias = sq_bias + self.same_anti.unsqueeze(-1).float() * self.anti_bias
        sq_bias = sq_bias + self.knight_rel.unsqueeze(-1).float() * self.knight_bias
        bias[:, c:, c:] = sq_bias.permute(2, 0, 1)  # (H, 64, 64)

        # Context-context
        bias[:, :c, :c] = self.ctx_ctx_bias

        # Context-square and square-context (broadcast)
        bias[:, :c, c:] = self.ctx_sq_bias.unsqueeze(-1).expand(-1, -1, 64)
        bias[:, c:, :c] = self.sq_ctx_bias.unsqueeze(-2).expand(-1, 64, -1)

        return bias


class ChessModel(nn.Module):
    """Full chess model: board encoder -> Qwen backbone -> move head.

    The Qwen transformer is used as a deep feature mixer over learned
    board embeddings. No text tokenization is involved.
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        encoder: nn.Module | None = None,
        encoder_dim: int = 256,
        encoder_blocks: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Extract the base transformer (not the LM head wrapper)
        # Qwen3ForCausalLM has .model = Qwen3Model (base transformer)
        if hasattr(qwen_model, 'model') and hasattr(qwen_model, 'lm_head'):
            base_model = qwen_model.model
        else:
            base_model = qwen_model

        if hasattr(base_model, 'config'):
            self.hidden_size = base_model.config.hidden_size
        else:
            self.hidden_size = 1024

        # Board encoder — use provided encoder or build default CNN encoder
        if encoder is not None:
            self.encoder = encoder
            encoder_dim = encoder.embed_dim
        else:
            self.encoder = BoardEncoder(embed_dim=encoder_dim, num_blocks=encoder_blocks)

        # Project encoder output to Qwen's hidden size
        self.input_proj = nn.Linear(encoder_dim, self.hidden_size)

        # The Qwen base transformer (used via inputs_embeds)
        self.backbone = base_model

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Policy head: move logits from final hidden states
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, VOCAB_SIZE),
        )

        # Value head: scalar evaluation
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # win/draw/loss
        )

    def forward(
        self,
        board_input,
        move_targets: torch.Tensor | None = None,
        value_targets: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            board_input: (B, 18, 8, 8) planes for CNN, or dict of token IDs for learned encoder
            move_targets: (B,) move vocabulary indices (for training)
            value_targets: (B,) value labels 0=loss/1=draw/2=win (for training)
            legal_masks: (B, VOCAB_SIZE) bool mask of legal moves (for inference)
        """
        if isinstance(board_input, dict):
            B = board_input["piece_ids"].shape[0]
        else:
            B = board_input.shape[0]

        # Encode board
        tokens = self.encoder(board_input)           # (B, N, encoder_dim)
        embeds = self.input_proj(tokens)             # (B, 65, hidden_size)

        # Match backbone dtype (bfloat16)
        backbone_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)

        # Pass through Qwen backbone
        outputs = self.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()   # (B, N, hidden_size) back to fp32

        # Use last token for policy and value — in causal attention,
        # only the last token has attended to all previous tokens
        global_hidden = hidden[:, -1, :]             # (B, hidden_size)

        # Policy
        policy_logits = self.policy_head(global_hidden)  # (B, VOCAB_SIZE)

        # Value
        value_logits = self.value_head(global_hidden)    # (B, 3)

        result = {
            "policy_logits": policy_logits,
            "value_logits": value_logits,
        }

        # Compute losses if targets provided
        device = board_input["piece_ids"].device if isinstance(board_input, dict) else board_input.device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            result["policy_loss"] = policy_loss
            total_loss = total_loss + policy_loss

        if value_targets is not None:
            value_loss = F.cross_entropy(value_logits, value_targets)
            result["value_loss"] = value_loss
            total_loss = total_loss + 0.5 * value_loss

        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def predict_move(self, board: chess.Board) -> tuple[chess.Move, torch.Tensor]:
        """Predict the best legal move for a position.

        Returns (best_move, policy_probs).
        """
        self.eval()
        device = next(self.parameters()).device

        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)

        result = self.forward(board_input)
        logits = result["policy_logits"][0]

        # Mask illegal moves to -inf
        logits[~mask] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        best_idx = probs.argmax().item()
        best_move = index_to_move(best_idx)

        return best_move, probs

    def trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        """Count all parameters."""
        return sum(p.numel() for p in self.parameters())
