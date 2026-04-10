"""exp002: Deep Move-History Transformer — LLaMA-style decoder for next-move prediction.

Upgrades over exp160 baseline:
  - LLaMA-style architecture: RMSNorm (pre-norm), SwiGLU FFN, no bias
  - Rotary Position Embeddings (RoPE) instead of absolute pos embeddings
  - Flash Attention via F.scaled_dot_product_attention
  - Gradient checkpointing for deep (24L+) training on 8GB VRAM
  - Cosine LR schedule with linear warmup
  - Proper weight initialization (small init for output projections)

Architecture: 24L / 512d / 8H / SwiGLU-2048 ≈ 78M params
Vocab: 1968 compact moves + 3 special tokens = 1971 total

The model sees ONLY move history (d2d4, e7e5, ...) — no board state, no piece
positions, no evaluation. It must learn to reconstruct board state implicitly
from the move sequence and predict the next legal move.

Usage:
  # Quick falsification (100 games, ~2min on RTX 4060):
  python experiments_history/exp002_deep_history.py \
    --train-pgn outputs/sf_games_5k.pgn \
    --output-path outputs/exp_history_002/best.pt \
    --train-max-games 100 --epochs 4

  # Full training (5K games):
  python experiments_history/exp002_deep_history.py \
    --train-pgn outputs/sf_games_5k.pgn \
    --output-path outputs/exp_history_002/best.pt

  # Play against Stockfish for Elo estimate:
  python experiments_history/exp002_deep_history.py \
    --play outputs/exp_history_002/best.pt --sf-elo 1500 --num-games 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import chess.engine
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from move_vocab import COMPACT_IDX_TO_UCI, COMPACT_UCI_TO_IDX, COMPACT_VOCAB_SIZE

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")


# ── Vocabulary ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MoveVocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    bos_id: int
    eos_id: int

    @classmethod
    def build(cls) -> "MoveVocabulary":
        tokens = [*SPECIAL_TOKENS, *COMPACT_IDX_TO_UCI]
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(
            token_to_id=token_to_id,
            id_to_token=tokens,
            pad_id=token_to_id["<pad>"],
            bos_id=token_to_id["<bos>"],
            eos_id=token_to_id["<eos>"],
        )

    @property
    def size(self) -> int:
        return len(self.id_to_token)


# ── Data Pipeline (reused from exp160, proven correct) ──────────────────────


@dataclass(slots=True)
class GameExample:
    inputs: list[int]
    targets: list[int]
    legal_target_ids: list[list[int]]


def read_games_from_pgn(path: Path, max_games: int | None = None) -> list[list[str]]:
    games: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            moves = [move.uci() for move in game.mainline_moves()]
            if moves:
                games.append(moves)
            if max_games is not None and len(games) >= max_games:
                break
    return games


def build_examples(
    games: Iterable[list[str]], vocab: MoveVocabulary, max_seq_len: int
) -> list[GameExample]:
    examples: list[GameExample] = []
    for moves in games:
        board = chess.Board()
        inputs = [vocab.bos_id]
        targets: list[int] = []
        legal_target_ids: list[list[int]] = []

        for uci_move in moves:
            # Collect legal moves BEFORE pushing — this is the legal mask for predicting this move
            legal_ids = []
            for lm in board.legal_moves:
                uci = lm.uci()
                if uci in vocab.token_to_id:
                    legal_ids.append(vocab.token_to_id[uci])
            if not legal_ids:
                break  # Safety: skip if no legal moves map to vocab
            legal_target_ids.append(legal_ids)
            targets.append(vocab.token_to_id.get(uci_move, vocab.pad_id))
            inputs.append(vocab.token_to_id.get(uci_move, vocab.pad_id))
            board.push_uci(uci_move)

        # EOS at end
        targets.append(vocab.eos_id)
        legal_target_ids.append([vocab.eos_id])

        sequence_inputs = inputs[:-1]
        # Chunk into max_seq_len windows (for very long games)
        for start in range(0, len(sequence_inputs), max_seq_len):
            stop = min(start + max_seq_len, len(sequence_inputs))
            examples.append(
                GameExample(
                    inputs=sequence_inputs[start:stop],
                    targets=targets[start:stop],
                    legal_target_ids=legal_target_ids[start:stop],
                )
            )
    return examples


class MoveHistoryDataset(Dataset[GameExample]):
    def __init__(self, examples: list[GameExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GameExample:
        return self.examples[index]


def collate_batch(
    batch: list[GameExample], pad_id: int, vocab_size: int
) -> dict[str, Tensor]:
    batch_size = len(batch)
    max_len = max(len(item.inputs) for item in batch)

    inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    legal_mask = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.bool)

    for row_index, item in enumerate(batch):
        seq_len = len(item.inputs)
        inputs[row_index, :seq_len] = torch.tensor(item.inputs, dtype=torch.long)
        targets[row_index, :seq_len] = torch.tensor(item.targets, dtype=torch.long)
        attention_mask[row_index, :seq_len] = True
        for t, allowed_ids in enumerate(item.legal_target_ids):
            if t >= max_len:
                break
            if allowed_ids:
                legal_mask[row_index, t, allowed_ids] = True
            legal_mask[row_index, t, pad_id] = True

    return {
        "inputs": inputs,
        "targets": targets,
        "attention_mask": attention_mask,
        "legal_mask": legal_mask,
    }


# ── Model Architecture: LLaMA-style Decoder ────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tensor:
    """Precompute RoPE frequency tensor [max_seq_len, dim//2, 2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # [T, dim//2]
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # [T, dim//2, 2]


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings. x: [B, H, T, D], freqs: [T, D//2, 2]."""
    B, H, T, D = x.shape
    x = x.reshape(B, H, T, D // 2, 2)
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2, 2]
    cos_f = freqs[..., 0]
    sin_f = freqs[..., 1]
    x0 = x[..., 0]
    x1 = x[..., 1]
    out = torch.stack([x0 * cos_f - x1 * sin_f, x0 * sin_f + x1 * cos_f], dim=-1)
    return out.reshape(B, H, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # Fused QKV projection (no bias, LLaMA-style)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, rope_freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: [B, T, H, D_h]
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Flash attention via PyTorch SDPA (causal=True handles the mask)
        drop = self.dropout if self.training else 0.0
        # Use attn_mask for padding if provided, is_causal for autoregressive
        if mask is not None:
            # mask: [B, T] bool (True = attend). Build [B, 1, T, T] combined mask
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            # padding_mask: [B, 1, 1, T] — False where padded
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            # Combined: positions that should NOT be attended to
            combined = causal.unsqueeze(0) | ~padding_mask  # [B, 1, T, T]
            attn_mask = torch.zeros_like(combined, dtype=q.dtype)
            attn_mask.masked_fill_(combined, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020, used in LLaMA)."""

    def __init__(self, d_model: int, dim_ff: int) -> None:
        super().__init__()
        # Gate + up projection, then down
        self.w_gate = nn.Linear(d_model, dim_ff, bias=False)
        self.w_up = nn.Linear(d_model, dim_ff, bias=False)
        self.w_down = nn.Linear(dim_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dim_ff)

    def forward(self, x: Tensor, rope_freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DeepMoveHistoryTransformer(nn.Module):
    """LLaMA-style causal decoder for move-sequence prediction."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Tie output weights to embedding (standard for small vocab LMs)
        self.output.weight = self.token_embedding.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        head_dim = d_model // nhead
        rope_freqs = precompute_rope_freqs(head_dim, max_seq_len)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following GPT-2 / LLaMA conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scale down residual projections by 1/sqrt(2*num_layers)
        scale = (2 * len(self.layers)) ** -0.5
        for layer in self.layers:
            nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.ffn.w_down.weight, mean=0.0, std=0.02 * scale)

    def forward(self, inputs: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = self.token_embedding(inputs)
        rope = self.rope_freqs

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, rope, attention_mask, use_reentrant=False
                )
            else:
                x = layer(x, rope, attention_mask)

        x = self.final_norm(x)
        return self.output(x)


# ── Loss / Masking ──────────────────────────────────────────────────────────


def apply_legal_mask(logits: Tensor, legal_mask: Tensor) -> Tensor:
    return logits.masked_fill(~legal_mask, float("-inf"))


def masked_cross_entropy(
    logits: Tensor, targets: Tensor, legal_mask: Tensor, pad_id: int,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross entropy with legal masking and correct label smoothing.

    Standard F.cross_entropy label_smoothing distributes mass to ALL classes
    including -inf masked ones, producing inf loss. Instead, we manually
    smooth only across legal moves.
    """
    legal_logits = apply_legal_mask(logits, legal_mask)
    B_T, V = legal_logits.reshape(-1, legal_logits.size(-1)).shape
    flat_logits = legal_logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)

    if label_smoothing <= 0.0:
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=pad_id)

    # Manual label smoothing over legal moves only
    log_probs = F.log_softmax(flat_logits, dim=-1)  # -inf stays -inf
    non_pad = flat_targets != pad_id

    if not non_pad.any():
        return torch.tensor(0.0, device=logits.device)

    # NLL component: -log_prob of target
    nll = F.nll_loss(log_probs, flat_targets, ignore_index=pad_id, reduction="none")

    # Smoothing component: average -log_prob over LEGAL moves only
    # Replace -inf with 0 before averaging (those entries are masked out)
    safe_log_probs = log_probs.clone()
    safe_log_probs[safe_log_probs == float("-inf")] = 0.0
    # Count legal moves per position
    n_legal = legal_mask.reshape(-1, V).sum(dim=-1).float().clamp(min=1)
    smooth_loss = -safe_log_probs.sum(dim=-1) / n_legal

    # Combine: (1 - ε) * NLL + ε * smooth
    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    return loss[non_pad].mean()


# ── Training Config ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class TrainConfig:
    train_pgn: Path
    eval_pgn: Path | None
    output_path: Path
    # Architecture
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 24
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    # Training
    batch_size: int = 32
    gradient_accumulation: int = 2
    epochs: int = 10
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    gradient_checkpointing: bool = True
    # Data
    seed: int = 7
    train_max_games: int | None = None
    eval_max_games: int | None = None
    eval_split: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── LR Schedule ─────────────────────────────────────────────────────────────


def cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    top3_correct = 0
    total_positions = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["attention_mask"].to(device)
        legal_mask = batch["legal_mask"].to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs, mask)

        loss = masked_cross_entropy(logits.float(), targets, legal_mask, pad_id)
        total_loss += loss.item()
        total_batches += 1

        legal_logits = apply_legal_mask(logits.float(), legal_mask)
        non_pad = targets != pad_id
        if non_pad.any():
            pred = legal_logits[non_pad].argmax(dim=-1)
            correct += (pred == targets[non_pad]).sum().item()
            top3 = legal_logits[non_pad].topk(min(3, legal_logits.size(-1)), dim=-1).indices
            top3_correct += (top3 == targets[non_pad].unsqueeze(-1)).any(dim=-1).sum().item()
            total_positions += non_pad.sum().item()

    return {
        "loss": total_loss / max(total_batches, 1),
        "top1": correct / max(total_positions, 1),
        "top3": top3_correct / max(total_positions, 1),
        "n": total_positions,
    }


# ── Gameplay (for Elo evaluation) ──────────────────────────────────────────


def play_move(
    model: nn.Module,
    move_history: list[str],
    board: chess.Board,
    vocab: MoveVocabulary,
    device: torch.device,
    temperature: float = 0.0,
    max_seq_len: int = 512,
) -> chess.Move:
    """Pick the next move given the move history so far."""
    model.eval()
    # Build input sequence: <bos> m1 m2 ... mk
    token_ids = [vocab.bos_id]
    for m in move_history:
        tid = vocab.token_to_id.get(m)
        if tid is not None:
            token_ids.append(tid)
    # Truncate from the LEFT if too long (keep most recent context)
    if len(token_ids) > max_seq_len:
        token_ids = [vocab.bos_id] + token_ids[-(max_seq_len - 1):]

    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(inputs, dtype=torch.bool)

    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs, mask)

    # Take logits at the last position
    last_logits = logits[0, -1].float()

    # Build legal mask for current position
    legal_ids = []
    for lm in board.legal_moves:
        tid = vocab.token_to_id.get(lm.uci())
        if tid is not None:
            legal_ids.append(tid)

    if not legal_ids:
        # Fallback: random legal move (should never happen with compact vocab)
        return random.choice(list(board.legal_moves))

    legal_mask = torch.full_like(last_logits, float("-inf"))
    legal_mask[legal_ids] = 0.0
    masked_logits = last_logits + legal_mask

    if temperature <= 0.0:
        move_id = masked_logits.argmax().item()
    else:
        probs = F.softmax(masked_logits / temperature, dim=-1)
        move_id = torch.multinomial(probs, 1).item()

    return chess.Move.from_uci(vocab.id_to_token[move_id])


def play_game_vs_stockfish(
    model: nn.Module,
    vocab: MoveVocabulary,
    device: torch.device,
    sf_path: str,
    sf_elo: int,
    model_color: chess.Color,
    max_seq_len: int = 512,
    temperature: float = 0.0,
    move_time: float = 0.1,
) -> tuple[str, int]:
    """Play one game. Returns (result_str, score) where score: 1=win, 0=loss, 0.5=draw for model."""
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})

    board = chess.Board()
    move_history: list[str] = []

    while not board.is_game_over(claim_draw=True):
        if board.turn == model_color:
            move = play_move(model, move_history, board, vocab, device, temperature, max_seq_len)
        else:
            result = engine.play(board, chess.engine.Limit(time=move_time))
            move = result.move

        move_history.append(move.uci())
        board.push(move)

    engine.quit()

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw", 0.5
    elif outcome.winner == model_color:
        return "win", 1.0
    else:
        return "loss", 0.0


def run_elo_eval(
    model: nn.Module,
    vocab: MoveVocabulary,
    device: torch.device,
    sf_path: str,
    sf_elo: int,
    num_games: int,
    max_seq_len: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Play num_games (half as white, half as black) and report score."""
    results = []
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        color_str = "white" if color == chess.WHITE else "black"
        result_str, score = play_game_vs_stockfish(
            model, vocab, device, sf_path, sf_elo, color, max_seq_len, temperature
        )
        results.append({"game": i + 1, "color": color_str, "result": result_str, "score": score})
        log(f"  Game {i+1}/{num_games} ({color_str}): {result_str}")

    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results)
    wins = sum(1 for r in results if r["result"] == "win")
    draws = sum(1 for r in results if r["result"] == "draw")
    losses = sum(1 for r in results if r["result"] == "loss")

    summary = {
        "sf_elo": sf_elo,
        "num_games": num_games,
        "score": f"{total_score}/{num_games}",
        "avg_score": avg_score,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "results": results,
    }
    log(f"Score vs SF{sf_elo}: {total_score}/{num_games} ({avg_score:.3f}) "
        f"[W:{wins} D:{draws} L:{losses}]")
    return summary


# ── Training Loop ───────────────────────────────────────────────────────────


def split_games(
    games: list[list[str]], eval_split: float, seed: int
) -> tuple[list[list[str]], list[list[str]]]:
    shuffled = games[:]
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, shuffled
    eval_count = max(1, int(len(shuffled) * eval_split))
    eval_count = min(eval_count, len(shuffled) - 1)
    return shuffled[eval_count:], shuffled[:eval_count]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def train(config: TrainConfig) -> Path:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    vocab = MoveVocabulary.build()
    log(f"Vocab: {vocab.size} tokens ({vocab.size - len(SPECIAL_TOKENS)} moves + "
        f"{len(SPECIAL_TOKENS)} special)")

    # ── Load data ──
    train_games = read_games_from_pgn(config.train_pgn, max_games=config.train_max_games)
    if config.eval_pgn is not None:
        eval_games = read_games_from_pgn(config.eval_pgn, max_games=config.eval_max_games)
    else:
        train_games, eval_games = split_games(train_games, config.eval_split, config.seed)

    log(f"Games: {len(train_games)} train, {len(eval_games)} eval")

    train_examples = build_examples(train_games, vocab, config.max_seq_len)
    eval_examples = build_examples(eval_games, vocab, config.max_seq_len)
    if not train_examples:
        raise ValueError("No training examples produced.")
    if not eval_examples:
        raise ValueError("No eval examples produced.")

    log(f"Examples: {len(train_examples)} train, {len(eval_examples)} eval")
    total_moves = sum(len(e.targets) for e in train_examples)
    avg_moves = total_moves / len(train_games) if train_games else 0
    log(f"Total train moves: {total_moves:,} (avg {avg_moves:.0f}/game)")

    train_dataset = MoveHistoryDataset(train_examples)
    eval_dataset = MoveHistoryDataset(eval_examples)
    collate = lambda batch: collate_batch(batch, vocab.pad_id, vocab.size)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # ── Model ──
    device = torch.device(config.device)
    model = DeepMoveHistoryTransformer(
        vocab_size=vocab.size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params/1e6:.1f}M params, {config.num_layers}L/{config.d_model}d/"
        f"{config.nhead}H, SwiGLU-{config.dim_feedforward}, RoPE, "
        f"grad_ckpt={config.gradient_checkpointing}")

    # ── Optimizer ──
    # Separate weight decay: no decay for embeddings, norms, biases
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95), fused=True)

    # No GradScaler needed with bfloat16 (same exponent range as float32)

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation
    log(f"Total steps: {total_steps} (warmup: {config.warmup_steps})")

    # ── Training ──
    best_eval_loss = math.inf
    best_checkpoint: dict | None = None
    global_step = 0
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["attention_mask"].to(device)
            legal_mask_t = batch["legal_mask"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs, mask)
                loss = masked_cross_entropy(
                    logits.float(), targets, legal_mask_t, vocab.pad_id,
                    label_smoothing=config.label_smoothing,
                )
                loss = loss / config.gradient_accumulation

            loss.backward()

            # Progress logging every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_t0
                rate = (batch_idx + 1) / elapsed
                log(f"  batch {batch_idx+1}/{num_batches} | "
                    f"loss={loss.item()*config.gradient_accumulation:.4f} | "
                    f"{rate:.1f} batch/s | "
                    f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                # Update LR
                lr = cosine_lr(global_step, config.warmup_steps, total_steps,
                               config.learning_rate, config.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            raw_loss = loss.item() * config.gradient_accumulation
            if math.isfinite(raw_loss):
                epoch_loss += raw_loss
                epoch_batches += 1

        epoch_time = time.time() - epoch_t0
        train_loss = epoch_loss / max(epoch_batches, 1)

        eval_metrics = evaluate(model, eval_loader, device, vocab.pad_id)
        eval_loss = eval_metrics["loss"]

        log(f"epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} | "
            f"top1={100*eval_metrics['top1']:.2f}% top3={100*eval_metrics['top3']:.2f}% | "
            f"lr={lr:.2e} | {epoch_time:.0f}s | step={global_step}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab": vocab.id_to_token,
                "config": {
                    "vocab_size": vocab.size,
                    "max_seq_len": config.max_seq_len,
                    "d_model": config.d_model,
                    "nhead": config.nhead,
                    "num_layers": config.num_layers,
                    "dim_feedforward": config.dim_feedforward,
                    "dropout": config.dropout,
                },
                "train_config": asdict(config),
                "best_eval_loss": eval_loss,
                "best_top1": eval_metrics["top1"],
                "best_top3": eval_metrics["top3"],
                "epoch": epoch,
                "global_step": global_step,
            }
            log(f"  ★ New best eval loss: {eval_loss:.4f}")

    total_time = time.time() - t0

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint produced.")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, config.output_path)
    log(f"Saved: {config.output_path}")
    log(f"  {n_params/1e6:.1f}M params | vocab={vocab.size} | "
        f"best_loss={best_eval_loss:.4f} | "
        f"top1={100*best_checkpoint['best_top1']:.2f}% "
        f"top3={100*best_checkpoint['best_top3']:.2f}% | "
        f"{total_time:.0f}s total")
    return config.output_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def find_stockfish() -> str:
    """Find Stockfish binary."""
    candidates = [
        "stockfish",
        r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\stockfish\stockfish.exe",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # Try PATH
    import shutil
    sf = shutil.which("stockfish")
    if sf:
        return sf
    raise FileNotFoundError("Stockfish not found. Install or set path.")


def load_model_for_play(
    checkpoint_path: Path, device: torch.device,
) -> tuple[DeepMoveHistoryTransformer, MoveVocabulary]:
    """Load a trained checkpoint for gameplay."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]
    vocab = MoveVocabulary.build()

    model = DeepMoveHistoryTransformer(
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0,  # No dropout at inference
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="exp002: Deep move-history transformer (LLaMA-style, 24L/512d/8H)"
    )
    # Training
    p.add_argument("--train-pgn", type=Path, help="PGN file for training")
    p.add_argument("--eval-pgn", type=Path)
    p.add_argument("--output-path", type=Path, help="Where to save best checkpoint")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-max-games", type=int)
    p.add_argument("--eval-max-games", type=int)
    p.add_argument("--eval-split", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    # Gameplay / Elo
    p.add_argument("--play", type=Path, help="Checkpoint to play with (skip training)")
    p.add_argument("--sf-elo", type=int, default=1500)
    p.add_argument("--num-games", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    if args.play:
        # Gameplay mode
        model, vocab = load_model_for_play(args.play, device)
        sf_path = find_stockfish()
        log(f"Playing {args.num_games} games vs SF{args.sf_elo} (temp={args.temperature})")
        summary = run_elo_eval(
            model, vocab, device, sf_path, args.sf_elo,
            args.num_games, args.max_seq_len, args.temperature,
        )
        # Save results
        out_dir = args.play.parent
        results_path = out_dir / f"elo_eval_sf{args.sf_elo}.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"Results saved: {results_path}")
        return

    # Training mode
    if not args.train_pgn or not args.output_path:
        build_parser().error("--train-pgn and --output-path required for training")

    config = TrainConfig(
        train_pgn=args.train_pgn.resolve(),
        eval_pgn=args.eval_pgn.resolve() if args.eval_pgn else None,
        output_path=args.output_path.resolve(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        train_max_games=args.train_max_games,
        eval_max_games=args.eval_max_games,
        eval_split=args.eval_split,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
