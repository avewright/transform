"""exp003: Deep Move-History Transformer trained on Lichess high-Elo games.

Same architecture as exp002 (LLaMA-style 24L/512d/8H, 101.7M params), but
trained on vastly more data streamed from Lichess/standard-chess-games on HF.

Data pipeline:
  1. Stream parquet from HuggingFace (Lichess/standard-chess-games)
  2. Filter: both players >= min_elo, "Normal" termination, >= 10 moves
  3. Convert SAN movetext -> UCI moves via python-chess
  4. Feed into the same build_examples / training pipeline as exp002

Usage:
  # Quick test (1K games, ~5min):
  python experiments_history/exp003_lichess_history.py \
    --num-games 1000 --epochs 3 \
    --output-path outputs/exp_history_003/best.pt

  # Full training (200K high-Elo games):
  python experiments_history/exp003_lichess_history.py \
    --num-games 200000 --min-elo 2000 --epochs 3 \
    --output-path outputs/exp_history_003/best.pt

  # Play against Stockfish:
  python experiments_history/exp003_lichess_history.py \
    --play outputs/exp_history_003/best.pt --sf-elo 1500 --num-games 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

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


# ── Lichess HF Data Loading ────────────────────────────────────────────────


_COMMENT_RE = re.compile(r"\{[^}]*\}")
_MOVENUM_RE = re.compile(r"\d+\.+\s*")
_RESULT_RE = re.compile(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$")


def movetext_to_uci(movetext: str) -> list[str] | None:
    """Convert SAN movetext '1. e4 e5 2. Nf3 Nc6...' to list of UCI moves.

    Returns None if the game is corrupt or too short (<10 moves).
    """
    # Strip comments like { [%eval 0.34] [%clk 0:05:00] }
    text = _COMMENT_RE.sub("", movetext)
    # Strip move numbers (1. 1... 12.)
    text = _MOVENUM_RE.sub("", text)
    # Strip result
    text = _RESULT_RE.sub("", text)

    tokens = text.split()
    if len(tokens) < 10:
        return None

    board = chess.Board()
    uci_moves: list[str] = []
    for san in tokens:
        san = san.strip()
        if not san:
            continue
        try:
            move = board.parse_san(san)
            uci_moves.append(move.uci())
            board.push(move)
        except (chess.IllegalMoveError, chess.AmbiguousMoveError,
                chess.InvalidMoveError, ValueError):
            # Corrupt game — return what we have if enough
            break

    return uci_moves if len(uci_moves) >= 10 else None


def stream_lichess_games(
    num_games: int,
    min_elo: int = 2000,
    min_moves: int = 10,
    termination: str = "Normal",
    year: int = 2024,
    month: int = 1,
) -> list[list[str]]:
    """Stream high-quality games from Lichess/standard-chess-games on HuggingFace.

    Targets a specific year/month partition to avoid slow full-dataset metadata
    resolution. The dataset is hive-partitioned: data/year=YYYY/month=MM/*.parquet

    Returns list of games, each game is a list of UCI move strings.
    """
    from datasets import load_dataset

    data_pattern = f"data/year={year}/month={month:02d}/*.parquet"
    log(f"Streaming from Lichess/standard-chess-games ({data_pattern}, "
        f"min_elo={min_elo}, target={num_games:,} games)...")

    ds = load_dataset(
        "Lichess/standard-chess-games",
        split="train",
        streaming=True,
        data_files={"train": data_pattern},
    )

    games: list[list[str]] = []
    checked = 0
    skipped_elo = 0
    skipped_term = 0
    skipped_parse = 0
    t0 = time.time()

    for row in ds:
        checked += 1

        # Filter by Elo
        w_elo = row.get("WhiteElo", 0) or 0
        b_elo = row.get("BlackElo", 0) or 0
        if w_elo < min_elo or b_elo < min_elo:
            skipped_elo += 1
            if checked % 50000 == 0:
                elapsed = time.time() - t0
                rate = checked / elapsed
                log(f"  Scanned {checked:,} rows, found {len(games):,}/{num_games:,} games "
                    f"({rate:.0f} rows/s, skip_elo={skipped_elo:,})")
            continue

        # Filter by termination (skip time forfeits, abandonments)
        term = row.get("Termination", "")
        if termination and term != termination:
            skipped_term += 1
            continue

        # Parse movetext
        movetext = row.get("movetext", "")
        if not movetext:
            skipped_parse += 1
            continue

        uci_moves = movetext_to_uci(movetext)
        if uci_moves is None:
            skipped_parse += 1
            continue

        games.append(uci_moves)

        if len(games) % 10000 == 0:
            elapsed = time.time() - t0
            rate = checked / elapsed
            log(f"  Collected {len(games):,}/{num_games:,} games "
                f"(scanned {checked:,}, {rate:.0f} rows/s)")

        if len(games) >= num_games:
            break

    elapsed = time.time() - t0
    total_moves = sum(len(g) for g in games)
    log(f"Collected {len(games):,} games ({total_moves:,} moves, avg {total_moves/max(len(games),1):.0f}/game)")
    log(f"  Scanned {checked:,} rows in {elapsed:.0f}s "
        f"(skip: elo={skipped_elo:,}, term={skipped_term:,}, parse={skipped_parse:,})")

    return games


# ── Data Pipeline (from exp002, proven correct) ────────────────────────────


@dataclass(slots=True)
class GameExample:
    inputs: list[int]
    targets: list[int]
    legal_target_ids: list[list[int]]


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
            legal_ids = []
            for lm in board.legal_moves:
                uci = lm.uci()
                if uci in vocab.token_to_id:
                    legal_ids.append(vocab.token_to_id[uci])
            if not legal_ids:
                break
            legal_target_ids.append(legal_ids)
            targets.append(vocab.token_to_id.get(uci_move, vocab.pad_id))
            inputs.append(vocab.token_to_id.get(uci_move, vocab.pad_id))
            try:
                board.push_uci(uci_move)
            except (chess.IllegalMoveError, ValueError):
                break

        targets.append(vocab.eos_id)
        legal_target_ids.append([vocab.eos_id])

        sequence_inputs = inputs[:-1]
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


# ── Model Architecture: LLaMA-style Decoder (from exp002) ──────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    B, H, T, D = x.shape
    x = x.reshape(B, H, T, D // 2, 2)
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)
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
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, rope_freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        drop = self.dropout if self.training else 0.0
        if mask is not None:
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            combined = causal.unsqueeze(0) | ~padding_mask
            attn_mask = torch.zeros_like(combined, dtype=q.dtype)
            attn_mask.masked_fill_(combined, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, dim_ff: int) -> None:
        super().__init__()
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
        self.output.weight = self.token_embedding.weight

        head_dim = d_model // nhead
        rope_freqs = precompute_rope_freqs(head_dim, max_seq_len)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
    legal_logits = apply_legal_mask(logits, legal_mask)
    B_T, V = legal_logits.reshape(-1, legal_logits.size(-1)).shape
    flat_logits = legal_logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)

    if label_smoothing <= 0.0:
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=pad_id)

    log_probs = F.log_softmax(flat_logits, dim=-1)
    non_pad = flat_targets != pad_id

    if not non_pad.any():
        return torch.tensor(0.0, device=logits.device)

    nll = F.nll_loss(log_probs, flat_targets, ignore_index=pad_id, reduction="none")

    safe_log_probs = log_probs.clone()
    safe_log_probs[safe_log_probs == float("-inf")] = 0.0
    n_legal = legal_mask.reshape(-1, V).sum(dim=-1).float().clamp(min=1)
    smooth_loss = -safe_log_probs.sum(dim=-1) / n_legal

    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    return loss[non_pad].mean()


# ── Training Config ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class TrainConfig:
    output_path: Path
    # Data source
    num_games: int = 200_000
    min_elo: int = 2000
    min_moves: int = 10
    year: int = 2024
    month: int = 1
    # Architecture
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 24
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    # Training
    batch_size: int = 16
    gradient_accumulation: int = 4
    epochs: int = 3
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    gradient_checkpointing: bool = True
    # Data
    seed: int = 7
    eval_split: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Optional: resume from checkpoint
    resume_from: Path | None = None
    # Optional: load from local PGN cache instead of streaming
    cache_pgn: Path | None = None


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
    model.eval()
    token_ids = [vocab.bos_id]
    for m in move_history:
        tid = vocab.token_to_id.get(m)
        if tid is not None:
            token_ids.append(tid)
    if len(token_ids) > max_seq_len:
        token_ids = [vocab.bos_id] + token_ids[-(max_seq_len - 1):]

    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(inputs, dtype=torch.bool)

    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs, mask)

    last_logits = logits[0, -1].float()

    legal_ids = []
    for lm in board.legal_moves:
        tid = vocab.token_to_id.get(lm.uci())
        if tid is not None:
            legal_ids.append(tid)

    if not legal_ids:
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
) -> tuple[str, float]:
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

    # ── Load data: stream from Lichess HF or local PGN cache ──
    if config.cache_pgn and config.cache_pgn.exists():
        log(f"Loading from local PGN cache: {config.cache_pgn}")
        all_games = _read_games_from_pgn(config.cache_pgn, max_games=config.num_games)
    else:
        all_games = stream_lichess_games(
            num_games=config.num_games,
            min_elo=config.min_elo,
            min_moves=config.min_moves,
            year=config.year,
            month=config.month,
        )

    if not all_games:
        raise ValueError("No games collected. Check network and filters.")

    train_games, eval_games = split_games(all_games, config.eval_split, config.seed)
    log(f"Games: {len(train_games):,} train, {len(eval_games):,} eval")

    # Build examples (this is the bottleneck — legal move generation is slow)
    log("Building training examples (legal move mask generation)...")
    t0 = time.time()
    train_examples = build_examples(train_games, vocab, config.max_seq_len)
    eval_examples = build_examples(eval_games, vocab, config.max_seq_len)
    elapsed = time.time() - t0
    log(f"Built {len(train_examples):,} train, {len(eval_examples):,} eval examples in {elapsed:.0f}s")

    if not train_examples:
        raise ValueError("No training examples produced.")

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

    start_epoch = 1
    global_step = 0

    # Resume from checkpoint if specified
    if config.resume_from and config.resume_from.exists():
        log(f"Resuming from {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        log(f"  Resumed at epoch {start_epoch}, step {global_step}")

    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation
    log(f"Total steps: {total_steps:,} (warmup: {config.warmup_steps})")

    # ── Training ──
    best_eval_loss = math.inf
    best_checkpoint: dict | None = None
    t0_total = time.time()
    lr = config.learning_rate

    for epoch in range(start_epoch, config.epochs + 1):
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

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_t0
                rate = (batch_idx + 1) / elapsed
                log(f"  batch {batch_idx+1}/{num_batches} | "
                    f"loss={loss.item()*config.gradient_accumulation:.4f} | "
                    f"{rate:.1f} batch/s | "
                    f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
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
                "optimizer_state_dict": optimizer.state_dict(),
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
                "train_config": {
                    "num_games": config.num_games,
                    "min_elo": config.min_elo,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "gradient_accumulation": config.gradient_accumulation,
                    "learning_rate": config.learning_rate,
                    "data_source": "Lichess/standard-chess-games",
                },
                "best_eval_loss": eval_loss,
                "best_top1": eval_metrics["top1"],
                "best_top3": eval_metrics["top3"],
                "epoch": epoch,
                "global_step": global_step,
                "total_train_moves": total_moves,
            }
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_checkpoint, config.output_path)
            log(f"  ★ New best eval loss: {eval_loss:.4f} — saved to {config.output_path}")

    total_time = time.time() - t0_total

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint produced.")

    log(f"Training complete in {total_time:.0f}s")
    log(f"  {n_params/1e6:.1f}M params | {len(train_games):,} games | {total_moves:,} moves | "
        f"best_loss={best_eval_loss:.4f} | "
        f"top1={100*best_checkpoint['best_top1']:.2f}% "
        f"top3={100*best_checkpoint['best_top3']:.2f}%")
    return config.output_path


# ── Utility: PGN reader for local cache ─────────────────────────────────────


def _read_games_from_pgn(path: Path, max_games: int | None = None) -> list[list[str]]:
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


# ── CLI ─────────────────────────────────────────────────────────────────────


def find_stockfish() -> str:
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
    import shutil
    sf = shutil.which("stockfish")
    if sf:
        return sf
    raise FileNotFoundError("Stockfish not found. Install or set path.")


def load_model_for_play(
    checkpoint_path: Path, device: torch.device,
) -> tuple[DeepMoveHistoryTransformer, MoveVocabulary]:
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
        dropout=0.0,
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, vocab


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="exp003: Deep move-history transformer trained on Lichess high-Elo games"
    )
    # Data
    p.add_argument("--num-games", type=int, default=200_000,
                    help="Number of high-Elo games to stream from Lichess HF")
    p.add_argument("--min-elo", type=int, default=2000,
                    help="Minimum Elo for both players")
    p.add_argument("--year", type=int, default=2024,
                    help="Year partition to stream from (default: 2024)")
    p.add_argument("--month", type=int, default=1,
                    help="Month partition to stream from (default: 1)")
    p.add_argument("--cache-pgn", type=Path,
                    help="Local PGN file to use instead of streaming")
    # Training
    p.add_argument("--output-path", type=Path, help="Where to save best checkpoint")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-split", type=float, default=0.02)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--resume-from", type=Path,
                    help="Resume training from a checkpoint")
    # Gameplay / Elo
    p.add_argument("--play", type=Path, help="Checkpoint to play with (skip training)")
    p.add_argument("--sf-elo", type=int, default=1500)
    p.add_argument("--num-elo-games", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    if args.play:
        model, vocab = load_model_for_play(args.play, device)
        sf_path = find_stockfish()
        log(f"Playing {args.num_elo_games} games vs SF{args.sf_elo} (temp={args.temperature})")
        summary = run_elo_eval(
            model, vocab, device, sf_path, args.sf_elo,
            args.num_elo_games, args.max_seq_len, args.temperature,
        )
        out_dir = args.play.parent
        results_path = out_dir / f"elo_eval_sf{args.sf_elo}.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"Results saved: {results_path}")
        return

    # Training mode
    if not args.output_path:
        build_parser().error("--output-path required for training")

    config = TrainConfig(
        output_path=args.output_path.resolve(),
        num_games=args.num_games,
        min_elo=args.min_elo,
        year=args.year,
        month=args.month,
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
        eval_split=args.eval_split,
        device=args.device,
        resume_from=args.resume_from,
        cache_pgn=args.cache_pgn,
    )
    train(config)


if __name__ == "__main__":
    main()
