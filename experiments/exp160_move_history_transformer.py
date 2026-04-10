"""exp160: Move-History Transformer — predict next move from game move sequence.

A causal decoder-only transformer that takes a sequence of UCI moves
(<bos> e2e4 e7e5 g1f3 ...) and predicts the next move. Uses the compact
1968-move vocabulary (geometrically reachable only) instead of 5504.

The model must implicitly reconstruct the board state from move history,
which is a harder task but captures game-level context (opening lines,
transpositions, tactical patterns).

Usage:
  # Generate training PGNs first (SF vs SF):
  python experiments/exp160_move_history_transformer.py \
    --train-pgn outputs/lichess_sf_games.pgn \
    --output-path outputs/exp160_move_history/best.pt

  # Quick test with fewer games:
  python experiments/exp160_move_history_transformer.py \
    --train-pgn outputs/lichess_sf_games.pgn \
    --output-path outputs/exp160_move_history/best.pt \
    --train-max-games 100 --epochs 4
"""
from __future__ import annotations

import argparse
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
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Use compact vocab from move_vocab.py (1968 moves vs legacy 5504)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from move_vocab import COMPACT_IDX_TO_UCI, COMPACT_UCI_TO_IDX, COMPACT_VOCAB_SIZE

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")


@dataclass(slots=True)
class MoveVocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    bos_id: int
    eos_id: int

    @classmethod
    def build(cls) -> "MoveVocabulary":
        # Use compact 1968-move vocab from move_vocab.py
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


def build_examples(games: Iterable[list[str]], vocab: MoveVocabulary, max_seq_len: int) -> list[GameExample]:
    examples: list[GameExample] = []

    for moves in games:
        board = chess.Board()
        inputs = [vocab.bos_id]
        targets: list[int] = []
        legal_target_ids: list[list[int]] = []

        for uci_move in moves:
            legal_target_ids.append([vocab.token_to_id[move.uci()] for move in board.legal_moves])
            targets.append(vocab.token_to_id[uci_move])
            inputs.append(vocab.token_to_id[uci_move])
            board.push_uci(uci_move)

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


def collate_batch(batch: list[GameExample], pad_id: int, vocab_size: int) -> dict[str, object]:
    batch_size = len(batch)
    max_len = max(len(item.inputs) for item in batch)

    inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    # Precompute dense legal mask on CPU — avoids thousands of GPU tensor ops per batch
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


class MoveHistoryTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        _, seq_len = inputs.shape
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand_as(inputs)
        hidden = self.token_embedding(inputs) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=inputs.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=~attention_mask)
        return self.output(hidden)


def apply_legal_mask(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Apply precomputed dense legal mask. legal_mask is bool (B, T, V) on same device."""
    return logits.masked_fill(~legal_mask, float("-inf"))


def masked_cross_entropy(logits: Tensor, targets: Tensor, legal_mask: Tensor, pad_id: int) -> Tensor:
    legal_logits = apply_legal_mask(logits, legal_mask)
    return F.cross_entropy(legal_logits.reshape(-1, legal_logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)


@dataclass(slots=True)
class TrainConfig:
    train_pgn: Path
    eval_pgn: Path | None
    output_path: Path
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_seq_len: int = 256
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    seed: int = 7
    train_max_games: int | None = None
    eval_max_games: int | None = None
    eval_split: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    label_smoothing: float = 0.05


def split_games(games: list[list[str]], eval_split: float, seed: int) -> tuple[list[list[str]], list[list[str]]]:
    shuffled = games[:]
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, shuffled
    eval_count = max(1, int(len(shuffled) * eval_split))
    eval_count = min(eval_count, len(shuffled) - 1)
    return shuffled[eval_count:], shuffled[:eval_count]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pad_id: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    top3_correct = 0
    total_positions = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["attention_mask"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(inputs, mask)

            loss = masked_cross_entropy(
                logits=logits.float(),
                targets=targets,
                legal_mask=legal_mask,
                pad_id=pad_id,
            )
            total_loss += loss.item()
            total_batches += 1

            # Accuracy on non-pad positions
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


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def train(config: TrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    vocab = MoveVocabulary.build()
    log(f"Vocab: {vocab.size} tokens ({vocab.size - len(SPECIAL_TOKENS)} moves + {len(SPECIAL_TOKENS)} special)")

    train_games = read_games_from_pgn(config.train_pgn, max_games=config.train_max_games)
    if config.eval_pgn is not None:
        eval_games = read_games_from_pgn(config.eval_pgn, max_games=config.eval_max_games)
    else:
        train_games, eval_games = split_games(train_games, config.eval_split, config.seed)

    log(f"Games: {len(train_games)} train, {len(eval_games)} eval")

    train_examples = build_examples(train_games, vocab, config.max_seq_len)
    eval_examples = build_examples(eval_games, vocab, config.max_seq_len)
    if not train_examples:
        raise ValueError("No training examples were produced from the PGN input.")
    if not eval_examples:
        raise ValueError("No eval examples were produced from the PGN input.")

    log(f"Examples: {len(train_examples)} train, {len(eval_examples)} eval")

    # Count total move positions for stats
    total_train_moves = sum(len(e.targets) for e in train_examples)
    avg_moves = total_train_moves / len(train_games) if train_games else 0
    log(f"Total train moves: {total_train_moves:,} (avg {avg_moves:.0f}/game)")

    train_dataset = MoveHistoryDataset(train_examples)
    eval_dataset = MoveHistoryDataset(eval_examples)
    collate = lambda batch: collate_batch(batch, vocab.pad_id, vocab.size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)

    device = torch.device(config.device)
    model = MoveHistoryTransformer(
        vocab_size=vocab.size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params/1e6:.1f}M params, {config.num_layers}L/{config.d_model}d/{config.nhead}H")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler("cuda")

    best_eval_loss = math.inf
    best_checkpoint: dict[str, object] | None = None
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_batches = 0
        epoch_t0 = time.time()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["attention_mask"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(inputs, mask)
                loss = masked_cross_entropy(
                    logits=logits.float(),
                    targets=targets,
                    legal_mask=legal_mask,
                    pad_id=vocab.pad_id,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_batches += 1

        epoch_time = time.time() - epoch_t0
        train_loss = total_train_loss / max(total_batches, 1)
        eval_metrics = evaluate(model, eval_loader, device, vocab.pad_id)
        eval_loss = eval_metrics["loss"]

        log(f"epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} | "
            f"top1={100*eval_metrics['top1']:.2f}% top3={100*eval_metrics['top3']:.2f}% | "
            f"{epoch_time:.0f}s")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab": vocab.id_to_token,
                "config": asdict(config),
                "best_eval_loss": eval_loss,
                "best_top1": eval_metrics["top1"],
                "best_top3": eval_metrics["top3"],
            }
            log(f"  ★ New best eval loss: {eval_loss:.4f}")

    total_time = time.time() - t0

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, config.output_path)
    log(f"Saved: {config.output_path}")
    log(f"vocab_size={vocab.size} ({vocab.size - len(SPECIAL_TOKENS)} moves)")
    log(f"train_examples={len(train_examples)}, eval_examples={len(eval_examples)}")
    log(f"best_eval_loss={best_eval_loss:.4f}, "
        f"best_top1={100*best_checkpoint['best_top1']:.2f}%, "
        f"best_top3={100*best_checkpoint['best_top3']:.2f}%")
    log(f"Total time: {total_time:.0f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move-history transformer experiment with per-ply legal masking.")
    parser.add_argument("--train-pgn", type=Path, required=True)
    parser.add_argument("--eval-pgn", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-max-games", type=int)
    parser.add_argument("--eval-max-games", type=int)
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        train_pgn=args.train_pgn.resolve(),
        eval_pgn=args.eval_pgn.resolve() if args.eval_pgn else None,
        output_path=args.output_path.resolve(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seed=args.seed,
        train_max_games=args.train_max_games,
        eval_max_games=args.eval_max_games,
        eval_split=args.eval_split,
        device=args.device,
        label_smoothing=args.label_smoothing,
    )
    train(config)


if __name__ == "__main__":
    main()