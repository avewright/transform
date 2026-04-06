"""exp152: Trajectory-Level Attention for Game Value Learning

Core idea: Instead of evaluating each chess position in isolation, attend across
multiple board states from the same game to learn richer value predictions.

The current value head sees: one board → WDL prediction.
This experiment adds: a sequence of boards from a game → per-position WDL predictions,
where each position can attend to previous positions via causal self-attention.

Architecture (two-level transformer):
  Level 1 — Position Encoder (frozen 204M backbone):
    Each board position → CLS embedding vector (1024d).
    Pre-computed once and cached for training efficiency.

  Level 2 — Trajectory Transformer (trainable, ~8M params):
    CLS embeddings from T positions in a game → causal self-attention.
    Each position attends to all PREVIOUS positions (+ itself).
    Ply embeddings provide temporal ordering.
    Per-position value head predicts WDL.

Hypothesis: Positions in context of their game history yield better value
predictions than isolated positions. The trajectory transformer should learn:
  - That early advantages propagate to wins
  - Which positions were critical turning points
  - Temporal patterns (opening structure → midgame pressure → endgame conversion)

Phases:
  1. Generate game trajectories (model vs SF at various ELOs)
  2. Pre-compute position embeddings using frozen 204M model
  3. Train trajectory value model on cached embeddings
  4. Evaluate: trajectory value accuracy vs single-position baseline

Usage:
  python experiments/exp152_trajectory_value.py --phase generate --num-games 200
  python experiments/exp152_trajectory_value.py --phase embed
  python experiments/exp152_trajectory_value.py --phase train --epochs 20
  python experiments/exp152_trajectory_value.py --phase eval
  python experiments/exp152_trajectory_value.py --all --num-games 200 --epochs 20
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import ChessTransformerConfig, build_model
from move_vocab import VOCAB_SIZE, move_to_index, legal_move_mask

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp152_trajectory_value"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── Stockfish resolution ──

def resolve_sf():
    candidates = [
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Stockfish not found. Tried: " + ", ".join(str(c) for c in candidates))


# ── Opening book ──

OPENINGS = [
    ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["c2c4", "e7e5"], ["g1f3", "d7d5"],
    ["e2e4", "c7c5"], ["d2d4", "g8f6"], ["e2e4", "e7e6"], ["d2d4", "d7d6"],
    ["e2e4", "c7c6"], ["g1f3", "g8f6"], ["c2c4", "c7c5"], ["e2e4", "g7g6"],
    ["d2d4", "e7e6"], ["c2c4", "g8f6"], ["b1c3", "d7d5"], ["g2g3", "d7d5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "c7c5", "g1f3"],
    ["d2d4", "g8f6", "c2c4"],
]


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Generate game trajectories
# ═══════════════════════════════════════════════════════════════════════

def generate_games(num_games: int, sf_elos: list[int], sims: int = 100,
                   ply_cap: int = 300) -> list[dict]:
    """Play games between the transformer model and Stockfish.

    Each game records:
      - fens: list of FEN strings for every position (both sides' moves)
      - outcome: 0=white_wins, 1=draw, 2=black_wins (absolute, not side-relative)
      - model_color: chess.WHITE or chess.BLACK
      - sf_elo: the SF ELO used
      - num_plies: total game length
    """
    from uci_engine import MCTSSearch, SyzygyProbe

    # Load model
    ckpt_path = ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt"
    if not ckpt_path.exists():
        # Try other common locations
        for alt in ["outputs/exp143_204m_lowlr/best_model.pt",
                     "outputs/exp142_204m_10m/best_model.pt"]:
            alt_path = ROOT / alt
            if alt_path.exists():
                ckpt_path = alt_path
                break

    log(f"Loading model from {ckpt_path}")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    syzygy = SyzygyProbe()
    mcts = MCTSSearch(
        model, DEVICE, syzygy,
        c_puct=2.5, batch_size=8,
        root_noise_alpha=0.3, root_noise_frac=0.25,
        use_fp16=True, use_transpositions=True,
    )

    sf_path = resolve_sf()
    games = []
    elo_idx = 0

    for game_i in range(num_games):
        sf_elo = sf_elos[game_i % len(sf_elos)]
        model_color = chess.WHITE if game_i % 2 == 0 else chess.BLACK
        opening = OPENINGS[game_i % len(OPENINGS)]

        # Start SF engine
        engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

        board = chess.Board()
        for uci in opening:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

        # Record all FENs (including the opening position)
        fens = [board.fen()]
        mcts.new_game()

        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
            if board.turn == model_color:
                tb = mcts.syzygy.get_move(board)
                if tb:
                    board.push(tb)
                else:
                    move, _ = mcts.search(board, max_sims=sims)
                    board.push(move)
                mcts.root = None
            else:
                sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
                if sf_move not in board.legal_moves:
                    sf_move = next(iter(board.legal_moves))
                board.push(sf_move)
            fens.append(board.fen())

        engine.quit()

        # Determine game outcome (absolute: 0=W wins, 1=draw, 2=B wins)
        outcome_obj = board.outcome(claim_draw=True)
        if outcome_obj is None or outcome_obj.winner is None:
            outcome = 1  # draw
        elif outcome_obj.winner == chess.WHITE:
            outcome = 0  # white wins
        else:
            outcome = 2  # black wins

        games.append({
            "fens": fens,
            "outcome": outcome,
            "model_color": model_color,
            "sf_elo": sf_elo,
            "num_plies": len(board.move_stack),
        })

        # Log progress
        result_str = ["W_WIN", "DRAW", "B_WIN"][outcome]
        side_str = "W" if model_color == chess.WHITE else "B"
        log(f"  Game {game_i+1}/{num_games}: model={side_str} vs SF{sf_elo} "
            f"→ {result_str} ({len(board.move_stack)}ply, {len(fens)} positions)")

    # Cleanup
    del model, mcts
    gc.collect()
    torch.cuda.empty_cache()

    return games


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Pre-compute position embeddings from frozen backbone
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(games: list[dict], batch_size: int = 64) -> list[dict]:
    """Run each position through the frozen 204M model, extract CLS token.

    Returns games with 'embeddings' key: list of (1024,) tensors per position.
    """
    # Load model
    ckpt_path = ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt"
    if not ckpt_path.exists():
        for alt in ["outputs/exp143_204m_lowlr/best_model.pt",
                     "outputs/exp142_204m_10m/best_model.pt"]:
            alt_path = ROOT / alt
            if alt_path.exists():
                ckpt_path = alt_path
                break

    log(f"Loading backbone from {ckpt_path}")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    # Collect all positions across all games
    all_fens = []
    game_indices = []  # (game_idx, position_idx)
    for gi, game in enumerate(games):
        for pi, fen in enumerate(game["fens"]):
            all_fens.append(fen)
            game_indices.append((gi, pi))

    log(f"Extracting embeddings for {len(all_fens)} positions across {len(games)} games")
    all_embeddings = []

    for start in range(0, len(all_fens), batch_size):
        end = min(start + batch_size, len(all_fens))
        batch_fens = all_fens[start:end]
        boards = [chess.Board(fen) for fen in batch_fens]

        board_input = batch_boards_to_fused_token_ids(boards, DEVICE)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Run through board encoder + input projection + transformer
            hidden = model.input_proj(model.encoder(board_input))
            B = hidden.shape[0]
            hidden = torch.cat([model.cls_token.expand(B, -1, -1), hidden], dim=1)
            if model.pos_embed is not None:
                hidden = hidden + model.pos_embed
            hidden = model.norm(model.transformer(hidden))
            cls_tokens = hidden[:, 0, :].float().cpu()  # (B, 1024)

        all_embeddings.append(cls_tokens)

        if (start // batch_size) % 50 == 0:
            log(f"  Embedded {end}/{len(all_fens)} positions")

    all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_positions, 1024)

    # Reassemble into per-game embedding lists
    for gi in range(len(games)):
        games[gi]["embeddings"] = []
    for idx, (gi, pi) in enumerate(game_indices):
        games[gi]["embeddings"].append(all_embeddings[idx])
    for gi in range(len(games)):
        games[gi]["embeddings"] = torch.stack(games[gi]["embeddings"])  # (T_i, 1024)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    log(f"Embeddings extracted. Shapes: {[g['embeddings'].shape for g in games[:5]]}")
    return games


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Value Model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryConfig:
    embed_dim: int = 1024       # matches backbone CLS dim
    traj_hidden: int = 512      # trajectory transformer hidden dim
    traj_layers: int = 6        # trajectory transformer layers
    traj_heads: int = 8         # attention heads
    traj_ffn_ratio: int = 4     # FFN expansion
    dropout: float = 0.1
    max_ply: int = 400          # max game length (half-moves)
    value_hidden: int = 256     # value head hidden
    window_size: int = 32       # training window size (positions per sample)


class TrajectoryValueModel(nn.Module):
    """Causal trajectory transformer for game-level value prediction.

    Takes a sequence of pre-computed position embeddings (from frozen backbone)
    and predicts WDL for each position using causal self-attention over the
    game trajectory.

    Key design:
      - Causal masking: each position attends only to past positions (+ itself)
      - Ply embeddings: learned temporal encoding for move number
      - Per-position value head: each position independently predicts WDL
      - At inference time: feed game history → get value for current position
    """

    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.config = config

        # Project backbone CLS tokens to trajectory transformer dim
        self.input_proj = nn.Linear(config.embed_dim, config.traj_hidden)

        # Ply (temporal position) embeddings
        self.ply_embed = nn.Embedding(config.max_ply, config.traj_hidden)

        # Causal trajectory transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.traj_hidden,
            nhead=config.traj_heads,
            dim_feedforward=config.traj_hidden * config.traj_ffn_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.traj_layers
        )
        self.norm = nn.LayerNorm(config.traj_hidden)

        # Per-position value head (WDL classification)
        self.value_head = nn.Sequential(
            nn.Linear(config.traj_hidden, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 3),  # [P(W wins), P(draw), P(B wins)]
        )

    def forward(
        self,
        embeddings: torch.Tensor,      # (B, T, embed_dim) pre-computed CLS tokens
        ply_indices: torch.Tensor,      # (B, T) half-move numbers
        padding_mask: torch.Tensor | None = None,  # (B, T) True=padded
    ) -> torch.Tensor:
        """
        Returns:
            value_logits: (B, T, 3) per-position WDL predictions
        """
        B, T, _ = embeddings.shape

        # Project + add temporal position
        x = self.input_proj(embeddings)  # (B, T, traj_hidden)
        ply_clamped = ply_indices.clamp(0, self.config.max_ply - 1)
        x = x + self.ply_embed(ply_clamped)

        # Causal mask: each position attends only to itself and past positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=embeddings.device, dtype=embeddings.dtype
        )

        # Run trajectory transformer with causal masking
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # Per-position value predictions
        value_logits = self.value_head(x)  # (B, T, 3)
        return value_logits

    def get_attention_weights(
        self,
        embeddings: torch.Tensor,
        ply_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract attention weights from each layer for visualization."""
        B, T, _ = embeddings.shape
        x = self.input_proj(embeddings)
        ply_clamped = ply_indices.clamp(0, self.config.max_ply - 1)
        x = x + self.ply_embed(ply_clamped)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=embeddings.device, dtype=embeddings.dtype
        )

        attn_weights = []
        for layer in self.transformer.layers:
            # Manual forward to capture attention weights
            # Self-attention with weight capture
            x2 = layer.norm1(x)
            x2, w = layer.self_attn(
                x2, x2, x2, attn_mask=causal_mask, need_weights=True,
                average_attn_weights=False,
            )
            attn_weights.append(w.detach().cpu())
            x = x + x2
            x = x + layer._ff_block(layer.norm2(x))

        return attn_weights  # list of (B, heads, T, T) per layer


# ═══════════════════════════════════════════════════════════════════════
# Single-Position Baseline (same capacity, no trajectory context)
# ═══════════════════════════════════════════════════════════════════════

class SinglePositionBaseline(nn.Module):
    """Baseline: predict WDL from each position independently (no trajectory).

    Capacity-matched to trajectory model via wider hidden layers.
    The trajectory model spends its extra params on cross-position attention;
    this baseline spends equivalent budget on deeper per-position processing.
    This isolates whether trajectory STRUCTURE helps, not just more params.
    """

    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.config = config
        # Match trajectory model param count (~20M) using wide MLP
        # Trajectory has ~19.8M: input_proj(524K) + ply_embed(200K) +
        #   6 transformer layers(~18M) + value_head(132K)
        # Baseline: 1024 → 2048 → 2048 → 2048 → 2048 → 256 → 3 ≈ 18.4M
        wide = config.traj_hidden * 4   # 2048 at default traj_hidden=512
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, wide),
            nn.GELU(),
            nn.LayerNorm(wide),
            nn.Linear(wide, wide),
            nn.GELU(),
            nn.LayerNorm(wide),
            nn.Linear(wide, wide),
            nn.GELU(),
            nn.LayerNorm(wide),
            nn.Linear(wide, wide),
            nn.GELU(),
            nn.LayerNorm(wide),
            nn.Linear(wide, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 3),
        )

    def forward(
        self,
        embeddings: torch.Tensor,      # (B, T, embed_dim)
        ply_indices: torch.Tensor,      # (B, T) — ignored
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(embeddings)  # (B, T, 3)


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    """Yields fixed-length windows from game trajectories.

    Each sample:
      - embeddings: (window_size, 1024) CLS tokens
      - ply_indices: (window_size,) half-move numbers
      - value_targets: (window_size,) outcome label for each position
      - mask: (window_size,) True for padded positions
    """

    def __init__(self, games: list[dict], window_size: int = 32,
                 stride: int | None = None, augment: bool = True):
        self.window_size = window_size
        self.stride = stride or (window_size // 2)
        self.augment = augment
        self.windows = []

        for game in games:
            embs = game["embeddings"]  # (T, 1024)
            T = embs.shape[0]
            outcome = game["outcome"]  # 0=W, 1=D, 2=B

            # Slide window across game
            for start in range(0, max(1, T - window_size // 2), self.stride):
                end = min(start + window_size, T)
                self.windows.append({
                    "embeddings": embs[start:end],
                    "ply_start": start,
                    "length": end - start,
                    "outcome": outcome,
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        T = w["length"]
        W = self.window_size

        # Pad to window_size
        embs = torch.zeros(W, w["embeddings"].shape[-1])
        embs[:T] = w["embeddings"][:T]

        ply_indices = torch.arange(w["ply_start"], w["ply_start"] + W, dtype=torch.long)
        value_targets = torch.full((W,), w["outcome"], dtype=torch.long)

        # Padding mask: True = padded (should be ignored)
        mask = torch.zeros(W, dtype=torch.bool)
        mask[T:] = True

        return {
            "embeddings": embs,
            "ply_indices": ply_indices,
            "value_targets": value_targets,
            "mask": mask,
        }


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Training
# ═══════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    lr: float,
    model_name: str,
):
    """Train a value model (trajectory or baseline) on game trajectories."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine LR schedule with warmup
    total_steps = epochs * len(train_loader)
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_eval_acc = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_valid = 0

        for batch_i, batch in enumerate(train_loader):
            embs = batch["embeddings"].to(DEVICE)          # (B, W, 1024)
            plys = batch["ply_indices"].to(DEVICE)          # (B, W)
            targets = batch["value_targets"].to(DEVICE)     # (B, W)
            mask = batch["mask"].to(DEVICE)                 # (B, W)

            logits = model(embs, plys, mask)                # (B, W, 3)

            # Flatten, ignoring padded positions
            valid = ~mask
            flat_logits = logits[valid]                     # (N_valid, 3)
            flat_targets = targets[valid]                   # (N_valid,)

            loss = F.cross_entropy(flat_logits, flat_targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * flat_targets.shape[0]
            preds = flat_logits.argmax(dim=-1)
            total_correct += (preds == flat_targets).sum().item()
            total_valid += flat_targets.shape[0]

        train_loss = total_loss / max(1, total_valid)
        train_acc = total_correct / max(1, total_valid)

        # Evaluate
        eval_loss, eval_acc, eval_details = evaluate_model(model, eval_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        is_best = eval_acc > best_eval_acc
        if is_best:
            best_eval_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        log(f"  [{model_name}] Epoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.3f}"
            f"{' ★' if is_best else ''}")

    return best_state, best_eval_acc, history


@torch.no_grad()
def evaluate_model(model, eval_loader):
    """Evaluate model on game trajectories."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    # Per-class stats
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    # Per-ply-bucket stats (early/mid/late game)
    ply_correct = defaultdict(int)
    ply_total = defaultdict(int)

    for batch in eval_loader:
        embs = batch["embeddings"].to(DEVICE)
        plys = batch["ply_indices"].to(DEVICE)
        targets = batch["value_targets"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        logits = model(embs, plys, mask)
        valid = ~mask
        flat_logits = logits[valid]
        flat_targets = targets[valid]
        flat_plys = plys[valid]

        loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
        total_loss += loss.item()

        preds = flat_logits.argmax(dim=-1)
        correct = (preds == flat_targets)
        total_correct += correct.sum().item()
        total_valid += flat_targets.shape[0]

        # Per-class
        for c in range(3):
            cls_mask = flat_targets == c
            per_class_correct[c] += correct[cls_mask].sum().item()
            per_class_total[c] += cls_mask.sum().item()

        # Per-ply bucket: opening (0-20), midgame (20-60), endgame (60+)
        for label, lo, hi in [("opening", 0, 20), ("midgame", 20, 60), ("endgame", 60, 999)]:
            bucket = (flat_plys >= lo) & (flat_plys < hi)
            ply_correct[label] += correct[bucket].sum().item()
            ply_total[label] += bucket.sum().item()

    model.train()

    eval_loss = total_loss / max(1, total_valid)
    eval_acc = total_correct / max(1, total_valid)

    details = {
        "per_class": {
            c: per_class_correct[c] / max(1, per_class_total[c])
            for c in range(3)
        },
        "per_phase": {
            label: ply_correct[label] / max(1, ply_total[label])
            for label in ["opening", "midgame", "endgame"]
        },
    }

    return eval_loss, eval_acc, details


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Evaluation & Analysis
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_attention(model: TrajectoryValueModel, games: list[dict],
                      n_games: int = 5):
    """Visualize which positions the trajectory model attends to most.

    For each game, extracts attention weights and identifies which
    past positions receive the highest attention — these are the
    positions the model considers most informative for value prediction.
    """
    model.eval()
    analyses = []

    for gi, game in enumerate(games[:n_games]):
        embs = game["embeddings"].unsqueeze(0).to(DEVICE)  # (1, T, 1024)
        T = embs.shape[1]
        plys = torch.arange(T, dtype=torch.long, device=DEVICE).unsqueeze(0)

        attn_weights = model.get_attention_weights(embs, plys)

        # Average attention across heads and layers
        avg_attn = torch.stack(attn_weights).mean(dim=(0, 1, 2))  # (T, T)

        # For each position, find which past positions it attends to most
        analysis = {
            "game_idx": gi,
            "outcome": ["W_WIN", "DRAW", "B_WIN"][game["outcome"]],
            "num_positions": T,
            "attention_peaks": [],
        }

        # Look at last 5 positions — which earlier positions do they attend to?
        for pos in range(max(0, T - 5), T):
            attn_row = avg_attn[pos, :pos + 1]  # attention to past (causal)
            if len(attn_row) > 1:
                top_k = min(3, len(attn_row))
                top_vals, top_idxs = attn_row.topk(top_k)
                peaks = [(idx.item(), val.item()) for idx, val in zip(top_idxs, top_vals)]
                analysis["attention_peaks"].append({
                    "position": pos,
                    "attends_to": peaks,
                })

        analyses.append(analysis)
        log(f"  Game {gi}: {analysis['outcome']} ({T} positions), "
            f"top attention peaks at: {[p['attends_to'][0][0] for p in analysis['attention_peaks']]}")

    return analyses


@torch.no_grad()
def compare_value_calibration(
    traj_model: TrajectoryValueModel,
    baseline_model: SinglePositionBaseline,
    eval_games: list[dict],
):
    """Compare per-position value predictions between trajectory and baseline.

    For each game, computes:
      - Per-position WDL prediction from both models
      - How predictions evolve over the course of the game
      - Whether trajectory model detects critical moments earlier
    """
    traj_model.eval()
    baseline_model.eval()

    results = []

    for gi, game in enumerate(eval_games):
        embs = game["embeddings"].unsqueeze(0).to(DEVICE)
        T = embs.shape[1]
        plys = torch.arange(T, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Trajectory model predictions
        traj_logits = traj_model(embs, plys)  # (1, T, 3)
        traj_probs = F.softmax(traj_logits[0], dim=-1).cpu()  # (T, 3)

        # Baseline predictions
        base_logits = baseline_model(embs, plys)
        base_probs = F.softmax(base_logits[0], dim=-1).cpu()

        outcome = game["outcome"]

        # Confidence in correct outcome over time
        traj_conf = traj_probs[:, outcome].numpy()
        base_conf = base_probs[:, outcome].numpy()

        results.append({
            "game_idx": gi,
            "outcome": outcome,
            "num_positions": T,
            "traj_mean_conf": float(traj_conf.mean()),
            "base_mean_conf": float(base_conf.mean()),
            "traj_final_conf": float(traj_conf[-1]),
            "base_final_conf": float(base_conf[-1]),
        })

    # Aggregate
    traj_mean = np.mean([r["traj_mean_conf"] for r in results])
    base_mean = np.mean([r["base_mean_conf"] for r in results])
    log(f"\nCalibration comparison ({len(results)} games):")
    log(f"  Trajectory avg confidence in correct outcome: {traj_mean:.3f}")
    log(f"  Baseline   avg confidence in correct outcome: {base_mean:.3f}")
    log(f"  Trajectory advantage: {traj_mean - base_mean:+.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp152: Trajectory Value Learning")
    parser.add_argument("--phase", choices=["generate", "embed", "train", "eval"],
                        help="Run a single phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")

    # Generation args
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of games to generate")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--sf-elos", type=str, default="1200,1500,1800,1900",
                        help="SF ELO levels (comma-separated)")

    # Training args
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--traj-layers", type=int, default=6)
    parser.add_argument("--traj-hidden", type=int, default=512)

    # Quick test mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 20 games, 5 epochs")

    args = parser.parse_args()

    global LOG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "log.txt"

    sf_elos = [int(e) for e in args.sf_elos.split(",")]

    if args.quick:
        args.num_games = 20
        args.epochs = 5
        args.sims = 50

    run_phases = []
    if args.all:
        run_phases = ["generate", "embed", "train", "eval"]
    elif args.phase:
        run_phases = [args.phase]
    else:
        parser.print_help()
        return

    log("=" * 70)
    log("exp152: Trajectory-Level Attention for Game Value Learning")
    log(f"Phases: {run_phases}")
    log(f"Config: {args.num_games} games, {args.epochs} epochs, "
        f"window={args.window_size}, layers={args.traj_layers}, "
        f"hidden={args.traj_hidden}")
    log("=" * 70)

    games_path = OUTPUT_DIR / "games.pt"
    embedded_path = OUTPUT_DIR / "games_embedded.pt"

    # ── Phase 1: Generate ──
    if "generate" in run_phases:
        log("\n═══ Phase 1: Generating game trajectories ═══")
        games = generate_games(
            num_games=args.num_games,
            sf_elos=sf_elos,
            sims=args.sims,
        )

        # Save (without embeddings — FENs are lightweight)
        save_data = []
        for g in games:
            save_data.append({
                "fens": g["fens"],
                "outcome": g["outcome"],
                "model_color": g["model_color"],
                "sf_elo": g["sf_elo"],
                "num_plies": g["num_plies"],
            })
        torch.save(save_data, games_path)
        log(f"Saved {len(games)} games to {games_path}")

        # Stats
        outcomes = [g["outcome"] for g in games]
        log(f"  W-wins={outcomes.count(0)}, Draws={outcomes.count(1)}, B-wins={outcomes.count(2)}")
        avg_ply = np.mean([g["num_plies"] for g in games])
        total_pos = sum(len(g["fens"]) for g in games)
        log(f"  Avg game length: {avg_ply:.0f} plies, Total positions: {total_pos}")

    # ── Phase 2: Embed ──
    if "embed" in run_phases:
        log("\n═══ Phase 2: Extracting backbone embeddings ═══")
        if not games_path.exists():
            log(f"ERROR: {games_path} not found. Run --phase generate first.")
            return
        games = torch.load(games_path, weights_only=False)
        games = extract_embeddings(games, batch_size=64)
        torch.save(games, embedded_path)
        log(f"Saved embedded games to {embedded_path}")

    # ── Phase 3: Train ──
    if "train" in run_phases:
        log("\n═══ Phase 3: Training trajectory value model ═══")
        if not embedded_path.exists():
            log(f"ERROR: {embedded_path} not found. Run --phase embed first.")
            return
        games = torch.load(embedded_path, weights_only=False)

        # Train/eval split (80/20 by game)
        n_train = int(len(games) * 0.8)
        train_games = games[:n_train]
        eval_games = games[n_train:]
        log(f"Split: {n_train} train games, {len(eval_games)} eval games")

        # Create datasets
        traj_config = TrajectoryConfig(
            traj_hidden=args.traj_hidden,
            traj_layers=args.traj_layers,
            window_size=args.window_size,
        )
        train_ds = TrajectoryDataset(train_games, window_size=args.window_size)
        eval_ds = TrajectoryDataset(eval_games, window_size=args.window_size,
                                     stride=args.window_size, augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, pin_memory=True)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)

        log(f"Train: {len(train_ds)} windows, Eval: {len(eval_ds)} windows")

        # ── Train trajectory model ──
        log("\n--- Training TRAJECTORY model ---")
        traj_model = TrajectoryValueModel(traj_config).to(DEVICE)
        n_params = sum(p.numel() for p in traj_model.parameters())
        log(f"Trajectory model: {n_params:,} params ({n_params/1e6:.1f}M)")

        traj_best, traj_acc, traj_history = train_model(
            traj_model, train_loader, eval_loader,
            epochs=args.epochs, lr=args.lr, model_name="TRAJ",
        )

        # ── Train baseline model ──
        log("\n--- Training BASELINE model (no trajectory context) ---")
        base_model = SinglePositionBaseline(traj_config).to(DEVICE)
        n_params_base = sum(p.numel() for p in base_model.parameters())
        log(f"Baseline model: {n_params_base:,} params ({n_params_base/1e6:.1f}M)")

        base_best, base_acc, base_history = train_model(
            base_model, train_loader, eval_loader,
            epochs=args.epochs, lr=args.lr, model_name="BASE",
        )

        # ── Summary ──
        log("\n" + "=" * 50)
        log("TRAINING RESULTS:")
        log(f"  Trajectory model: best eval acc = {traj_acc:.4f}")
        log(f"  Baseline model:   best eval acc = {base_acc:.4f}")
        improvement = traj_acc - base_acc
        log(f"  Trajectory advantage: {improvement:+.4f} "
            f"({'BETTER' if improvement > 0 else 'WORSE'})")
        log("=" * 50)

        # Save results
        results = {
            "config": asdict(traj_config),
            "args": vars(args),
            "trajectory": {
                "best_acc": traj_acc,
                "history": traj_history,
                "n_params": sum(p.numel() for p in TrajectoryValueModel(traj_config).parameters()),
            },
            "baseline": {
                "best_acc": base_acc,
                "history": base_history,
                "n_params": sum(p.numel() for p in SinglePositionBaseline(traj_config).parameters()),
            },
        }

        # Save models
        torch.save(traj_best, OUTPUT_DIR / "traj_best.pt")
        torch.save(base_best, OUTPUT_DIR / "base_best.pt")
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"Saved results to {OUTPUT_DIR / 'results.json'}")

    # ── Phase 4: Eval ──
    if "eval" in run_phases:
        log("\n═══ Phase 4: Evaluation & Analysis ═══")
        if not embedded_path.exists():
            log(f"ERROR: {embedded_path} not found.")
            return

        games = torch.load(embedded_path, weights_only=False)
        eval_games = games[int(len(games) * 0.8):]

        traj_config = TrajectoryConfig(
            traj_hidden=args.traj_hidden,
            traj_layers=args.traj_layers,
            window_size=args.window_size,
        )

        traj_path = OUTPUT_DIR / "traj_best.pt"
        base_path = OUTPUT_DIR / "base_best.pt"
        if not traj_path.exists() or not base_path.exists():
            log("ERROR: Model checkpoints not found. Run --phase train first.")
            return

        traj_model = TrajectoryValueModel(traj_config).to(DEVICE)
        traj_model.load_state_dict(torch.load(traj_path, weights_only=True))

        base_model = SinglePositionBaseline(traj_config).to(DEVICE)
        base_model.load_state_dict(torch.load(base_path, weights_only=True))

        # Attention analysis
        log("\n--- Attention Pattern Analysis ---")
        analyses = analyze_attention(traj_model, eval_games, n_games=5)

        # Calibration comparison
        log("\n--- Value Calibration Comparison ---")
        calibration = compare_value_calibration(traj_model, base_model, eval_games)

        # Save
        with open(OUTPUT_DIR / "eval_analysis.json", "w") as f:
            json.dump({"attention": analyses, "calibration": calibration}, f, indent=2, default=str)
        log(f"Saved analysis to {OUTPUT_DIR / 'eval_analysis.json'}")


if __name__ == "__main__":
    main()
