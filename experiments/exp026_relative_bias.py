"""exp026: Chess Transformer with Relative Position Attention Bias.

Hypothesis: Adding a learned relative position attention bias (based on
rank/file distances between squares) to the chess transformer will improve
move prediction accuracy by encoding chess spatial geometry.

Background:
  - exp023: chess transformer 8L/512d, 50K → 40.5% (10 epochs, loss declining)
  - exp024: same, 460K → 48.7% (3 epochs, loss declining)
  - Current encoder uses absolute square embeddings but no explicit spatial bias
  - Chess has strong spatial structure: knight moves, bishop diagonals, rook lines
  - Relative position bias lets the model learn "squares N ranks / M files apart
    should attend differently" — this is a natural inductive bias for chess

Changes from exp023:
  - Custom transformer block that adds learned (15,15) relative bias to QK^T
  - Bias is per-head, added inside the scaled dot-product attention
  - Previous attempt via src_mask failed (collapsed training) — now done properly

Primary metric: top-1 accuracy (best across epochs)
Comparison: exp023 baseline at 40.5% on 50K/10ep
Time budget: ~10 min (5 epochs × ~2 min/epoch on 50K)
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp026_relative_bias")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN = 50000
NUM_EVAL = 500
EPOCHS = 5
BATCH_SIZE = 128
LR = 3e-4
WARMUP_STEPS = 200
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 6
GAME_SF_DEPTH = 3


# === Relative Position Bias ===

class ChessRelativePositionBias(nn.Module):
    """Per-head learned attention bias based on (rank_diff, file_diff).

    For the 64 board squares: bias[h,i,j] = rel_table[h, rank_i-rank_j+7, file_i-file_j+7]
    For the 3 context tokens: separate learned biases per head.
    """

    def __init__(self, num_heads, num_context=3):
        super().__init__()
        self.num_heads = num_heads
        # Per-head bias table: (num_heads, 15, 15) — 15 = max rank/file diff * 2 + 1
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, 15, 15))
        self.context_sq_bias = nn.Parameter(torch.zeros(num_heads, num_context, 64))
        self.sq_context_bias = nn.Parameter(torch.zeros(num_heads, 64, num_context))
        self.context_context_bias = nn.Parameter(torch.zeros(num_heads, num_context, num_context))

        # Pre-compute relative position indices for all 64x64 square pairs
        ranks = torch.arange(64) // 8
        files = torch.arange(64) % 8
        dr = ranks.unsqueeze(1) - ranks.unsqueeze(0) + 7  # (64, 64), values 0-14
        df = files.unsqueeze(1) - files.unsqueeze(0) + 7  # (64, 64), values 0-14
        self.register_buffer('dr_idx', dr)
        self.register_buffer('df_idx', df)

    def forward(self):
        """Build (num_heads, 67, 67) attention bias."""
        # (num_heads, 64, 64) via advanced indexing
        sq_bias = self.rel_bias[:, self.dr_idx, self.df_idx]

        H = self.num_heads
        bias = torch.zeros(H, 67, 67, device=self.rel_bias.device)
        bias[:, 3:, 3:] = sq_bias
        bias[:, :3, 3:] = self.context_sq_bias
        bias[:, 3:, :3] = self.sq_context_bias
        bias[:, :3, :3] = self.context_context_bias
        return bias  # (num_heads, 67, 67)


class MultiHeadAttentionWithBias(nn.Module):
    """Multi-head attention that adds a learned bias to QK^T scores."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: (B, S, D)
            attn_bias: (num_heads, S, S) or None
        Returns:
            (B, S, D)
        """
        B, S, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(0)  # broadcast over batch

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlockWithBias(nn.Module):
    """Pre-norm transformer block with optional attention bias."""

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttentionWithBias(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.ff(self.norm2(x))
        return x


# === Architecture ===

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
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


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer('from_sqs', from_sqs)
        self.register_buffer('to_sqs', to_sqs)
        self.register_buffer('promo_types', promo_types)

    def forward(self, hidden_states):
        sq_hidden = hidden_states[:, 3:67, :]
        global_hidden = hidden_states[:, 0, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(global_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformerWithBias(nn.Module):
    """Chess transformer with learned per-head relative position attention bias."""

    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 67, hidden_dim) * 0.02)

        # Custom transformer blocks that accept attention bias
        self.blocks = nn.ModuleList([
            TransformerBlockWithBias(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Per-head relative position bias — shared across all layers
        self.rel_bias = ChessRelativePositionBias(num_heads=num_heads, num_context=3)

        self.policy_head = SpatialPolicyHead(hidden_dim, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim
        self._init_weights()

    def _init_weights(self):
        # Skip bias parameters — they must start at zero so model
        # is equivalent to no-bias baseline at initialization
        bias_params = set(id(p) for p in self.rel_bias.parameters())
        for p in self.parameters():
            if p.dim() > 1 and id(p) not in bias_params:
                nn.init.xavier_uniform_(p)

    def forward(self, board_input, move_targets=None, value_targets=None, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed

        # Compute per-head relative position bias once, reuse for all layers
        attn_bias = self.rel_bias()  # (num_heads, 67, 67)

        for block in self.blocks:
            hidden = block(hidden, attn_bias=attn_bias)
        hidden = self.norm(hidden)

        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)

        result = {"policy_logits": policy_logits, "value_pred": value_logits}

        device = board_input["piece_ids"].device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            total_loss = total_loss + F.cross_entropy(policy_logits, move_targets)
        if value_targets is not None:
            total_loss = total_loss + 0.5 * F.cross_entropy(value_logits, value_targets)
        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def predict_move(self, board):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Baseline (no bias) for fair comparison ===

class ChessTransformerBaseline(nn.Module):
    """Same custom architecture WITHOUT relative bias — for fair A/B comparison."""

    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 67, hidden_dim) * 0.02)

        # Same custom transformer blocks, but no bias passed
        self.blocks = nn.ModuleList([
            TransformerBlockWithBias(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(hidden_dim, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, board_input, move_targets=None, value_targets=None, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        for block in self.blocks:
            hidden = block(hidden)  # No bias
        hidden = self.norm(hidden)

        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)

        result = {"policy_logits": policy_logits, "value_pred": value_logits}

        device = board_input["piece_ids"].device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            total_loss = total_loss + F.cross_entropy(policy_logits, move_targets)
        if value_targets is not None:
            total_loss = total_loss + 0.5 * F.cross_entropy(value_logits, value_targets)
        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def predict_move(self, board):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Data ===

def build_old_move_mapping():
    PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    def inside(idx, f_from): return 0 <= idx < 64 and abs((idx % 8) - f_from) <= 2
    ucis = set()
    for f in range(64):
        ff = f % 8
        for d in (+8, -8, +1, -1):
            t = f + d
            while inside(t, ff): ucis.add(chess.Move(f, t).uci()); t += d
        for d in (+9, -9, +7, -7):
            t = f + d
            while inside(t, ff): ucis.add(chess.Move(f, t).uci()); t += d
        for off in (+17, +15, +10, +6, -6, -10, -15, -17):
            t = f + off
            if inside(t, ff): ucis.add(chess.Move(f, t).uci())
    for f in range(64):
        file_, rank = f % 8, f // 8
        if rank == 6:
            for df in (-9, -8, -7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES: ucis.add(chess.Move(f, t, promotion=p).uci())
        if rank == 1:
            for df in (+9, +8, +7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES: ucis.add(chess.Move(f, t, promotion=p).uci())
    return sorted(ucis)


_PIECE2CHESS = {
    1: (chess.PAWN, chess.WHITE), 2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE), 4: (chess.ROOK, chess.WHITE),
    5: (chess.QUEEN, chess.WHITE), 6: (chess.KING, chess.WHITE),
    7: (chess.PAWN, chess.BLACK), 8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK), 10: (chess.ROOK, chess.BLACK),
    11: (chess.QUEEN, chess.BLACK), 12: (chess.KING, chess.BLACK),
}


def hf_sample_to_board(board_flat, turn_raw):
    board = chess.Board(fen=None)
    board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8; col = idx % 8
            rank = 7 - row; sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = bool(turn_raw)
    return board


def prepare_hf_data(dataset, old_sorted_uci, n, offset=0):
    data, skipped = [], 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n: break
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX: skipped += 1; continue
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError: skipped += 1; continue
        if move not in board.legal_moves: skipped += 1; continue
        winner = s["winner"]
        if winner == 0: vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK): vt = 2
        else: vt = 0
        data.append({"board": board, "move": move, "value_target": vt})
    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


# === Training ===

def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk],
                                    dtype=torch.long, device=device)
        value_targets = torch.tensor([d["value_target"] for d in chunk],
                                     dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_accuracy(model, eval_data, device, n=None, batch_size=128):
    model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, target_idx in enumerate(targets):
                total += 1
                if preds[j] == target_idx: correct += 1
                if target_idx in top3s[j]: top3_correct += 1
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "top3_accuracy": round(top3_correct / max(total, 1), 4),
        "total": total,
    }


def train_variant(name, model, train_data, eval_data, device):
    """Train a single model variant and return history."""
    print(f"\n  --- Training {name} ---")
    trainable = model.trainable_params()
    print(f"  Trainable params: {trainable:,}")

    pre = evaluate_accuracy(model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre['accuracy']:.1%}")

    params = list(model.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    history = []
    best_acc = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0
        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = model(batch_input, move_targets=move_targets,
                          value_targets=value_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += result["loss"].item()
            steps += 1

        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_accuracy(model, eval_data, device)
        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
        elapsed = time.time() - t0
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})
        marker = " *BEST*" if ev["accuracy"] >= best_acc else ""
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} acc={ev['accuracy']:.1%} "
              f"top3={ev['top3_accuracy']:.1%}{marker} [{elapsed:.0f}s]")

    return {
        "name": name,
        "trainable": trainable,
        "best_acc": best_acc,
        "final": history[-1] if history else {},
        "history": history,
        "time_s": time.time() - t0,
    }


def play_game_vs_stockfish(model, sf_depth, model_color, device, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = model.predict_move(board)
            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None: break
            board.push(chess.Move.from_uci(sf_uci))
    result = board.result()
    winner = "white" if result == "1-0" else "black" if result == "0-1" else "draw"
    model_result = (
        "win" if (winner == "white" and model_color == chess.WHITE) or
                 (winner == "black" and model_color == chess.BLACK)
        else "loss" if winner != "draw" else "draw"
    )
    term = board.outcome().termination.name if board.outcome() else "max_moves"
    return {
        "model_color": "white" if model_color else "black",
        "result": result, "model_result": model_result,
        "moves": board.fullmove_number, "termination": term,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp026_relative_bias")
    print(f"Hypothesis: Relative position attention bias improves chess accuracy")

    # --- Load data ---
    print("\n[1/4] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    print(f"  Dataset: {len(ds):,} samples")

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)

    # --- Build both models with identical init seeds ---
    print(f"\n[2/4] Building models (8L/{HIDDEN_DIM}d/{NUM_HEADS}h)")

    # Variant A: Baseline (no relative bias)
    torch.manual_seed(SEED)
    model_a = ChessTransformerBaseline(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)

    # Variant B: With relative position bias
    torch.manual_seed(SEED)
    model_b = ChessTransformerWithBias(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)

    print(f"  A (baseline): {model_a.trainable_params():,} params")
    print(f"  B (rel bias): {model_b.trainable_params():,} params")
    print(f"  Bias adds: {model_b.trainable_params() - model_a.trainable_params():,} params")

    # --- Train both ---
    print(f"\n[3/4] Training on {len(train_data)} positions, {EPOCHS} epochs each")

    result_a = train_variant("baseline", model_a, train_data, eval_data, device)
    result_b = train_variant("rel_bias", model_b, train_data, eval_data, device)

    # --- Compare ---
    delta = result_b["best_acc"] - result_a["best_acc"]
    print(f"\n[4/4] Comparison:")
    print(f"  Baseline best: {result_a['best_acc']:.1%}")
    print(f"  Rel bias best: {result_b['best_acc']:.1%}")
    print(f"  Delta: {delta:+.1%}")
    print(f"  exp023 ref:    40.5%")

    # --- Play games with the better model ---
    better_model = model_b if result_b["best_acc"] >= result_a["best_acc"] else model_a
    better_name = "rel_bias" if result_b["best_acc"] >= result_a["best_acc"] else "baseline"
    print(f"\n  Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH} with {better_name}")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(better_model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv ({r['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")

    # --- Save results ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp026_relative_bias",
        "hypothesis": "Relative position attention bias improves chess accuracy",
        "primary_metric": "top-1 accuracy (best across epochs)",
        "seed": SEED,
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "model": {
            "type": "chess_transformer",
            "layers": NUM_LAYERS, "hidden": HIDDEN_DIM, "heads": NUM_HEADS,
        },
        "training": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "warmup_steps": WARMUP_STEPS, "schedule": "cosine",
        },
        "results": {
            "baseline": result_a,
            "rel_bias": result_b,
            "delta_pp": round(delta * 100, 2),
            "exp023_ref": 0.405,
        },
        "games": {
            "model_used": better_name,
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time},
        "device": str(device),
        "command": "python experiments/exp026_relative_bias.py",
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp026_relative_bias")
    print(f"{'='*60}")
    print(f"  Baseline: {result_a['best_acc']:.1%} best ({result_a['time_s']:.0f}s)")
    print(f"  Rel bias: {result_b['best_acc']:.1%} best ({result_b['time_s']:.0f}s)")
    print(f"  Delta:    {delta:+.1%} {'IMPROVED' if delta > 0.01 else 'TIE' if abs(delta) <= 0.01 else 'REGRESSED'}")
    print(f"  vs exp023 ref: 40.5%")
    print(f"  Games vs SF d{GAME_SF_DEPTH}: W{wins}/D{draws}/L{losses}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Results: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
