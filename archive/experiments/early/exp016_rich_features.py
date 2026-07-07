"""exp016: Enriched Board Features — Attack maps, mobility, pawn structure.

Hypothesis: Adding chess-specific derived features (attack maps, pin info,
mobility, pawn structure) to the board encoder significantly improves
accuracy by giving the network information it otherwise must compute
implicitly across many transformer layers.

Reference: AlphaZero used 119 input planes including history; our encoder
uses only 67 tokens with basic piece/position/context info. The model must
learn attack patterns, pins, and material from scratch. Pre-computing these
gives an immediate signal.

New features (per square, 64 tokens):
  - Base: piece_id, color_id, square_pos (existing)
  - Attack: is_attacked_by_white, is_attacked_by_black (2 features)
  - Defense: is_defended_by_white, is_defended_by_black (2 features)
  - Pawn: is_passed, is_isolated, is_doubled (3 features per pawn)
  - Check: is_in_check (global, added to context)

New context tokens (on top of turn, castling, ep):
  - material_balance: continuous embedding
  - mobility_white: number of legal moves if it were white's turn
  - mobility_black: same for black
  - is_check: is side to move in check
  - game_phase: opening/middlegame/endgame indicator

Approach:
  1. Build EnrichedBoardEncoder extending LearnedBoardEncoder
  2. Compare enriched vs base encoder on same 5K Stockfish data
  3. Same training setup as exp012b for fair comparison

Memory: ~3GB VRAM. Fits 8GB easily, 18GB trivially.
Time: ~8 min (train 2 models × 10 epochs).
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

from chess_features import (
    batch_boards_to_token_ids, board_to_token_ids,
    NUM_PIECE_TYPES, NUM_COLORS, NUM_CASTLING_STATES, NUM_EP_STATES,
)
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, move_to_index, legal_move_mask
from config import Config

# --- Configuration ---
OUTPUT_DIR = Path("outputs/exp016_rich_features")
CACHE_FILE = Path("outputs/exp012_stockfish_supervised/labeled_data.json")

NUM_TRAIN = 5000
NUM_EVAL = 500
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42

# Number of enriched context tokens
NUM_EXTRA_CONTEXT = 4  # material_balance, mobility, game_phase, is_check


def compute_attack_maps(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute which squares are attacked by each side.

    Returns: (white_attacks, black_attacks) each (64,) bool tensors.
    """
    w_attacks = torch.zeros(64, dtype=torch.bool)
    b_attacks = torch.zeros(64, dtype=torch.bool)

    for sq in range(64):
        if board.is_attacked_by(chess.WHITE, sq):
            w_attacks[sq] = True
        if board.is_attacked_by(chess.BLACK, sq):
            b_attacks[sq] = True

    return w_attacks, b_attacks


def compute_material_balance(board: chess.Board) -> float:
    """Material balance from white's perspective, normalized to [-1, 1]."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }
    balance = 0
    for sq, piece in board.piece_map().items():
        if piece.piece_type in piece_values:
            val = piece_values[piece.piece_type]
            balance += val if piece.color == chess.WHITE else -val

    # Normalize: max possible imbalance is about ±39 (9Q+2R+2B+2N+8P)
    return max(-1.0, min(1.0, balance / 39.0))


def compute_game_phase(board: chess.Board) -> int:
    """Estimate game phase: 0=opening, 1=middlegame, 2=endgame.

    Based on total material on the board.
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }
    total = 0
    for piece in board.piece_map().values():
        if piece.piece_type in piece_values:
            total += piece_values[piece.piece_type]

    # Full board ~78 total material
    if total > 60:
        return 0  # opening
    elif total > 30:
        return 1  # middlegame
    else:
        return 2  # endgame


def compute_pawn_features(board: chess.Board) -> dict[str, torch.Tensor]:
    """Compute pawn structure features per square.

    Returns:
        is_passed (64,) bool: Is this a passed pawn?
        is_isolated (64,) bool: Is this pawn isolated (no friendly pawns on adjacent files)?
        is_doubled (64,) bool: Is there another friendly pawn on the same file?
    """
    is_passed = torch.zeros(64, dtype=torch.bool)
    is_isolated = torch.zeros(64, dtype=torch.bool)
    is_doubled = torch.zeros(64, dtype=torch.bool)

    for sq, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            continue

        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        color = piece.color

        # Check doubled
        same_file_pawns = 0
        for r in range(8):
            if r != rank:
                other_sq = chess.square(file, r)
                other = board.piece_at(other_sq)
                if other and other.piece_type == chess.PAWN and other.color == color:
                    same_file_pawns += 1
        if same_file_pawns > 0:
            is_doubled[sq] = True

        # Check isolated
        has_adjacent_friend = False
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file <= 7:
                for r in range(8):
                    adj_sq = chess.square(adj_file, r)
                    adj_piece = board.piece_at(adj_sq)
                    if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == color:
                        has_adjacent_friend = True
                        break
            if has_adjacent_friend:
                break
        if not has_adjacent_friend:
            is_isolated[sq] = True

        # Check passed: no enemy pawns on same or adjacent files ahead
        is_pass = True
        if color == chess.WHITE:
            check_ranks = range(rank + 1, 8)
        else:
            check_ranks = range(0, rank)

        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                for r in check_ranks:
                    check_sq = chess.square(check_file, r)
                    blocker = board.piece_at(check_sq)
                    if blocker and blocker.piece_type == chess.PAWN and blocker.color != color:
                        is_pass = False
                        break
            if not is_pass:
                break
        if is_pass:
            is_passed[sq] = True

    return {"is_passed": is_passed, "is_isolated": is_isolated, "is_doubled": is_doubled}


def board_to_enriched_ids(board: chess.Board) -> dict[str, torch.Tensor]:
    """Convert board to enriched token IDs including attack/pawn/material features."""
    base = board_to_token_ids(board)

    # Attack maps
    w_attacks, b_attacks = compute_attack_maps(board)
    base["white_attacks"] = w_attacks.long()   # (64,)
    base["black_attacks"] = b_attacks.long()   # (64,)

    # Pawn features
    pawn = compute_pawn_features(board)
    base["is_passed"] = pawn["is_passed"].long()    # (64,)
    base["is_isolated"] = pawn["is_isolated"].long() # (64,)
    base["is_doubled"] = pawn["is_doubled"].long()   # (64,)

    # Global context
    base["material_balance"] = torch.tensor([compute_material_balance(board)], dtype=torch.float32)
    base["game_phase"] = torch.tensor([compute_game_phase(board)], dtype=torch.long)
    base["is_check"] = torch.tensor([1 if board.is_check() else 0], dtype=torch.long)

    # Mobility (approximate: count pseudo-legal moves for each side)
    # Full legality check is expensive; we use a simpler heuristic
    base["mobility"] = torch.tensor(
        [len(list(board.legal_moves))], dtype=torch.long
    )

    return base


def batch_boards_to_enriched_ids(boards: list[chess.Board], device=None) -> dict[str, torch.Tensor]:
    """Batch version of enriched encoding."""
    batch = [board_to_enriched_ids(b) for b in boards]
    result = {}
    for key in batch[0]:
        if batch[0][key].dtype == torch.float32:
            result[key] = torch.stack([b[key] for b in batch])
        elif batch[0][key].dim() == 0 or (batch[0][key].dim() == 1 and batch[0][key].shape[0] == 1):
            result[key] = torch.cat([b[key] for b in batch])
        else:
            result[key] = torch.stack([b[key] for b in batch])
    if device is not None:
        result = {k: v.to(device) for k, v in result.items()}
    return result


class EnrichedBoardEncoder(nn.Module):
    """Board encoder with attack maps, pawn structure, material, game phase.

    Extends LearnedBoardEncoder with per-square binary features and
    additional context tokens for global game state.

    Output shape: (batch, 71, embed_dim)
      = 64 squares + 3 base context (turn, castling, ep)
      + 4 enriched context (material, game_phase, is_check, mobility)
    """

    NUM_BASE_CONTEXT = 3
    NUM_EXTRA_CONTEXT = 4
    TOTAL_CONTEXT = NUM_BASE_CONTEXT + NUM_EXTRA_CONTEXT

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 64 + self.TOTAL_CONTEXT  # 71

        # Base embeddings (same as LearnedBoardEncoder)
        self.piece_embed = nn.Embedding(NUM_PIECE_TYPES, embed_dim)
        self.color_proj = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(NUM_COLORS)]
        )
        self.square_embed = nn.Embedding(64, embed_dim)

        # Base context
        self.turn_embed = nn.Embedding(2, embed_dim)
        self.castling_embed = nn.Embedding(NUM_CASTLING_STATES, embed_dim)
        self.ep_embed = nn.Embedding(NUM_EP_STATES, embed_dim)

        # Enriched per-square features (binary → learned embedding)
        self.attack_w_embed = nn.Embedding(2, embed_dim)  # attacked by white
        self.attack_b_embed = nn.Embedding(2, embed_dim)  # attacked by black
        self.passed_embed = nn.Embedding(2, embed_dim)
        self.isolated_embed = nn.Embedding(2, embed_dim)
        self.doubled_embed = nn.Embedding(2, embed_dim)

        # Enriched context tokens
        self.material_proj = nn.Linear(1, embed_dim)       # continuous material
        self.phase_embed = nn.Embedding(3, embed_dim)       # opening/mid/end
        self.check_embed = nn.Embedding(2, embed_dim)       # is_check
        self.mobility_embed = nn.Embedding(100, embed_dim)  # discretized mobility (0-99)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        piece_ids = token_ids["piece_ids"]
        color_ids = token_ids["color_ids"]
        B = piece_ids.shape[0]

        # Base piece embeddings
        piece_emb = self.piece_embed(piece_ids)
        proj_stack = torch.stack(
            [proj(piece_emb) for proj in self.color_proj], dim=2
        )
        color_sel = F.one_hot(color_ids, NUM_COLORS).float()
        sq_emb = (proj_stack * color_sel.unsqueeze(-1)).sum(dim=2)
        sq_idx = torch.arange(64, device=piece_ids.device)
        sq_emb = sq_emb + self.square_embed(sq_idx)

        # Add enriched per-square features
        sq_emb = sq_emb + self.attack_w_embed(token_ids["white_attacks"])
        sq_emb = sq_emb + self.attack_b_embed(token_ids["black_attacks"])
        sq_emb = sq_emb + self.passed_embed(token_ids["is_passed"])
        sq_emb = sq_emb + self.isolated_embed(token_ids["is_isolated"])
        sq_emb = sq_emb + self.doubled_embed(token_ids["is_doubled"])

        # Base context tokens
        turn_tok = self.turn_embed(token_ids["turn"]).unsqueeze(1)
        castle_tok = self.castling_embed(token_ids["castling"]).unsqueeze(1)
        ep_tok = self.ep_embed(token_ids["ep_file"]).unsqueeze(1)

        # Enriched context tokens
        material = token_ids["material_balance"].unsqueeze(-1)  # (B, 1)
        material_tok = self.material_proj(material).unsqueeze(1)  # (B, 1, D)
        phase_tok = self.phase_embed(token_ids["game_phase"]).unsqueeze(1)
        check_tok = self.check_embed(token_ids["is_check"]).unsqueeze(1)
        mobility_clamped = token_ids["mobility"].clamp(0, 99)
        mobility_tok = self.mobility_embed(mobility_clamped).unsqueeze(1)

        # [turn, castling, ep, material, phase, check, mobility, sq0...sq63]
        tokens = torch.cat([
            turn_tok, castle_tok, ep_tok,
            material_tok, phase_tok, check_tok, mobility_tok,
            sq_emb,
        ], dim=1)  # (B, 71, D)

        return self.norm(tokens)

    def prepare_input(self, board: chess.Board, device: torch.device):
        return batch_boards_to_enriched_ids([board], device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        return batch_boards_to_enriched_ids(boards, device)


def cp_to_wdl_idx(cp, eval_type, board_turn):
    if eval_type == "mate":
        return 0 if cp > 0 else 2 if cp < 0 else 1
    if not board_turn:
        cp = -cp
    return 0 if cp > 100 else 2 if cp < -100 else 1


def prepare_data(labeled):
    data = []
    for e in labeled:
        board = chess.Board(e["fen"])
        move = chess.Move.from_uci(e["uci"])
        wdl = cp_to_wdl_idx(e["eval_value"], e["eval_type"], board.turn)
        data.append({"board": board, "move": move, "wdl_idx": wdl})
    return data


def make_batches_base(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], dtype=torch.long, device=device)
        value_targets = torch.tensor([d["wdl_idx"] for d in chunk], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def make_batches_enriched(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_enriched_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], dtype=torch.long, device=device)
        value_targets = torch.tensor([d["wdl_idx"] for d in chunk], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_model(chess_model, eval_data, device, n=200):
    chess_model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for entry in eval_data[:n]:
            board, target = entry["board"], entry["move"]
            pred, probs = chess_model.predict_move(board)
            total += 1
            if pred == target:
                correct += 1
            top3 = probs.topk(min(3, probs.shape[0])).indices.cpu().tolist()
            if move_to_index(target) in top3:
                top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def train_model(chess_model, train_data, eval_data, device, epochs, make_batch_fn):
    params = [p for p in chess_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = epochs * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    for epoch in range(epochs):
        chess_model.train()
        batches = make_batch_fn(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0

        for batch_input, move_targets, value_targets in batches:
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            steps += 1

        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_model(chess_model, eval_data, device)
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%}")

    return history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading cached Stockfish labels...")
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    train_data = prepare_data(cache["train"][:NUM_TRAIN])
    eval_data = prepare_data(cache["eval"][:NUM_EVAL])
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Load backbone
    print("Loading Qwen3-0.6B backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # ---- Variant A: Base encoder (67 tokens) ----
    print(f"\n{'=' * 60}")
    print(f" BASELINE: LearnedBoardEncoder (67 tokens)")
    print(f"{'=' * 60}")

    encoder_base = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_base = ChessModel(full_model, encoder=encoder_base, freeze_backbone=True).to(device)
    base_params = model_base.trainable_params()
    print(f"  Trainable: {base_params:,}, Encoder: {sum(p.numel() for p in encoder_base.parameters()):,}")

    base_history = train_model(model_base, train_data, eval_data, device, EPOCHS, make_batches_base)

    del model_base, encoder_base
    torch.cuda.empty_cache()

    # ---- Variant B: Enriched encoder (71 tokens) ----
    print(f"\n{'=' * 60}")
    print(f" ENRICHED: EnrichedBoardEncoder (71 tokens, +attacks/pawns/material)")
    print(f"{'=' * 60}")

    encoder_rich = EnrichedBoardEncoder(embed_dim=ENCODER_DIM)
    model_rich = ChessModel(full_model, encoder=encoder_rich, freeze_backbone=True).to(device)
    rich_params = model_rich.trainable_params()
    print(f"  Trainable: {rich_params:,}, Encoder: {sum(p.numel() for p in encoder_rich.parameters()):,}")

    rich_history = train_model(model_rich, train_data, eval_data, device, EPOCHS, make_batches_enriched)

    # ---- Compare ----
    print(f"\n{'=' * 60}")
    print(f" RESULTS COMPARISON")
    print(f"{'=' * 60}")

    base_best = max(h["accuracy"] for h in base_history)
    rich_best = max(h["accuracy"] for h in rich_history)
    base_final = base_history[-1]
    rich_final = rich_history[-1]

    print(f"  Base:     acc={base_final['accuracy']:.1%} top3={base_final['top3_accuracy']:.1%} "
          f"loss={base_final['loss']:.4f} (best: {base_best:.1%})")
    print(f"  Enriched: acc={rich_final['accuracy']:.1%} top3={rich_final['top3_accuracy']:.1%} "
          f"loss={rich_final['loss']:.4f} (best: {rich_best:.1%})")

    diff = rich_best - base_best
    winner = "ENRICHED" if diff > 0.01 else "BASE" if diff < -0.01 else "TIE"
    print(f"\n  Winner: {winner} (delta: {diff:+.1%})")

    # Save
    elapsed = time.time() - t0
    results = {
        "experiment": "exp016_rich_features",
        "hypothesis": "Attack maps + pawn structure + material context improve accuracy",
        "base": {
            "tokens": 67, "trainable": base_params,
            "best_accuracy": base_best, "final": base_final, "history": base_history,
        },
        "enriched": {
            "tokens": 71, "trainable": rich_params,
            "best_accuracy": rich_best, "final": rich_final, "history": rich_history,
        },
        "winner": winner,
        "delta": round(diff, 4),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
