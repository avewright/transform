"""exp013: Action-Value Training — Learn the value of EVERY legal move.

Hypothesis: Training on Stockfish evaluations for ALL legal moves (not just
the best move) provides ~30x more gradient signal per position and will
dramatically improve both policy accuracy and value estimation.

Reference: Ruoss et al., "Grandmaster-Level Chess Without Search" (2024)
used action-value prediction as their core training signal to reach 2895 Elo
WITHOUT search. This is the single biggest architectural insight from that work.

Approach:
  1. For each position, run Stockfish on every legal move (push, eval, pop)
  2. Convert evals to Q-values (win probability from side-to-move perspective)
  3. Train the model to predict Q(s,a) for all legal moves simultaneously
  4. Loss = MSE(predicted_Q, stockfish_Q) masked to legal moves only
  5. Compare: action-value loss vs policy-only cross-entropy baseline

Key insight: Standard policy training gives 1 gradient signal per position
(correct/wrong on the best move). Action-value training gives N signals
where N = number of legal moves (~30 on average). This is a ~30x increase
in information density per position.

Time budget: ~10 min (label 2K positions with all-move evals, train, compare).
Memory: ~3GB VRAM for Qwen3-0.6B bf16 + heads. Fits in 8GB easily, 18GB plenty.
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
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, index_to_move, legal_move_mask
from config import Config

# --- Configuration ---
STOCKFISH_PATH = "stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR = Path("outputs/exp013_action_value")
POLICY_CACHE = Path("outputs/exp012_stockfish_supervised/labeled_data.json")
AV_CACHE = OUTPUT_DIR / "action_value_data.json"

# Data
NUM_POSITIONS = 2000       # positions to label with all-move evals
SF_DEPTH = 8               # depth per move (lower than d10 since we eval ALL moves)
SF_THREADS = 2
MAX_LEGAL_MOVES = 218      # theoretical max; avg ~30

# Training
EPOCHS = 10
BATCH_SIZE = 32            # 18GB: can go to 64. 8GB: use 16-32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42

# Comparison
BASELINE_EPOCHS = 10       # policy-only baseline for comparison


def generate_diverse_positions(n: int, seed: int = 42) -> list[chess.Board]:
    """Generate diverse positions via random play with stratified ply depths."""
    random.seed(seed)
    positions = []
    seen = set()
    ply_ranges = [(4, 15, 0.3), (16, 40, 0.45), (41, 80, 0.25)]

    for min_ply, max_ply, fraction in ply_ranges:
        target = int(n * fraction)
        collected = 0
        attempts = 0
        while collected < target and attempts < target * 10:
            board = chess.Board()
            ply = random.randint(min_ply, max_ply)
            for _ in range(ply):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))
            if not board.is_game_over() and list(board.legal_moves):
                key = board.board_fen() + (" w" if board.turn else " b")
                if key not in seen:
                    seen.add(key)
                    positions.append(board.copy())
                    collected += 1
            attempts += 1

    random.shuffle(positions)
    return positions[:n]


def label_action_values(positions: list[chess.Board], depth: int = SF_DEPTH) -> list[dict]:
    """Label every legal move in each position with Stockfish evaluation.

    For each position, pushes every legal move, evaluates the resulting
    position, and records the eval. This gives a full action-value map.

    Returns list of dicts with:
        fen, best_uci, move_values: [{uci, eval_cp, eval_type}]
    """
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=depth,
        parameters={"Threads": SF_THREADS, "Hash": 128},
    )

    labeled = []
    t0 = time.time()

    for i, board in enumerate(positions):
        fen = board.fen()
        legal_moves = list(board.legal_moves)

        move_values = []
        best_val = -float("inf")
        best_uci = None

        for move in legal_moves:
            board.push(move)
            child_fen = board.fen()
            board.pop()

            try:
                sf.set_fen_position(child_fen)
                ev = sf.get_evaluation()
                eval_type = ev.get("type", "cp")
                eval_value = ev.get("value", 0)

                # Flip sign: child position is from opponent's perspective
                # so negate to get value from current player's perspective
                if eval_type == "cp":
                    val_from_mover = -eval_value
                elif eval_type == "mate":
                    val_from_mover = -eval_value
                else:
                    val_from_mover = 0

                move_values.append({
                    "uci": move.uci(),
                    "eval_type": eval_type,
                    "eval_value": val_from_mover,
                })

                if eval_type == "mate":
                    # Mate values: positive mate-in-N is very good
                    sort_val = 100000 - abs(val_from_mover) if val_from_mover > 0 else -100000 + abs(val_from_mover)
                else:
                    sort_val = val_from_mover

                if sort_val > best_val:
                    best_val = sort_val
                    best_uci = move.uci()

            except Exception:
                continue

        if move_values and best_uci:
            labeled.append({
                "fen": fen,
                "best_uci": best_uci,
                "move_values": move_values,
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(positions) - i - 1) / rate
            print(f"  Labeled {i + 1}/{len(positions)} ({rate:.1f} pos/s, ETA {eta:.0f}s)")

    return labeled


def cp_to_q(cp: int, eval_type: str = "cp") -> float:
    """Convert centipawn eval to Q-value in [0, 1] (win probability).

    Uses sigmoid scaling consistent with LC0/Lichess WDL model.
    """
    if eval_type == "mate":
        if cp > 0:
            return 1.0 - 0.001 * min(abs(cp), 50)  # mate-in-1 = 0.999
        elif cp < 0:
            return 0.001 * min(abs(cp), 50)
        else:
            return 0.5
    # Sigmoid scaling
    k = 1.0 / 111.7  # Standard LC0 constant
    return 1.0 / (1.0 + math.exp(-k * cp))


def make_av_batches(data: list[dict], batch_size: int, device: torch.device):
    """Build training batches for action-value training.

    Each batch contains:
      - board_input: token IDs for the encoder
      - q_targets: (B, VOCAB_SIZE) float tensor with Q-values for legal moves
      - q_mask: (B, VOCAB_SIZE) bool mask for legal moves
      - best_move_idx: (B,) for policy accuracy tracking
    """
    random.shuffle(data)
    batches = []

    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]

        batch_input = batch_boards_to_token_ids(boards, device)

        q_targets = torch.zeros(len(chunk), VOCAB_SIZE, device=device)
        q_mask = torch.zeros(len(chunk), VOCAB_SIZE, dtype=torch.bool, device=device)
        best_move_idx = torch.zeros(len(chunk), dtype=torch.long, device=device)

        for j, d in enumerate(chunk):
            for mv in d["move_values"]:
                uci = mv["uci"]
                if uci in UCI_TO_IDX:
                    idx = UCI_TO_IDX[uci]
                    q_targets[j, idx] = cp_to_q(mv["eval_value"], mv["eval_type"])
                    q_mask[j, idx] = True
            if d["best_uci"] in UCI_TO_IDX:
                best_move_idx[j] = UCI_TO_IDX[d["best_uci"]]

        batches.append((batch_input, q_targets, q_mask, best_move_idx))

    return batches


def make_policy_batches(data: list[dict], batch_size: int, device: torch.device):
    """Build standard policy-only training batches (for baseline comparison)."""
    random.shuffle(data)
    batches = []

    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        targets = torch.tensor(
            [UCI_TO_IDX.get(d["best_uci"], 0) for d in chunk],
            dtype=torch.long, device=device,
        )
        batches.append((batch_input, targets))

    return batches


class ActionValueHead(nn.Module):
    """Predicts Q(s,a) for each move in the vocabulary.

    Replaces the standard policy head with a value-per-action head.
    Same architecture but sigmoid output instead of log-softmax.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))  # Q-values in [0, 1]


def evaluate_model(chess_model, eval_data, device, n=200):
    """Evaluate policy accuracy (top-1, top-3) on held-out data."""
    chess_model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for d in eval_data[:n]:
            board = chess.Board(d["fen"])
            pred_move, probs = chess_model.predict_move(board)
            target_idx = UCI_TO_IDX.get(d["best_uci"], -1)
            total += 1
            if pred_move.uci() == d["best_uci"]:
                correct += 1
            top3 = probs.topk(min(3, probs.shape[0])).indices.cpu().tolist()
            if target_idx in top3:
                top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def train_action_value(chess_model, av_head, train_data, eval_data, device, epochs=EPOCHS):
    """Train with action-value loss: MSE on Q(s,a) for all legal moves."""
    params = list(chess_model.encoder.parameters()) + \
             list(chess_model.input_proj.parameters()) + \
             list(chess_model.policy_head.parameters()) + \
             list(chess_model.value_head.parameters()) + \
             list(av_head.parameters())

    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = epochs * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    for epoch in range(epochs):
        chess_model.train()
        av_head.train()
        batches = make_av_batches(train_data, BATCH_SIZE, device)
        ep_loss = 0.0
        steps = 0

        for batch_input, q_targets, q_mask, best_move_idx in batches:
            # Single forward pass: encoder -> projection -> backbone
            tokens = chess_model.encoder(batch_input)
            embeds = chess_model.input_proj(tokens)
            backbone_dtype = next(chess_model.backbone.parameters()).dtype
            embeds = embeds.to(backbone_dtype)
            outputs = chess_model.backbone(inputs_embeds=embeds, use_cache=False)
            hidden = outputs.last_hidden_state.float()
            global_hidden = hidden[:, 0, :]

            # Policy logits from chess_model's policy head
            policy_logits = chess_model.policy_head(global_hidden)

            # Q-value predictions from action-value head
            q_pred = av_head(global_hidden)  # (B, VOCAB_SIZE) in [0,1]

            # Action-value loss: MSE only on legal moves
            av_loss = ((q_pred - q_targets) ** 2 * q_mask.float()).sum() / q_mask.float().sum()

            # Policy loss: cross-entropy on best move (auxiliary)
            policy_loss = F.cross_entropy(policy_logits, best_move_idx)

            # Combined loss: action-value primary, policy auxiliary
            loss = av_loss + 0.3 * policy_loss

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


def train_policy_baseline(chess_model, train_data, eval_data, device, epochs=BASELINE_EPOCHS):
    """Standard policy-only training (cross-entropy on best move) for comparison."""
    params = [p for p in chess_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = epochs * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    for epoch in range(epochs):
        chess_model.train()
        batches = make_policy_batches(train_data, BATCH_SIZE, device)
        ep_loss = 0.0
        steps = 0

        for batch_input, targets in batches:
            result = chess_model(batch_input, move_targets=targets)
            loss = result["policy_loss"]

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

    # ---- Step 1: Generate action-value labeled data ----
    if AV_CACHE.exists():
        print(f"Loading cached action-value data from {AV_CACHE}...")
        with open(AV_CACHE) as f:
            av_data = json.load(f)
        print(f"  Loaded {len(av_data)} positions")
    else:
        print(f"Generating {NUM_POSITIONS} diverse positions...")
        positions = generate_diverse_positions(NUM_POSITIONS, seed=SEED)
        print(f"  Generated {len(positions)} positions")

        print(f"Labeling all legal moves with Stockfish depth {SF_DEPTH}...")
        av_data = label_action_values(positions, depth=SF_DEPTH)
        print(f"  Labeled {len(av_data)} positions")

        # Cache for reuse
        with open(AV_CACHE, "w") as f:
            json.dump(av_data, f)
        print(f"  Cached to {AV_CACHE}")

    # Split into train/eval
    split = int(len(av_data) * 0.9)
    train_data = av_data[:split]
    eval_data = av_data[split:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Stats
    avg_moves = sum(len(d["move_values"]) for d in av_data) / len(av_data)
    print(f"  Avg legal moves per position: {avg_moves:.1f}")
    print(f"  Total move evaluations: {sum(len(d['move_values']) for d in av_data):,}")

    # ---- Step 2: Load model ----
    print("\nLoading Qwen3-0.6B backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # ---- Step 3: Train policy-only baseline ----
    print(f"\n{'=' * 60}")
    print(f" BASELINE: Policy-only cross-entropy ({BASELINE_EPOCHS} epochs)")
    print(f"{'=' * 60}")
    encoder_baseline = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_baseline = ChessModel(full_model, encoder=encoder_baseline, freeze_backbone=True).to(device)
    print(f"  Trainable params: {model_baseline.trainable_params():,}")

    pre_eval = evaluate_model(model_baseline, eval_data, device)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    baseline_history = train_policy_baseline(model_baseline, train_data, eval_data, device)

    # ---- Step 4: Train action-value model ----
    print(f"\n{'=' * 60}")
    print(f" ACTION-VALUE: Q(s,a) for all legal moves ({EPOCHS} epochs)")
    print(f"{'=' * 60}")

    # Fresh model
    encoder_av = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_av = ChessModel(full_model, encoder=encoder_av, freeze_backbone=True).to(device)
    av_head = ActionValueHead(model_av.hidden_size, VOCAB_SIZE).to(device)
    total_trainable = model_av.trainable_params() + sum(p.numel() for p in av_head.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_trainable:,} (includes AV head)")

    pre_eval_av = evaluate_model(model_av, eval_data, device)
    print(f"  Pre-train: acc={pre_eval_av['accuracy']:.1%} top3={pre_eval_av['top3_accuracy']:.1%}")

    av_history = train_action_value(model_av, av_head, train_data, eval_data, device)

    # ---- Step 5: Compare ----
    print(f"\n{'=' * 60}")
    print(f" RESULTS COMPARISON")
    print(f"{'=' * 60}")

    bl_final = baseline_history[-1] if baseline_history else {}
    av_final = av_history[-1] if av_history else {}

    print(f"  Policy-only: acc={bl_final.get('accuracy', 0):.1%} "
          f"top3={bl_final.get('top3_accuracy', 0):.1%} "
          f"loss={bl_final.get('loss', 0):.4f}")
    print(f"  Action-value: acc={av_final.get('accuracy', 0):.1%} "
          f"top3={av_final.get('top3_accuracy', 0):.1%} "
          f"loss={av_final.get('loss', 0):.4f}")

    bl_best = max((h["accuracy"] for h in baseline_history), default=0)
    av_best = max((h["accuracy"] for h in av_history), default=0)
    diff = av_best - bl_best
    winner = "ACTION-VALUE" if diff > 0 else "POLICY-ONLY" if diff < 0 else "TIE"
    print(f"\n  Best accuracy — Baseline: {bl_best:.1%}, Action-Value: {av_best:.1%}")
    print(f"  Winner: {winner} (delta: {diff:+.1%})")

    # ---- Save results ----
    elapsed = time.time() - t0
    results = {
        "experiment": "exp013_action_value",
        "hypothesis": "Action-value Q(s,a) training beats policy-only cross-entropy",
        "data": {
            "positions": len(av_data),
            "train": len(train_data),
            "eval": len(eval_data),
            "avg_legal_moves": round(avg_moves, 1),
            "total_move_evals": sum(len(d["move_values"]) for d in av_data),
            "sf_depth": SF_DEPTH,
        },
        "baseline_policy": {
            "epochs": BASELINE_EPOCHS,
            "final": bl_final,
            "best_accuracy": bl_best,
            "history": baseline_history,
        },
        "action_value": {
            "epochs": EPOCHS,
            "final": av_final,
            "best_accuracy": av_best,
            "history": av_history,
        },
        "winner": winner,
        "delta": round(diff, 4),
        "elapsed_seconds": round(elapsed, 1),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
