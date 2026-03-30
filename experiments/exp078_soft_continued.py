"""exp078: Continued pretraining on lichess-sf (832M positions) with replay mixing.

Hypothesis: The 200M model's ~46% accuracy ceiling is data-limited (trained on 47.5K
positions). Continuing on the much larger lichess-sf dataset (832M deep-labeled positions,
depth 22) with proper anti-forgetting measures will break through the ceiling.

exp076 failed from catastrophic forgetting due to:
  1. No replay of original training data
  2. Distribution shift between source-sharded files
  3. Fresh optimizer without warmup

Fixes applied here:
  1. Mix in 20% replay from original dataset (chess-positions, 47.5K)
  2. Large shuffle buffer (50K) to smooth distribution shifts
  3. Very low LR (3e-6 peak) with 500-step linear warmup
  4. Gradient norm monitoring with auto-stop
  5. Eval every 1000 steps to catch divergence early

Also using soft targets from the original dataset's multi-PV top_moves (KL loss)
mixed with hard CE targets from lichess-sf.

Primary metric: move accuracy on 1000-position eval set.
Secondary: Elo via SF calibration at 1320/1450/1600.
Device: RTX 4060 Laptop (8GB VRAM), batch 4 + accum 32 = effective 128.
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import (
    VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX,
    index_to_move, legal_move_mask,
)

# ---- Paths ----
OUTPUT_DIR = Path("outputs/exp078_soft_continued")
CHECKPOINT_PATH = Path("outputs/hf/chess-transformer-200m-latest/best_model.pt")
STOCKFISH_PATH = Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")

# ---- Dataset config ----
MAIN_REPO = "avewright/chess-positions"              # 47.5K, multi-PV
LARGE_REPO = "avewright/chess-positions-lichess-sf"  # 832M, depth 22

# ---- Training config ----
TRAIN_BATCH = 4              # Tiny batches for 8GB VRAM
TRAIN_ACCUM = 32             # Effective batch = 128
MAX_LR = 3e-6                # Conservative peak LR
MIN_LR = 1e-7                # Floor
WARMUP_STEPS = 500           # Linear warmup
TOTAL_STEPS = 20000          # ~2.56M positions (128 * 20000)
REPLAY_RATIO = 0.20          # 20% of each batch from original dataset
SHUFFLE_BUFFER = 50000       # Large buffer to smooth distribution shifts
GRAD_CLIP = 0.5              # Tight gradient clipping
MAX_GNORM = 10.0             # Auto-stop if grad norm exceeds this

# ---- Soft target config ----
SOFT_TARGET_TAU = 150.0      # Temperature for CP → probability (centipawns)
SOFT_LOSS_WEIGHT = 0.3       # Weight for KL loss (0.7 for CE)

# ---- Eval config ----
EVAL_EVERY = 1000            # Steps between evals
EVAL_POSITIONS = 1000        # More positions for reliable metrics
EVAL_BATCH = 4               # Same as training for VRAM safety
SAVE_EVERY = 2000            # Checkpoint every N steps
PATIENCE = 5                 # Stop if no improvement for this many evals

# ---- SF calibration ----
SF_ELOS = [1320, 1450, 1600]
SF_GAMES_PER_LEVEL = 8       # 4 as white + 4 as black
SF_MOVETIME = 0.05

# ---- Model architecture (200M) ----
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

LOG_FILE = None


def log(msg: str):
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ===========================================================================
# Model definition (mirrors play.py ChessTransformer200M)
# ===========================================================================

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


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=512):
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

    def forward(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(cls_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


from chess_model import FusedBoardEncoder


class ChessTransformer200M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO, dropout=DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 3),
        )

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ===========================================================================
# Data loading
# ===========================================================================

def _load_hf_token():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


HF_TOKEN = _load_hf_token()


def stream_large_dataset(shuffle_seed=42):
    """Stream positions from the large lichess-sf dataset."""
    from datasets import load_dataset

    ds = load_dataset(LARGE_REPO, split="train", streaming=True, token=HF_TOKEN)
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=SHUFFLE_BUFFER)

    for row in ds:
        try:
            fen = row["fen"]
            move_uci = row["best_move"]
            if move_uci not in UCI_TO_IDX:
                continue

            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                continue

            # Parse eval value (may be string in this dataset)
            eval_value = int(row.get("eval_value", 0))
            eval_type = row.get("eval_type", "cp")

            # Parse top_moves if available
            top_moves_raw = row.get("top_moves", "")
            top_moves = []
            if top_moves_raw:
                try:
                    top_moves = json.loads(top_moves_raw) if isinstance(top_moves_raw, str) else top_moves_raw
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse WDL
            try:
                wdl = (float(row.get("wdl_win", 0.33)),
                       float(row.get("wdl_draw", 0.34)),
                       float(row.get("wdl_loss", 0.33)))
            except (ValueError, TypeError):
                wdl = (0.33, 0.34, 0.33)

            yield {
                "board": board,
                "move": move,
                "move_idx": UCI_TO_IDX[move_uci],
                "eval_type": eval_type,
                "eval_value": eval_value,
                "wdl": wdl,
                "phase": row.get("phase", "unknown"),
                "top_moves": top_moves,
                "source": "lichess_sf",
            }
        except Exception:
            continue


def load_replay_data():
    """Load the original small dataset (47.5K) into memory for replay."""
    from datasets import load_dataset

    ds = load_dataset(MAIN_REPO, split="train", token=HF_TOKEN)
    data = []
    for row in ds:
        try:
            move_uci = row["best_move"]
            if move_uci not in UCI_TO_IDX:
                continue
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                continue

            top_moves_raw = row.get("top_moves", "")
            top_moves = []
            if top_moves_raw:
                try:
                    top_moves = json.loads(top_moves_raw) if isinstance(top_moves_raw, str) else top_moves_raw
                except (json.JSONDecodeError, TypeError):
                    pass

            data.append({
                "board": board,
                "move": move,
                "move_idx": UCI_TO_IDX[move_uci],
                "eval_type": row.get("eval_type", "cp"),
                "eval_value": int(row.get("eval_value", 0)),
                "wdl": (float(row.get("wdl_win", 0.33)),
                        float(row.get("wdl_draw", 0.34)),
                        float(row.get("wdl_loss", 0.33))),
                "phase": row.get("phase", "unknown"),
                "top_moves": top_moves,
                "source": "replay",
            })
        except Exception:
            continue

    return data


def load_eval_data(n=EVAL_POSITIONS):
    """Load eval set from the test split of the original dataset."""
    from datasets import load_dataset

    ds = load_dataset(MAIN_REPO, split="test", token=HF_TOKEN)
    data = []
    for row in ds:
        try:
            move_uci = row["best_move"]
            if move_uci not in UCI_TO_IDX:
                continue
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                continue
            data.append({
                "board": board,
                "move": move,
                "move_idx": UCI_TO_IDX[move_uci],
            })
            if len(data) >= n:
                break
        except Exception:
            continue
    return data


def make_mixed_batch(large_stream, replay_data, batch_size=TRAIN_BATCH):
    """Create a training batch mixing large dataset + replay data."""
    n_replay = max(1, int(batch_size * REPLAY_RATIO))
    n_new = batch_size - n_replay

    batch = []

    # Sample replay positions
    replay_samples = random.sample(replay_data, min(n_replay, len(replay_data)))
    batch.extend(replay_samples)

    # Stream new positions
    for item in large_stream:
        batch.append(item)
        if len(batch) >= batch_size:
            break

    random.shuffle(batch)
    return batch


# ===========================================================================
# Loss computation
# ===========================================================================

def compute_soft_target(top_moves, tau=SOFT_TARGET_TAU):
    """Convert top_moves CP values to a soft probability distribution over vocab."""
    if not top_moves or len(top_moves) < 2:
        return None

    # Build sparse target
    indices = []
    cps = []
    for m in top_moves:
        uci = m.get("uci", m.get("move", ""))
        cp = m.get("cp", 0)
        if uci in UCI_TO_IDX:
            indices.append(UCI_TO_IDX[uci])
            cps.append(float(cp))

    if len(indices) < 2:
        return None

    # Softmax over CP values (higher CP = higher probability)
    cps_tensor = torch.tensor(cps, dtype=torch.float32)
    probs = F.softmax(cps_tensor / tau, dim=0)

    # Create full-vocab target
    target = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for idx, prob in zip(indices, probs):
        target[idx] = prob

    return target


def compute_loss(model, boards, targets, soft_targets, wdl_targets, device):
    """Compute combined CE + soft target loss + WDL value loss."""
    board_input = batch_boards_to_fused_token_ids(boards, device)
    output = model(board_input)
    logits = output["policy_logits"]
    value_logits = output["value_logits"]

    B = logits.shape[0]
    target_indices = torch.tensor(targets, dtype=torch.long, device=device)

    # Hard CE loss (always computed)
    ce_loss = F.cross_entropy(logits, target_indices)

    # Soft KL loss (where available)
    has_soft = [i for i, s in enumerate(soft_targets) if s is not None]
    kl_loss = torch.tensor(0.0, device=device)
    if has_soft:
        soft_logits = logits[has_soft]
        soft_target_batch = torch.stack([soft_targets[i] for i in has_soft]).to(device)
        log_probs = F.log_softmax(soft_logits, dim=-1)
        kl_loss = F.kl_div(log_probs, soft_target_batch, reduction="batchmean")

    # WDL value loss
    wdl_tensor = torch.tensor(wdl_targets, dtype=torch.float32, device=device)
    value_loss = F.cross_entropy(value_logits, wdl_tensor)

    # Combined loss
    if has_soft:
        policy_loss = (1 - SOFT_LOSS_WEIGHT) * ce_loss + SOFT_LOSS_WEIGHT * kl_loss
    else:
        policy_loss = ce_loss

    total_loss = policy_loss + 0.1 * value_loss

    return total_loss, ce_loss.item(), kl_loss.item(), value_loss.item()


# ===========================================================================
# Learning rate schedule
# ===========================================================================

def get_lr(step):
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return MIN_LR + (MAX_LR - MIN_LR) * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return MIN_LR + (MAX_LR - MIN_LR) * cosine


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(model, eval_data, device, batch_size=EVAL_BATCH):
    """Evaluate move prediction accuracy."""
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        boards = [d["board"] for d in batch]
        targets = [d["move_idx"] for d in batch]

        board_input = batch_boards_to_fused_token_ids(boards, device)
        output = model(board_input)
        logits = output["policy_logits"]

        # Mask illegal moves
        for j, board in enumerate(boards):
            mask = legal_move_mask(board).to(device)
            logits[j] = logits[j].masked_fill(~mask, float("-inf"))

        preds = logits.argmax(dim=-1)
        top3 = logits.topk(3, dim=-1).indices

        for j, target in enumerate(targets):
            total += 1
            if preds[j].item() == target:
                correct += 1
            if target in top3[j].tolist():
                top3_correct += 1

        if total % 200 == 0:
            log(f"    eval: {total}/{len(eval_data)} positions...")

    acc = correct / total if total > 0 else 0
    top3_acc = top3_correct / total if total > 0 else 0
    model.train()
    return acc, top3_acc


# ===========================================================================
# Stockfish calibration
# ===========================================================================

@torch.no_grad()
def sf_calibration(model, device, elos=SF_ELOS, games_per=SF_GAMES_PER_LEVEL):
    """Play games against Stockfish at various Elo levels."""
    model.eval()
    results = {}

    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    except Exception as e:
        log(f"  SF calibration failed: {e}")
        return {}

    for elo in elos:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        score = 0.0

        for g in range(games_per):
            model_is_white = (g % 2 == 0)
            board = chess.Board()

            for ply in range(200):
                if board.is_game_over():
                    break

                model_turn = (board.turn == chess.WHITE) == model_is_white

                if model_turn:
                    # Model move
                    board_input = batch_boards_to_fused_token_ids([board], device)
                    output = model(board_input)
                    logits = output["policy_logits"][0]
                    mask = legal_move_mask(board).to(device)
                    logits = logits.masked_fill(~mask, float("-inf"))
                    move_idx = logits.argmax().item()
                    move = index_to_move(move_idx)
                    if move not in board.legal_moves:
                        move = random.choice(list(board.legal_moves))
                else:
                    # Stockfish move
                    result = engine.play(board, chess.engine.Limit(time=SF_MOVETIME))
                    move = result.move

                board.push(move)

            # Score
            result = board.result()
            if result == "1-0":
                score += 1.0 if model_is_white else 0.0
            elif result == "0-1":
                score += 0.0 if model_is_white else 1.0
            else:
                score += 0.5

        results[elo] = score / games_per
        log(f"    vs SF {elo}: {score}/{games_per} ({score/games_per*100:.1f}%)")

    engine.quit()
    model.train()
    return results


# ===========================================================================
# Main training loop
# ===========================================================================

def main():
    global LOG_FILE

    random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = str(OUTPUT_DIR / "exp078.log")

    log("=" * 70)
    log("exp078: Continued pretraining on lichess-sf with replay mixing")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"Checkpoint: {CHECKPOINT_PATH}")
    log(f"Large dataset: {LARGE_REPO}")
    log(f"Replay dataset: {MAIN_REPO}")
    log(f"Total steps: {TOTAL_STEPS}")
    log(f"Effective batch: {TRAIN_BATCH * TRAIN_ACCUM}")
    log(f"LR: {MAX_LR} (warmup {WARMUP_STEPS} steps)")
    log(f"Replay ratio: {REPLAY_RATIO}")
    log(f"Eval every: {EVAL_EVERY} steps")
    log("")

    # Load model
    log("Loading model...")
    model = ChessTransformer200M()
    state = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.train()
    log(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Load replay data
    log("Loading replay data...")
    replay_data = load_replay_data()
    log(f"Replay data: {len(replay_data)} positions")

    # Load eval data
    log("Loading eval data...")
    eval_data = load_eval_data()
    log(f"Eval data: {len(eval_data)} positions")

    # Baseline evaluation
    log("\n--- Baseline evaluation ---")
    baseline_acc, baseline_top3 = evaluate(model, eval_data, DEVICE)
    log(f"Baseline accuracy: {baseline_acc*100:.1f}%")
    log(f"Baseline top-3: {baseline_top3*100:.1f}%")

    log("\n--- Baseline SF calibration ---")
    baseline_sf = sf_calibration(model, DEVICE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)

    # Initialize data stream
    log("\nInitializing data stream from lichess-sf...")
    large_stream = stream_large_dataset(shuffle_seed=SEED)

    best_acc = baseline_acc
    best_step = 0
    no_improve_count = 0
    step = 0
    accum_loss = 0.0
    accum_ce = 0.0
    accum_kl = 0.0
    accum_val = 0.0
    accum_count = 0
    t_start = time.time()

    log(f"\n--- Training ({TOTAL_STEPS} steps) ---")

    for step in range(1, TOTAL_STEPS + 1):
        # Set learning rate
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Accumulate gradients
        optimizer.zero_grad()
        step_loss = 0.0
        step_ce = 0.0
        step_kl = 0.0
        step_val = 0.0

        for micro in range(TRAIN_ACCUM):
            # Get mixed batch
            batch = make_mixed_batch(large_stream, replay_data, TRAIN_BATCH)

            if len(batch) < TRAIN_BATCH:
                # Stream exhausted — restart
                log("  Data stream exhausted, restarting...")
                large_stream = stream_large_dataset(shuffle_seed=SEED + step)
                batch = make_mixed_batch(large_stream, replay_data, TRAIN_BATCH)

            boards = [d["board"] for d in batch]
            targets = [d["move_idx"] for d in batch]
            soft_targets = [compute_soft_target(d.get("top_moves", [])) for d in batch]
            wdl_targets = [list(d.get("wdl", (0.33, 0.34, 0.33))) for d in batch]

            loss, ce, kl, val = compute_loss(model, boards, targets, soft_targets, wdl_targets, DEVICE)
            loss = loss / TRAIN_ACCUM
            loss.backward()

            step_loss += loss.item()
            step_ce += ce / TRAIN_ACCUM
            step_kl += kl / TRAIN_ACCUM
            step_val += val / TRAIN_ACCUM

        # Gradient clipping and step
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Check for divergence
        if gnorm > MAX_GNORM:
            log(f"\n  WARNING: grad norm {gnorm:.1f} > {MAX_GNORM} at step {step}!")
            log("  Potential divergence detected. Reducing LR by 10x for next 100 steps.")
            # Emergency LR reduction
            for pg in optimizer.param_groups:
                pg["lr"] = lr * 0.1

        if math.isnan(step_loss) or math.isinf(step_loss):
            log(f"\n  FATAL: NaN/Inf loss at step {step}. Stopping.")
            break

        optimizer.step()

        accum_loss += step_loss
        accum_ce += step_ce
        accum_kl += step_kl
        accum_val += step_val
        accum_count += 1

        # Logging every 100 steps
        if step % 100 == 0:
            avg_loss = accum_loss / accum_count
            avg_ce = accum_ce / accum_count
            avg_kl = accum_kl / accum_count
            avg_val = accum_val / accum_count
            elapsed = time.time() - t_start
            pos_per_sec = (step * TRAIN_BATCH * TRAIN_ACCUM) / elapsed
            log(f"  step {step}/{TOTAL_STEPS}: loss={avg_loss:.4f} ce={avg_ce:.4f} "
                f"kl={avg_kl:.4f} val={avg_val:.4f} gnorm={gnorm:.2f} "
                f"lr={lr:.2e} pos/s={pos_per_sec:.0f}")
            accum_loss = 0.0
            accum_ce = 0.0
            accum_kl = 0.0
            accum_val = 0.0
            accum_count = 0

        # Evaluation
        if step % EVAL_EVERY == 0:
            log(f"\n--- Eval at step {step} ---")
            acc, top3 = evaluate(model, eval_data, DEVICE)
            delta = (acc - baseline_acc) * 100
            log(f"  Accuracy: {acc*100:.1f}% (baseline: {baseline_acc*100:.1f}%, "
                f"delta={delta:+.1f}pp)")
            log(f"  Top-3: {top3*100:.1f}% (baseline: {baseline_top3*100:.1f}%)")

            if acc > best_acc:
                best_acc = acc
                best_step = step
                no_improve_count = 0
                # Save best checkpoint
                ckpt_path = OUTPUT_DIR / "best_model.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "accuracy": acc,
                    "top3": top3,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, str(ckpt_path))
                log(f"  NEW BEST! Saved to {ckpt_path}")
            else:
                no_improve_count += 1
                log(f"  No improvement ({no_improve_count}/{PATIENCE})")

            if no_improve_count >= PATIENCE:
                log(f"\n  Early stopping: no improvement for {PATIENCE} evals")
                break

        # Periodic checkpoint
        if step % SAVE_EVERY == 0:
            ckpt_path = OUTPUT_DIR / f"checkpoint_step{step}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
            }, str(ckpt_path))
            log(f"  Checkpoint saved: {ckpt_path}")

    # Final evaluation
    total_time = time.time() - t_start
    log(f"\n--- Training complete ({total_time:.0f}s, {step} steps) ---")
    log(f"\n--- Final evaluation ---")
    final_acc, final_top3 = evaluate(model, eval_data, DEVICE)
    log(f"  Final accuracy: {final_acc*100:.1f}% (baseline: {baseline_acc*100:.1f}%)")
    log(f"  Final top-3: {final_top3*100:.1f}% (baseline: {baseline_top3*100:.1f}%)")
    log(f"  Best accuracy: {best_acc*100:.1f}% at step {best_step}")

    log("\n--- Final SF calibration ---")
    final_sf = sf_calibration(model, DEVICE)

    # Save final checkpoint
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "accuracy": final_acc,
        "top3": final_top3,
    }, str(final_path))
    log(f"Final model saved: {final_path}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Baseline: acc={baseline_acc*100:.1f}% top3={baseline_top3*100:.1f}%")
    log(f"  Final: acc={final_acc*100:.1f}% top3={final_top3*100:.1f}%")
    log(f"  Best: acc={best_acc*100:.1f}% at step {best_step}")
    log(f"  Delta: {(final_acc - baseline_acc)*100:+.1f}pp accuracy")
    log(f"  Steps: {step}/{TOTAL_STEPS}")
    log(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    log(f"  Positions trained: {step * TRAIN_BATCH * TRAIN_ACCUM:,}")

    log("\n  SF calibration comparison:")
    for elo in SF_ELOS:
        b = baseline_sf.get(elo, 0)
        f = final_sf.get(elo, 0)
        log(f"    {elo}: {b*100:.1f}% -> {f*100:.1f}%")

    # Save results JSON
    results = {
        "experiment": "exp078_soft_continued",
        "baseline_acc": baseline_acc,
        "baseline_top3": baseline_top3,
        "final_acc": final_acc,
        "final_top3": final_top3,
        "best_acc": best_acc,
        "best_step": best_step,
        "total_steps": step,
        "total_time": total_time,
        "total_positions": step * TRAIN_BATCH * TRAIN_ACCUM,
        "baseline_sf": baseline_sf,
        "final_sf": final_sf,
        "config": {
            "max_lr": MAX_LR,
            "warmup_steps": WARMUP_STEPS,
            "train_batch": TRAIN_BATCH,
            "train_accum": TRAIN_ACCUM,
            "replay_ratio": REPLAY_RATIO,
            "soft_target_tau": SOFT_TARGET_TAU,
            "soft_loss_weight": SOFT_LOSS_WEIGHT,
            "grad_clip": GRAD_CLIP,
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log("\nResults saved to outputs/exp078_soft_continued/results.json")
    log("Done!")


if __name__ == "__main__":
    main()
