"""exp081: Confidence-weighted cached continuation tuned for 8GB VRAM.

Hypothesis: The stable exp079 continuation path becomes more productive if policy
supervision is weighted by Stockfish label confidence, so clear tactical or high-margin
positions contribute more gradient than nearly-equal positions.

Key design:
  - Same local cached 200K lichess-sf path as exp079
  - Same 8GB-friendly microbatching (batch 4, accum 32)
  - Replay mixing from the original 47.5K dataset
  - Soft targets where multi-PV data is available
  - Confidence-weighted CE/KL losses
  - Richer eval slices by phase and confidence bucket

Primary metric: move accuracy on 1000-position eval set.
Secondary: top-3, phase/confidence slices, Elo via SF calibration.
Device: RTX 4060 Laptop (8GB VRAM), batch 4 + accum 32 = effective 128.
"""

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
OUTPUT_DIR = Path("outputs/exp081_confidence_continue")
CHECKPOINT_PATH = Path("outputs/hf/chess-transformer-200m-latest/best_model.pt")
STOCKFISH_PATH = Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")
CACHED_DATA_PATH = Path("data/lichess_sf_cached_200k.jsonl")

# ---- Dataset config ----
MAIN_REPO = "avewright/chess-positions"  # 47.5K, multi-PV (replay)

# ---- Training config ----
TRAIN_BATCH = 4
TRAIN_ACCUM = 32             # Effective batch = 128
NUM_EPOCHS = 3               # Epochs over the 200K data
MAX_LR = 4e-6                # Slightly lower for weighted continuation
MIN_LR = 1e-7
WARMUP_STEPS = 200
REPLAY_RATIO = 0.25          # 25% of each batch from original dataset
GRAD_CLIP = 0.5
MAX_GNORM = 10.0

# ---- Soft target config ----
SOFT_TARGET_TAU = 150.0      # Temperature for CP → probability
SOFT_LOSS_WEIGHT = 0.3       # Weight for KL loss vs CE
VALUE_LOSS_WEIGHT = 0.1

# ---- Confidence weighting ----
MIN_CONF_WEIGHT = 0.35
MAX_CONF_WEIGHT = 1.75
CONF_BUCKET_BOUNDS = (0.75, 1.25)

# ---- Eval config ----
EVAL_EVERY_STEPS = 300       # Eval more frequently to catch improvements early
EVAL_POSITIONS = 1000
EVAL_BATCH = 4
PATIENCE = 6                 # Early stop if no improvement for this many evals

# ---- SF calibration ----
SF_ELOS = [1320, 1450, 1600]
SF_GAMES = 8                 # Per Elo level
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
# Model (mirrors play.py ChessTransformer200M)
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


def parse_top_moves(row):
    top_moves = []
    top_raw = row.get("top_moves", "[]")
    try:
        top_moves = json.loads(top_raw) if isinstance(top_raw, str) else (top_raw or [])
    except (json.JSONDecodeError, TypeError):
        pass
    return top_moves


def compute_confidence_weight(eval_type, eval_value, top_moves):
    """Bounded confidence score for weighting policy supervision."""
    base = 0.0

    if eval_type == "mate":
        try:
            mate_dist = abs(int(eval_value))
        except (TypeError, ValueError):
            mate_dist = 10
        base = 1.6 if mate_dist <= 3 else 1.35
    else:
        try:
            cp_abs = abs(float(eval_value))
        except (TypeError, ValueError):
            cp_abs = 0.0
        base = min(cp_abs / 300.0, 1.0)

    cps = []
    for move_info in top_moves[:2]:
        cp = move_info.get("cp")
        if cp is None:
            continue
        try:
            cps.append(float(cp))
        except (TypeError, ValueError):
            continue

    margin_bonus = 0.0
    if len(cps) >= 2:
        margin_bonus = min(abs(cps[0] - cps[1]) / 200.0, 0.8)

    weight = 0.55 + 0.75 * base + 0.55 * margin_bonus
    return max(MIN_CONF_WEIGHT, min(MAX_CONF_WEIGHT, weight))


def confidence_bucket(weight):
    if weight < CONF_BUCKET_BOUNDS[0]:
        return "low"
    if weight < CONF_BUCKET_BOUNDS[1]:
        return "medium"
    return "high"


def load_cached_data(path=CACHED_DATA_PATH):
    """Load locally cached lichess-sf positions."""
    data = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                move_uci = row["best_move"]
                if move_uci not in UCI_TO_IDX:
                    skipped += 1
                    continue

                board = chess.Board(row["fen"])
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    skipped += 1
                    continue

                top_moves = parse_top_moves(row)

                # Parse WDL
                try:
                    wdl = (float(row.get("wdl_win", 0.33)),
                           float(row.get("wdl_draw", 0.34)),
                           float(row.get("wdl_loss", 0.33)))
                except (ValueError, TypeError):
                    wdl = (0.33, 0.34, 0.33)

                conf_weight = compute_confidence_weight(
                    row.get("eval_type", "cp"),
                    int(row.get("eval_value", 0)),
                    top_moves,
                )

                data.append({
                    "board": board,
                    "move": move,
                    "move_idx": UCI_TO_IDX[move_uci],
                    "eval_type": row.get("eval_type", "cp"),
                    "eval_value": int(row.get("eval_value", 0)),
                    "wdl": wdl,
                    "phase": row.get("phase", "unknown"),
                    "top_moves": top_moves,
                    "source": "lichess_sf",
                    "confidence_weight": conf_weight,
                    "confidence_bucket": confidence_bucket(conf_weight),
                })
            except Exception:
                skipped += 1
                continue

    return data, skipped


def load_replay_data():
    """Load original small dataset for replay."""
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

            top_moves = parse_top_moves(row)

            conf_weight = compute_confidence_weight(
                row.get("eval_type", "cp"),
                int(row.get("eval_value", 0)),
                top_moves,
            )

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
                "confidence_weight": conf_weight,
                "confidence_bucket": confidence_bucket(conf_weight),
            })
        except Exception:
            continue
    return data


def load_eval_data(n=EVAL_POSITIONS):
    """Load eval set from test split."""
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
            top_moves = parse_top_moves(row)
            conf_weight = compute_confidence_weight(
                row.get("eval_type", "cp"),
                int(row.get("eval_value", 0)),
                top_moves,
            )

            data.append({
                "board": board,
                "move": move,
                "move_idx": UCI_TO_IDX[move_uci],
                "phase": row.get("phase", "unknown"),
                "confidence_weight": conf_weight,
                "confidence_bucket": confidence_bucket(conf_weight),
            })
            if len(data) >= n:
                break
        except Exception:
            continue
    return data


# ===========================================================================
# Loss computation
# ===========================================================================

def compute_soft_target(top_moves, tau=SOFT_TARGET_TAU):
    """Convert multi-PV CP values to soft probability distribution."""
    if not top_moves or len(top_moves) < 2:
        return None

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

    cps_tensor = torch.tensor(cps, dtype=torch.float32)
    probs = F.softmax(cps_tensor / tau, dim=0)

    target = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for idx, prob in zip(indices, probs):
        target[idx] = prob
    return target


def compute_loss(model, boards, targets, soft_targets, wdl_targets, confidence_weights, device):
    """Combined confidence-weighted CE + soft KL + WDL value loss."""
    board_input = batch_boards_to_fused_token_ids(boards, device)
    output = model(board_input)
    logits = output["policy_logits"]
    value_logits = output["value_logits"]

    target_indices = torch.tensor(targets, dtype=torch.long, device=device)
    weights = torch.tensor(confidence_weights, dtype=torch.float32, device=device)
    ce_per = F.cross_entropy(logits, target_indices, reduction="none")
    ce_loss = (ce_per * weights).sum() / weights.sum().clamp_min(1e-6)

    # Soft KL loss where available
    has_soft = [i for i, s in enumerate(soft_targets) if s is not None]
    kl_loss = torch.tensor(0.0, device=device)
    if has_soft:
        soft_logits = logits[has_soft]
        soft_batch = torch.stack([soft_targets[i] for i in has_soft]).to(device)
        soft_weights = weights[has_soft]
        log_probs = F.log_softmax(soft_logits, dim=-1)
        kl_per = F.kl_div(log_probs, soft_batch, reduction="none").sum(dim=-1)
        kl_loss = (kl_per * soft_weights).sum() / soft_weights.sum().clamp_min(1e-6)

    # WDL value loss
    wdl_tensor = torch.tensor(wdl_targets, dtype=torch.float32, device=device)
    value_log_probs = F.log_softmax(value_logits, dim=-1)
    value_loss = -(wdl_tensor * value_log_probs).sum(dim=-1).mean()

    if has_soft:
        policy_loss = (1 - SOFT_LOSS_WEIGHT) * ce_loss + SOFT_LOSS_WEIGHT * kl_loss
    else:
        policy_loss = ce_loss

    total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
    return total_loss, ce_loss.item(), kl_loss.item(), value_loss.item()


# ===========================================================================
# LR schedule
# ===========================================================================

def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return MIN_LR + (MAX_LR - MIN_LR) * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return MIN_LR + (MAX_LR - MIN_LR) * cosine


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(model, eval_data, device, batch_size=EVAL_BATCH):
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0
    phase_stats = {}
    conf_stats = {}

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        boards = [d["board"] for d in batch]
        targets = [d["move_idx"] for d in batch]

        board_input = batch_boards_to_fused_token_ids(boards, device)
        output = model(board_input)
        logits = output["policy_logits"]

        for j, board in enumerate(boards):
            mask = legal_move_mask(board).to(device)
            logits[j] = logits[j].masked_fill(~mask, float("-inf"))

        preds = logits.argmax(dim=-1)
        top3 = logits.topk(3, dim=-1).indices

        for j, target in enumerate(targets):
            total += 1
            hit = preds[j].item() == target
            top3_hit = target in top3[j].tolist()
            if hit:
                correct += 1
            if top3_hit:
                top3_correct += 1

            phase = batch[j].get("phase", "unknown")
            phase_stats.setdefault(phase, {"correct": 0, "total": 0})
            phase_stats[phase]["total"] += 1
            phase_stats[phase]["correct"] += int(hit)

            conf_bucket = batch[j].get("confidence_bucket", "unknown")
            conf_stats.setdefault(conf_bucket, {"correct": 0, "total": 0})
            conf_stats[conf_bucket]["total"] += 1
            conf_stats[conf_bucket]["correct"] += int(hit)

        if total % 200 == 0:
            log(f"    eval: {total}/{len(eval_data)}...")

    acc = correct / total if total > 0 else 0
    top3 = top3_correct / total if total > 0 else 0
    model.train()
    return {
        "accuracy": acc,
        "top3_accuracy": top3,
        "phase_accuracy": {
            phase: round(stats["correct"] / stats["total"], 4)
            for phase, stats in sorted(phase_stats.items())
        },
        "confidence_accuracy": {
            bucket: round(stats["correct"] / stats["total"], 4)
            for bucket, stats in sorted(conf_stats.items())
        },
    }


# ===========================================================================
# SF calibration
# ===========================================================================

@torch.no_grad()
def sf_calibration(model, device):
    model.eval()
    results = {}

    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    except Exception as e:
        log(f"  SF calibration failed: {e}")
        return {}

    for elo in SF_ELOS:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        score = 0.0
        for g in range(SF_GAMES):
            model_white = (g % 2 == 0)
            board = chess.Board()
            for _ in range(200):
                if board.is_game_over():
                    break
                model_turn = (board.turn == chess.WHITE) == model_white
                if model_turn:
                    inp = batch_boards_to_fused_token_ids([board], device)
                    out = model(inp)
                    logits = out["policy_logits"][0]
                    mask = legal_move_mask(board).to(device)
                    logits = logits.masked_fill(~mask, float("-inf"))
                    move = index_to_move(logits.argmax().item())
                    if move not in board.legal_moves:
                        move = random.choice(list(board.legal_moves))
                else:
                    result = engine.play(board, chess.engine.Limit(time=SF_MOVETIME))
                    move = result.move
                board.push(move)

            r = board.result()
            if r == "1-0":
                score += 1.0 if model_white else 0.0
            elif r == "0-1":
                score += 0.0 if model_white else 1.0
            else:
                score += 0.5

        results[elo] = score / SF_GAMES
        log(f"    vs SF {elo}: {score}/{SF_GAMES} ({score/SF_GAMES*100:.1f}%)")

    engine.quit()
    model.train()
    return results


# ===========================================================================
# Training
# ===========================================================================

def make_epoch_batches(new_data, replay_data, batch_size=TRAIN_BATCH):
    """Create shuffled batches, biased toward higher-confidence new positions."""
    n_replay_per_batch = max(1, int(batch_size * REPLAY_RATIO))
    n_new_per_batch = batch_size - n_replay_per_batch

    indices = list(range(len(new_data)))
    random.shuffle(indices)
    indices.sort(key=lambda idx: random.random() / max(new_data[idx]["confidence_weight"], 1e-6))

    batches = []
    pos = 0
    while pos + n_new_per_batch <= len(indices):
        batch = []
        # New data
        for i in range(n_new_per_batch):
            batch.append(new_data[indices[pos + i]])
        pos += n_new_per_batch

        # Replay data
        replay_sample = random.sample(replay_data, min(n_replay_per_batch, len(replay_data)))
        batch.extend(replay_sample)

        random.shuffle(batch)
        batches.append(batch)

    return batches


def summarize_confidence(data):
    counts = {"low": 0, "medium": 0, "high": 0}
    for item in data:
        counts[item.get("confidence_bucket", "unknown")] = counts.get(item.get("confidence_bucket", "unknown"), 0) + 1
    return counts


def main():
    global LOG_FILE

    random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = str(OUTPUT_DIR / "exp081.log")

    log("=" * 70)
    log("exp081: Confidence-weighted cached continuation")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"Checkpoint: {CHECKPOINT_PATH}")
    log(f"Cached data: {CACHED_DATA_PATH}")
    log(f"Epochs: {NUM_EPOCHS}, Effective batch: {TRAIN_BATCH * TRAIN_ACCUM}")
    log(f"LR: {MAX_LR} (warmup {WARMUP_STEPS}), Replay: {REPLAY_RATIO}")
    log(f"Confidence weights: {MIN_CONF_WEIGHT}..{MAX_CONF_WEIGHT}")
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
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Model loaded ({n_params:.1f}M params)")

    # Load data
    log("Loading cached lichess-sf data...")
    new_data, skipped = load_cached_data()
    log(f"New data: {len(new_data):,} positions ({skipped} skipped)")
    log(f"New-data confidence buckets: {summarize_confidence(new_data)}")

    log("Loading replay data...")
    replay_data = load_replay_data()
    log(f"Replay data: {len(replay_data):,} positions")
    log(f"Replay confidence buckets: {summarize_confidence(replay_data)}")

    log("Loading eval data...")
    eval_data = load_eval_data()
    log(f"Eval data: {len(eval_data)} positions")
    log(f"Eval confidence buckets: {summarize_confidence(eval_data)}")

    # Baseline
    log("\n--- Baseline evaluation ---")
    baseline_eval = evaluate(model, eval_data, DEVICE)
    baseline_acc = baseline_eval["accuracy"]
    baseline_top3 = baseline_eval["top3_accuracy"]
    log(f"Baseline: acc={baseline_acc*100:.1f}% top3={baseline_top3*100:.1f}%")
    log(f"Baseline phase slices: {baseline_eval['phase_accuracy']}")
    log(f"Baseline confidence slices: {baseline_eval['confidence_accuracy']}")

    log("\n--- Baseline SF calibration ---")
    baseline_sf = sf_calibration(model, DEVICE)

    # Compute training plan
    batches_per_epoch = len(new_data) // (TRAIN_BATCH - max(1, int(TRAIN_BATCH * REPLAY_RATIO)))
    steps_per_epoch = batches_per_epoch // TRAIN_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    log(f"\nTraining plan: {batches_per_epoch} batches/epoch, {steps_per_epoch} steps/epoch, "
        f"{total_steps} total steps across {NUM_EPOCHS} epochs")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)

    best_acc = baseline_acc
    best_step = 0
    no_improve_count = 0
    global_step = 0
    early_stopped = False
    t_start = time.time()

    # Running window for loss logging
    window_loss = 0.0
    window_ce = 0.0
    window_kl = 0.0
    window_val = 0.0
    window_steps = 0

    log(f"\n--- Training ({NUM_EPOCHS} epochs, ~{total_steps} steps) ---")

    for epoch in range(1, NUM_EPOCHS + 1):
        log(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        epoch_batches = make_epoch_batches(new_data, replay_data, TRAIN_BATCH)
        log(f"  Batches this epoch: {len(epoch_batches)}")

        micro_count = 0
        step_loss = 0.0
        step_ce = 0.0
        step_kl = 0.0
        step_val = 0.0

        optimizer.zero_grad()

        for bi, batch in enumerate(epoch_batches):
            boards = [d["board"] for d in batch]
            targets = [d["move_idx"] for d in batch]
            soft_targets = [compute_soft_target(d.get("top_moves", [])) for d in batch]
            wdl_targets = [list(d.get("wdl", (0.33, 0.34, 0.33))) for d in batch]
            confidence_weights = [d.get("confidence_weight", 1.0) for d in batch]

            loss, ce, kl, val = compute_loss(
                model,
                boards,
                targets,
                soft_targets,
                wdl_targets,
                confidence_weights,
                DEVICE,
            )
            loss = loss / TRAIN_ACCUM
            loss.backward()

            step_loss += loss.item() * TRAIN_ACCUM  # Undo the division for logging
            step_ce += ce
            step_kl += kl
            step_val += val
            micro_count += 1

            if micro_count >= TRAIN_ACCUM:
                # Optimizer step
                global_step += 1
                lr = get_lr(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                if gnorm > MAX_GNORM:
                    log(f"  WARNING: gnorm={gnorm:.1f} at step {global_step}")

                avg_step_loss = step_loss / TRAIN_ACCUM
                if math.isnan(avg_step_loss) or math.isinf(avg_step_loss):
                    log(f"  FATAL: NaN/Inf loss at step {global_step}")
                    early_stopped = True
                    break

                optimizer.step()
                optimizer.zero_grad()

                # Accumulate window stats
                window_loss += step_loss / TRAIN_ACCUM
                window_ce += step_ce / TRAIN_ACCUM
                window_kl += step_kl / TRAIN_ACCUM
                window_val += step_val / TRAIN_ACCUM
                window_steps += 1

                # Reset per-step accumulators
                step_loss = 0.0
                step_ce = 0.0
                step_kl = 0.0
                step_val = 0.0
                micro_count = 0

                # Logging every 50 steps
                if global_step % 50 == 0 and window_steps > 0:
                    avg_loss = window_loss / window_steps
                    avg_ce = window_ce / window_steps
                    avg_kl = window_kl / window_steps
                    elapsed = time.time() - t_start
                    pos_total = global_step * TRAIN_BATCH * TRAIN_ACCUM
                    pos_per_sec = pos_total / elapsed
                    log(f"  step {global_step}/{total_steps}: loss={avg_loss:.4f} "
                        f"ce={avg_ce:.4f} kl={avg_kl:.4f} gnorm={gnorm:.2f} "
                        f"lr={lr:.2e} pos/s={pos_per_sec:.0f}")
                    window_loss = 0.0
                    window_ce = 0.0
                    window_kl = 0.0
                    window_val = 0.0
                    window_steps = 0

                # Evaluation
                if global_step % EVAL_EVERY_STEPS == 0:
                    log(f"\n--- Eval at step {global_step} (epoch {epoch}) ---")
                    ev = evaluate(model, eval_data, DEVICE)
                    acc = ev["accuracy"]
                    top3 = ev["top3_accuracy"]
                    delta = (acc - baseline_acc) * 100
                    log(f"  Accuracy: {acc*100:.1f}% (baseline: {baseline_acc*100:.1f}%, "
                        f"delta={delta:+.1f}pp)")
                    log(f"  Top-3: {top3*100:.1f}% (baseline: {baseline_top3*100:.1f}%)")
                    log(f"  Phase slices: {ev['phase_accuracy']}")
                    log(f"  Confidence slices: {ev['confidence_accuracy']}")

                    if acc > best_acc:
                        best_acc = acc
                        best_step = global_step
                        no_improve_count = 0
                        ckpt = OUTPUT_DIR / "best_model.pt"
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "step": global_step,
                            "epoch": epoch,
                            "accuracy": acc,
                            "top3": top3,
                            "optimizer_state_dict": optimizer.state_dict(),
                        }, str(ckpt))
                        log(f"  NEW BEST! Saved to {ckpt}")
                    else:
                        no_improve_count += 1
                        log(f"  No improvement ({no_improve_count}/{PATIENCE})")

                    if no_improve_count >= PATIENCE:
                        log(f"  Early stopping: no improvement for {PATIENCE} evals")
                        early_stopped = True
                        break

        if early_stopped:
            break

        # End-of-epoch eval
        log(f"\n--- End of Epoch {epoch} ---")
        ev = evaluate(model, eval_data, DEVICE)
        acc = ev["accuracy"]
        top3 = ev["top3_accuracy"]
        delta = (acc - baseline_acc) * 100
        log(f"  Accuracy: {acc*100:.1f}% (delta={delta:+.1f}pp)")
        log(f"  Top-3: {top3*100:.1f}%")
        log(f"  Phase slices: {ev['phase_accuracy']}")
        log(f"  Confidence slices: {ev['confidence_accuracy']}")

        # Save epoch checkpoint
        ckpt = OUTPUT_DIR / f"epoch{epoch}_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": global_step,
            "epoch": epoch,
            "accuracy": acc,
        }, str(ckpt))
        log(f"  Checkpoint: {ckpt}")

    # Final results
    total_time = time.time() - t_start
    log(f"\n--- Training complete ({total_time:.0f}s / {total_time/60:.1f}min) ---")
    log(f"  Total steps: {global_step}")
    log(f"  Positions seen: {global_step * TRAIN_BATCH * TRAIN_ACCUM:,}")

    log("\n--- Final evaluation ---")
    final_eval = evaluate(model, eval_data, DEVICE)
    final_acc = final_eval["accuracy"]
    final_top3 = final_eval["top3_accuracy"]
    log(f"  Final: acc={final_acc*100:.1f}% top3={final_top3*100:.1f}%")
    log(f"  Final phase slices: {final_eval['phase_accuracy']}")
    log(f"  Final confidence slices: {final_eval['confidence_accuracy']}")
    log(f"  Best: acc={best_acc*100:.1f}% at step {best_step}")

    log("\n--- Final SF calibration ---")
    final_sf = sf_calibration(model, DEVICE)

    # Save final
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(final_path))

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Baseline: acc={baseline_acc*100:.1f}% top3={baseline_top3*100:.1f}%")
    log(f"  Final: acc={final_acc*100:.1f}% top3={final_top3*100:.1f}%")
    log(f"  Best: acc={best_acc*100:.1f}% at step {best_step}")
    log(f"  Delta: {(final_acc - baseline_acc)*100:+.1f}pp")
    log(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    for elo in SF_ELOS:
        b = baseline_sf.get(elo, 0)
        f = final_sf.get(elo, 0)
        log(f"  SF {elo}: {b*100:.1f}% -> {f*100:.1f}%")

    # Save results JSON
    results = {
        "experiment": "exp081_confidence_continue",
        "baseline_eval": baseline_eval,
        "final_eval": final_eval,
        "best_acc": best_acc,
        "best_step": best_step,
        "total_steps": global_step,
        "total_time": total_time,
        "baseline_sf": {str(k): v for k, v in baseline_sf.items()},
        "final_sf": {str(k): v for k, v in final_sf.items()},
        "config": {
            "max_lr": MAX_LR,
            "epochs": NUM_EPOCHS,
            "batch": TRAIN_BATCH,
            "accum": TRAIN_ACCUM,
            "replay_ratio": REPLAY_RATIO,
            "soft_tau": SOFT_TARGET_TAU,
            "soft_weight": SOFT_LOSS_WEIGHT,
            "value_loss_weight": VALUE_LOSS_WEIGHT,
            "min_conf_weight": MIN_CONF_WEIGHT,
            "max_conf_weight": MAX_CONF_WEIGHT,
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log("\nDone!")


if __name__ == "__main__":
    main()
