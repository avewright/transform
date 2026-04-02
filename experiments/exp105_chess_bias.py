"""exp105: Chess-relative attention bias with cross-file shuffling.

Hypothesis: Adding chess-geometry-aware attention biases (rank/file distance,
diagonal alignment, knight-move relationships) will help the transformer learn
chess-specific patterns faster and reach higher accuracy than position-embedding-only models.

Key differences from exp104:
  - Adds ChessRelativeBias: learned per-head biases based on board geometry
  - Biases are zero-initialized so checkpoint loading is backward-compatible
  - Init from exp101 v2 best (0.1726 acc) instead of HF init (0.1628)
  - Same cross-file shuffled data loading as exp104

New parameters: ~0.1M (rank_bias + file_bias + diag/anti/knight + ctx biases)
Total: ~204.1M parameters

Architecture: ChessTransformer200M + ChessRelativeBias
"""

import gc
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, move_to_index, legal_move_mask
from data_loader import (
    get_eval_batch_input, compute_wdl, compute_phase,
    build_eval_from_hf, get_hf_dataset_layout, _tokenize_parquet,
    board_array_to_fused, ep_square_to_file,
)

# ── Config ──
INIT_CHECKPOINT = Path("outputs/exp101_long_v2/best_model.pt")
HF_DATASET = "avewright/chess-positions-lichess-sf"
OUTPUT_DIR = Path("/root/transform/outputs/exp105_chess_bias")

BATCH_SIZE = 256
ACCUM_STEPS = 2          # effective batch = 512
LR = 2e-5
WARMUP_STEPS = 200
MIN_LR_FRAC = 0.10
VALUE_WEIGHT = 0.50
LABEL_SMOOTHING = 0.10   # spread 10% of probability across legal moves
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
SEED = 42

EMA_DECAY = 0.999
EMA_START_STEP = 50

# Model dims (must match checkpoint)
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

# Logging
LOG_INTERVAL = 25
EVAL_INTERVAL = 200      # less frequent, but 10K eval is more reliable
SAVE_INTERVAL = 200
N_EVAL = 10000            # 10K eval set for reliable measurement

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Graceful shutdown ──
SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Graceful shutdown requested. Saving checkpoint...", flush=True)

signal.signal(signal.SIGINT, _signal_handler)


# ── Logging ──
LOG_FILE = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


# ── Model (identical to exp101) ──

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


class ChessRelativeBias(nn.Module):
    """Chess-aware relative geometry bias for attention.

    For each pair of squares (i, j), computes a learned bias based on:
      - rank distance (0..7)
      - file distance (0..7)
      - same diagonal (bool)
      - same anti-diagonal (bool)
      - knight-move relationship (bool)

    All parameters zero-initialized for backward-compatible checkpoint loading.
    """

    def __init__(self, num_heads, n_ctx=4):
        super().__init__()
        self.num_heads = num_heads
        self.n_ctx = n_ctx

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

        # Learned bias tables — zero init for backward compat
        self.rank_bias = nn.Embedding(8, num_heads)
        self.file_bias = nn.Embedding(8, num_heads)
        self.diag_bias = nn.Parameter(torch.zeros(num_heads))
        self.anti_bias = nn.Parameter(torch.zeros(num_heads))
        self.knight_bias = nn.Parameter(torch.zeros(num_heads))
        self.ctx_ctx_bias = nn.Parameter(torch.zeros(num_heads, n_ctx, n_ctx))
        self.ctx_sq_bias = nn.Parameter(torch.zeros(num_heads, n_ctx))
        self.sq_ctx_bias = nn.Parameter(torch.zeros(num_heads, n_ctx))

        # Zero-init embeddings
        nn.init.zeros_(self.rank_bias.weight)
        nn.init.zeros_(self.file_bias.weight)

    def forward(self):
        seq_len = self.n_ctx + 64
        bias = torch.zeros(self.num_heads, seq_len, seq_len,
                           device=self.rank_dist.device)
        c = self.n_ctx
        rb = self.rank_bias(self.rank_dist)
        fb = self.file_bias(self.file_dist)
        sq_bias = rb + fb
        sq_bias = sq_bias + self.same_diag.unsqueeze(-1).float() * self.diag_bias
        sq_bias = sq_bias + self.same_anti.unsqueeze(-1).float() * self.anti_bias
        sq_bias = sq_bias + self.knight_rel.unsqueeze(-1).float() * self.knight_bias
        bias[:, c:, c:] = sq_bias.permute(2, 0, 1)
        bias[:, :c, :c] = self.ctx_ctx_bias
        bias[:, :c, c:] = self.ctx_sq_bias.unsqueeze(-1).expand(-1, -1, 64)
        bias[:, c:, :c] = self.sq_ctx_bias.unsqueeze(-2).expand(-1, 64, -1)
        return bias


class ChessTransformer200M(nn.Module):
    def __init__(self, use_chess_bias=True):
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
        # Chess-relative attention bias
        self.use_chess_bias = use_chess_bias
        if use_chess_bias:
            self.chess_bias = ChessRelativeBias(NUM_HEADS, n_ctx=4)

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed

        # Apply chess-relative attention bias
        mask = None
        if self.use_chess_bias:
            # bias shape: (num_heads, 68, 68) → (B*num_heads, 68, 68)
            bias = self.chess_bias()  # (H, 68, 68)
            mask = bias.repeat(B, 1, 1)  # (B*H, 68, 68) - more memory efficient than expand+reshape

        hidden = self.transformer(hidden, mask=mask)
        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ── EMA ──

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.data, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.data)

    def apply_shadow(self, model):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self._backup)
        del self._backup

    def save_weights(self, path, model):
        self.apply_shadow(model)
        torch.save(model.state_dict(), path)
        self.restore(model)

    def load_state_dict(self, d):
        if "shadow" in d:
            self.shadow = d["shadow"]
        else:
            self.shadow = d


# ── Multi-file data loading ──

def load_parquet_files(repo_id, n_files, seed=42, file_pattern="src"):
    """Load N parquet files into CPU tensors, concatenated and shuffled."""
    from huggingface_hub import hf_hub_download

    layout = get_hf_dataset_layout(repo_id)
    revision = layout["revision"]

    if file_pattern == "src":
        all_files = list(layout["train_src"])
    else:
        all_files = sorted([
            f for f in layout["all_files"]
            if f.startswith("data/train") and f.endswith(".parquet")
        ])

    # Deterministic file selection — pick n_files spread across the dataset
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_files), generator=rng).tolist()
    selected = perm[:n_files]

    log(f"Loading {n_files} parquet files from HF (out of {len(all_files)} total)...")

    all_fused = []
    all_turn = []
    all_castling = []
    all_ep = []
    all_move_idx = []
    all_wdl = []
    total_positions = 0

    for i, file_idx in enumerate(selected):
        pq_name = all_files[file_idx]
        try:
            local_path = hf_hub_download(
                repo_id, pq_name, repo_type="dataset",
                revision=revision,
            )
            raw = _tokenize_parquet(local_path)
            n = raw["board_array"].shape[0]

            fused = board_array_to_fused(torch.from_numpy(raw["board_array"]))
            turn = torch.from_numpy(raw["turn"]).long()
            castling = torch.from_numpy(raw["castling"]).long()
            ep = ep_square_to_file(torch.from_numpy(raw["ep_square"]).long())
            move_idx = torch.from_numpy(raw["move_idx"]).long()
            wdl = compute_wdl(
                torch.from_numpy(raw["cp"]),
                torch.from_numpy(raw["mate"])
            )
            del raw

            all_fused.append(fused)
            all_turn.append(turn)
            all_castling.append(castling)
            all_ep.append(ep)
            all_move_idx.append(move_idx)
            all_wdl.append(wdl)
            total_positions += n

            if (i + 1) % 10 == 0 or i == 0:
                log(f"  Loaded {i+1}/{n_files} files, {total_positions:,} positions so far")

        except Exception as e:
            log(f"  Error loading {pq_name}: {e}")
            continue

    # Concatenate all
    log(f"Concatenating {total_positions:,} positions from {n_files} files...")
    data = {
        "fused_ids": torch.cat(all_fused, dim=0),
        "turn": torch.cat(all_turn, dim=0),
        "castling": torch.cat(all_castling, dim=0),
        "ep_file": torch.cat(all_ep, dim=0),
        "move_idx": torch.cat(all_move_idx, dim=0),
        "wdl": torch.cat(all_wdl, dim=0),
    }
    del all_fused, all_turn, all_castling, all_ep, all_move_idx, all_wdl
    gc.collect()

    # Global shuffle
    log("Shuffling across all files...")
    rng2 = torch.Generator().manual_seed(seed + 777)
    perm = torch.randperm(total_positions, generator=rng2)
    for key in data:
        data[key] = data[key][perm]

    log(f"Data ready: {total_positions:,} positions, globally shuffled")
    return data, total_positions


class ShuffledBatchIterator:
    """Iterate through pre-loaded shuffled data in minibatches."""

    def __init__(self, data, batch_size, device, seed=42):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.n = data["fused_ids"].shape[0]
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Reshuffle each epoch
        rng = torch.Generator().manual_seed(self.seed + self.epoch * 1000)
        perm = torch.randperm(self.n, generator=rng)
        self.epoch += 1

        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            if end - start < self.batch_size:
                break  # drop last incomplete batch

            idx = perm[start:end]
            batch_input = {
                "fused_ids": self.data["fused_ids"][idx].to(self.device),
                "turn": self.data["turn"][idx].to(self.device),
                "castling": self.data["castling"][idx].to(self.device),
                "ep_file": self.data["ep_file"][idx].to(self.device),
            }
            yield (
                batch_input,
                self.data["move_idx"][idx].to(self.device),
                self.data["wdl"][idx].float().to(self.device),
            )


# ── Model utils ──

def load_model(path, device):
    model = ChessTransformer200M(use_chess_bias=True)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
    model.to(device)
    if missing:
        log(f"  New parameters (zero-init): {[k.split('.')[-1] for k in missing[:5]]}... ({len(missing)} total)")
    log(f"  Loaded checkpoint: {path}")
    return model


def save_model_weights(path, model):
    torch.save(model.state_dict(), path)


# ── Eval ──

def evaluate(model, eval_data, eval_tensors, device, batch_size=128):
    """Evaluate model on eval set with legal move masking (matches exp101)."""
    model.eval()
    correct = top3_correct = total = 0
    val_correct = val_total = 0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            n = len(chunk)
            idx = slice(i, i + n)
            batch_input = get_eval_batch_input(eval_tensors, idx, "fused", device)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()

            for j, d in enumerate(chunk):
                board, true_move = d["board"], d["move"]
                phase = d.get("phase", "unknown")
                l = logits[j].clone()
                mask = legal_move_mask(board).to(device)
                l[~mask] = float("-inf")

                pred_idx = l.argmax().item()
                true_idx = move_to_index(true_move)
                hit = pred_idx == true_idx
                if hit:
                    correct += 1
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk:
                    top3_correct += 1
                total += 1

                if phase not in phase_stats:
                    phase_stats[phase] = {"correct": 0, "total": 0}
                phase_stats[phase]["total"] += 1
                if hit:
                    phase_stats[phase]["correct"] += 1

            wdl_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                pred_class = wdl_logits[j].argmax().item()
                true_wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: true_wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1

    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": {p: round(s["correct"] / max(s["total"], 1), 4) for p, s in phase_stats.items()},
        "n_eval": total,
    }


# ── LR schedule ──

def cosine_lr(step, total_steps, warmup_steps, peak_lr, min_frac=0.10):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_frac + (1 - min_frac) * cosine_decay)


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


# ── Status persistence ──

def save_status(out_dir, step, positions, best_acc, metrics=None, cursor=None):
    status = {
        "experiment": "exp105_chess_bias",
        "step": step,
        "positions_seen": positions,
        "best_accuracy": best_acc,
        "timestamp": datetime.now().isoformat(),
    }
    if metrics:
        status["latest_eval"] = metrics
    tmp = out_dir / "status.json.tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(str(tmp), str(out_dir / "status.json"))


# ── Main ──

def main():
    global LOG_FILE, SHUTDOWN_REQUESTED

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, default=str(INIT_CHECKPOINT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--value-weight", type=float, default=VALUE_WEIGHT)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--n-files", type=int, default=100,
                        help="Number of parquet files to load into RAM")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for speedup")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(out_dir / "exp105.log", "a")

    log("=" * 60)
    log("exp105: Chess-relative attention bias + cross-file shuffling")
    log(f"  init: {args.init_checkpoint}")
    log(f"  output: {out_dir}")
    log(f"  lr={args.lr}, batch={args.batch_size}, accum={args.accum_steps}")
    log(f"  value_weight={args.value_weight}, label_smoothing={args.label_smoothing}")
    log(f"  ema_decay={args.ema_decay}, warmup={args.warmup_steps}")
    log(f"  n_files={args.n_files}, max_steps={args.max_steps}, seed={args.seed}")
    log(f"  eval_interval={args.eval_interval}, n_eval={args.n_eval}")
    log(f"  compile={args.compile}")
    log(f"  device={DEVICE}")

    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # ── Load eval data ──
    log(f"Loading {args.n_eval}-position eval set...")
    eval_data, eval_tensors = build_eval_from_hf(HF_DATASET, n_eval=args.n_eval, encoder_type="fused")
    log(f"  Eval: {len(eval_data)} positions")

    # ── Load model ──
    log("Loading model...")
    model = load_model(args.init_checkpoint, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {n_params / 1e6:.1f}M parameters")

    if args.compile:
        log("Compiling model with torch.compile...")
        model = torch.compile(model)
        log("  Compilation done.")

    # ── Optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda') if DEVICE == "cuda" else None

    # ── EMA ──
    ema = EMAModel(model, decay=args.ema_decay)
    log(f"  EMA: decay={args.ema_decay}, start_step={EMA_START_STEP}")

    # ── Initial eval ──
    log("Initial eval...")
    init_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  Accuracy: {init_metrics['accuracy']:.4f}, Top3: {init_metrics['top3_accuracy']:.4f}, "
        f"Value: {init_metrics['value_accuracy']:.4f}")
    log(f"  Phase: {init_metrics['phase_accuracy']}")
    best_acc = init_metrics['accuracy']

    # ── Load training data ──
    data, n_positions = load_parquet_files(
        HF_DATASET, args.n_files, seed=args.seed
    )

    # Create batch iterator
    loader = ShuffledBatchIterator(data, args.batch_size, DEVICE, seed=args.seed)

    # Compute total steps
    batches_per_epoch = n_positions // args.batch_size
    steps_per_epoch = batches_per_epoch // args.accum_steps
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = steps_per_epoch * 3  # ~3 epochs default
    log(f"  {n_positions:,} positions, {batches_per_epoch} batches/epoch, "
        f"{steps_per_epoch} steps/epoch, cos_period={total_steps}")

    # ── Training loop ──
    model.train()
    optimizer.zero_grad()
    global_step = 0
    positions_seen = 0
    accum_count = 0
    running_loss = running_ce = running_val = running_gnorm = 0.0
    log_count = 0
    t_start = time.time()
    t_last_log = time.time()
    nan_count = 0
    epoch = 0

    log("Starting training loop...")

    while global_step < total_steps and not SHUTDOWN_REQUESTED:
        epoch += 1
        log(f"  Epoch {epoch}")

        for batch_input, move_targets, wdl_targets in loader:
            if SHUTDOWN_REQUESTED:
                break

            B = move_targets.shape[0]

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                policy_logits = result["policy_logits"]
                value_logits = result["value_logits"]

                # Policy loss with label smoothing
                ce_loss = F.cross_entropy(
                    policy_logits, move_targets,
                    label_smoothing=args.label_smoothing,
                )

                # Value loss
                value_loss = F.cross_entropy(value_logits, wdl_targets)

                total_loss = ce_loss + args.value_weight * value_loss
                scaled_loss = total_loss / args.accum_steps

            # NaN guard
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                nan_count += 1
                if nan_count > 10:
                    log(f"ERROR: too many NaN losses ({nan_count}), stopping")
                    SHUTDOWN_REQUESTED = True
                    break
                optimizer.zero_grad()
                accum_count = 0
                continue
            nan_count = 0

            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_count += 1
            running_ce += ce_loss.item()
            running_val += value_loss.item()
            running_loss += total_loss.item()
            positions_seen += B

            if accum_count >= args.accum_steps:
                # LR schedule
                lr = cosine_lr(global_step, total_steps, args.warmup_steps, args.lr, MIN_LR_FRAC)
                set_lr(optimizer, lr)

                if scaler:
                    scaler.unscale_(optimizer)
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                if global_step >= EMA_START_STEP:
                    ema.update(model)

                running_gnorm += gnorm if not math.isnan(gnorm) and not math.isinf(gnorm) else 0.0
                log_count += 1
                global_step += 1
                accum_count = 0

                # Heartbeat
                if global_step <= 5:
                    log(f"  [heartbeat] step={global_step} ce={running_ce/(log_count*args.accum_steps):.4f} "
                        f"pos={positions_seen:,}")

                # Max steps check
                if global_step >= total_steps:
                    break

                # Log
                if global_step % LOG_INTERVAL == 0 and log_count > 0:
                    avg_loss = running_loss / (log_count * args.accum_steps)
                    avg_ce = running_ce / (log_count * args.accum_steps)
                    avg_val = running_val / (log_count * args.accum_steps)
                    avg_gnorm = running_gnorm / log_count
                    elapsed = time.time() - t_last_log
                    pos_per_s = (log_count * args.accum_steps * args.batch_size) / max(elapsed, 0.1)

                    log(f"step={global_step} loss={avg_loss:.4f} ce={avg_ce:.4f} "
                        f"val={avg_val:.4f} gnorm={avg_gnorm:.2f} lr={lr:.2e} "
                        f"pos={positions_seen:,} epoch={epoch} "
                        f"speed={pos_per_s:.0f}pos/s")

                    running_loss = running_ce = running_val = running_gnorm = 0.0
                    log_count = 0
                    t_last_log = time.time()

                # Eval
                if global_step % args.eval_interval == 0:
                    log("Eval (live)...")
                    live_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
                    log(f"  live: acc={live_metrics['accuracy']:.4f} "
                        f"top3={live_metrics['top3_accuracy']:.4f} "
                        f"val_acc={live_metrics['value_accuracy']:.4f}")
                    log(f"  phase: {live_metrics['phase_accuracy']}")

                    ema_metrics = None
                    if global_step >= EMA_START_STEP:
                        log("Eval (EMA)...")
                        ema.apply_shadow(model)
                        ema_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
                        ema.restore(model)
                        log(f"  ema:  acc={ema_metrics['accuracy']:.4f} "
                            f"top3={ema_metrics['top3_accuracy']:.4f} "
                            f"val_acc={ema_metrics['value_accuracy']:.4f}")
                        log(f"  phase: {ema_metrics['phase_accuracy']}")

                    best_metrics = ema_metrics if ema_metrics and ema_metrics['accuracy'] >= live_metrics['accuracy'] else live_metrics
                    is_ema_better = ema_metrics and ema_metrics['accuracy'] >= live_metrics['accuracy']

                    if best_metrics['accuracy'] > best_acc:
                        best_acc = best_metrics['accuracy']
                        if is_ema_better:
                            ema.save_weights(out_dir / "best_model.pt", model)
                        else:
                            save_model_weights(out_dir / "best_model.pt", model)
                        log(f"  NEW BEST: acc={best_acc:.4f} ({'ema' if is_ema_better else 'live'})")

                    save_status(out_dir, global_step, positions_seen, best_acc, metrics=best_metrics)
                    model.train()

                # Save checkpoint
                if global_step % args.save_interval == 0:
                    save_model_weights(out_dir / "latest_model.pt", model)
                    ema.save_weights(out_dir / "ema_model.pt", model)
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "positions_seen": positions_seen,
                        "best_acc": best_acc,
                        "ema_state_dict": ema.__dict__,
                    }
                    if scaler:
                        ckpt["scaler_state_dict"] = scaler.state_dict()
                    tmp = out_dir / "latest_checkpoint.pt.tmp"
                    torch.save(ckpt, tmp)
                    os.replace(str(tmp), str(out_dir / "latest_checkpoint.pt"))

    # ── Final eval ──
    elapsed_total = time.time() - t_start
    log(f"\nTraining complete. {global_step} steps, {positions_seen:,} positions, "
        f"{elapsed_total / 3600:.1f} hours, {epoch} epochs")

    log("Final eval (live)...")
    final_live = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  live: acc={final_live['accuracy']:.4f} top3={final_live['top3_accuracy']:.4f}")

    log("Final eval (EMA)...")
    ema.apply_shadow(model)
    final_ema = evaluate(model, eval_data, eval_tensors, DEVICE)
    ema.restore(model)
    log(f"  ema:  acc={final_ema['accuracy']:.4f} top3={final_ema['top3_accuracy']:.4f}")

    save_model_weights(out_dir / "final_live_model.pt", model)
    ema.save_weights(out_dir / "final_ema_model.pt", model)

    best_final = final_ema if final_ema['accuracy'] >= final_live['accuracy'] else final_live
    if best_final['accuracy'] > best_acc:
        best_acc = best_final['accuracy']
        if final_ema['accuracy'] >= final_live['accuracy']:
            ema.save_weights(out_dir / "best_model.pt", model)
        else:
            save_model_weights(out_dir / "best_model.pt", model)
    log(f"  Final best accuracy: {best_acc:.4f}")

    save_status(out_dir, global_step, positions_seen, best_acc, metrics=best_final)

    log("Done.")
    LOG_FILE.close()


if __name__ == "__main__":
    main()
