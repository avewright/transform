"""exp073: 204M parameter model — sharded training, 1 epoch.

Hypothesis: A ~204M parameter model trained on 48M positions from
avewright/chess-positions-lichess-sf using the sharded data pipeline
will significantly outperform smaller models.

Architecture:
  - FusedBoardEncoder 256d → 1024d transformer
  - 16 layers, 16 heads, FFN 4× (4096), GELU, norm_first
  - SpatialPolicyHead (head_dim=512)
  - WDL value head (1024 → 512 → 3)
  - ~204M total parameters

Data: Pretokenized parquet shards from avewright/chess-positions-lichess-sf
  - 16 shards × ~3M rows = ~48M positions (architecture supports 876M+)
  - Shards loaded one at a time — never full materialization
GPU: NVIDIA A40 46GB

Strategy:
  Phase 1: Pretokenize parquet → compact .pt shards (one-time, ~8 min).
  Phase 2: Train via ShardedChessLoader (streaming, resumable).

The dataset is pre-cleaned — no need for chess.Board() during data loading.
"""

import gc
import json
import math
import os
import random
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# ── GPU performance flags (A40 = Ampere, supports TF32) ──
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, move_to_index, legal_move_mask
from data_loader import (
    get_eval_batch_input, compute_wdl, compute_phase,
    pretokenize_parquet_to_shards, build_eval_from_pretokenized,
    ShardedChessLoader,
)

OUTPUT_DIR = Path("outputs/exp073_200m_full_epoch")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SHARD_DIR = OUTPUT_DIR / "shards"

# Where we expect parquet files (downloaded by earlier HF cache)
PARQUET_DIR = OUTPUT_DIR / "hf_cache" / "hub" / "datasets--avewright--chess-positions-lichess-sf" / "snapshots"

# ── Config ──
EVAL_POSITIONS = 5_000

# Training
BATCH_SIZE = 256
ACCUM_STEPS = 4           # effective batch = 1024
EPOCHS = 1
LR = 2e-4                 # slightly lower for larger model
WARMUP_FRAC = 0.03        # 3% warmup
MIN_LR_FRAC = 0.05        # cosine decays to 5% of peak
VALUE_WEIGHT = 0.5
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
SEED = 42

# Model dims — ~200M parameters
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

# Logging/checkpoint intervals (in optimizer steps, not micro-steps)
LOG_INTERVAL = 200
EVAL_INTERVAL = 2000
SAVE_INTERVAL = 5000
HEARTBEAT_INTERVAL = 50

# ── Graceful shutdown ──
SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Graceful shutdown requested (signal {signum}). "
          "Will save checkpoint after current step...", flush=True)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Find parquet files ──

def find_parquet_shards():
    """Locate downloaded parquet train shards."""
    import glob as globmod

    patterns = [
        str(PARQUET_DIR / "*/data/train-*-of-*.parquet"),
        str(OUTPUT_DIR / "parquet_cache" / "*/snapshots/*/data/train-*-of-*.parquet"),
    ]
    for pattern in patterns:
        files = sorted(globmod.glob(pattern))
        if files:
            return files

    # Try downloading
    print("  No local parquet shards found. Downloading from HF...")
    from huggingface_hub import snapshot_download
    cache_dir = str(OUTPUT_DIR / "hf_cache")
    snapshot_download(
        "avewright/chess-positions-lichess-sf",
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns="data/train-*-of-*.parquet",
    )
    for pattern in patterns:
        files = sorted(globmod.glob(pattern))
        if files:
            return files

    raise FileNotFoundError("Could not find or download parquet shards")


# ── Model ──

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
    """Move prediction via from-square × to-square spatial features."""
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


class ChessTransformer200M(nn.Module):
    """~200M parameter chess-native transformer."""
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_LAYERS,
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(
            HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM,
        )
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 3),
        )

    def forward(self, board_input):
        tokens = self.encoder(board_input)           # (B, 67, 256)
        hidden = self.input_proj(tokens)             # (B, 67, 1024)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)     # (B, 68, 1024)
        hidden = hidden + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ── Evaluation ──

def evaluate(model, eval_data, eval_tensors, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    sf_rank_sum = 0.0
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

                sorted_indices = l.argsort(descending=True).tolist()
                rank = sorted_indices.index(true_idx) + 1 if true_idx in sorted_indices else len(sorted_indices)
                sf_rank_sum += rank
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
    phase_accuracy = {p: round(s["correct"] / max(s["total"], 1), 4)
                      for p, s in phase_stats.items()}
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": phase_accuracy,
        "n_eval": total,
    }


# ── Checkpoint management ──

def save_checkpoint(model, optimizer, scaler, scheduler, global_step,
                    positions_seen, best_acc, results_log, tag=""):
    """Save a full training checkpoint for resumption."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"step_{global_step}" if not tag else tag
    ckpt_path = CHECKPOINT_DIR / f"{name}.pt"

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "positions_seen": positions_seen,
        "best_acc": best_acc,
        "results_log": results_log,
        "config": {
            "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "ffn_ratio": FFN_RATIO,
            "encoder_dim": ENCODER_DIM, "batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS, "lr": LR, "seed": SEED,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(ckpt, ckpt_path)
    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"    [CKPT] Saved {ckpt_path.name} ({size_mb:.0f} MB)", flush=True)

    # Also save a "latest" symlink/copy for easy resume
    latest_path = CHECKPOINT_DIR / "latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    # Use copy instead of symlink for robustness
    import shutil
    shutil.copy2(ckpt_path, latest_path)

    return ckpt_path


def load_checkpoint(model, optimizer, scaler, scheduler, device):
    """Load latest checkpoint if available. Returns (global_step, positions_seen, best_acc, results_log)."""
    latest_path = CHECKPOINT_DIR / "latest.pt"
    if not latest_path.exists():
        return 0, 0, 0.0, []

    print(f"  Resuming from checkpoint: {latest_path}")
    ckpt = torch.load(latest_path, map_location=device, weights_only=False)

    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    global_step = ckpt["global_step"]
    positions_seen = ckpt["positions_seen"]
    best_acc = ckpt["best_acc"]
    results_log = ckpt.get("results_log", [])
    ts = ckpt.get("timestamp", "unknown")
    print(f"  Resumed: step={global_step}, positions={positions_seen:,}, "
          f"best_acc={best_acc:.1%}, saved={ts}")
    return global_step, positions_seen, best_acc, results_log


# ── Training ──

def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ──
    print(f"\n{'='*72}")
    print(f" EXP073: 200M PARAMETER MODEL — FULL DATASET, 1 EPOCH")
    print(f"{'='*72}")
    print(f"  Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Device:     {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:       {vram:.1f} GB")
    print(f"  Model:      FusedBoardEncoder {ENCODER_DIM}d → {HIDDEN_DIM}d, "
          f"{NUM_LAYERS}L, {NUM_HEADS}H, FFN {FFN_RATIO}×")
    print(f"  Pipeline:   ShardedChessLoader (pretokenized parquet shards)")
    print(f"  Training:   batch={BATCH_SIZE} × accum={ACCUM_STEPS} "
          f"(eff={BATCH_SIZE * ACCUM_STEPS}), lr={LR}")
    print(f"  Schedule:   cosine, {WARMUP_FRAC*100:.0f}% warmup, "
          f"min_lr={MIN_LR_FRAC*100:.0f}%")
    print(f"  Grad clip:  {GRAD_CLIP}, weight_decay: {WEIGHT_DECAY}")
    print(f"  Seed:       {SEED}")
    print(f"  Output:     {OUTPUT_DIR}")
    print()

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── Phase 1: Pretokenize (concurrent with training) ──
    print("=" * 72)
    print(" PHASE 1: DATA PREPARATION")
    print("=" * 72)
    t_data_start = time.time()

    eval_path = SHARD_DIR / "eval.pt"
    shard_files = sorted(SHARD_DIR.glob("shard_*.pt"))
    pretok_thread = None
    expected_shards = None

    if not shard_files:
        print("  No pretokenized shards found. Starting background pretokenization...")
        parquet_files = find_parquet_shards()
        n_parquet = len(parquet_files)
        print(f"  Found {n_parquet} parquet shards")
        # Estimate expected shards: each parquet ~3M rows → 1 shard per parquet
        expected_shards = n_parquet  # conservative 1:1 estimate

        pretok_error = [None]  # mutable container for thread exception

        def _pretok_worker():
            try:
                pretokenize_parquet_to_shards(
                    parquet_files, SHARD_DIR,
                    n_eval=EVAL_POSITIONS,
                    rows_per_shard=3_000_000,
                )
            except Exception as e:
                pretok_error[0] = e
                traceback.print_exc()

        pretok_thread = threading.Thread(target=_pretok_worker, daemon=True)
        pretok_thread.start()
        print("  Pretokenization running in background thread")

        # Wait for eval.pt (written after all shards of first parquet)
        print("  Waiting for eval.pt...", flush=True)
        t_wait = time.time()
        while not eval_path.exists():
            if pretok_error[0] is not None:
                raise RuntimeError(f"Pretokenization failed: {pretok_error[0]}")
            time.sleep(0.5)
        print(f"  eval.pt ready ({time.time() - t_wait:.1f}s)")
    else:
        print(f"  Found {len(shard_files)} pretokenized shards in {SHARD_DIR}")

    # Build eval data
    eval_data, eval_tensors = build_eval_from_pretokenized(eval_path, "fused")

    data_time = time.time() - t_data_start
    print(f"  Data ready in {data_time:.0f}s ({data_time/60:.1f}m)")
    print(f"  Eval: {len(eval_data):,} positions")

    # ── Phase 2: Build model ──
    print(f"\n{'='*72}")
    print(f" PHASE 2: MODEL + TRAINING")
    print(f"{'='*72}")

    print("\nBuilding model...")
    model = ChessTransformer200M().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Trainable params: {n_trainable:,} ({n_trainable/1e6:.1f}M)")

    for name, module in [("encoder", model.encoder), ("input_proj", model.input_proj),
                         ("transformer", model.transformer), ("norm", model.norm),
                         ("policy_head", model.policy_head), ("value_head", model.value_head)]:
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:15s}: {n:>12,} ({n/1e6:.1f}M)")
    cls_n = model.cls_token.numel() + model.pos_embed.numel()
    print(f"    {'cls+pos_embed':15s}: {cls_n:>12,}")

    # ── Create loader — supports concurrent pretokenization ──
    if expected_shards is not None:
        # Background pretokenization still running: estimate total positions
        # Each parquet shard has ~3M rows → ~3M positions per shard
        n_train_estimate = expected_shards * 3_000_000
        print(f"\n  Estimated train positions: ~{n_train_estimate:,} (concurrent mode)")
        n_train = n_train_estimate
    else:
        loader_probe = ShardedChessLoader(
            SHARD_DIR, batch_size=BATCH_SIZE, encoder_type="fused",
            device=device, seed=SEED, drop_last=True, skip_positions=0,
        )
        n_train = loader_probe.total_positions
        del loader_probe

    # ── Optimizer & scheduler ──
    steps_per_epoch = n_train // BATCH_SIZE
    total_opt_steps = steps_per_epoch // ACCUM_STEPS
    warmup_steps = max(int(total_opt_steps * WARMUP_FRAC), 100)
    print(f"\n  Train positions: {n_train:,}")
    print(f"  Steps/epoch: {steps_per_epoch:,} micro, {total_opt_steps:,} optimizer")
    print(f"  Warmup: {warmup_steps} steps")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                      betas=(0.9, 0.95))
    scaler = GradScaler('cuda')

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Resume from checkpoint ──
    global_step, positions_seen, best_acc, results_log = load_checkpoint(
        model, optimizer, scaler, scheduler, device
    )
    best_state = None

    # ── torch.compile for ~1.4x throughput (benchmarked: 477 vs 342 pos/s on A40) ──
    print("  Compiling model with torch.compile...")
    model = torch.compile(model)

    # ── Create loader with resume skip ──
    loader = ShardedChessLoader(
        SHARD_DIR, batch_size=BATCH_SIZE, encoder_type="fused",
        device=device, seed=SEED, drop_last=True,
        skip_positions=positions_seen,
        expected_shards=expected_shards,
    )

    # ── Save config ──
    config_dict = {
        "experiment": "exp073_200m_full_epoch",
        "model": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "ffn_ratio": FFN_RATIO, "dropout": DROPOUT,
            "policy_head_dim": POLICY_HEAD_DIM,
            "value_hidden": VALUE_HIDDEN,
            "total_params": n_params,
        },
        "training": {
            "n_train": n_train, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * ACCUM_STEPS,
            "lr": LR, "warmup_frac": WARMUP_FRAC,
            "min_lr_frac": MIN_LR_FRAC,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "value_weight": VALUE_WEIGHT,
        },
        "data": {
            "dataset": "avewright/chess-positions-lichess-sf",
            "pipeline": "ShardedChessLoader",
            "shard_dir": str(SHARD_DIR),
            "eval_positions": EVAL_POSITIONS,
        },
        "seed": SEED,
        "total_opt_steps": total_opt_steps,
        "warmup_steps": warmup_steps,
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Initial eval ──
    print("\n  Initial evaluation...")
    ev0 = evaluate(model, eval_data, eval_tensors, device)
    print(f"  Baseline:  acc={ev0['accuracy']:.1%} "
          f"top3={ev0['top3_accuracy']:.1%} "
          f"sf_rank={ev0['mean_sf_rank']:.1f} "
          f"val={ev0['value_accuracy']:.1%}")
    if global_step == 0:
        results_log.append({
            "step": 0, "positions_seen": 0,
            "type": "initial",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev0.items()},
        })
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(results_log, f, indent=2)

    # ── Training loop ──
    model.train()
    t_start = time.time()
    micro_step = global_step * ACCUM_STEPS
    running_pl = 0.0
    running_vl = 0.0
    running_batches = 0
    grad_norm = 0.0
    start_positions = positions_seen
    n_train_finalized = (expected_shards is None)  # track if we've got true total

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'─'*72}")
    print(f" Training started ({n_train:,} positions, {total_opt_steps:,} opt steps)")
    if positions_seen > 0:
        print(f" Resuming from position {positions_seen:,}")
    print(f"{'─'*72}\n")

    for batch_input, move_targets, wdl_targets in loader:
        if SHUTDOWN_REQUESTED:
            break

        bs = move_targets.shape[0]

        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            policy_loss = F.cross_entropy(result["policy_logits"], move_targets)
            value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
            value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")
            loss = (policy_loss + VALUE_WEIGHT * value_loss) / ACCUM_STEPS

        # NaN guard: skip this micro-batch if loss is NaN/Inf
        if not torch.isfinite(loss):
            nan_count = getattr(train, '_nan_count', 0) + 1
            train._nan_count = nan_count
            print(f"\n  [WARN] NaN/Inf loss at micro_step {micro_step} "
                  f"(pl={policy_loss.item():.4f}, vl={value_loss.item():.4f}, "
                  f"scale={scaler.get_scale():.0f}). "
                  f"Skipping batch ({nan_count} total).", flush=True)
            optimizer.zero_grad()  # discard poisoned gradients
            micro_step += 1
            positions_seen += bs
            if nan_count >= 20:
                print("  [ERROR] 20 consecutive NaN losses. Saving & stopping.", flush=True)
                save_checkpoint(model, optimizer, scaler, scheduler,
                                global_step, positions_seen, best_acc, results_log, "nan_crash")
                return
            continue
        else:
            train._nan_count = 0  # reset counter on good batch

        scaler.scale(loss).backward()
        micro_step += 1

        running_pl += policy_loss.item()
        running_vl += value_loss.item()
        running_batches += 1
        positions_seen += bs

        # Optimizer step after accumulation
        if micro_step % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # ── Finalize n_train once pretokenization completes ──
            if not n_train_finalized and pretok_thread is not None:
                if not pretok_thread.is_alive():
                    pretok_thread.join()
                    loader.update_total()
                    n_train = loader.total_positions
                    steps_per_epoch = n_train // BATCH_SIZE
                    total_opt_steps = steps_per_epoch // ACCUM_STEPS
                    print(f"\n  ** Pretokenization done. Actual: {n_train:,} positions, "
                          f"{total_opt_steps:,} opt steps")
                    n_train_finalized = True

            # ── Heartbeat ──
            if global_step % HEARTBEAT_INTERVAL == 0:
                elapsed = time.time() - t_start
                tp = (positions_seen - start_positions) / max(elapsed, 0.1)
                progress = positions_seen / n_train * 100
                remaining = n_train - positions_seen
                eta_s = remaining / max(tp, 1)
                eta_h = eta_s / 3600
                sys.stdout.write(
                    f"\r    step {global_step:>7,}/{total_opt_steps:,} | "
                    f"pos {positions_seen:>11,}/{n_train:,} ({progress:5.1f}%) | "
                    f"{tp:.0f} pos/s | ETA {eta_h:.1f}h"
                )
                sys.stdout.flush()

            # ── Detailed log ──
            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                tp = (positions_seen - start_positions) / max(elapsed, 0.1)
                lr_now = scheduler.get_last_lr()[0]
                avg_pl = running_pl / max(running_batches, 1)
                avg_vl = running_vl / max(running_batches, 1)
                progress = positions_seen / n_train * 100
                peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                print(f"\n    [{global_step:>7,}/{total_opt_steps:,}] "
                      f"pl={avg_pl:.4f} vl={avg_vl:.4f} "
                      f"| lr={lr_now:.2e} | gnorm={grad_norm:.2f} "
                      f"| {tp:.0f} pos/s | {progress:.1f}% "
                      f"| mem={peak_mem:.1f}GB",
                      flush=True)

                running_pl = 0.0
                running_vl = 0.0
                running_batches = 0

            # ── Evaluation ──
            if global_step % EVAL_INTERVAL == 0:
                ev = evaluate(model, eval_data, eval_tensors, device)
                elapsed = time.time() - t_start
                print(f"\n    ** EVAL step {global_step}: "
                      f"acc={ev['accuracy']:.1%} "
                      f"top3={ev['top3_accuracy']:.1%} "
                      f"sf_rank={ev['mean_sf_rank']:.1f} "
                      f"val={ev['value_accuracy']:.1%} "
                      f"({elapsed/60:.0f}m elapsed)")
                for phase, acc in sorted(ev['phase_accuracy'].items()):
                    print(f"       {phase}: {acc:.1%}")

                results_log.append({
                    "step": global_step,
                    "positions_seen": positions_seen,
                    "type": "eval",
                    "elapsed_s": round(elapsed),
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in ev.items()},
                })

                if ev["accuracy"] > best_acc:
                    best_acc = ev["accuracy"]
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    print(f"    ** New best: {best_acc:.1%}")
                    torch.save(best_state, OUTPUT_DIR / "best_model.pt")

                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(results_log, f, indent=2)

                model.train()

            # ── Periodic checkpoint ──
            if global_step % SAVE_INTERVAL == 0:
                save_checkpoint(
                    model, optimizer, scaler, scheduler,
                    global_step, positions_seen, best_acc, results_log,
                )

    # ── End of training ──
    total_time = time.time() - t_start
    print(f"\n\n{'='*72}")

    if SHUTDOWN_REQUESTED:
        print(" GRACEFUL SHUTDOWN — saving final state")
    else:
        print(" TRAINING COMPLETE")

    # Final evaluation
    ev_final = evaluate(model, eval_data, eval_tensors, device)
    print(f"\n  Final eval: acc={ev_final['accuracy']:.1%} "
          f"top3={ev_final['top3_accuracy']:.1%} "
          f"sf_rank={ev_final['mean_sf_rank']:.1f} "
          f"val={ev_final['value_accuracy']:.1%}")
    for phase, acc in sorted(ev_final['phase_accuracy'].items()):
        print(f"    {phase}: {acc:.1%}")

    results_log.append({
        "step": global_step,
        "positions_seen": positions_seen,
        "type": "final",
        "elapsed_s": round(total_time),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev_final.items()},
    })

    if ev_final["accuracy"] > best_acc:
        best_acc = ev_final["accuracy"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save everything
    save_checkpoint(
        model, optimizer, scaler, scheduler,
        global_step, positions_seen, best_acc, results_log, tag="final",
    )

    if best_state is not None:
        torch.save(best_state, OUTPUT_DIR / "best_model.pt")
        print(f"\n  Best model saved ({best_acc:.1%})")

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    print(f"  Final model saved")

    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(results_log, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f" SUMMARY")
    print(f"{'='*72}")
    print(f"  Model:           {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  Positions seen:  {positions_seen:,}")
    print(f"  Optimizer steps: {global_step:,}")
    print(f"  Best accuracy:   {best_acc:.1%}")
    print(f"  Total time:      {total_time:.0f}s ({total_time/3600:.1f}h)")
    if total_time > 0:
        print(f"  Throughput:      {positions_seen / total_time:.0f} pos/s")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU mem:    {peak:.1f} GB")
    print(f"  Output:          {OUTPUT_DIR}")
    print(f"{'='*72}\n")

    # Write final summary
    summary = {
        "experiment": "exp073_200m_full_epoch",
        "model_params": n_params,
        "positions_seen": positions_seen,
        "global_steps": global_step,
        "best_accuracy": round(best_acc, 4),
        "total_time_s": round(total_time),
        "throughput_pos_s": round(positions_seen / max(total_time, 1)),
        "results_log": results_log,
        "config": config_dict,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results_log


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        # Emergency checkpoint
        print("Attempting emergency save...")
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_DIR / "crash_log.txt", "w") as f:
                f.write(f"Crash at {datetime.now(timezone.utc).isoformat()}\n")
                f.write(traceback.format_exc())
            print("Crash log saved.")
        except Exception:
            pass
        sys.exit(1)
