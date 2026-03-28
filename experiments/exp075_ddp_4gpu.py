"""exp075: Continue training the 200M v2 model on 4xA40 GPUs via Local SGD.

Hypothesis: 4x GPU throughput on the full source-sharded corpus (~832M
positions) will train significantly further than single-GPU exp074.

Architecture: identical to exp073/074 -- ChessTransformer200M
  - FusedBoardEncoder 256d -> 1024d transformer
  - 16 layers, 16 heads, FFN 4x (4096), GELU, norm_first
  - SpatialPolicyHead (head_dim=512), WDL value head
  - ~204M parameters

Strategy:
  - Local SGD: 4 independent processes, one per GPU
  - Each process trains on 1/4 of the 3275 source parquet files
  - Download best_model.pt from avewright/chess-transformer-200m-v2 on HF
  - Every SYNC_INTERVAL optimizer steps, workers write their weights to
    shared directory; a simple file-based barrier syncs them; each worker
    loads the averaged weights and continues.
  - torch.compile for max single-GPU throughput
  - Batch 256, gradient accumulation 4 (effective 1024 per worker)
  - Lower LR (1e-4 cosine -> 5% floor) since we're continuing

GPU: 4x NVIDIA A40 46GB

Usage:
  python experiments/exp075_ddp_4gpu.py
"""

import gc
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, move_to_index, legal_move_mask
from data_loader import (
    get_eval_batch_input, compute_wdl, compute_phase,
    StreamingHFChessLoader, build_eval_from_hf, get_hf_dataset_layout,
)

# ── Paths ──
OUTPUT_DIR = Path("outputs/exp075_ddp_4gpu")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SYNC_DIR = OUTPUT_DIR / "sync"
HF_DATASET = "avewright/chess-positions-lichess-sf"
HF_MODEL = "avewright/chess-transformer-200m-v2"

# ── Config ──
EVAL_POSITIONS = 5_000
NUM_GPUS = min(torch.cuda.device_count(), 4)

# Training -- each worker does batch=256 x accum=4 = eff 1024
BATCH_SIZE = 256
ACCUM_STEPS = 4
LR = 1e-4
WARMUP_FRAC = 0.01
MIN_LR_FRAC = 0.05
VALUE_WEIGHT = 0.5
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
SEED = 42

# Local SGD sync interval (in optimizer steps per worker)
SYNC_INTERVAL = 500

# Model dims -- must match exp073/074 exactly
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

# Logging/checkpoint intervals (in optimizer steps)
LOG_INTERVAL = 100
EVAL_INTERVAL = 1000
SAVE_INTERVAL = 250
HEARTBEAT_INTERVAL = 25

# ── Graceful shutdown ──
SHUTDOWN_REQUESTED = False


def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Graceful shutdown requested (signal {signum}). "
          "Will save checkpoint after current step...", flush=True)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Model (identical to exp073/074) ──

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
                    positions_seen, best_acc, results_log, tag="",
                    data_cursor=None, worker_id=0):
    ckpt_dir = CHECKPOINT_DIR / f"worker{worker_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{global_step}" if not tag else tag
    ckpt_path = ckpt_dir / f"{name}.pt"

    sd = {k.replace("_orig_mod.", ""): v
          for k, v in model.state_dict().items()}

    ckpt = {
        "model_state_dict": sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "positions_seen": positions_seen,
        "best_acc": best_acc,
        "results_log": results_log,
        "data_cursor": data_cursor,
        "worker_id": worker_id,
        "config": {
            "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS, "ffn_ratio": FFN_RATIO,
            "encoder_dim": ENCODER_DIM, "batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS, "lr": LR, "seed": SEED,
            "num_gpus": NUM_GPUS,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(ckpt, ckpt_path)
    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"    [W{worker_id}][CKPT] Saved {ckpt_path.name} ({size_mb:.0f} MB)", flush=True)

    latest_path = ckpt_dir / "latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    import shutil
    shutil.copy2(ckpt_path, latest_path)


def load_checkpoint(model, optimizer, scaler, device, worker_id=0):
    latest_path = CHECKPOINT_DIR / f"worker{worker_id}" / "latest.pt"
    if not latest_path.exists():
        return None

    print(f"  [W{worker_id}] Resuming from checkpoint: {latest_path}", flush=True)
    ckpt = torch.load(latest_path, map_location=device, weights_only=False)

    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    return {
        "global_step": ckpt["global_step"],
        "positions_seen": ckpt["positions_seen"],
        "best_acc": ckpt["best_acc"],
        "results_log": ckpt.get("results_log", []),
        "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
        "data_cursor": ckpt.get("data_cursor"),
    }


def load_best_model_from_hf(model, device):
    from huggingface_hub import hf_hub_download

    cache_dir = OUTPUT_DIR / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading best_model.pt from {HF_MODEL}...", flush=True)
    downloaded = hf_hub_download(
        HF_MODEL, "best_model.pt",
        local_dir=str(cache_dir),
    )
    best_path = Path(downloaded)

    state_dict = torch.load(best_path, map_location=device, weights_only=True)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    print(f"  Loaded v2 weights OK ({best_path.stat().st_size / 1e6:.0f} MB)", flush=True)


# ── Local SGD weight averaging ──

def write_worker_weights(model, worker_id, sync_step):
    """Write this worker's state_dict to shared sync directory."""
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    path = SYNC_DIR / f"worker{worker_id}_step{sync_step}.pt"
    sd = {k.replace("_orig_mod.", ""): v.cpu()
          for k, v in model.state_dict().items()}
    torch.save(sd, path)
    (SYNC_DIR / f"worker{worker_id}_step{sync_step}.ready").touch()
    return path


def wait_and_average_weights(model, worker_id, sync_step, device, timeout=300):
    """Wait for all workers to write weights, then load averaged weights.
    
    Loads all state dicts to CPU to avoid GPU OOM, averages on CPU,
    then loads the averaged result back to the model on GPU.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        ready = sum(
            1 for i in range(NUM_GPUS)
            if (SYNC_DIR / f"worker{i}_step{sync_step}.ready").exists()
        )
        if ready >= NUM_GPUS:
            break
        time.sleep(0.5)
    else:
        print(f"    [W{worker_id}] Sync timeout at step {sync_step}, "
              f"only {ready}/{NUM_GPUS} ready. Continuing without sync.", flush=True)
        return False

    # Load and average all worker weights ON CPU to avoid GPU OOM
    avg_state = None
    for i in range(NUM_GPUS):
        path = SYNC_DIR / f"worker{i}_step{sync_step}.pt"
        sd = torch.load(path, map_location='cpu', weights_only=True)
        if avg_state is None:
            avg_state = sd
        else:
            for k in avg_state:
                avg_state[k] += sd[k]
        del sd

    for k in avg_state:
        avg_state[k] /= NUM_GPUS

    # Move averaged weights to device and load into model
    avg_state = {k: v.to(device) for k, v in avg_state.items()}

    # Handle torch.compile prefix
    current_keys = set(model.state_dict().keys())
    if any(k.startswith("_orig_mod.") for k in current_keys):
        avg_state = {"_orig_mod." + k: v for k, v in avg_state.items()}

    model.load_state_dict(avg_state)
    del avg_state
    gc.collect()
    torch.cuda.empty_cache()

    # Cleanup ALL sync files for this step
    for f in SYNC_DIR.glob("*step*"):
        f.unlink(missing_ok=True)

    return True


# ── Worker training function ──

def train_worker(worker_id):
    """Train on 1/NUM_GPUS of the data on a single GPU."""
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    torch.manual_seed(SEED + worker_id)
    random.seed(SEED + worker_id)
    torch.cuda.manual_seed(SEED + worker_id)

    worker_dir = OUTPUT_DIR / f"worker{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_path = worker_dir / "train.log"

    log_file = open(log_path, "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"\n\n{'#'*72}", flush=True)
    print(f"# Worker {worker_id} starting at {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"{'#'*72}", flush=True)

    try:
        _train_worker_inner(worker_id, device, worker_dir)
    except Exception as e:
        print(f"\n  [W{worker_id}] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
    finally:
        log_file.close()


def _train_worker_inner(worker_id, device, worker_dir):
    # Clean stale sync files from previous runs (worker 0 does this)
    if worker_id == 0:
        for f in SYNC_DIR.glob("*"):
            f.unlink(missing_ok=True)

    print(f"\n{'='*72}")
    print(f" EXP075 WORKER {worker_id} on GPU {worker_id} "
          f"({torch.cuda.get_device_name(worker_id)})")
    print(f"{'='*72}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(worker_id).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(worker_id), 'total_mem')
          else f"  VRAM: {torch.cuda.get_device_properties(worker_id).total_memory / 1e9:.1f} GB")
    print(flush=True)

    # Data split -- deterministic across all workers
    layout = get_hf_dataset_layout(HF_DATASET)
    all_src_files = sorted(layout["train_src"])
    n_files = len(all_src_files)

    files_per_worker = n_files // NUM_GPUS
    my_start = worker_id * files_per_worker
    my_end = my_start + files_per_worker if worker_id < NUM_GPUS - 1 else n_files

    print(f"  Data split: files [{my_start}:{my_end}) = "
          f"{my_end - my_start}/{n_files} files, "
          f"~{(my_end - my_start) * 254_000 / 1e6:.0f}M positions")

    # Build model
    model = ChessTransformer200M().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params", flush=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                      betas=(0.9, 0.95), fused=True)
    scaler = GradScaler('cuda', init_scale=2**14)

    own_ckpt = load_checkpoint(model, optimizer, scaler, device, worker_id)
    resume_cursor = None

    if own_ckpt is not None:
        global_step = own_ckpt["global_step"]
        positions_seen = own_ckpt["positions_seen"]
        best_acc = own_ckpt["best_acc"]
        results_log = own_ckpt["results_log"]
        resume_cursor = own_ckpt.get("data_cursor")
        print(f"  Resumed: step={global_step}, pos={positions_seen:,}, "
              f"best={best_acc:.1%}", flush=True)
    else:
        load_best_model_from_hf(model, device)
        global_step = 0
        positions_seen = 0
        best_acc = 0.0
        results_log = []
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                          betas=(0.9, 0.95), fused=True)
        scaler = GradScaler('cuda', init_scale=2**14)

    # No torch.compile -- saves ~20GB VRAM, allows larger batch size
    # model = torch.compile(model)

    # Eval data (worker 0 only)
    eval_data = eval_tensors = None
    if worker_id == 0:
        eval_data, eval_tensors = build_eval_from_hf(
            HF_DATASET, n_eval=EVAL_POSITIONS, encoder_type="fused"
        )

    # Streaming data loader for this worker's file slice
    loader = StreamingHFChessLoader(
        HF_DATASET, batch_size=BATCH_SIZE, encoder_type="fused",
        device=device, seed=SEED + worker_id * 1000, drop_last=True,
        file_pattern="src",
        start_file=my_start,
        max_files=my_end - my_start,
        resume_cursor=resume_cursor,
    )
    n_train = loader.total_positions

    steps_per_epoch = n_train // BATCH_SIZE
    total_opt_steps = steps_per_epoch // ACCUM_STEPS
    warmup_steps = max(int(total_opt_steps * WARMUP_FRAC), 50)

    print(f"  Training: ~{n_train/1e6:.0f}M positions, ~{total_opt_steps:,} opt steps")
    print(f"  Batch: {BATCH_SIZE} x accum={ACCUM_STEPS} "
          f"(eff={BATCH_SIZE * ACCUM_STEPS}), lr={LR}")
    print(f"  Warmup: {warmup_steps} steps, sync every {SYNC_INTERVAL} steps")

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if own_ckpt is not None and own_ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(own_ckpt["scheduler_state_dict"])

    best_state = None

    # Initial eval (worker 0 only)
    if worker_id == 0 and global_step == 0:
        print("\n  Initial evaluation...", flush=True)
        ev0 = evaluate(model, eval_data, eval_tensors, device)
        print(f"  Loaded model: acc={ev0['accuracy']:.1%} "
              f"top3={ev0['top3_accuracy']:.1%} "
              f"sf_rank={ev0['mean_sf_rank']:.1f} "
              f"val={ev0['value_accuracy']:.1%}")
        for phase, acc in sorted(ev0['phase_accuracy'].items()):
            print(f"    {phase}: {acc:.1%}")

        best_acc = ev0["accuracy"]
        results_log.append({
            "step": 0, "positions_seen": 0,
            "type": "initial_from_v2",
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in ev0.items()},
        })
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(results_log, f, indent=2)
        best_state = {k.replace("_orig_mod.", ""): v.cpu().clone()
                      for k, v in model.state_dict().items()}
        torch.save(best_state, OUTPUT_DIR / "best_model.pt")
        print(f"  Saved baseline as best_model.pt (acc={best_acc:.1%})", flush=True)

    # Training loop
    model.train()
    t_start = time.time()
    micro_step = global_step * ACCUM_STEPS
    running_pl = 0.0
    running_vl = 0.0
    running_batches = 0
    grad_norm = 0.0
    start_positions = positions_seen
    nan_count = 0
    next_sync_step = ((global_step // SYNC_INTERVAL) + 1) * SYNC_INTERVAL

    torch.cuda.reset_peak_memory_stats(device)

    print(f"\n{'---'*24}")
    print(f" [W{worker_id}] Training started (~{n_train/1e6:.0f}M positions, "
          f"~{total_opt_steps:,} opt steps)")
    print(f"{'---'*24}\n", flush=True)

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

        # NaN guard
        if not torch.isfinite(loss):
            nan_count += 1
            print(f"\n  [W{worker_id}][WARN] NaN/Inf loss at micro_step {micro_step} "
                  f"(pl={policy_loss.item():.4f}, vl={value_loss.item():.4f}, "
                  f"scale={scaler.get_scale():.0f}). "
                  f"Skipping ({nan_count} total).", flush=True)
            optimizer.zero_grad()
            micro_step += 1
            positions_seen += bs
            if nan_count >= 20:
                print(f"  [W{worker_id}][ERROR] 20 NaN losses. Saving & stopping.",
                      flush=True)
                save_checkpoint(model, optimizer, scaler, scheduler,
                                global_step, positions_seen, best_acc,
                                results_log, "nan_crash",
                                data_cursor=loader.get_cursor(),
                                worker_id=worker_id)
                return
            continue
        else:
            nan_count = 0

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

            # Heartbeat
            if global_step % HEARTBEAT_INTERVAL == 0:
                elapsed = time.time() - t_start
                new_positions = positions_seen - start_positions
                tp = new_positions / max(elapsed, 0.1)
                progress = positions_seen / n_train * 100
                remaining_pos = max(n_train - positions_seen, 0)
                eta_h = (remaining_pos / max(tp, 1)) / 3600
                sys.stdout.write(
                    f"\r    [W{worker_id}] step {global_step:>7,}/{total_opt_steps:,} | "
                    f"pos {positions_seen:>11,}/{n_train:,} ({progress:5.1f}%) | "
                    f"{tp:.0f} pos/s | ETA {eta_h:.1f}h"
                )
                sys.stdout.flush()

            # Detailed log
            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                new_positions = positions_seen - start_positions
                tp = new_positions / max(elapsed, 0.1)
                lr_now = scheduler.get_last_lr()[0]
                avg_pl = running_pl / max(running_batches, 1)
                avg_vl = running_vl / max(running_batches, 1)
                progress = positions_seen / n_train * 100
                peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

                print(f"\n    [W{worker_id}][{global_step:>7,}/{total_opt_steps:,}] "
                      f"pl={avg_pl:.4f} vl={avg_vl:.4f} "
                      f"| lr={lr_now:.2e} | gnorm={grad_norm:.2f} "
                      f"| {tp:.0f} pos/s | {progress:.1f}% "
                      f"| pos={positions_seen:,} "
                      f"| mem={peak_mem:.1f}GB",
                      flush=True)

                running_pl = 0.0
                running_vl = 0.0
                running_batches = 0

            # Local SGD sync + checkpoint (always save at sync points)
            if global_step >= next_sync_step:
                sync_id = next_sync_step  # use planned step as sync key
                print(f"\n    [W{worker_id}] Sync at step {global_step} "
                      f"(sync_id={sync_id})...", flush=True)
                write_worker_weights(model, worker_id, sync_id)
                synced = wait_and_average_weights(model, worker_id, sync_id, device)
                if synced:
                    print(f"    [W{worker_id}] Weights averaged with {NUM_GPUS} workers",
                          flush=True)
                # Always save checkpoint at sync points for safety
                save_checkpoint(
                    model, optimizer, scaler, scheduler,
                    global_step, positions_seen, best_acc, results_log,
                    data_cursor=loader.get_cursor(),
                    worker_id=worker_id,
                )
                next_sync_step += SYNC_INTERVAL

            # Evaluation (worker 0 only)
            if worker_id == 0 and global_step % EVAL_INTERVAL == 0:
                ev = evaluate(model, eval_data, eval_tensors, device)
                elapsed = time.time() - t_start
                print(f"\n    [W0] ** EVAL step {global_step}: "
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
                    "data_cursor": loader.get_cursor(),
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in ev.items()},
                })

                if ev["accuracy"] > best_acc:
                    best_acc = ev["accuracy"]
                    best_state = {k.replace("_orig_mod.", ""): v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    print(f"    ** New best: {best_acc:.1%}")
                    torch.save(best_state, OUTPUT_DIR / "best_model.pt")

                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(results_log, f, indent=2)

                model.train()

            # Periodic checkpoint
            if global_step % SAVE_INTERVAL == 0:
                save_checkpoint(
                    model, optimizer, scaler, scheduler,
                    global_step, positions_seen, best_acc, results_log,
                    data_cursor=loader.get_cursor(),
                    worker_id=worker_id,
                )

    # End of training
    total_time = time.time() - t_start
    print(f"\n\n{'='*72}")

    if SHUTDOWN_REQUESTED:
        print(f" [W{worker_id}] GRACEFUL SHUTDOWN")
    else:
        print(f" [W{worker_id}] TRAINING COMPLETE")

    if worker_id == 0 and eval_data is not None:
        ev_final = evaluate(model, eval_data, eval_tensors, device)
        print(f"\n  Final eval: acc={ev_final['accuracy']:.1%} "
              f"top3={ev_final['top3_accuracy']:.1%} "
              f"sf_rank={ev_final['mean_sf_rank']:.1f} "
              f"val={ev_final['value_accuracy']:.1%}")

        results_log.append({
            "step": global_step,
            "positions_seen": positions_seen,
            "type": "final",
            "elapsed_s": round(total_time),
            "data_cursor": loader.get_cursor(),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in ev_final.items()},
        })

        if ev_final["accuracy"] > best_acc:
            best_acc = ev_final["accuracy"]
            best_state = {k.replace("_orig_mod.", ""): v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if best_state is not None:
            torch.save(best_state, OUTPUT_DIR / "best_model.pt")
            print(f"\n  Best model saved ({best_acc:.1%})")

        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(results_log, f, indent=2)

    save_checkpoint(
        model, optimizer, scaler, scheduler,
        global_step, positions_seen, best_acc, results_log, tag="final",
        data_cursor=loader.get_cursor(),
        worker_id=worker_id,
    )

    final_sd = {k.replace("_orig_mod.", ""): v
                for k, v in model.state_dict().items()}
    torch.save(final_sd, worker_dir / "final_model.pt")

    new_positions = positions_seen - start_positions
    print(f"\n  [W{worker_id}] Summary:")
    print(f"    Positions: {positions_seen:,} (new: {new_positions:,})")
    print(f"    Steps: {global_step:,}")
    print(f"    Time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    if total_time > 0:
        print(f"    Throughput: {new_positions / total_time:.0f} pos/s")
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"    Peak GPU mem: {peak:.1f} GB")
    print(flush=True)


# ── Orchestrator ──

def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SYNC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f" EXP075: LOCAL SGD -- {NUM_GPUS}xA40 MULTI-GPU TRAINING")
    print(f"{'='*72}")
    print(f"  Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    print(f"  GPUs:       {NUM_GPUS}x {torch.cuda.get_device_name(0)}")
    for i in range(NUM_GPUS):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}, {vram:.1f} GB")
    print(f"  Model:      FusedBoardEncoder {ENCODER_DIM}d -> {HIDDEN_DIM}d, "
          f"{NUM_LAYERS}L, {NUM_HEADS}H, FFN {FFN_RATIO}x")
    print(f"  Resume from: {HF_MODEL}")
    print(f"  Strategy:   Local SGD -- {NUM_GPUS} independent workers, "
          f"sync every {SYNC_INTERVAL} steps")
    print(f"  Per-worker: batch={BATCH_SIZE} x accum={ACCUM_STEPS} "
          f"(eff={BATCH_SIZE * ACCUM_STEPS}), lr={LR}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(flush=True)

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    processes = []
    for i in range(NUM_GPUS):
        p = mp.Process(target=train_worker, args=(i,), name=f"worker-{i}")
        p.start()
        processes.append(p)
        print(f"  Launched worker {i} (PID {p.pid}) on GPU {i}")
        time.sleep(2)

    print(f"\n  All {NUM_GPUS} workers launched. Monitoring...\n", flush=True)

    while True:
        alive = sum(1 for p in processes if p.is_alive())
        if alive == 0:
            break
        time.sleep(30)

        status = []
        for i, p in enumerate(processes):
            log_path = OUTPUT_DIR / f"worker{i}" / "train.log"
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                    for line in reversed(lines):
                        if "step" in line and "pos/s" in line:
                            status.append(f"  W{i}: {line.strip()}")
                            break
                    else:
                        if lines:
                            status.append(f"  W{i}: {lines[-1].strip()[:80]}")
                        else:
                            status.append(f"  W{i}: empty log")
                except Exception:
                    status.append(f"  W{i}: ???")
            else:
                status.append(f"  W{i}: no log yet")

        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
              f"{alive}/{NUM_GPUS} workers alive")
        for s in status:
            print(s)
        sys.stdout.flush()

    for i, p in enumerate(processes):
        p.join()
        print(f"  Worker {i} exited with code {p.exitcode}")

    print(f"\n{'='*72}")
    print(f" ALL WORKERS COMPLETE")
    print(f"{'='*72}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Worker logs: {OUTPUT_DIR}/worker*/train.log")
    print(f"  Best model: {OUTPUT_DIR}/best_model.pt")


if __name__ == "__main__":
    train()
