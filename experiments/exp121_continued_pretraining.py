"""exp121: Continued pre-training on 832M HF positions.

Background: The current best model (~1831 ELO) was pre-trained on 832M positions
(exp076/exp083) then fine-tuned on 224K multi-PV positions (exp084). Every attempt
to improve ELO via small-data fine-tuning (exp101-116) has failed.

Strategy: Go back to what works — continued pre-training on the full 832M corpus
with the same loss formulation (CE on best_move + KL on WDL). The model may not be
fully converged after the original ~2 epochs.

Key parameters (derived from exp076/exp083 lessons):
  - LR=1e-5 (exp083c proved safe for continuation)
  - MIN_LR_FRAC=0.20 (20% floor = 2e-6)
  - VALUE_WEIGHT=0.5 (same as original pre-training)
  - Loss = CE(policy, best_move) + 0.5 * KL_div(value_logits, wdl_targets)
  - WDL targets from cp/mate (White-absolute, matching pre-training convention)

Hardware: 1× NVIDIA A40 46GB
Expected throughput: ~900-1000 pos/s
Expected time for full corpus: ~10 days (1 pass)

Success criterion: ELO improvement over baseline 1831 on 128-game eval vs SF1900.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# ── GPU performance flags ──
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, move_to_index, legal_move_mask
from data_loader import (
    get_eval_batch_input, compute_wdl, compute_phase,
    StreamingHFChessLoader, build_eval_from_hf, get_hf_dataset_layout,
)

# ── Paths ──
OUTPUT_DIR = Path("outputs/exp121_continued_pretraining")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HF_DATASET = "avewright/chess-positions-lichess-sf"
INIT_CHECKPOINT = Path("outputs/hf_checkpoint/best_model.pt")

# ── Config ──
EVAL_POSITIONS = 20_000

# Training — ultra-safe LR for continuation (exp083c proved 1e-5 is safe)
BATCH_SIZE = 256
ACCUM_STEPS = 4           # effective batch = 1024
LR = 1e-5                 # exp083c proved this is safe
WARMUP_STEPS = 200        # short fixed warmup (model already trained)
MIN_LR_FRAC = 0.20        # cosine decays to 20% of peak (2e-6)
VALUE_WEIGHT = 0.5        # same as original pre-training
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
SEED = 121

# Model dims — must match exactly
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

# Logging/checkpoint intervals (in optimizer steps)
LOG_INTERVAL = 200
EVAL_INTERVAL = 5000       # eval every 5K opt steps (~5M positions)
SAVE_INTERVAL = 2000
HEARTBEAT_INTERVAL = 50
ELO_EVAL_INTERVAL = 50_000  # quick ELO check every 50K steps
ELO_GAMES = 32             # quick check, 128 for definitive

# HF upload
UPLOAD_TO_HF = True
HF_MODEL_REPO = "avewright/chess-transformer-200m-latest"

# ── Graceful shutdown ──
SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Graceful shutdown requested (signal {signum}). "
          "Will save checkpoint after current step...", flush=True)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Model (identical to exp076/083) ──

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


# ── ELO evaluation (quick check during training) ──

def quick_elo_eval(model, device, n_games=32, sf_elo=1900):
    """Run a quick ELO evaluation during training."""
    import subprocess
    import tempfile

    # Save model to temp file
    tmp_path = OUTPUT_DIR / "tmp_elo_model.pt"
    state_dict = {k.replace("_orig_mod.", ""): v.cpu().clone()
                  for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": state_dict}, tmp_path)

    try:
        result = subprocess.run(
            [sys.executable, "elo_eval_latest.py",
             str(tmp_path), "exp121_quick",
             "--elos", str(sf_elo),
             "--games-per-opening-per-color", "2",
             "--stop-after-bracket"],
            capture_output=True, text=True, timeout=600,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        # Parse output for score
        for line in result.stdout.split('\n'):
            if 'Score:' in line or 'score=' in line:
                print(f"    [ELO] {line.strip()}", flush=True)
        return result.stdout
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    [ELO] Quick eval failed: {e}", flush=True)
        return None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Checkpoint management ──

def save_checkpoint(model, optimizer, scaler, scheduler, global_step,
                    positions_seen, best_acc, results_log, tag="",
                    data_cursor=None):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"step_{global_step}" if not tag else tag
    ckpt_path = CHECKPOINT_DIR / f"{name}.pt"

    ckpt = {
        "model_state_dict": {k.replace("_orig_mod.", ""): v
                             for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "positions_seen": positions_seen,
        "best_acc": best_acc,
        "results_log": results_log,
        "data_cursor": data_cursor,
        "dataset_fingerprint": data_cursor.get("fingerprint") if data_cursor else None,
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
    cursor_info = ""
    if data_cursor:
        cursor_info = (f", cursor: file {data_cursor['files_completed']}/"
                       f"batch {data_cursor['batches_in_current_file']}")
    print(f"    [CKPT] Saved {ckpt_path.name} ({size_mb:.0f} MB{cursor_info})",
          flush=True)

    # Maintain latest.pt
    import shutil
    latest_path = CHECKPOINT_DIR / "latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    shutil.copy2(ckpt_path, latest_path)

    # Keep only last 5 step checkpoints
    if not tag:
        step_ckpts = sorted(CHECKPOINT_DIR.glob("step_*.pt"),
                           key=lambda p: p.stat().st_mtime)
        while len(step_ckpts) > 5:
            old = step_ckpts.pop(0)
            if old.name != "latest.pt":
                old.unlink()
                print(f"    [CKPT] Pruned old checkpoint: {old.name}", flush=True)

    return ckpt_path


def load_checkpoint(model, optimizer, scaler, device):
    """Load exp121's own latest checkpoint if resuming."""
    latest_path = CHECKPOINT_DIR / "latest.pt"
    if not latest_path.exists():
        return None

    print(f"  Resuming from own checkpoint: {latest_path}")
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
        "data_cursor": ckpt.get("data_cursor"),
        "scheduler_state_dict": ckpt.get("scheduler_state_dict"),
    }


def load_init_model(model, device):
    """Load model weights from the current best checkpoint."""
    if not INIT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {INIT_CHECKPOINT}")

    print(f"  Loading init model: {INIT_CHECKPOINT}")
    ckpt = torch.load(INIT_CHECKPOINT, map_location=device, weights_only=False)

    # Handle both raw state dict and wrapped format
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    print(f"  Loaded weights OK ({INIT_CHECKPOINT.stat().st_size / 1e6:.0f} MB)")


def _hf_token():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def upload_best_to_hf(model_path, step, accuracy, results_log):
    if not UPLOAD_TO_HF:
        return
    try:
        from huggingface_hub import HfApi, create_repo
        token = _hf_token()
        if not token:
            print("    [HF] No token, skipping upload", flush=True)
            return

        api = HfApi(token=token)
        try:
            create_repo(HF_MODEL_REPO, exist_ok=True, repo_type="model", token=token)
        except Exception:
            pass

        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="best_model.pt",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )

        log_path = OUTPUT_DIR / "training_log.json"
        if log_path.exists():
            api.upload_file(
                path_or_fileobj=str(log_path),
                path_in_repo="training_log.json",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )

        print(f"    [HF] Uploaded best_model.pt (step {step}, acc={accuracy:.1%})",
              flush=True)
    except Exception as e:
        print(f"    [HF] Upload failed: {e}", flush=True)


# ── Training ──

def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ──
    print(f"\n{'='*72}")
    print(f" EXP121: CONTINUED PRE-TRAINING ON 832M HF POSITIONS")
    print(f"{'='*72}")
    print(f"  Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Device:     {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:       {vram:.1f} GB")
    print(f"  Model:      ChessTransformer200M — FusedBoardEncoder {ENCODER_DIM}d → "
          f"{HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H, FFN {FFN_RATIO}×")
    print(f"  Init from:  {INIT_CHECKPOINT}")
    print(f"  Training:   batch={BATCH_SIZE} × accum={ACCUM_STEPS} "
          f"(eff={BATCH_SIZE * ACCUM_STEPS}), lr={LR}")
    print(f"  Loss:       CE(policy) + {VALUE_WEIGHT} × KL(value WDL)")
    print(f"  LR:         {LR} → {LR * MIN_LR_FRAC} (cosine, {WARMUP_STEPS} warmup)")
    print(f"  Grad clip:  {GRAD_CLIP}")
    print(f"  Seed:       {SEED}")
    print(f"  Output:     {OUTPUT_DIR}")
    print()

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── Data ──
    print("=" * 72)
    print(" DATA PREPARATION")
    print("=" * 72)

    eval_data, eval_tensors = build_eval_from_hf(
        HF_DATASET, n_eval=EVAL_POSITIONS, encoder_type="fused"
    )

    # ── Build model ──
    print(f"\n{'='*72}")
    print(f" MODEL + OPTIMIZER")
    print(f"{'='*72}")

    print("\nBuilding model...")
    model = ChessTransformer200M().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Trainable params: {n_trainable:,} ({n_trainable/1e6:.1f}M)")

    # ── Load weights ──
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                      betas=(0.9, 0.95), fused=True)
    scaler = GradScaler('cuda', init_scale=2**14)

    own_ckpt = load_checkpoint(model, optimizer, scaler, device)
    resume_cursor = None

    if own_ckpt is not None:
        global_step = own_ckpt["global_step"]
        positions_seen = own_ckpt["positions_seen"]
        best_acc = own_ckpt["best_acc"]
        results_log = own_ckpt["results_log"]
        resume_cursor = own_ckpt.get("data_cursor")
        print(f"  Resumed: step={global_step}, pos={positions_seen:,}, "
              f"best={best_acc:.1%}")
    else:
        load_init_model(model, device)
        global_step = 0
        positions_seen = 0
        best_acc = 0.0
        results_log = []
        # Re-create optimizer fresh
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                          betas=(0.9, 0.95), fused=True)
        scaler = GradScaler('cuda', init_scale=2**14)

    # Streaming loader
    loader = StreamingHFChessLoader(
        HF_DATASET, batch_size=BATCH_SIZE, encoder_type="fused",
        device=device, seed=SEED, drop_last=True,
        file_pattern="src",
        resume_cursor=resume_cursor,
    )
    n_train = loader.total_positions

    # Validate fingerprint on resume
    if resume_cursor and resume_cursor.get("fingerprint"):
        if resume_cursor["fingerprint"] != loader.fingerprint:
            raise RuntimeError(
                f"Dataset fingerprint mismatch! "
                f"Checkpoint: {resume_cursor['fingerprint']}, "
                f"Current: {loader.fingerprint}"
            )
        print(f"  Dataset fingerprint OK: {loader.fingerprint}")

    # ── Schedule ──
    steps_per_epoch = n_train // BATCH_SIZE
    total_opt_steps = steps_per_epoch // ACCUM_STEPS
    warmup_steps = WARMUP_STEPS

    print(f"\n  Estimated training positions: ~{n_train/1e6:.0f}M (streaming)")
    print(f"  Source files: {loader.num_files:,}")
    print(f"  Estimated optimizer steps: ~{total_opt_steps:,}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Peak LR: {LR:.1e} → min {LR * MIN_LR_FRAC:.1e}")

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    if own_ckpt is not None and own_ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(own_ckpt["scheduler_state_dict"])
        print(f"  Scheduler restored (lr={scheduler.get_last_lr()[0]:.2e})")

    best_state = None

    # ── torch.compile ──
    print("  Compiling model with torch.compile...")
    model = torch.compile(model)

    # ── Save config ──
    config_dict = {
        "experiment": "exp121_continued_pretraining",
        "parent": str(INIT_CHECKPOINT),
        "baseline_elo": 1831,
        "model": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "ffn_ratio": FFN_RATIO, "dropout": DROPOUT,
            "policy_head_dim": POLICY_HEAD_DIM,
            "value_hidden": VALUE_HIDDEN,
            "total_params": n_params,
        },
        "training": {
            "data_pipeline": "StreamingHFChessLoader",
            "file_pattern": "src",
            "n_train_est": n_train,
            "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * ACCUM_STEPS,
            "lr": LR, "warmup_steps": WARMUP_STEPS,
            "min_lr_frac": MIN_LR_FRAC,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "value_weight": VALUE_WEIGHT,
            "loss": "CE(policy) + 0.5 * KL(value WDL)",
        },
        "seed": SEED,
        "total_opt_steps": total_opt_steps,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Initial eval ──
    print("\n  Initial evaluation (from loaded weights)...")
    ev0 = evaluate(model, eval_data, eval_tensors, device)
    print(f"  Init: acc={ev0['accuracy']:.1%} "
          f"top3={ev0['top3_accuracy']:.1%} "
          f"sf_rank={ev0['mean_sf_rank']:.1f} "
          f"val={ev0['value_accuracy']:.1%}")
    for phase, acc in sorted(ev0['phase_accuracy'].items()):
        print(f"    {phase}: {acc:.1%}")

    if global_step == 0:
        best_acc = ev0["accuracy"]
        results_log.append({
            "step": 0, "positions_seen": 0,
            "type": "initial",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev0.items()},
        })
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(results_log, f, indent=2)
        best_state = {k.replace("_orig_mod.", ""): v.cpu().clone()
                      for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": best_state}, OUTPUT_DIR / "best_model.pt")
        print(f"  Saved baseline as best_model.pt (acc={best_acc:.1%})")

    # ── Training loop ──
    model.train()
    t_start = time.time()
    micro_step = global_step * ACCUM_STEPS
    running_pl = 0.0
    running_vl = 0.0
    running_batches = 0
    grad_norm = 0.0
    start_positions = positions_seen
    nan_count = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'─'*72}")
    print(f" Training started (~{n_train/1e6:.0f}M positions streaming, "
          f"~{total_opt_steps:,} opt steps)")
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

        # NaN guard
        if not torch.isfinite(loss):
            nan_count += 1
            print(f"\n  [WARN] NaN/Inf loss at micro_step {micro_step} "
                  f"(pl={policy_loss.item():.4f}, vl={value_loss.item():.4f}, "
                  f"scale={scaler.get_scale():.0f}). "
                  f"Skipping batch ({nan_count} total).", flush=True)
            optimizer.zero_grad()
            micro_step += 1
            positions_seen += bs
            if nan_count >= 20:
                print("  [ERROR] 20 NaN losses. Saving & stopping.", flush=True)
                save_checkpoint(model, optimizer, scaler, scheduler,
                                global_step, positions_seen, best_acc,
                                results_log, "nan_crash",
                                data_cursor=loader.get_cursor())
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

            # ── Heartbeat ──
            if global_step % HEARTBEAT_INTERVAL == 0:
                elapsed = time.time() - t_start
                new_positions = positions_seen - start_positions
                tp = new_positions / max(elapsed, 0.1)
                progress_total = positions_seen / n_train * 100
                remaining_pos = max(n_train - positions_seen, 0)
                eta_s = remaining_pos / max(tp, 1)
                eta_h = eta_s / 3600
                sys.stdout.write(
                    f"\r    step {global_step:>7,}/{total_opt_steps:,} | "
                    f"pos {positions_seen:>11,}/{n_train:,} ({progress_total:5.1f}%) | "
                    f"{tp:.0f} pos/s | ETA {eta_h:.1f}h"
                )
                sys.stdout.flush()

            # ── Detailed log ──
            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                new_positions = positions_seen - start_positions
                tp = new_positions / max(elapsed, 0.1)
                lr_now = scheduler.get_last_lr()[0]
                avg_pl = running_pl / max(running_batches, 1)
                avg_vl = running_vl / max(running_batches, 1)
                progress_total = positions_seen / n_train * 100
                peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                log_entry = {
                    "step": global_step,
                    "positions_seen": positions_seen,
                    "type": "train",
                    "policy_loss": round(avg_pl, 4),
                    "value_loss": round(avg_vl, 4),
                    "lr": lr_now,
                    "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                    "throughput": round(tp),
                    "elapsed_s": round(elapsed),
                    "peak_mem_gb": round(peak_mem, 1),
                }

                print(f"\n    [{global_step:>7,}/{total_opt_steps:,}] "
                      f"pl={avg_pl:.4f} vl={avg_vl:.4f} "
                      f"| lr={lr_now:.2e} | gnorm={grad_norm:.2f} "
                      f"| {tp:.0f} pos/s | {progress_total:.1f}% "
                      f"| pos_total={positions_seen:,} "
                      f"| mem={peak_mem:.1f}GB",
                      flush=True)

                results_log.append(log_entry)
                running_pl = 0.0
                running_vl = 0.0
                running_batches = 0

                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(results_log, f, indent=2)

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
                    "data_cursor": loader.get_cursor(),
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in ev.items()},
                })

                if ev["accuracy"] > best_acc:
                    best_acc = ev["accuracy"]
                    best_state = {k.replace("_orig_mod.", ""): v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    print(f"    ** New best: {best_acc:.1%}")
                    torch.save({"model_state_dict": best_state},
                               OUTPUT_DIR / "best_model.pt")
                    upload_best_to_hf(
                        OUTPUT_DIR / "best_model.pt",
                        global_step, best_acc, results_log
                    )

                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(results_log, f, indent=2)

                model.train()

            # ── Periodic checkpoint ──
            if global_step % SAVE_INTERVAL == 0:
                save_checkpoint(
                    model, optimizer, scaler, scheduler,
                    global_step, positions_seen, best_acc, results_log,
                    data_cursor=loader.get_cursor(),
                )

    # ── End of training ──
    total_time = time.time() - t_start
    print(f"\n\n{'='*72}")

    if SHUTDOWN_REQUESTED:
        print(" GRACEFUL SHUTDOWN — saving final state")
    else:
        print(" TRAINING COMPLETE")

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
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev_final.items()},
    })

    if ev_final["accuracy"] > best_acc:
        best_acc = ev_final["accuracy"]
        best_state = {k.replace("_orig_mod.", ""): v.cpu().clone()
                      for k, v in model.state_dict().items()}

    save_checkpoint(
        model, optimizer, scaler, scheduler,
        global_step, positions_seen, best_acc, results_log, tag="final",
        data_cursor=loader.get_cursor(),
    )

    if best_state:
        torch.save({"model_state_dict": best_state}, OUTPUT_DIR / "best_model.pt")

    print(f"\n  Total time: {total_time/3600:.1f}h")
    print(f"  Positions seen: {positions_seen:,}")
    print(f"  Best accuracy: {best_acc:.1%}")
    print(f"  Final accuracy: {ev_final['accuracy']:.1%}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    train()
