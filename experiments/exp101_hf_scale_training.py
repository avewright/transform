"""exp101: HF-scale training on 4M+ diverse positions.

Hypothesis: Training on millions of diverse positions (all game phases)
from avewright/chess-positions-lichess-sf will break the ~1700 ELO ceiling
that results from training only on opening positions (ply 8-23).

Key differences from exp093:
  - Uses StreamingHFChessLoader (one parquet at a time, ~254K per file)
  - Hard CE only (no soft targets — HF data has only best_move)
  - WDL value targets computed from CP/mate via sigmoid model
  - 4M+ positions vs 115K (35x more data, all game phases)
  - Init from exp093-d8 EMA model (best known ELO ~1666)

Architecture: ChessTransformer200M (~204M params)
  FusedBoardEncoder 256d → 1024d, 16L/16H, FFN 4x, GELU, norm_first
  SpatialPolicyHead (head_dim=512), WDL value head

Expected: RTX 4060 8GB, ~200-400 pos/s streaming, ~3-6 hours for 4M positions
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
    StreamingHFChessLoader, build_eval_from_hf,
)

# ── Paths ──
# Use local non-OneDrive path to avoid file-locking issues
OUTPUT_DIR = Path(r"C:\temp\chess_training\exp101")
INIT_CHECKPOINT = Path("outputs/exp093_ema_curriculum_d8/ema_model.pt")
HF_DATASET = "avewright/chess-positions-lichess-sf"

# ── Config ──
BATCH_SIZE = 16            # optimal for RTX 4060 8GB (~31 pos/s)
ACCUM_STEPS = 32           # effective batch = 512
LR = 5e-5                  # moderate LR — finetuning from strong checkpoint
WARMUP_STEPS = 200         # short warmup since model is already trained
MIN_LR_FRAC = 0.10         # cosine floor = 10% of peak
VALUE_WEIGHT = 0.25        # value head training weight
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
SEED = 42

# EMA
EMA_DECAY = 0.999
EMA_START_STEP = 50

# Model dims
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

# Logging intervals (in optimizer steps)
LOG_INTERVAL = 25
EVAL_INTERVAL = 100          # eval every ~100 steps (~4h per parquet)
SAVE_INTERVAL = 50           # checkpoint every ~50 steps (~2h per parquet)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Graceful shutdown ──
SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Graceful shutdown requested. Saving checkpoint...", flush=True)

signal.signal(signal.SIGINT, _signal_handler)


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


# ── EMA ──

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def save_weights(self, path, model):
        self.apply_shadow(model)
        try:
            save_model_weights(path, model)
        finally:
            self.restore(model)


# ── Utils ──

def save_model_weights(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save({"model_state_dict": model.state_dict()}, tmp)
    os.replace(str(tmp), str(path))


def load_model(checkpoint_path, device):
    model = ChessTransformer200M()
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        # Strip _orig_mod. prefixes from torch.compile
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    model.to(device)
    return model


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_frac):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine_decay)


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Evaluation ──

def evaluate(model, eval_data, eval_tensors, device, batch_size=128):
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


# ── Logging ──

LOG_FILE = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


def save_status(step, positions, best_acc, metrics=None, cursor=None):
    status = {
        "experiment": "exp101_hf_scale",
        "step": step,
        "positions_seen": positions,
        "best_accuracy": best_acc,
        "timestamp": datetime.now().isoformat(),
    }
    if metrics:
        status["latest_eval"] = metrics
    if cursor:
        status["data_cursor"] = cursor
    tmp = OUTPUT_DIR / "status.json.tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(str(tmp), str(OUTPUT_DIR / "status.json"))


# ── Main training ──

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
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of HF parquet files (for smoke tests)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output-dir")
    args = parser.parse_args()

    OUTPUT_DIR_ACTUAL = Path(args.output_dir)
    OUTPUT_DIR_ACTUAL.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(OUTPUT_DIR_ACTUAL / "exp101.log", "a")

    log("=" * 60)
    log("exp101: HF-scale training on 4M+ diverse positions")
    log(f"  init_checkpoint: {args.init_checkpoint}")
    log(f"  output_dir: {OUTPUT_DIR_ACTUAL}")
    log(f"  lr={args.lr}, batch={args.batch_size}, accum={args.accum_steps}")
    log(f"  value_weight={args.value_weight}, ema_decay={args.ema_decay}")
    log(f"  max_files={args.max_files}, seed={args.seed}")
    log(f"  device={DEVICE}")

    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # ── Load eval data from HF ──
    log("Loading eval data from HF...")
    eval_data, eval_tensors = build_eval_from_hf(HF_DATASET, n_eval=2500, encoder_type="fused")
    log(f"  Eval: {len(eval_data)} positions")

    # ── Load model ──
    log("Loading model...")
    model = load_model(args.init_checkpoint, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {n_params / 1e6:.1f}M parameters")

    # ── Optimizer + scaler ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda') if DEVICE == "cuda" else None

    # ── EMA ──
    ema = EMAModel(model, decay=args.ema_decay)
    log(f"  EMA: decay={args.ema_decay}, start_step={EMA_START_STEP}")

    # ── Resume state ──
    global_step = 0
    positions_seen = 0
    best_acc = 0.0
    resume_cursor = None

    if args.resume:
        resume_ckpt = OUTPUT_DIR_ACTUAL / "latest_checkpoint.pt"
        if resume_ckpt.exists():
            log(f"  Resuming from {resume_ckpt}...")
            ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            global_step = ckpt.get("global_step", 0)
            positions_seen = ckpt.get("positions_seen", 0)
            best_acc = ckpt.get("best_acc", 0.0)
            resume_cursor = ckpt.get("data_cursor", None)
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            log(f"  Resumed: step={global_step}, positions={positions_seen:,}, best_acc={best_acc:.4f}")

    # ── Initial eval ──
    log("Initial eval...")
    init_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  Accuracy: {init_metrics['accuracy']:.4f}, Top3: {init_metrics['top3_accuracy']:.4f}, "
        f"Value: {init_metrics['value_accuracy']:.4f}")
    log(f"  Phase: {init_metrics['phase_accuracy']}")
    if init_metrics['accuracy'] > best_acc:
        best_acc = init_metrics['accuracy']

    # ── Create streaming loader ──
    log("Creating streaming data loader...")
    loader = StreamingHFChessLoader(
        repo_id=HF_DATASET,
        batch_size=args.batch_size,
        encoder_type="fused",
        device=DEVICE,
        seed=args.seed,
        drop_last=True,
        file_pattern="src",
        max_files=args.max_files,
        resume_cursor=resume_cursor,
    )
    log(f"  Files: {loader.num_files}, est ~{loader.total_positions / 1e6:.1f}M positions")

    # Estimate total steps for LR schedule
    est_total_positions = loader.total_positions
    est_total_steps = est_total_positions // (args.batch_size * args.accum_steps)
    log(f"  Estimated total steps: ~{est_total_steps:,}")

    # ── Training loop ──
    model.train()
    optimizer.zero_grad()
    accum_count = 0
    running_loss = 0.0
    running_ce = 0.0
    running_val = 0.0
    running_gnorm = 0.0
    log_count = 0
    t_start = time.time()
    t_last_log = time.time()
    nan_count = 0

    log("Starting training loop...")

    for batch_input, move_targets, wdl_targets in loader:
        if SHUTDOWN_REQUESTED:
            break

        B = move_targets.shape[0]

        # Forward
        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            policy_logits = result["policy_logits"]
            value_logits = result["value_logits"]

            # Policy loss: hard cross-entropy
            ce_loss = F.cross_entropy(policy_logits, move_targets)

            # Value loss: cross-entropy against WDL distribution
            value_loss = F.cross_entropy(value_logits, wdl_targets)

            total_loss = ce_loss + args.value_weight * value_loss
            scaled_loss = total_loss / args.accum_steps

        # NaN guard
        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            nan_count += 1
            if nan_count > 10:
                log(f"ERROR: too many NaN losses ({nan_count}), stopping")
                break
            log(f"WARNING: NaN loss at step {global_step}, skipping batch")
            optimizer.zero_grad()
            accum_count = 0
            continue
        nan_count = 0

        # Backward
        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        accum_count += 1
        running_ce += ce_loss.item()
        running_val += value_loss.item()
        running_loss += total_loss.item()
        positions_seen += B

        # Optimizer step
        if accum_count >= args.accum_steps:
            # LR schedule
            lr = cosine_lr(global_step, est_total_steps, WARMUP_STEPS, args.lr, MIN_LR_FRAC)
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

            # EMA update
            if global_step >= EMA_START_STEP:
                ema.update(model)

            # Accumulate for logging
            running_gnorm += gnorm if not math.isnan(gnorm) and not math.isinf(gnorm) else 0.0
            log_count += 1

            global_step += 1
            accum_count = 0

            # Heartbeat every step for first 5 steps
            if global_step <= 5:
                log(f"  [heartbeat] step={global_step} ce={running_ce/(log_count*args.accum_steps):.4f} "
                    f"pos={positions_seen:,}")

            # Log
            if global_step % LOG_INTERVAL == 0 and log_count > 0:
                avg_loss = running_loss / (log_count * args.accum_steps)
                avg_ce = running_ce / (log_count * args.accum_steps)
                avg_val = running_val / (log_count * args.accum_steps)
                avg_gnorm = running_gnorm / log_count
                elapsed = time.time() - t_last_log
                pos_per_s = (log_count * args.accum_steps * args.batch_size) / max(elapsed, 0.1)
                cursor = loader.get_cursor()
                file_prog = f"{cursor['files_completed']}/{loader.num_files}"

                log(f"step={global_step} loss={avg_loss:.4f} ce={avg_ce:.4f} "
                    f"val={avg_val:.4f} gnorm={avg_gnorm:.2f} lr={lr:.2e} "
                    f"pos={positions_seen:,} files={file_prog} "
                    f"speed={pos_per_s:.0f}pos/s")

                running_loss = running_ce = running_val = running_gnorm = 0.0
                log_count = 0
                t_last_log = time.time()

            # Eval
            if global_step % EVAL_INTERVAL == 0:
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

                # Pick winner
                best_metrics = ema_metrics if ema_metrics and ema_metrics['accuracy'] >= live_metrics['accuracy'] else live_metrics
                is_ema_better = ema_metrics and ema_metrics['accuracy'] >= live_metrics['accuracy']

                if best_metrics['accuracy'] > best_acc:
                    best_acc = best_metrics['accuracy']
                    if is_ema_better:
                        ema.save_weights(OUTPUT_DIR_ACTUAL / "best_model.pt", model)
                    else:
                        save_model_weights(OUTPUT_DIR_ACTUAL / "best_model.pt", model)
                    log(f"  NEW BEST: acc={best_acc:.4f} ({'ema' if is_ema_better else 'live'})")

                save_status(global_step, positions_seen, best_acc,
                            metrics=best_metrics, cursor=loader.get_cursor())
                model.train()

            # Save checkpoint
            if global_step % SAVE_INTERVAL == 0:
                save_model_weights(OUTPUT_DIR_ACTUAL / "latest_model.pt", model)
                ema.save_weights(OUTPUT_DIR_ACTUAL / "ema_model.pt", model)

                # Full resumable checkpoint
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "positions_seen": positions_seen,
                    "best_acc": best_acc,
                    "data_cursor": loader.get_cursor(),
                    "ema_state_dict": ema.__dict__,
                }
                if scaler:
                    ckpt["scaler_state_dict"] = scaler.state_dict()
                tmp = OUTPUT_DIR_ACTUAL / "latest_checkpoint.pt.tmp"
                torch.save(ckpt, tmp)
                os.replace(str(tmp), str(OUTPUT_DIR_ACTUAL / "latest_checkpoint.pt"))

    # ── Final eval ──
    elapsed_total = time.time() - t_start
    log(f"\nTraining complete. {global_step} steps, {positions_seen:,} positions, "
        f"{elapsed_total / 3600:.1f} hours")

    log("Final eval (live)...")
    final_live = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  live: acc={final_live['accuracy']:.4f} top3={final_live['top3_accuracy']:.4f}")

    log("Final eval (EMA)...")
    ema.apply_shadow(model)
    final_ema = evaluate(model, eval_data, eval_tensors, DEVICE)
    ema.restore(model)
    log(f"  ema:  acc={final_ema['accuracy']:.4f} top3={final_ema['top3_accuracy']:.4f}")

    # Save final models
    save_model_weights(OUTPUT_DIR_ACTUAL / "final_live_model.pt", model)
    ema.save_weights(OUTPUT_DIR_ACTUAL / "final_ema_model.pt", model)

    # Save best if improved
    best_final = final_ema if final_ema['accuracy'] >= final_live['accuracy'] else final_live
    if best_final['accuracy'] > best_acc:
        best_acc = best_final['accuracy']
        if final_ema['accuracy'] >= final_live['accuracy']:
            ema.save_weights(OUTPUT_DIR_ACTUAL / "best_model.pt", model)
        else:
            save_model_weights(OUTPUT_DIR_ACTUAL / "best_model.pt", model)
    log(f"  Final best accuracy: {best_acc:.4f}")
    log(f"  Phase accuracy (EMA): {final_ema['phase_accuracy']}")

    save_status(global_step, positions_seen, best_acc,
                metrics=best_final, cursor=loader.get_cursor())

    LOG_FILE.close()
    log("Done.")


if __name__ == "__main__":
    main()
