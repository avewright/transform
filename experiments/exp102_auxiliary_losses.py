"""exp102: Auxiliary losses — material count + game phase prediction.

Source: alphazero/possible_improvements.md §9 (Auxiliary Losses)
        stockfish_md/improvements.md §12 (Endgame Corrections)

Hypothesis: Adding auxiliary training objectives forces the transformer trunk
to encode basic positional facts (material balance, game phase) that are
currently only learned implicitly. This should:
  1. Improve value head accuracy (material awareness = better evaluation)
  2. Improve endgame play (phase-aware features help the model shift strategy)
  3. Potentially enable search (exp094/097 showed value head is too weak)

The model already has a WDL value head (alphazero §3 ✓) and a transformer
backbone (alphazero §1 ✓). Auxiliary losses add dense per-position supervision
beyond the sparse game-outcome signal.

Implementation:
  - material_head: predict centipawn material balance from CLS token
  - phase_head: predict game phase (opening/middlegame/endgame) from CLS token
  - piece_count_head: predict total non-king piece count (regression)
  These share the transformer trunk but have independent small MLPs.
  Total auxiliary loss weight: 0.1 (light touch — don't distort policy learning)

Data: streams from avewright/chess-positions-lichess-sf (4M+ positions).
      Material/phase labels are computed on-the-fly from FEN (free labels).

Init: from exp093-d8 EMA model (best known ELO ~1666).

Architecture: ChessTransformer200M + 3 auxiliary heads (204M + ~0.1M new params)

Usage:
    python experiments/exp102_auxiliary_losses.py --max-files 5
    python experiments/exp102_auxiliary_losses.py --max-files 1  # smoke test
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
import numpy as np
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
    board_array_to_fused, ep_square_to_file, _fast_parse_fen, _reconstruct_fen,
    PIECE_MAP,
)

# ── Paths ──
OUTPUT_DIR = Path("/root/transform/outputs/exp102_aux")
INIT_CHECKPOINT = Path("outputs/hf_checkpoint/best_model.pt")
HF_DATASET = "avewright/chess-positions-lichess-sf"

# ── Config ──
BATCH_SIZE = 128
ACCUM_STEPS = 4           # effective batch = 512
LR = 3e-5
WARMUP_STEPS = 200
MIN_LR_FRAC = 0.10
VALUE_WEIGHT = 0.25
AUX_WEIGHT = 0.10         # total weight for all auxiliary losses
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
SEED = 42

# EMA
EMA_DECAY = 0.999
EMA_START_STEP = 50

# Model dims (must match ChessTransformer200M)
ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

LOG_INTERVAL = 25
EVAL_INTERVAL = 100          # eval every ~100 steps
SAVE_INTERVAL = 50           # checkpoint every ~50 steps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHUTDOWN_REQUESTED = False
def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Shutdown requested.", flush=True)
signal.signal(signal.SIGINT, _signal_handler)


# ── Auxiliary target computation (from FEN, free labels) ──

PIECE_VALUES = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900,
                'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900,
                'K': 0, 'k': 0}

def compute_material_balance(board_array):
    """Compute material balance in centipawns from board_array tensor.
    board_array: (B, 64) with values 0-12.
    Returns: (B,) float tensor of material balance (positive = white ahead).
    """
    # Mapping: 0=empty, 1-6=white P,N,B,R,Q,K, 7-12=black p,n,b,r,q,k
    values = torch.tensor([0, 100, 320, 330, 500, 900, 0, -100, -320, -330, -500, -900, 0],
                          dtype=torch.float32, device=board_array.device)
    piece_vals = values[board_array.long()]  # (B, 64)
    return piece_vals.sum(dim=1)  # (B,)


def compute_piece_count(board_array):
    """Count non-king, non-empty pieces. Returns (B,) float."""
    ba = board_array.long()
    # Non-empty: 1-5 (white non-king) + 7-11 (black non-king)
    non_king = ((ba >= 1) & (ba <= 5)) | ((ba >= 7) & (ba <= 11))
    return non_king.sum(dim=1).float()


def compute_phase_labels(board_array):
    """Compute game phase from piece count. Returns (B,) long tensor.
    0=opening (>=14 non-king pieces), 1=middlegame (6-13), 2=endgame (<6)
    """
    counts = compute_piece_count(board_array)
    labels = torch.ones_like(counts, dtype=torch.long)  # default middlegame
    labels[counts >= 14] = 0  # opening
    labels[counts < 6] = 2    # endgame
    return labels


# ── Model with auxiliary heads ──

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


class ChessTransformer200M_Aux(nn.Module):
    """ChessTransformer200M with auxiliary prediction heads."""
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

        # Primary heads
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, VALUE_HIDDEN), nn.ReLU(), nn.Linear(VALUE_HIDDEN, 3),
        )

        # Auxiliary heads (from alphazero/possible_improvements.md §9)
        self.material_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, 3),
        )
        self.piece_count_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, 1),
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
            "material_pred": self.material_head(cls_hidden).squeeze(-1),
            "phase_logits": self.phase_head(cls_hidden),
            "piece_count_pred": self.piece_count_head(cls_hidden).squeeze(-1),
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
    # Save only the base model weights (strip aux heads for compatibility)
    sd = model.state_dict()
    torch.save({"model_state_dict": sd}, tmp)
    os.replace(str(tmp), str(path))


def load_model(checkpoint_path, device):
    model = ChessTransformer200M_Aux()
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        # Load with strict=False since checkpoint won't have aux heads
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
    # Aux head accumulators
    mat_se_sum = phase_correct = phase_total = pcount_se_sum = aux_total = 0

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            n = len(chunk)
            idx = slice(i, i + n)
            batch_input = get_eval_batch_input(eval_tensors, idx, "fused", device)
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()

            # --- Policy accuracy ---
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

            # --- Value accuracy ---
            wdl_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                pred_class = wdl_logits[j].argmax().item()
                true_wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: true_wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1

            # --- Aux head metrics ---
            fused_ids = batch_input["fused_ids"]
            mat_true = compute_material_balance(fused_ids) / 900.0
            phase_true = compute_phase_labels(fused_ids)
            pcount_true = compute_piece_count(fused_ids) / 30.0

            mat_pred = result["material_pred"].float()
            phase_pred = result["phase_logits"].float().argmax(dim=-1)
            pcount_pred = result["piece_count_pred"].float()

            mat_se_sum += ((mat_pred - mat_true) ** 2).sum().item()
            phase_correct += (phase_pred == phase_true).sum().item()
            phase_total += phase_true.shape[0]
            pcount_se_sum += ((pcount_pred - pcount_true) ** 2).sum().item()
            aux_total += mat_true.shape[0]

    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": {p: round(s["correct"] / max(s["total"], 1), 4) for p, s in phase_stats.items()},
        "material_mse": mat_se_sum / max(aux_total, 1),
        "aux_phase_accuracy": phase_correct / max(phase_total, 1),
        "piece_count_mse": pcount_se_sum / max(aux_total, 1),
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


def save_status(out, step, positions, best_acc, metrics=None, cursor=None):
    status = {
        "experiment": "exp102_auxiliary_losses",
        "step": step,
        "positions_seen": positions,
        "best_accuracy": best_acc,
        "timestamp": datetime.now().isoformat(),
    }
    if metrics:
        status["latest_eval"] = metrics
    if cursor:
        status["data_cursor"] = cursor
    tmp = out / "status.json.tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(str(tmp), str(out / "status.json"))


# ── Streaming loader that also provides board_array for aux targets ──

class AuxStreamingLoader:
    """Wraps StreamingHFChessLoader and also returns raw board_array for aux targets."""

    def __init__(self, **kwargs):
        self.inner = StreamingHFChessLoader(**kwargs)

    @property
    def num_files(self):
        return self.inner.num_files

    @property
    def total_positions(self):
        return self.inner.total_positions

    def get_cursor(self):
        return self.inner.get_cursor()

    def __iter__(self):
        for batch_input, move_targets, wdl_targets in self.inner:
            # Extract fused_ids from batch_input to compute aux targets
            fused_ids = batch_input["fused_ids"]  # (B, 64) on device
            yield batch_input, move_targets, wdl_targets, fused_ids


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
    parser.add_argument("--aux-weight", type=float, default=AUX_WEIGHT)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps")
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output-dir")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(out / "exp102.log", "a")

    log("=" * 60)
    log("exp102: Auxiliary losses (material + phase + piece_count)")
    log(f"  init_checkpoint: {args.init_checkpoint}")
    log(f"  lr={args.lr}, batch={args.batch_size}, accum={args.accum_steps}")
    log(f"  value_weight={args.value_weight}, aux_weight={args.aux_weight}")
    log(f"  max_files={args.max_files}, max_steps={args.max_steps}, seed={args.seed}")
    log(f"  eval_interval={args.eval_interval}, save_interval={args.save_interval}")
    log(f"  device={DEVICE}")

    # Load eval
    log("Loading eval data from HF...")
    eval_data, eval_tensors = build_eval_from_hf(HF_DATASET, n_eval=2500, encoder_type="fused")
    log(f"  Eval: {len(eval_data)} positions")

    # Load model with aux heads
    log("Loading model...")
    model = load_model(args.init_checkpoint, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {n_params / 1e6:.1f}M parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda') if DEVICE == "cuda" else None
    ema = EMAModel(model, decay=args.ema_decay)

    # ── Resume state ──
    global_step = 0
    positions_seen = 0
    best_acc = 0.0
    resume_cursor = None

    if args.resume:
        resume_ckpt = out / "latest_checkpoint.pt"
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
            if "ema_shadow" in ckpt:
                ema.shadow = ckpt["ema_shadow"]
            log(f"  Resumed: step={global_step}, positions={positions_seen:,}, best_acc={best_acc:.4f}")

    # Initial eval
    log("Initial eval...")
    init_m = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  acc={init_m['accuracy']:.4f} top3={init_m['top3_accuracy']:.4f} "
        f"val_acc={init_m['value_accuracy']:.4f}")
    log(f"  aux: mat_mse={init_m['material_mse']:.4f} phase_acc={init_m['aux_phase_accuracy']:.4f} "
        f"pcount_mse={init_m['piece_count_mse']:.4f}")
    if init_m['accuracy'] > best_acc:
        best_acc = init_m['accuracy']

    # Streaming loader
    log("Creating streaming loader...")
    loader = AuxStreamingLoader(
        repo_id=HF_DATASET, batch_size=args.batch_size, encoder_type="fused",
        device=DEVICE, seed=args.seed, drop_last=True, file_pattern="src",
        max_files=args.max_files, resume_cursor=resume_cursor,
    )
    est_total_steps = loader.total_positions // (args.batch_size * args.accum_steps)
    if args.max_steps:
        est_total_steps = min(est_total_steps, args.max_steps)
    log(f"  Files: {loader.num_files}, est ~{est_total_steps:,} steps (cosine LR period)")

    # Training loop
    model.train()
    optimizer.zero_grad()
    global_step = 0
    accum_count = 0
    positions_seen = 0
    running = {"loss": 0, "ce": 0, "val": 0, "mat": 0, "phase": 0, "pcount": 0, "gnorm": 0}
    log_count = 0
    t_last = time.time()

    log("Starting training...")

    for batch_input, move_targets, wdl_targets, fused_ids in loader:
        if SHUTDOWN_REQUESTED:
            break

        B = move_targets.shape[0]

        # Compute aux targets from board_array (fused_ids = board_array on device)
        material_targets = compute_material_balance(fused_ids) / 900.0  # normalize by queen value
        phase_targets = compute_phase_labels(fused_ids)
        pcount_targets = compute_piece_count(fused_ids) / 30.0  # normalize

        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)

            # Primary losses
            ce_loss = F.cross_entropy(result["policy_logits"], move_targets)
            value_loss = F.cross_entropy(result["value_logits"], wdl_targets)

            # Auxiliary losses
            mat_loss = F.mse_loss(result["material_pred"], material_targets)
            phase_loss = F.cross_entropy(result["phase_logits"], phase_targets)
            pcount_loss = F.mse_loss(result["piece_count_pred"], pcount_targets)

            aux_loss = mat_loss + phase_loss + pcount_loss
            total_loss = ce_loss + args.value_weight * value_loss + args.aux_weight * aux_loss
            scaled = total_loss / args.accum_steps

        if torch.isnan(scaled) or torch.isinf(scaled):
            optimizer.zero_grad()
            accum_count = 0
            continue

        if scaler:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        accum_count += 1
        positions_seen += B
        running["ce"] += ce_loss.item()
        running["val"] += value_loss.item()
        running["mat"] += mat_loss.item()
        running["phase"] += phase_loss.item()
        running["pcount"] += pcount_loss.item()
        running["loss"] += total_loss.item()

        if accum_count >= args.accum_steps:
            lr = cosine_lr(global_step, est_total_steps, args.warmup_steps, args.lr, MIN_LR_FRAC)
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

            running["gnorm"] += gnorm if not math.isnan(gnorm) else 0.0
            log_count += 1
            global_step += 1
            accum_count = 0

            if global_step <= 3:
                log(f"  [heartbeat] step={global_step} pos={positions_seen:,}")

            # Check max-steps early exit
            if args.max_steps and global_step >= args.max_steps:
                log(f"Reached --max-steps={args.max_steps}, stopping.")
                SHUTDOWN_REQUESTED = True
                break

            if global_step % LOG_INTERVAL == 0 and log_count > 0:
                n = log_count * args.accum_steps
                elapsed = time.time() - t_last
                speed = (log_count * args.accum_steps * args.batch_size) / max(elapsed, 0.1)
                log(f"step={global_step} ce={running['ce']/n:.4f} val={running['val']/n:.4f} "
                    f"mat={running['mat']/n:.4f} phase={running['phase']/n:.4f} "
                    f"pcount={running['pcount']/n:.4f} gnorm={running['gnorm']/log_count:.2f} "
                    f"lr={lr:.2e} pos={positions_seen:,} {speed:.0f}pos/s")
                running = {k: 0 for k in running}
                log_count = 0
                t_last = time.time()

            if global_step % args.eval_interval == 0:
                log("Eval...")
                m = evaluate(model, eval_data, eval_tensors, DEVICE)
                log(f"  live: acc={m['accuracy']:.4f} top3={m['top3_accuracy']:.4f} "
                    f"val_acc={m['value_accuracy']:.4f}")
                log(f"  aux: mat_mse={m['material_mse']:.4f} phase_acc={m['aux_phase_accuracy']:.4f} "
                    f"pcount_mse={m['piece_count_mse']:.4f}")
                log(f"  phase: {m['phase_accuracy']}")
                if global_step >= EMA_START_STEP:
                    ema.apply_shadow(model)
                    em = evaluate(model, eval_data, eval_tensors, DEVICE)
                    ema.restore(model)
                    log(f"  ema:  acc={em['accuracy']:.4f} top3={em['top3_accuracy']:.4f} "
                        f"val_acc={em['value_accuracy']:.4f}")
                    log(f"  ema aux: mat_mse={em['material_mse']:.4f} phase_acc={em['aux_phase_accuracy']:.4f} "
                        f"pcount_mse={em['piece_count_mse']:.4f}")
                    best_m = em if em['accuracy'] >= m['accuracy'] else m
                else:
                    best_m = m
                if best_m['accuracy'] > best_acc:
                    best_acc = best_m['accuracy']
                    if best_m is em:
                        ema.save_weights(out / "best_model.pt", model)
                    else:
                        save_model_weights(out / "best_model.pt", model)
                    log(f"  NEW BEST: {best_acc:.4f}")
                save_status(out, global_step, positions_seen, best_acc,
                            metrics=best_m, cursor=loader.get_cursor())
                model.train()

            if global_step % args.save_interval == 0:
                save_model_weights(out / "latest_model.pt", model)
                ema.save_weights(out / "ema_model.pt", model)

                # Full resumable checkpoint
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "positions_seen": positions_seen,
                    "best_acc": best_acc,
                    "data_cursor": loader.get_cursor(),
                    "ema_shadow": ema.shadow,
                }
                if scaler:
                    ckpt["scaler_state_dict"] = scaler.state_dict()
                tmp = out / "latest_checkpoint.pt.tmp"
                torch.save(ckpt, tmp)
                os.replace(str(tmp), str(out / "latest_checkpoint.pt"))

    # Final
    log(f"Done. {global_step} steps, {positions_seen:,} positions")
    save_model_weights(out / "final_model.pt", model)
    ema.save_weights(out / "final_ema_model.pt", model)
    final = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"Final: acc={final['accuracy']:.4f} top3={final['top3_accuracy']:.4f} "
        f"val_acc={final['value_accuracy']:.4f}")
    log(f"Final aux: mat_mse={final['material_mse']:.4f} "
        f"phase_acc={final['aux_phase_accuracy']:.4f} "
        f"pcount_mse={final['piece_count_mse']:.4f}")
    if final['accuracy'] > best_acc:
        best_acc = final['accuracy']
        ema.save_weights(out / "best_model.pt", model)
    save_status(out, global_step, positions_seen, best_acc,
                metrics=final, cursor=loader.get_cursor())
    LOG_FILE.close()


if __name__ == "__main__":
    main()
