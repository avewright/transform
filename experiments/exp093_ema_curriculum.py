"""exp093: EMA + Curriculum training on depth-8 relabeled data.

Key features over exp084:
  1. EMA (Exponential Moving Average) — maintains a shadow model for stable checkpoints
  2. Curriculum learning — sorts positions by confidence (cp_gap), trains easy→hard
  3. Cosine LR schedule with linear warmup — adapts to dataset size
  4. Evaluates both live model and EMA model, saves whichever is better

Usage:
    python experiments/exp093_ema_curriculum.py \
        --output-dir outputs/exp093_ema_curriculum_d8 \
        --init-checkpoint outputs/exp090_full_legal_temp05_continue_ckpt/checkpoints/latest.pt \
        --dataset-glob "outputs/exp087_relabeled_d8/dataset/positions_*.jsonl" \
        --teacher-temp 0.5 --epochs 1 --lr 1.5e-6 \
        --hard-ce-weight 0.5 --soft-top-k 8 \
        --kl-conf-scale 80 --kl-conf-min 0.1 --kl-conf-max 1.0 \
        --ema-decay 0.999 \
        --curriculum-phases 3 --curriculum-key cp_gap_top1_top2 \
        --warmup-frac 0.05 \
        --save-weights-only-checkpoints --no-upload-to-hf
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import UCI_TO_IDX, VOCAB_SIZE, legal_move_mask
from play import ChessTransformer200M

OUTPUT_DIR = Path("outputs/exp093_ema_curriculum_d8")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "exp093.log"
CONFIG_PATH = OUTPUT_DIR / "config.json"
STATUS_PATH = OUTPUT_DIR / "status.json"
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
BEST_STATE_PATH = CHECKPOINT_DIR / "best.pt"
LATEST_MODEL_PATH = OUTPUT_DIR / "latest_model.pt"
BEST_PATH = OUTPUT_DIR / "best_model.pt"
EMA_MODEL_PATH = OUTPUT_DIR / "ema_model.pt"

INIT_CHECKPOINT_CANDIDATES = [
    Path("outputs/exp090_full_legal_temp05_continue_ckpt/checkpoints/latest.pt"),
    Path("outputs/exp092_full_legal_confkl_top8/latest_model.pt"),
    Path("outputs/hf/chess-transformer-200m-latest/best_model.pt"),
]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

BATCH_SIZE = 4
ACCUM_STEPS = 16
EPOCHS = 1
LR = 1.5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
HARD_CE_WEIGHT = 0.5
VALUE_LOSS_WEIGHT = 0.10
TEACHER_TEMP = 0.5
SOFT_TOP_K = 8
KL_CONF_SCALE = 80.0
KL_CONF_MIN = 0.10
KL_CONF_MAX = 1.00
EVAL_FRACTION = 0.05
MAX_EVAL_RECORDS = 2048
LOG_INTERVAL = 25
SAVE_INTERVAL = 200

# --- EMA ---
EMA_DECAY = 0.999
EMA_START_STEP = 50  # start EMA tracking after warmup settles

# --- Curriculum ---
CURRICULUM_PHASES = 3
CURRICULUM_KEY = "cp_gap_top1_top2"

# --- LR schedule ---
WARMUP_FRAC = 0.05
MIN_LR_FRAC = 0.10

UPLOAD_TO_HF = False
SAVE_CHECKPOINTS = True
SAVE_FINAL_ONLY = False
SAVE_WEIGHTS_ONLY_CHECKPOINTS = True

LOG_FILE = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(stamped + "\n")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_model_weights(path: Path, model: ChessTransformer200M) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save({"model_state_dict": _model_state_dict_cpu(model)}, tmp)
    os.replace(tmp, path)


def _model_state_dict_cpu(model) -> dict:
    return {
        k.replace("_orig_mod.", ""): v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: ChessTransformer200M, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: ChessTransformer200M) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: ChessTransformer200M) -> None:
        """Swap model weights with EMA shadow (call before eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: ChessTransformer200M) -> None:
        """Restore original weights (call after eval)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.to(DEVICE) for k, v in state["shadow"].items()}

    def save_weights(self, path: Path, model: ChessTransformer200M) -> None:
        """Save EMA weights as a standalone model checkpoint."""
        self.apply_shadow(model)
        try:
            save_model_weights(path, model)
        finally:
            self.restore(model)


# ---------------------------------------------------------------------------
# Curriculum sorting
# ---------------------------------------------------------------------------

def sort_by_curriculum(records: list[dict], key: str, phases: int) -> list[list[dict]]:
    """Split records into curriculum phases sorted by confidence (high gap = easy, low gap = hard).

    Phase 0 = highest confidence (easiest), Phase N-1 = lowest (hardest).
    """
    keyed = []
    for r in records:
        val = float(r.get(key, 0))
        keyed.append((val, r))
    keyed.sort(key=lambda x: x[0], reverse=True)  # high confidence first

    sorted_records = [r for _, r in keyed]
    phase_size = max(1, len(sorted_records) // phases)
    batches = []
    for i in range(phases):
        start = 0  # each phase includes ALL prior phases + new slice
        end = min((i + 1) * phase_size, len(sorted_records))
        if i == phases - 1:
            end = len(sorted_records)  # last phase = everything
        batches.append(sorted_records[:end])
    return batches


# ---------------------------------------------------------------------------
# Cosine LR Schedule with warmup
# ---------------------------------------------------------------------------

def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_frac: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine_decay)


def set_lr(optimizer: AdamW, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        log(f"skip malformed row path={path} line={line_no}")
    return records


def stable_train_eval_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []
    eval_target = min(MAX_EVAL_RECORDS, max(512, int(len(records) * EVAL_FRACTION)))
    keyed = []
    for item in records:
        digest = hashlib.blake2b(item["fen"].encode("utf-8"), digest_size=8).digest()
        keyed.append((int.from_bytes(digest, "big"), item))
    keyed.sort(key=lambda pair: pair[0])
    eval_records = [item for _, item in keyed[:eval_target]]
    train_records = [item for _, item in keyed[eval_target:]]
    return train_records, eval_records


# ---------------------------------------------------------------------------
# Loss computation (same as exp084)
# ---------------------------------------------------------------------------

def _temperature_scale_probs(targets: list[dict], teacher_temp: float) -> list[float]:
    probs = [max(float(t["prob"]), 1e-12) for t in targets]
    if abs(teacher_temp - 1.0) < 1e-9:
        total = sum(probs)
        return [p / total for p in probs]
    scaled = [p ** (1.0 / teacher_temp) for p in probs]
    total = sum(scaled)
    if total <= 0:
        return [1.0 / len(targets)] * len(targets)
    return [p / total for p in scaled]


def _select_soft_targets(item: dict, soft_top_k: int) -> list[dict]:
    targets = item["soft_targets"]
    if soft_top_k > 0:
        return targets[:soft_top_k]
    return targets


def sparse_soft_targets_to_dense(
    batch: list[dict], teacher_temp: float, soft_top_k: int,
) -> torch.Tensor:
    dense = torch.zeros(len(batch), VOCAB_SIZE, dtype=torch.float32)
    for row_idx, item in enumerate(batch):
        selected = _select_soft_targets(item, soft_top_k)
        scaled = _temperature_scale_probs(selected, teacher_temp)
        for t, p in zip(selected, scaled):
            dense[row_idx, UCI_TO_IDX[t["uci"]]] = float(p)
    return dense


def batch_kl_confidence_weights(
    batch: list[dict], *, kl_conf_scale: float, kl_conf_min: float, kl_conf_max: float,
) -> torch.Tensor:
    if kl_conf_scale <= 0:
        return torch.ones(len(batch), dtype=torch.float32)
    weights = []
    for item in batch:
        cp_gap = max(float(item.get("cp_gap_top1_top2", 0.0)), 0.0)
        conf = max(kl_conf_min, min(kl_conf_max, cp_gap / kl_conf_scale))
        weights.append(conf)
    return torch.tensor(weights, dtype=torch.float32)


def compute_policy_losses(
    logits: torch.Tensor, best_moves: torch.Tensor,
    soft_targets: torch.Tensor, kl_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hard_ce = F.cross_entropy(logits, best_moves)
    log_probs = F.log_softmax(logits, dim=-1)
    kl_per_sample = F.kl_div(log_probs, soft_targets, reduction="none").sum(dim=-1)
    kl = (kl_per_sample * kl_weights).mean()
    return hard_ce, kl


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer200M:
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: ChessTransformer200M, eval_records: list[dict], *,
    teacher_temp: float, hard_ce_weight: float, value_loss_weight: float,
    soft_top_k: int, kl_conf_scale: float, kl_conf_min: float, kl_conf_max: float,
) -> dict:
    model.eval()
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    total = correct = top3 = 0

    with torch.no_grad():
        for idx in range(0, len(eval_records), BATCH_SIZE):
            batch = eval_records[idx:idx + BATCH_SIZE]
            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
            value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
            soft_targets = sparse_soft_targets_to_dense(batch, teacher_temp=teacher_temp, soft_top_k=soft_top_k).to(DEVICE)
            kl_weights = batch_kl_confidence_weights(batch, kl_conf_scale=kl_conf_scale, kl_conf_min=kl_conf_min, kl_conf_max=kl_conf_max).to(DEVICE)
            out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
            logits = out["policy_logits"].float()
            value_logits = out["value_logits"].float()

            hard_ce, kl = compute_policy_losses(logits, best_moves, soft_targets, kl_weights)
            value_loss = F.cross_entropy(value_logits, value_targets)
            total_loss = (1.0 - hard_ce_weight) * kl + hard_ce_weight * hard_ce + value_loss_weight * value_loss

            loss_sum += total_loss.item() * len(batch)
            ce_sum += hard_ce.item() * len(batch)
            kl_sum += kl.item() * len(batch)
            val_sum += value_loss.item() * len(batch)

            for row_idx, board in enumerate(boards):
                masked = logits[row_idx].clone()
                mask = legal_move_mask(board).to(DEVICE)
                masked[~mask] = float("-inf")
                pred_idx = masked.argmax().item()
                true_idx = best_moves[row_idx].item()
                if pred_idx == true_idx:
                    correct += 1
                topk = masked.topk(min(3, int(mask.sum().item()))).indices.tolist()
                if true_idx in topk:
                    top3 += 1
                total += 1

    return {
        "loss": loss_sum / max(total, 1),
        "ce": ce_sum / max(total, 1),
        "kl": kl_sum / max(total, 1),
        "value": val_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "top3": top3 / max(total, 1),
        "n": total,
    }


# ---------------------------------------------------------------------------
# Training with EMA + curriculum + cosine LR
# ---------------------------------------------------------------------------

def train_curriculum_phase(
    phase_idx: int,
    phase_records: list[dict],
    model: ChessTransformer200M,
    ema: EMAModel,
    optimizer: AdamW,
    scaler: GradScaler,
    eval_records: list[dict],
    state: dict,
    args: argparse.Namespace,
    total_steps_so_far: int,
    total_steps_all: int,
    warmup_steps: int,
) -> int:
    """Train one curriculum phase. Returns steps trained."""
    model.train()
    random.shuffle(phase_records)
    steps_this_phase = math.ceil(len(phase_records) / (BATCH_SIZE * ACCUM_STEPS))
    cursor = 0
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    best_eval = state.get("best_eval_loss", float("inf"))

    for step_idx in range(steps_this_phase):
        global_step = total_steps_so_far + step_idx + 1

        # Cosine LR with warmup
        current_lr = cosine_lr(global_step, total_steps_all, warmup_steps, args.lr, MIN_LR_FRAC)
        set_lr(optimizer, current_lr)

        optimizer.zero_grad(set_to_none=True)
        step_loss = step_ce = step_kl = step_val = 0.0

        for _ in range(ACCUM_STEPS):
            batch = phase_records[cursor:cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            if len(batch) < BATCH_SIZE:
                needed = BATCH_SIZE - len(batch)
                batch = batch + phase_records[:needed]
                cursor = needed

            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
            value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
            soft_targets = sparse_soft_targets_to_dense(batch, teacher_temp=args.teacher_temp, soft_top_k=args.soft_top_k).to(DEVICE)
            kl_weights = batch_kl_confidence_weights(batch, kl_conf_scale=args.kl_conf_scale, kl_conf_min=args.kl_conf_min, kl_conf_max=args.kl_conf_max).to(DEVICE)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
                out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
                logits = out["policy_logits"]
                value_logits = out["value_logits"]
                hard_ce, kl = compute_policy_losses(logits, best_moves, soft_targets, kl_weights)
                value_loss = F.cross_entropy(value_logits, value_targets)
                total_loss = ((1.0 - args.hard_ce_weight) * kl + args.hard_ce_weight * hard_ce + args.value_loss_weight * value_loss) / ACCUM_STEPS

            scaler.scale(total_loss).backward()
            step_loss += total_loss.item() * ACCUM_STEPS
            step_ce += hard_ce.item()
            step_kl += kl.item()
            step_val += value_loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # EMA update
        if global_step >= EMA_START_STEP:
            ema.update(model)

        state["train_steps"] = global_step
        loss_sum += step_loss
        ce_sum += step_ce / ACCUM_STEPS
        kl_sum += step_kl / ACCUM_STEPS
        val_sum += step_val / ACCUM_STEPS

        done_steps = step_idx + 1
        if global_step % LOG_INTERVAL == 0:
            log(
                f"phase={phase_idx} step={global_step} "
                f"loss={loss_sum / done_steps:.4f} ce={ce_sum / done_steps:.4f} "
                f"kl={kl_sum / done_steps:.4f} value={val_sum / done_steps:.4f} "
                f"gnorm={float(grad_norm):.2f} lr={current_lr:.2e}"
            )

        if global_step % SAVE_INTERVAL == 0 or step_idx == steps_this_phase - 1:
            # Eval live model
            eval_live = evaluate(
                model, eval_records,
                teacher_temp=args.teacher_temp, hard_ce_weight=args.hard_ce_weight,
                value_loss_weight=args.value_loss_weight, soft_top_k=args.soft_top_k,
                kl_conf_scale=args.kl_conf_scale, kl_conf_min=args.kl_conf_min, kl_conf_max=args.kl_conf_max,
            )
            log(
                f"eval_live phase={phase_idx} step={global_step} loss={eval_live['loss']:.4f} "
                f"ce={eval_live['ce']:.4f} kl={eval_live['kl']:.4f} value={eval_live['value']:.4f} "
                f"acc={eval_live['acc']:.4f} top3={eval_live['top3']:.4f}"
            )

            # Eval EMA model
            ema.apply_shadow(model)
            eval_ema = evaluate(
                model, eval_records,
                teacher_temp=args.teacher_temp, hard_ce_weight=args.hard_ce_weight,
                value_loss_weight=args.value_loss_weight, soft_top_k=args.soft_top_k,
                kl_conf_scale=args.kl_conf_scale, kl_conf_min=args.kl_conf_min, kl_conf_max=args.kl_conf_max,
            )
            ema.restore(model)
            log(
                f"eval_ema  phase={phase_idx} step={global_step} loss={eval_ema['loss']:.4f} "
                f"ce={eval_ema['ce']:.4f} kl={eval_ema['kl']:.4f} value={eval_ema['value']:.4f} "
                f"acc={eval_ema['acc']:.4f} top3={eval_ema['top3']:.4f}"
            )

            # Decide which is better
            best_loss = min(eval_live["loss"], eval_ema["loss"])
            winner = "ema" if eval_ema["loss"] <= eval_live["loss"] else "live"
            is_best = best_loss < best_eval
            if is_best:
                best_eval = best_loss
                state["best_eval_loss"] = best_eval
                log(f"new_best loss={best_loss:.4f} source={winner}")

            state["last_eval_live"] = eval_live
            state["last_eval_ema"] = eval_ema
            state["last_eval"] = eval_ema if winner == "ema" else eval_live

            # Save checkpoints
            if SAVE_CHECKPOINTS:
                save_model_weights(LATEST_MODEL_PATH, model)
                ema.save_weights(EMA_MODEL_PATH, model)
                if is_best:
                    if winner == "ema":
                        ema.save_weights(BEST_PATH, model)
                    else:
                        save_model_weights(BEST_PATH, model)

            atomic_write_json(STATUS_PATH, {
                "updated_at": utcnow_iso(),
                "phase": phase_idx,
                "train_steps": global_step,
                "best_eval_loss": best_eval,
                "best_source": winner,
                "eval_live": eval_live,
                "eval_ema": eval_ema,
                "curriculum_phase_records": len(phase_records),
                "lr": current_lr,
            })
            model.train()

    return steps_this_phase


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp093: EMA + Curriculum + Cosine LR trainer")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--dataset-glob", type=str, default=None)
    p.add_argument("--dataset-path", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--teacher-temp", type=float, default=TEACHER_TEMP)
    p.add_argument("--hard-ce-weight", type=float, default=HARD_CE_WEIGHT)
    p.add_argument("--value-loss-weight", type=float, default=VALUE_LOSS_WEIGHT)
    p.add_argument("--soft-top-k", type=int, default=SOFT_TOP_K)
    p.add_argument("--kl-conf-scale", type=float, default=KL_CONF_SCALE)
    p.add_argument("--kl-conf-min", type=float, default=KL_CONF_MIN)
    p.add_argument("--kl-conf-max", type=float, default=KL_CONF_MAX)
    p.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    p.add_argument("--curriculum-phases", type=int, default=CURRICULUM_PHASES)
    p.add_argument("--curriculum-key", type=str, default=CURRICULUM_KEY)
    p.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    p.add_argument("--no-upload-to-hf", action="store_true")
    p.add_argument("--no-save-checkpoints", action="store_true")
    p.add_argument("--save-final-only", action="store_true")
    p.add_argument("--save-weights-only-checkpoints", action="store_true")
    return p.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset_path is not None:
        return [args.dataset_path]
    if args.dataset_glob:
        paths = sorted(Path().glob(args.dataset_glob))
    else:
        # Default: try relabeled d8 first, then fall back to exp087
        d8_paths = sorted(Path("outputs/exp087_relabeled_d8/dataset").glob("positions_*.jsonl"))
        if d8_paths:
            paths = d8_paths
        else:
            paths = sorted(Path("outputs/exp087_full_legal_harvest/dataset").glob("positions_*.jsonl"))
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise FileNotFoundError("No dataset files found.")
    return paths


def resolve_init_checkpoint(args: argparse.Namespace) -> Path:
    if args.init_checkpoint is not None:
        p = args.init_checkpoint.resolve()
        if p.exists():
            return p
    for c in INIT_CHECKPOINT_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError("No init checkpoint found.")


def main() -> None:
    global OUTPUT_DIR, CHECKPOINT_DIR, LOG_PATH, CONFIG_PATH, STATUS_PATH
    global LATEST_PATH, BEST_STATE_PATH, LATEST_MODEL_PATH, BEST_PATH, EMA_MODEL_PATH
    global LOG_FILE, SAVE_CHECKPOINTS, SAVE_FINAL_ONLY, SAVE_WEIGHTS_ONLY_CHECKPOINTS

    args = parse_args()
    SAVE_CHECKPOINTS = not args.no_save_checkpoints and not args.save_final_only
    SAVE_FINAL_ONLY = args.save_final_only
    SAVE_WEIGHTS_ONLY_CHECKPOINTS = args.save_weights_only_checkpoints

    OUTPUT_DIR = args.output_dir
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_PATH = OUTPUT_DIR / "exp093.log"
    CONFIG_PATH = OUTPUT_DIR / "config.json"
    STATUS_PATH = OUTPUT_DIR / "status.json"
    LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
    BEST_STATE_PATH = CHECKPOINT_DIR / "best.pt"
    LATEST_MODEL_PATH = OUTPUT_DIR / "latest_model.pt"
    BEST_PATH = OUTPUT_DIR / "best_model.pt"
    EMA_MODEL_PATH = OUTPUT_DIR / "ema_model.pt"

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_PATH

    # Load data
    dataset_paths = resolve_dataset_paths(args)
    records = load_jsonl_dataset(dataset_paths)
    train_records, eval_records = stable_train_eval_split(records)

    # Curriculum phases
    curriculum = sort_by_curriculum(train_records, args.curriculum_key, args.curriculum_phases)

    # Compute total steps for LR schedule
    total_steps = 0
    for phase_records in curriculum:
        total_steps += math.ceil(len(phase_records) / (BATCH_SIZE * ACCUM_STEPS))
    total_steps *= args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    # Load model
    init_checkpoint = resolve_init_checkpoint(args)
    model = load_model(init_checkpoint, DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=AMP_ENABLED)
    ema = EMAModel(model, decay=args.ema_decay)

    state = {
        "epoch": 1,
        "train_steps": 0,
        "best_eval_loss": float("inf"),
        "dataset_records": len(records),
        "last_eval": None,
    }

    # Config dump
    atomic_write_json(CONFIG_PATH, {
        "started_at": utcnow_iso(),
        "init_checkpoint": str(init_checkpoint),
        "dataset_files": [str(p) for p in dataset_paths],
        "dataset_records": len(records),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "effective_batch": BATCH_SIZE * ACCUM_STEPS,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "min_lr_frac": MIN_LR_FRAC,
        "teacher_temp": args.teacher_temp,
        "soft_top_k": args.soft_top_k,
        "kl_conf_scale": args.kl_conf_scale,
        "kl_conf_min": args.kl_conf_min,
        "kl_conf_max": args.kl_conf_max,
        "hard_ce_weight": args.hard_ce_weight,
        "value_loss_weight": args.value_loss_weight,
        "ema_decay": args.ema_decay,
        "ema_start_step": EMA_START_STEP,
        "curriculum_phases": args.curriculum_phases,
        "curriculum_key": args.curriculum_key,
        "curriculum_phase_sizes": [len(phase) for phase in curriculum],
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "device": str(DEVICE),
    })

    log("=" * 72)
    log("exp093: EMA + Curriculum + Cosine LR training")
    log("=" * 72)
    log(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    log(f"dataset={len(records)} train={len(train_records)} eval={len(eval_records)} files={len(dataset_paths)}")
    log(f"effective_batch={BATCH_SIZE * ACCUM_STEPS} epochs={args.epochs} total_steps={total_steps}")
    log(f"lr={args.lr} warmup={warmup_steps}steps cosine→{MIN_LR_FRAC * args.lr:.2e}")
    log(f"ema_decay={args.ema_decay} start_step={EMA_START_STEP}")
    log(f"curriculum: {args.curriculum_phases} phases on '{args.curriculum_key}' "
        f"sizes={[len(p) for p in curriculum]}")
    log(f"teacher_temp={args.teacher_temp} hard_ce={args.hard_ce_weight} soft_top_k={args.soft_top_k}")
    log(f"kl_conf_scale={args.kl_conf_scale} kl_conf_min={args.kl_conf_min} kl_conf_max={args.kl_conf_max}")
    log(f"init_checkpoint={init_checkpoint}")

    # Initial eval
    initial_eval = evaluate(
        model, eval_records,
        teacher_temp=args.teacher_temp, hard_ce_weight=args.hard_ce_weight,
        value_loss_weight=args.value_loss_weight, soft_top_k=args.soft_top_k,
        kl_conf_scale=args.kl_conf_scale, kl_conf_min=args.kl_conf_min, kl_conf_max=args.kl_conf_max,
    )
    log(
        f"initial_eval loss={initial_eval['loss']:.4f} ce={initial_eval['ce']:.4f} "
        f"kl={initial_eval['kl']:.4f} value={initial_eval['value']:.4f} "
        f"acc={initial_eval['acc']:.4f} top3={initial_eval['top3']:.4f}"
    )
    state["best_eval_loss"] = initial_eval["loss"]
    state["last_eval"] = initial_eval

    if SAVE_CHECKPOINTS:
        save_model_weights(LATEST_MODEL_PATH, model)
        save_model_weights(BEST_PATH, model)

    # Training loop: epochs × curriculum phases
    steps_so_far = 0
    for epoch in range(1, args.epochs + 1):
        state["epoch"] = epoch
        log(f"--- epoch {epoch}/{args.epochs} ---")

        for phase_idx, phase_records in enumerate(curriculum):
            log(f"curriculum phase {phase_idx}/{len(curriculum)-1}: "
                f"{len(phase_records)} records "
                f"(cp_gap range: {_gap_range(phase_records, args.curriculum_key)})")

            steps_trained = train_curriculum_phase(
                phase_idx=phase_idx,
                phase_records=phase_records,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                eval_records=eval_records,
                state=state,
                args=args,
                total_steps_so_far=steps_so_far,
                total_steps_all=total_steps,
                warmup_steps=warmup_steps,
            )
            steps_so_far += steps_trained

    # Final eval
    log("--- final evaluation ---")
    final_live = evaluate(
        model, eval_records,
        teacher_temp=args.teacher_temp, hard_ce_weight=args.hard_ce_weight,
        value_loss_weight=args.value_loss_weight, soft_top_k=args.soft_top_k,
        kl_conf_scale=args.kl_conf_scale, kl_conf_min=args.kl_conf_min, kl_conf_max=args.kl_conf_max,
    )
    log(f"final_live loss={final_live['loss']:.4f} acc={final_live['acc']:.4f} top3={final_live['top3']:.4f}")

    ema.apply_shadow(model)
    final_ema = evaluate(
        model, eval_records,
        teacher_temp=args.teacher_temp, hard_ce_weight=args.hard_ce_weight,
        value_loss_weight=args.value_loss_weight, soft_top_k=args.soft_top_k,
        kl_conf_scale=args.kl_conf_scale, kl_conf_min=args.kl_conf_min, kl_conf_max=args.kl_conf_max,
    )
    ema.restore(model)
    log(f"final_ema  loss={final_ema['loss']:.4f} acc={final_ema['acc']:.4f} top3={final_ema['top3']:.4f}")

    winner = "ema" if final_ema["loss"] <= final_live["loss"] else "live"
    log(f"final winner: {winner}")

    # Save final
    save_model_weights(LATEST_MODEL_PATH, model)
    ema.save_weights(EMA_MODEL_PATH, model)
    best_final = min(final_live["loss"], final_ema["loss"])
    if best_final <= state["best_eval_loss"]:
        if winner == "ema":
            ema.save_weights(BEST_PATH, model)
        else:
            save_model_weights(BEST_PATH, model)

    atomic_write_json(STATUS_PATH, {
        "updated_at": utcnow_iso(),
        "done": True,
        "train_steps": steps_so_far,
        "best_eval_loss": min(state["best_eval_loss"], best_final),
        "final_live": final_live,
        "final_ema": final_ema,
        "winner": winner,
    })
    log("done")


def _gap_range(records: list[dict], key: str) -> str:
    """Helper to show min-max of curriculum key for logging."""
    vals = [float(r.get(key, 0)) for r in records]
    if not vals:
        return "empty"
    return f"{min(vals):.0f}-{max(vals):.0f}"


if __name__ == "__main__":
    main()
