"""exp098: Strong value head training with SF eval targets.

Problem: exp094/097 showed search HURTS ELO because value head is too weak
(trained at 10% weight with noisy game-outcome targets).

Solution: Train value head with:
  1. Stockfish centipawn evaluation (best_cp) converted to WDL probabilities
  2. Much higher value weight (50% of total loss)
  3. Continue from best policy model (exp093-d8 when available)

If this produces a good value head, even simple 1-ply search should add 100+ ELO.

Usage:
    python experiments/exp098_strong_value_head.py \
        --output-dir outputs/exp098_strong_value_d8 \
        --init-checkpoint outputs/exp093_ema_curriculum_d8/best_model.pt \
        --dataset-glob "outputs/exp087_relabeled_d8/dataset/positions_*.jsonl" \
        --value-weight 0.50 --epochs 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
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

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

# Defaults
BATCH_SIZE = 4
ACCUM_STEPS = 16
EPOCHS = 2
LR = 1e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
HARD_CE_WEIGHT = 0.5
VALUE_LOSS_WEIGHT = 0.50  # 5x higher than exp093's 0.10
TEACHER_TEMP = 0.5
SOFT_TOP_K = 8
KL_CONF_SCALE = 80.0
KL_CONF_MIN = 0.10
KL_CONF_MAX = 1.00
EVAL_FRACTION = 0.05
MAX_EVAL_RECORDS = 2048
LOG_INTERVAL = 25
SAVE_INTERVAL = 200
SEED = 42

# EMA
EMA_DECAY = 0.999
EMA_START_STEP = 50

WARMUP_FRAC = 0.05
MIN_LR_FRAC = 0.10

LOG_FILE = None


# ---------------------------------------------------------------------------
# CP to WDL conversion
# ---------------------------------------------------------------------------

def cp_to_wdl(cp: float, ply: int = 30) -> tuple[float, float, float]:
    """Convert centipawn evaluation to (loss, draw, win) probabilities.

    Uses LC0-style conversion: win_rate = sigmoid(cp / k) where k depends
    on game phase (approximated by ply). This is more informative than
    binary game outcome for middlegame positions.

    Returns (loss_prob, draw_prob, win_prob) from side-to-move perspective.
    """
    # Clamp extreme values
    cp = max(-1500, min(1500, cp))

    # k factor: increases with ply (more uncertain in middlegame)
    # These values roughly match LC0's WDL model
    k = 111.714 + 0.2 * max(0, 50 - ply)  # ~120 in early game, ~112 in endgame

    win_prob = 1.0 / (1.0 + math.exp(-cp / k))
    loss_prob = 1.0 - win_prob

    # Estimate draw probability using a simple model:
    # draws are more likely when eval is near 0
    draw_width = 0.5 * math.exp(-abs(cp) / 200.0)
    draw_prob = draw_width * min(win_prob, loss_prob) * 4  # peaks at eval=0

    # Redistribute
    total = win_prob + loss_prob
    if total > 0:
        win_prob = win_prob / total * (1.0 - draw_prob)
        loss_prob = loss_prob / total * (1.0 - draw_prob)

    # Ensure valid probabilities
    s = win_prob + draw_prob + loss_prob
    if s > 0:
        return (loss_prob / s, draw_prob / s, win_prob / s)
    return (0.333, 0.334, 0.333)


# ---------------------------------------------------------------------------
# Utilities (same as exp093)
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


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
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


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_frac):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine_decay)


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(paths: list[Path]) -> list[dict]:
    records = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def record_to_tensors(item: dict, device: torch.device) -> dict | None:
    try:
        board = chess.Board(item["fen"])
    except Exception:
        return None

    soft_targets = item.get("soft_targets", [])
    if not soft_targets:
        return None

    raw_logits = torch.full((VOCAB_SIZE,), -1e9, dtype=torch.float32)
    for entry in soft_targets:
        idx = UCI_TO_IDX.get(entry["uci"])
        if idx is not None:
            raw_logits[idx] = float(entry["cp"])

    mask = legal_move_mask(board)
    raw_logits[~mask] = -1e9

    topk = torch.topk(raw_logits, min(SOFT_TOP_K, int(mask.sum().item())))
    sparse_logits = torch.full_like(raw_logits, -1e9)
    sparse_logits[topk.indices] = topk.values
    teacher_probs = F.softmax(sparse_logits / max(TEACHER_TEMP, 1e-6), dim=-1)

    hard_target = UCI_TO_IDX.get(item.get("best_move", ""))
    if hard_target is None:
        return None

    # CP-based WDL target (better than game outcome)
    best_cp = item.get("best_cp", 0)
    ply = item.get("ply", 30)
    wdl = cp_to_wdl(float(best_cp), int(ply))
    value_target = torch.tensor(wdl, dtype=torch.float32)

    cp_gap = max(float(item.get("cp_gap_top1_top2", 0.0)), 0.0)

    inp = batch_boards_to_fused_token_ids([board], device)

    return {
        "input": inp,
        "teacher_probs": teacher_probs.to(device),
        "hard_target": torch.tensor(hard_target, dtype=torch.long).to(device),
        "value_target": value_target.to(device),
        "mask": mask.to(device),
        "cp_gap": cp_gap,
    }


def sort_by_curriculum(records, key, phases):
    keyed = [(float(r.get(key, 0)), r) for r in records]
    keyed.sort(key=lambda x: x[0], reverse=True)
    sorted_records = [r for _, r in keyed]
    phase_size = max(1, len(sorted_records) // phases)
    batches = []
    for i in range(phases):
        end = min((i + 1) * phase_size, len(sorted_records))
        if i == phases - 1:
            end = len(sorted_records)
        batches.append(sorted_records[:end])
    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(model, batch, optimizer, scaler, accum_step, total_accum):
    is_last = (accum_step + 1) == total_accum

    with autocast("cuda", enabled=AMP_ENABLED):
        result = model(batch["input"])
        policy_logits = result["policy_logits"]
        value_logits = result["value_logits"]

        log_probs = F.log_softmax(policy_logits.float(), dim=-1)
        teacher = batch["teacher_probs"].unsqueeze(0)
        kl_per_item = F.kl_div(log_probs, teacher, reduction="none").sum(-1)

        cp_gap = batch["cp_gap"]
        conf = min(KL_CONF_MAX, max(KL_CONF_MIN, cp_gap / KL_CONF_SCALE))
        kl_loss = kl_per_item.mean() * conf

        ce_loss = F.cross_entropy(policy_logits.float(), batch["hard_target"].unsqueeze(0))

        # CP-based WDL value loss (much better than game outcome)
        value_loss = F.cross_entropy(value_logits.float(), batch["value_target"].unsqueeze(0))

        loss = kl_loss + HARD_CE_WEIGHT * ce_loss + VALUE_LOSS_WEIGHT * value_loss
        loss = loss / total_accum

    scaler.scale(loss).backward()

    if is_last:
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    else:
        gnorm = 0.0

    return {
        "loss": loss.item() * total_accum,
        "ce": ce_loss.item(),
        "kl": kl_per_item.mean().item(),
        "value": value_loss.item(),
        "gnorm": gnorm,
    }


@torch.no_grad()
def evaluate(model, eval_data, device):
    model.eval()
    metrics = {"loss": 0, "ce": 0, "kl": 0, "value": 0, "correct": 0, "top3": 0, "n": 0}

    for item in eval_data:
        t = record_to_tensors(item, device)
        if t is None:
            continue

        result = model(t["input"])
        logits = result["policy_logits"].float()
        value_logits = result["value_logits"].float()

        log_probs = F.log_softmax(logits, dim=-1)
        teacher = t["teacher_probs"].unsqueeze(0)
        kl = F.kl_div(log_probs, teacher, reduction="batchmean").item()
        ce = F.cross_entropy(logits, t["hard_target"].unsqueeze(0)).item()
        val = F.cross_entropy(value_logits, t["value_target"].unsqueeze(0)).item()

        loss = kl + HARD_CE_WEIGHT * ce + VALUE_LOSS_WEIGHT * val

        masked = logits.clone()
        masked[0][~t["mask"]] = float("-inf")
        pred = masked[0].argmax().item()
        top3_preds = masked[0].topk(3).indices.tolist()

        target = t["hard_target"].item()
        metrics["loss"] += loss
        metrics["ce"] += ce
        metrics["kl"] += kl
        metrics["value"] += val
        metrics["correct"] += int(pred == target)
        metrics["top3"] += int(target in top3_preds)
        metrics["n"] += 1

    model.train()
    n = max(metrics["n"], 1)
    return {
        "loss": metrics["loss"] / n,
        "ce": metrics["ce"] / n,
        "kl": metrics["kl"] / n,
        "value": metrics["value"] / n,
        "acc": metrics["correct"] / n,
        "top3": metrics["top3"] / n,
        "n": metrics["n"],
    }


def load_model(checkpoint_path, device):
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device)


def main():
    global LOG_FILE, SOFT_TOP_K, TEACHER_TEMP, KL_CONF_SCALE, KL_CONF_MIN, KL_CONF_MAX
    global VALUE_LOSS_WEIGHT, HARD_CE_WEIGHT

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-glob", type=str, required=True)
    parser.add_argument("--value-weight", type=float, default=VALUE_LOSS_WEIGHT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--curriculum-phases", type=int, default=3)
    parser.add_argument("--save-weights-only-checkpoints", action="store_true")
    parser.add_argument("--no-upload-to-hf", action="store_true")
    args = parser.parse_args()

    VALUE_LOSS_WEIGHT = args.value_weight
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG_FILE = output_dir / "exp098.log"
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    import glob as globmod
    files = sorted(Path(p) for p in globmod.glob(args.dataset_glob))
    if not files:
        log(f"ERROR: no files match {args.dataset_glob}")
        return

    log("exp098: Strong Value Head + EMA + Curriculum training")
    log("=" * 72)
    log(f"device={DEVICE}")

    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"gpu={props.name} vram_gb={props.total_memory / 1e9:.1f}")

    all_records = load_jsonl_dataset(files)
    random.seed(SEED)
    random.shuffle(all_records)

    n_eval = min(MAX_EVAL_RECORDS, max(1, int(len(all_records) * EVAL_FRACTION)))
    eval_records = all_records[:n_eval]
    train_records = all_records[n_eval:]

    log(f"dataset={len(all_records)} train={len(train_records)} eval={len(eval_records)} files={len(files)}")

    total_steps_per_epoch = len(train_records) // (BATCH_SIZE * ACCUM_STEPS)
    curriculum_phases = args.curriculum_phases
    steps_per_phase = total_steps_per_epoch // curriculum_phases
    total_steps = total_steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))

    log(f"effective_batch={BATCH_SIZE * ACCUM_STEPS} epochs={args.epochs} total_steps={total_steps}")
    log(f"lr={args.lr} warmup={warmup_steps}steps")
    log(f"value_weight={VALUE_LOSS_WEIGHT} (5x normal) — CP-based WDL targets")
    log(f"ema_decay={args.ema_decay} start_step={EMA_START_STEP}")
    log(f"curriculum: {curriculum_phases} phases on 'cp_gap_top1_top2'")
    log(f"init_checkpoint={args.init_checkpoint.resolve()}")

    model = load_model(args.init_checkpoint, DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=AMP_ENABLED)
    ema = EMAModel(model, decay=args.ema_decay)

    # Initial eval
    init_metrics = evaluate(model, eval_records, DEVICE)
    log(f"initial_eval loss={init_metrics['loss']:.4f} ce={init_metrics['ce']:.4f} "
        f"kl={init_metrics['kl']:.4f} value={init_metrics['value']:.4f} "
        f"acc={init_metrics['acc']:.4f} top3={init_metrics['top3']:.4f}")

    curriculum = sort_by_curriculum(train_records, "cp_gap_top1_top2", curriculum_phases)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        log(f"--- epoch {epoch}/{args.epochs} ---")

        for phase_idx, phase_records in enumerate(curriculum):
            phase_steps = len(phase_records) // (BATCH_SIZE * ACCUM_STEPS)
            random.shuffle(phase_records)

            gap_vals = [float(r.get("cp_gap_top1_top2", 0)) for r in phase_records]
            log(f"curriculum phase {phase_idx}/{curriculum_phases - 1}: "
                f"{len(phase_records)} records (cp_gap range: {int(min(gap_vals))}-{int(max(gap_vals))})")

            idx = 0
            accum_metrics = {"loss": 0, "ce": 0, "kl": 0, "value": 0, "gnorm": 0, "count": 0}

            for step_in_phase in range(phase_steps):
                for accum_step in range(ACCUM_STEPS):
                    batch_items = phase_records[idx:idx + BATCH_SIZE]
                    idx += BATCH_SIZE
                    if idx >= len(phase_records):
                        random.shuffle(phase_records)
                        idx = 0

                    for item in batch_items:
                        t = record_to_tensors(item, DEVICE)
                        if t is None:
                            continue

                        metrics = train_step(model, t, optimizer, scaler, accum_step, ACCUM_STEPS)
                        for k in ["loss", "ce", "kl", "value", "gnorm"]:
                            accum_metrics[k] += metrics[k]
                        accum_metrics["count"] += 1

                global_step += 1

                # LR schedule
                lr = cosine_lr(global_step, total_steps, warmup_steps, args.lr, MIN_LR_FRAC)
                set_lr(optimizer, lr)

                # EMA update
                if global_step >= EMA_START_STEP:
                    ema.update(model)

                # Logging
                if global_step % LOG_INTERVAL == 0:
                    n = max(accum_metrics["count"], 1)
                    log(f"phase={phase_idx} step={global_step} "
                        f"loss={accum_metrics['loss']/n:.4f} "
                        f"ce={accum_metrics['ce']/n:.4f} "
                        f"kl={accum_metrics['kl']/n:.4f} "
                        f"value={accum_metrics['value']/n:.4f} "
                        f"gnorm={accum_metrics['gnorm']/n:.2f} "
                        f"lr={lr:.2e}")
                    accum_metrics = {"loss": 0, "ce": 0, "kl": 0, "value": 0, "gnorm": 0, "count": 0}

                # Eval + Save
                if global_step % SAVE_INTERVAL == 0 or global_step == total_steps:
                    live_metrics = evaluate(model, eval_records, DEVICE)
                    log(f"eval_live phase={phase_idx} step={global_step} "
                        f"loss={live_metrics['loss']:.4f} ce={live_metrics['ce']:.4f} "
                        f"kl={live_metrics['kl']:.4f} value={live_metrics['value']:.4f} "
                        f"acc={live_metrics['acc']:.4f} top3={live_metrics['top3']:.4f}")

                    ema.apply_shadow(model)
                    ema_metrics = evaluate(model, eval_records, DEVICE)
                    ema.restore(model)
                    log(f"eval_ema  phase={phase_idx} step={global_step} "
                        f"loss={ema_metrics['loss']:.4f} ce={ema_metrics['ce']:.4f} "
                        f"kl={ema_metrics['kl']:.4f} value={ema_metrics['value']:.4f} "
                        f"acc={ema_metrics['acc']:.4f} top3={ema_metrics['top3']:.4f}")

                    # Save best (either live or ema)
                    for tag, m in [("live", live_metrics), ("ema", ema_metrics)]:
                        if m["loss"] < best_loss:
                            best_loss = m["loss"]
                            if tag == "ema":
                                ema.save_weights(output_dir / "best_model.pt", model)
                            else:
                                save_model_weights(output_dir / "best_model.pt", model)
                            log(f"new best: {tag} loss={m['loss']:.4f}")

                    save_model_weights(output_dir / "latest_model.pt", model)
                    ema.save_weights(output_dir / "ema_model.pt", model)

    # Final evaluation
    log("--- final evaluation ---")
    final_live = evaluate(model, eval_records, DEVICE)
    log(f"final_live loss={final_live['loss']:.4f} acc={final_live['acc']:.4f} "
        f"top3={final_live['top3']:.4f} value={final_live['value']:.4f}")

    ema.apply_shadow(model)
    final_ema = evaluate(model, eval_records, DEVICE)
    ema.restore(model)
    log(f"final_ema  loss={final_ema['loss']:.4f} acc={final_ema['acc']:.4f} "
        f"top3={final_ema['top3']:.4f} value={final_ema['value']:.4f}")

    winner = "ema" if final_ema["loss"] < final_live["loss"] else "live"
    log(f"final winner: {winner}")
    log("done")


if __name__ == "__main__":
    main()
