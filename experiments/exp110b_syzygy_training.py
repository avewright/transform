"""exp110b: Endgame-enriched training with perfect Syzygy labels.

Hypothesis: Adding 50K+ perfectly-labeled endgame positions from Syzygy tables
on top of the exp110 diverse training will push endgame accuracy significantly
and improve overall ELO. Perfect labels (depth=999) give zero label noise
in endgames.

Strategy:
  - Init from BEST checkpoint of exp110 diverse training
  - Data: exp085 (224K) + diverse (7.5K) + syzygy (50K) + tablebase (1.3K)
  - Give syzygy positions 2x weight in batch sampling (perfect labels)
  - Shallower LR (2e-6) since we're 2nd-stage fine-tuning
  - Focus: endgame accuracy, preserving opening/middlegame knowledge

Architecture: ChessTransformer 200M (unchanged)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import chess
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import UCI_TO_IDX, VOCAB_SIZE, legal_move_mask
from play import ChessTransformer200M

# ── Config ──
OUTPUT_DIR = Path("outputs/exp110b_syzygy_training")
DATASET_DIRS = [
    Path("outputs/exp085_hf_data/dataset"),           # 224K opening-heavy multi-PV
    Path("outputs/exp110_diverse_harvest/dataset"),    # 7.5K diverse middlegame/endgame
    Path("outputs/exp110_diverse_harvest_v2/dataset"), # 15K extra diverse positions
    Path("outputs/exp110_diverse_harvest_v3/dataset"), # extra diverse positions
    Path("outputs/exp110_diverse_harvest_v4/dataset"), # extra diverse positions (seed 400)
    Path("outputs/exp110_syzygy/dataset"),             # 50K perfect endgame (Syzygy)
    Path("outputs/exp110_tablebase/dataset"),          # 1.3K deep endgame (Stockfish)
    Path("outputs/exp110_puzzle_harvest/dataset"),      # 30K tactical puzzles
]

# This will be set dynamically at runtime (best from exp110 diverse training)
INIT_CHECKPOINT = Path("outputs/exp110_diverse_training/best_model.pt")
FALLBACK_CHECKPOINT = Path("outputs/hf_checkpoint/best_model.pt")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

BATCH_SIZE = 8
ACCUM_STEPS = 8      # effective batch = 64
EPOCHS = 2            # faster iteration, most learning happens in epoch 1
LR = 2e-6             # lower LR for 2nd-stage fine-tuning
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5

# Loss weights
HARD_CE_WEIGHT = 0.20   # 20% hard CE + 80% soft KL (more emphasis on soft targets)
VALUE_WEIGHT = 0.50      # strong value head
TEACHER_TEMP = 0.5
SOFT_TOP_K = 8

# KL confidence scaling
KL_CONF_SCALE = 80.0
KL_CONF_MIN = 0.10
KL_CONF_MAX = 1.00

# Syzygy positions get 2x weight (perfect labels)
SYZYGY_WEIGHT = 2.0

# EMA
EMA_DECAY = 0.999
EMA_START_STEP = 50

# LR schedule
WARMUP_FRAC = 0.05
MIN_LR_FRAC = 0.10

# Eval
EVAL_FRACTION = 0.03
MAX_EVAL_RECORDS = 2048

# Logging
LOG_INTERVAL = 25
EVAL_INTERVAL = 100
SAVE_INTERVAL = 200

STOP_REQUESTED = False


def signal_handler(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[SIGNAL] Shutdown requested...", flush=True)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def log(msg: str):
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    log_path = OUTPUT_DIR / "exp110b.log"
    with open(log_path, "a") as f:
        f.write(stamped + "\n")


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
            save_model(path, model)
        finally:
            self.restore(model)


def save_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        k.replace("_orig_mod.", ""): v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    torch.save({"model_state_dict": state}, str(path))


def cosine_lr(step, total, warmup, base_lr, min_frac):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ── Data Loading ──
def load_dataset(dataset_dirs: list[Path]) -> list[dict]:
    records = []
    source_counts = Counter()
    phase_counts = Counter()

    for ddir in dataset_dirs:
        if not ddir.exists():
            log(f"  WARNING: {ddir} not found, skipping")
            continue
        files = sorted(ddir.glob("positions_*.jsonl"))
        dir_count = 0
        for f in files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if rec.get("soft_targets") and rec.get("best_move"):
                                records.append(rec)
                                dir_count += 1
                                source_counts[rec.get("source", "unknown")] += 1
                                phase_counts[rec.get("phase", "unknown")] += 1
                        except json.JSONDecodeError:
                            pass
        log(f"  Loaded {dir_count:,} from {ddir}")

    log(f"  Total: {len(records):,} records")
    log(f"  Sources: {dict(source_counts)}")
    log(f"  Phases: {dict(phase_counts)}")
    return records


def weighted_oversample(records: list[dict]) -> list[dict]:
    """Oversample high-quality data (syzygy/tablebase) and balance phases."""
    result = []
    for r in records:
        source = r.get("source", "")
        # Syzygy data gets extra weight
        if "syzygy" in source or "tablebase" in source:
            n_copies = int(SYZYGY_WEIGHT)
            result.extend([r] * n_copies)
        else:
            result.append(r)
    
    random.shuffle(result)
    return result


def split_train_eval(records: list[dict]) -> tuple[list[dict], list[dict]]:
    eval_n = min(MAX_EVAL_RECORDS, max(512, int(len(records) * EVAL_FRACTION)))
    keyed = []
    for r in records:
        h = hashlib.blake2b(r["fen"].encode(), digest_size=8).digest()
        keyed.append((int.from_bytes(h, "big"), r))
    keyed.sort(key=lambda x: x[0])
    eval_recs = [r for _, r in keyed[:eval_n]]
    train_recs = [r for _, r in keyed[eval_n:]]
    return train_recs, eval_recs


# ── Loss Functions ──
def build_soft_targets(batch, teacher_temp, top_k):
    dense = torch.zeros(len(batch), VOCAB_SIZE, dtype=torch.float32)
    for i, item in enumerate(batch):
        targets = item["soft_targets"][:top_k]
        probs = [max(float(t["prob"]), 1e-12) for t in targets]
        if abs(teacher_temp - 1.0) > 1e-9:
            probs = [p ** (1.0 / teacher_temp) for p in probs]
        total = sum(probs)
        for t, p in zip(targets, probs):
            if t["uci"] in UCI_TO_IDX:
                dense[i, UCI_TO_IDX[t["uci"]]] = p / total
    return dense


def kl_confidence_weights(batch):
    weights = []
    for item in batch:
        cp_gap = abs(item.get("cp_gap_top1_top2", 0))
        w = min(max(cp_gap / KL_CONF_SCALE, KL_CONF_MIN), KL_CONF_MAX)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def build_value_targets(batch, device):
    targets = []
    for item in batch:
        vt = item.get("value_target", 1)
        targets.append(vt)
    return torch.tensor(targets, dtype=torch.long, device=device)


def build_batch(batch, device):
    boards = []
    labels = []
    masks = []
    for item in batch:
        fen = item["fen"]
        board = chess.Board(fen)
        boards.append(board)
        best_uci = item["best_move"]
        idx = UCI_TO_IDX.get(best_uci, -1)
        labels.append(idx)
        masks.append(legal_move_mask(board))

    token_ids = batch_boards_to_fused_token_ids(boards)
    token_ids = token_ids.to(device=device, non_blocking=True)
    label_t = torch.tensor(labels, dtype=torch.long, device=device)
    mask_t = torch.stack(masks).to(device=device, non_blocking=True)
    return boards, token_ids, label_t, mask_t


def compute_loss(model, token_ids, label_t, mask_t, soft_t, conf_w, value_t, device):
    out = model(token_ids)
    logits = out["policy_logits"].float()
    value_logits = out["value_logits"].float()
    logits = logits + (mask_t.float() - 1.0) * 1e9
    log_probs = F.log_softmax(logits, dim=-1)

    # Hard CE
    ce_hard = F.cross_entropy(logits, label_t, reduction="mean")

    # Soft KL
    soft_t = soft_t.to(device=device)
    conf_w = conf_w.to(device=device)
    raw_kl = F.kl_div(log_probs, soft_t, reduction="none").sum(dim=-1)
    kl_soft = (raw_kl * conf_w).mean()

    # Value
    value_loss = F.cross_entropy(value_logits, value_t, reduction="mean")

    policy_loss = (1.0 - HARD_CE_WEIGHT) * kl_soft + HARD_CE_WEIGHT * ce_hard
    total = policy_loss + VALUE_WEIGHT * value_loss

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc = (preds == label_t).float().mean()
        _, top3 = logits.topk(3, dim=-1)
        top3_acc = (top3 == label_t.unsqueeze(-1)).any(dim=-1).float().mean()

    return total, ce_hard.item(), kl_soft.item(), value_loss.item(), acc.item(), top3_acc.item()


@torch.no_grad()
def evaluate(model, eval_data, device, max_batches=64):
    model.eval()
    metrics = {"loss": 0.0, "acc": 0.0, "top3": 0.0, "value": 0.0, "n": 0}
    for i in range(0, min(len(eval_data), max_batches * BATCH_SIZE), BATCH_SIZE):
        batch = eval_data[i : i + BATCH_SIZE]
        if len(batch) < 2:
            continue
        try:
            _, token_ids, label_t, mask_t = build_batch(batch, device)
            soft_t = build_soft_targets(batch, TEACHER_TEMP, SOFT_TOP_K)
            conf_w = kl_confidence_weights(batch)
            value_t = build_value_targets(batch, device)
            loss, ce, kl, vl, acc, t3 = compute_loss(model, token_ids, label_t, mask_t, soft_t, conf_w, value_t, device)
            bs = len(batch)
            metrics["loss"] += loss.item() * bs
            metrics["acc"] += acc * bs
            metrics["top3"] += t3 * bs
            metrics["value"] += vl * bs
            metrics["n"] += bs
        except Exception:
            pass
    model.train()
    n = max(metrics["n"], 1)
    return {k: v / n for k, v in metrics.items() if k != "n"}


def main():
    global STOP_REQUESTED

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    log("=" * 60)
    log("exp110b: Endgame-enriched training with Syzygy labels")
    log("=" * 60)

    # Load model
    ckpt_path = INIT_CHECKPOINT if INIT_CHECKPOINT.exists() else FALLBACK_CHECKPOINT
    log(f"Loading checkpoint: {ckpt_path}")
    model = ChessTransformer200M()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model = torch.compile(model)
    log(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load data
    log("Loading datasets...")
    all_records = load_dataset(DATASET_DIRS)
    if not all_records:
        log("ERROR: No data loaded!")
        return

    # Weighted oversample (Syzygy gets 2x)
    all_records = weighted_oversample(all_records)
    log(f"After weighting: {len(all_records):,} records")

    train_data, eval_data = split_train_eval(all_records)
    log(f"Train: {len(train_data):,}, Eval: {len(eval_data):,}")

    # Phase distribution
    phase_dist = Counter(r.get("phase", "?") for r in train_data)
    log(f"Train phases: {dict(phase_dist)}")
    source_dist = Counter(r.get("source", "?") for r in train_data)
    log(f"Train sources: {dict(source_dist)}")

    # Compute steps
    steps_per_epoch = len(train_data) // (BATCH_SIZE * ACCUM_STEPS)
    max_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(max_steps * WARMUP_FRAC)
    log(f"Steps: {max_steps} ({steps_per_epoch}/epoch × {EPOCHS} epochs, warmup={warmup_steps})")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    scaler = GradScaler(device="cuda", enabled=AMP_ENABLED)
    ema = EMAModel(model, decay=EMA_DECAY)

    # Training
    model.train()
    step = 0
    best_live_acc = 0.0
    best_ema_acc = 0.0
    t0 = time.time()
    epoch_idx = 0

    log(f"\nStarting training: bs={BATCH_SIZE}×{ACCUM_STEPS}, lr={LR}, value_w={VALUE_WEIGHT}")
    log(f"  hard_ce_w={HARD_CE_WEIGHT}, syzygy_w={SYZYGY_WEIGHT}")

    for epoch in range(EPOCHS):
        if STOP_REQUESTED:
            break
        epoch_idx = epoch
        random.shuffle(train_data)

        accum_count = 0
        loss_accum = 0.0
        ce_accum = 0.0
        kl_accum = 0.0
        val_accum = 0.0

        for i in range(0, len(train_data) - BATCH_SIZE, BATCH_SIZE):
            if STOP_REQUESTED:
                break

            batch = train_data[i : i + BATCH_SIZE]
            try:
                _, token_ids, label_t, mask_t = build_batch(batch, DEVICE)
                soft_t = build_soft_targets(batch, TEACHER_TEMP, SOFT_TOP_K)
                conf_w = kl_confidence_weights(batch)
                value_t = build_value_targets(batch, DEVICE)

                with autocast(device_type="cuda", enabled=AMP_ENABLED):
                    loss, ce, kl, vl, acc, t3 = compute_loss(
                        model, token_ids, label_t, mask_t, soft_t, conf_w, value_t, DEVICE
                    )
                    loss = loss / ACCUM_STEPS

                scaler.scale(loss).backward()
                loss_accum += loss.item() * ACCUM_STEPS
                ce_accum += ce
                kl_accum += kl
                val_accum += vl
                accum_count += 1

            except Exception as e:
                log(f"  batch error: {e}")
                continue

            if accum_count >= ACCUM_STEPS:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                lr = cosine_lr(step, max_steps, warmup_steps, LR, MIN_LR_FRAC)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if step >= EMA_START_STEP:
                    ema.update(model)

                if step % LOG_INTERVAL == 0:
                    elapsed = (time.time() - t0) / 60
                    log(
                        f"ep={epoch} step={step} "
                        f"loss={loss_accum / ACCUM_STEPS:.4f} "
                        f"ce={ce_accum / ACCUM_STEPS:.4f} "
                        f"kl={kl_accum / ACCUM_STEPS:.4f} "
                        f"val={val_accum / ACCUM_STEPS:.4f} "
                        f"gnorm={gnorm:.2f} lr={lr:.2e} "
                        f"elapsed={elapsed:.1f}m"
                    )

                if step % EVAL_INTERVAL == 0:
                    # Eval live
                    ev = evaluate(model, eval_data, DEVICE)
                    log(f"LIVE eval step={step}: acc={ev['acc']:.4f} top3={ev['top3']:.4f} "
                        f"loss={ev['loss']:.4f} value={ev['value']:.4f}")
                    if ev["acc"] > best_live_acc:
                        best_live_acc = ev["acc"]
                        save_model(OUTPUT_DIR / "best_model.pt", model)
                        log(f"  NEW BEST: LIVE acc={best_live_acc:.4f} at step {step}")

                    # Eval EMA
                    if step >= EMA_START_STEP:
                        ema.apply_shadow(model)
                        ev_ema = evaluate(model, eval_data, DEVICE)
                        ema.restore(model)
                        log(f"EMA  eval step={step}: acc={ev_ema['acc']:.4f} top3={ev_ema['top3']:.4f} "
                            f"loss={ev_ema['loss']:.4f} value={ev_ema['value']:.4f}")
                        if ev_ema["acc"] > best_ema_acc:
                            best_ema_acc = ev_ema["acc"]
                            ema.save_weights(OUTPUT_DIR / "best_ema_model.pt", model)
                            log(f"  NEW BEST: EMA acc={best_ema_acc:.4f} at step {step}")

                if step % SAVE_INTERVAL == 0:
                    save_model(OUTPUT_DIR / f"checkpoint_step{step}.pt", model)

                loss_accum = 0.0
                ce_accum = 0.0
                kl_accum = 0.0
                val_accum = 0.0
                accum_count = 0

                if step >= max_steps:
                    break

    # Final eval
    elapsed_total = (time.time() - t0) / 60
    log(f"\n=== TRAINING COMPLETE ===")
    log(f"  Steps: {step}, Epochs: {epoch_idx + 1}")
    log(f"  Time: {elapsed_total:.1f}m")
    log(f"  Best LIVE acc: {best_live_acc:.4f}")
    log(f"  Best EMA acc: {best_ema_acc:.4f}")

    # Save final
    save_model(OUTPUT_DIR / "final_model.pt", model)
    ema.save_weights(OUTPUT_DIR / "final_ema_model.pt", model)

    status = {
        "completed": True,
        "steps": step,
        "epochs": epoch_idx + 1,
        "best_live_acc": best_live_acc,
        "best_ema_acc": best_ema_acc,
        "elapsed_min": round(elapsed_total, 1),
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2))
    log(f"Done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
