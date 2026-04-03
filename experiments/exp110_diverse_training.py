"""exp110: Diverse-phase multi-PV training + strong value head.

Hypothesis: Training on 200K diverse positions (20% opening, 45% middlegame,
35% endgame) with multi-PV soft targets at depth 12 and 50% value weight
will push ELO past 1900 by:
  1. Closing middlegame/endgame gaps (model was mostly trained on openings)
  2. Strengthening the value head enough to enable search
  3. KL divergence on soft targets sharpens the policy beyond hard-label ceiling

Architecture: ChessTransformer 200M (unchanged)
Init: avewright/chess-transformer-200m-latest (best checkpoint, ~1850 ELO)
Data: outputs/exp110_diverse_harvest/dataset/*.jsonl
Loss: (1-α)·KL(soft) + α·CE(hard) + β·CE(value)  where α=0.25, β=0.50
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
OUTPUT_DIR = Path("outputs/exp110_diverse_training")
DATASET_DIRS = [
    Path("outputs/exp110_diverse_harvest/dataset"),  # diverse middlegame/endgame
    Path("outputs/exp085_hf_data/dataset"),           # exp085 opening-heavy multi-PV
]
INIT_CHECKPOINT = Path("outputs/hf_checkpoint/best_model.pt")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

BATCH_SIZE = 8
ACCUM_STEPS = 8      # effective batch = 64
EPOCHS = 3
LR = 3e-6             # conservative for fine-tuning
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5

# Loss weights
HARD_CE_WEIGHT = 0.25   # 25% hard CE + 75% soft KL
VALUE_WEIGHT = 0.50      # 50% value — 5x stronger than previous (10%)
TEACHER_TEMP = 0.5
SOFT_TOP_K = 8

# KL confidence scaling
KL_CONF_SCALE = 80.0
KL_CONF_MIN = 0.10
KL_CONF_MAX = 1.00

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
    log_path = OUTPUT_DIR / "exp110.log"
    with open(log_path, "a") as f:
        f.write(stamped + "\n")


# ── EMA ──
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


# ── Utilities ──
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
    for ddir in dataset_dirs:
        if not ddir.exists():
            continue
        files = sorted(ddir.glob("positions_*.jsonl"))
        for f in files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if rec.get("soft_targets") and rec.get("best_move"):
                                records.append(rec)
                        except json.JSONDecodeError:
                            pass
    return records


def phase_balanced_oversample(records: list[dict], target_ratio: dict[str, float] = None) -> list[dict]:
    """Oversample underrepresented phases to achieve target ratio."""
    if target_ratio is None:
        target_ratio = {"opening": 0.40, "middlegame": 0.35, "endgame": 0.25}

    from collections import Counter
    phase_buckets = {}
    for r in records:
        phase = r.get("phase", "opening")
        phase_buckets.setdefault(phase, []).append(r)

    # Find the target count for each phase
    total = len(records)
    target_total = int(total * 1.2)  # allow some expansion
    phase_targets = {p: max(1, int(target_total * f)) for p, f in target_ratio.items()}

    result = []
    for phase, target in phase_targets.items():
        bucket = phase_buckets.get(phase, [])
        if not bucket:
            continue
        if len(bucket) >= target:
            # Subsample
            rng = random.Random(42)
            result.extend(rng.sample(bucket, target))
        else:
            # Oversample: include all originals + repeats
            repeats = target // len(bucket)
            remainder = target % len(bucket)
            result.extend(bucket * repeats)
            rng = random.Random(42)
            result.extend(rng.sample(bucket, remainder))

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
        gap = max(float(item.get("cp_gap_top1_top2", 0.0)), 0.0)
        w = max(KL_CONF_MIN, min(KL_CONF_MAX, gap / KL_CONF_SCALE))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def compute_losses(logits, best_moves, soft_targets, kl_weights, value_logits, value_targets):
    hard_ce = F.cross_entropy(logits, best_moves)
    log_probs = F.log_softmax(logits, dim=-1)
    kl_per = F.kl_div(log_probs, soft_targets, reduction="none").sum(dim=-1)
    kl = (kl_per * kl_weights).mean()
    value_loss = F.cross_entropy(value_logits, value_targets)
    total = (1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_WEIGHT * value_loss
    return total, hard_ce, kl, value_loss


# ── Evaluation ──
@torch.no_grad()
def evaluate(model, eval_records):
    model.eval()
    total = correct = top3_correct = 0
    loss_sum = ce_sum = kl_sum = val_sum = 0.0

    for idx in range(0, len(eval_records), BATCH_SIZE):
        batch = eval_records[idx:idx + BATCH_SIZE]
        boards = [chess.Board(r["fen"]) for r in batch]
        best_moves = torch.tensor([UCI_TO_IDX[r["best_move"]] for r in batch], dtype=torch.long, device=DEVICE)
        value_targets = torch.tensor([r["value_target"] for r in batch], dtype=torch.long, device=DEVICE)
        soft = build_soft_targets(batch, TEACHER_TEMP, SOFT_TOP_K).to(DEVICE)
        kl_w = kl_confidence_weights(batch).to(DEVICE)

        out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
        logits = out["policy_logits"].float()
        vlogs = out["value_logits"].float()

        loss, ce, kl, vl = compute_losses(logits, best_moves, soft, kl_w, vlogs, value_targets)
        n = len(batch)
        loss_sum += loss.item() * n
        ce_sum += ce.item() * n
        kl_sum += kl.item() * n
        val_sum += vl.item() * n

        for i, board in enumerate(boards):
            masked = logits[i].clone()
            mask = legal_move_mask(board).to(DEVICE)
            masked[~mask] = float("-inf")
            pred = masked.argmax().item()
            true = best_moves[i].item()
            if pred == true:
                correct += 1
            topk = masked.topk(min(3, int(mask.sum().item()))).indices.tolist()
            if true in topk:
                top3_correct += 1
            total += 1

    n = max(total, 1)
    # Also compute value accuracy
    return {
        "loss": loss_sum / n,
        "ce": ce_sum / n,
        "kl": kl_sum / n,
        "value_loss": val_sum / n,
        "acc": correct / n,
        "top3": top3_correct / n,
        "n": total,
    }


# ── Main Training Loop ──
def main():
    global STOP_REQUESTED

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    log("exp110: Diverse multi-PV training + strong value head")
    log(f"Device: {DEVICE}")

    # Load data
    log(f"Loading datasets from {len(DATASET_DIRS)} sources...")
    records = load_dataset(DATASET_DIRS)
    log(f"Loaded {len(records)} records total")

    if len(records) < 1000:
        log("ERROR: Not enough data. Wait for harvest to complete.")
        return

    train_records, eval_records = split_train_eval(records)
    log(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

    # Phase distribution before balancing
    from collections import Counter
    phases = Counter(r.get("phase", "unknown") for r in train_records)
    log(f"Phase distribution (raw): {dict(phases)}")

    # Oversample to balance phases
    train_records = phase_balanced_oversample(train_records)
    phases_balanced = Counter(r.get("phase", "unknown") for r in train_records)
    log(f"Phase distribution (balanced): {dict(phases_balanced)}, total={len(train_records)}")

    # Load model
    log(f"Loading model from {INIT_CHECKPOINT}...")
    model = ChessTransformer200M()
    state = torch.load(str(INIT_CHECKPOINT), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(DEVICE)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Model loaded ({params:.0f}M params)")

    # Compile model for speed
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log("Model compiled with torch.compile")
        except Exception as e:
            log(f"torch.compile failed: {e}")

    # EMA
    ema = EMAModel(model, EMA_DECAY)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)

    # Schedule
    steps_per_epoch = math.ceil(len(train_records) / (BATCH_SIZE * ACCUM_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    log(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    # Initial eval
    init_eval = evaluate(model, eval_records)
    log(f"Init eval: acc={init_eval['acc']:.4f} top3={init_eval['top3']:.4f} "
        f"loss={init_eval['loss']:.4f} ce={init_eval['ce']:.4f} value={init_eval['value_loss']:.4f}")

    best_acc = init_eval["acc"]
    best_step = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        if STOP_REQUESTED:
            break

        random.shuffle(train_records)
        cursor = 0
        model.train()

        loss_sum = ce_sum = kl_sum = val_sum = 0.0
        steps_this_epoch = 0

        for step_in_epoch in range(steps_per_epoch):
            if STOP_REQUESTED:
                break

            global_step += 1
            lr = cosine_lr(global_step, total_steps, warmup_steps, LR, MIN_LR_FRAC)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            step_loss = step_ce = step_kl = step_val = 0.0

            for _ in range(ACCUM_STEPS):
                batch = train_records[cursor:cursor + BATCH_SIZE]
                cursor += BATCH_SIZE
                if len(batch) < BATCH_SIZE:
                    needed = BATCH_SIZE - len(batch)
                    batch = batch + train_records[:needed]
                    cursor = needed

                boards = [chess.Board(r["fen"]) for r in batch]
                best_moves = torch.tensor([UCI_TO_IDX[r["best_move"]] for r in batch], dtype=torch.long, device=DEVICE)
                value_targets = torch.tensor([r["value_target"] for r in batch], dtype=torch.long, device=DEVICE)
                soft = build_soft_targets(batch, TEACHER_TEMP, SOFT_TOP_K).to(DEVICE)
                kl_w = kl_confidence_weights(batch).to(DEVICE)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
                    out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
                    logits = out["policy_logits"]
                    vlogs = out["value_logits"]
                    loss, ce, kl, vl = compute_losses(logits, best_moves, soft, kl_w, vlogs, value_targets)
                    loss_scaled = loss / ACCUM_STEPS

                scaler.scale(loss_scaled).backward()
                step_loss += loss.item()
                step_ce += ce.item()
                step_kl += kl.item()
                step_val += vl.item()

            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            if global_step >= EMA_START_STEP:
                ema.update(model)

            loss_sum += step_loss / ACCUM_STEPS
            ce_sum += step_ce / ACCUM_STEPS
            kl_sum += step_kl / ACCUM_STEPS
            val_sum += step_val / ACCUM_STEPS
            steps_this_epoch += 1

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                n_done = steps_this_epoch
                log(f"ep={epoch} step={global_step} loss={loss_sum/n_done:.4f} "
                    f"ce={ce_sum/n_done:.4f} kl={kl_sum/n_done:.4f} val={val_sum/n_done:.4f} "
                    f"gnorm={float(gnorm):.2f} lr={lr:.2e} elapsed={elapsed/60:.1f}m")

            if global_step % EVAL_INTERVAL == 0:
                # Evaluate live model
                live_eval = evaluate(model, eval_records)
                log(f"LIVE eval step={global_step}: acc={live_eval['acc']:.4f} top3={live_eval['top3']:.4f} "
                    f"loss={live_eval['loss']:.4f} value={live_eval['value_loss']:.4f}")

                # Evaluate EMA model
                ema.apply_shadow(model)
                ema_eval = evaluate(model, eval_records)
                ema.restore(model)
                log(f"EMA  eval step={global_step}: acc={ema_eval['acc']:.4f} top3={ema_eval['top3']:.4f} "
                    f"loss={ema_eval['loss']:.4f} value={ema_eval['value_loss']:.4f}")

                # Save best
                better_eval = ema_eval if ema_eval["acc"] >= live_eval["acc"] else live_eval
                is_ema_best = ema_eval["acc"] >= live_eval["acc"]

                if better_eval["acc"] > best_acc:
                    best_acc = better_eval["acc"]
                    best_step = global_step
                    tag = "EMA" if is_ema_best else "LIVE"
                    log(f"  NEW BEST: {tag} acc={best_acc:.4f} at step {global_step}")
                    if is_ema_best:
                        ema.save_weights(OUTPUT_DIR / "best_model.pt", model)
                    else:
                        save_model(OUTPUT_DIR / "best_model.pt", model)

                model.train()

            if global_step % SAVE_INTERVAL == 0:
                save_model(OUTPUT_DIR / "latest_model.pt", model)
                ema.save_weights(OUTPUT_DIR / "ema_model.pt", model)

    # Final save
    save_model(OUTPUT_DIR / "latest_model.pt", model)
    ema.save_weights(OUTPUT_DIR / "ema_model.pt", model)

    # Final evaluation
    final_live = evaluate(model, eval_records)
    ema.apply_shadow(model)
    final_ema = evaluate(model, eval_records)
    ema.restore(model)

    elapsed = time.time() - t0
    log(f"\n=== TRAINING COMPLETE ===")
    log(f"  Epochs: {EPOCHS}, Steps: {global_step}, Time: {elapsed/60:.1f}m")
    log(f"  Best acc: {best_acc:.4f} at step {best_step}")
    log(f"  Final LIVE: acc={final_live['acc']:.4f} top3={final_live['top3']:.4f} value={final_live['value_loss']:.4f}")
    log(f"  Final EMA:  acc={final_ema['acc']:.4f} top3={final_ema['top3']:.4f} value={final_ema['value_loss']:.4f}")

    # Save final result
    result = {
        "experiment": "exp110",
        "hypothesis": "Diverse-phase multi-PV training + strong value head",
        "init_checkpoint": str(INIT_CHECKPOINT),
        "dataset": str(DATASET_DIR),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "epochs": EPOCHS,
        "total_steps": global_step,
        "best_acc": best_acc,
        "best_step": best_step,
        "init_acc": init_eval["acc"],
        "final_live_acc": final_live["acc"],
        "final_ema_acc": final_ema["acc"],
        "final_live_eval": final_live,
        "final_ema_eval": final_ema,
        "elapsed_sec": round(elapsed),
        "config": {
            "lr": LR, "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
            "hard_ce_weight": HARD_CE_WEIGHT, "value_weight": VALUE_WEIGHT,
            "teacher_temp": TEACHER_TEMP, "ema_decay": EMA_DECAY,
            "soft_top_k": SOFT_TOP_K,
        },
    }
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    log(f"Results saved to {OUTPUT_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
