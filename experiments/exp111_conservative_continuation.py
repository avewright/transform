"""exp111: Conservative continuation training.

Insight: exp110 regressed from ~1850 to ~1600 ELO because it used 5x higher
value weight (0.50 vs 0.10) and 10x higher LR (3e-6 vs 3e-7) than baseline.

This experiment uses the EXACT same loss formulation as the baseline (exp084):
  total = 0.75 * KL(soft_targets) + 0.25 * CE(best_move) + 0.10 * CE(value)

Key design: add diverse/puzzle/syzygy data as ~40% of training mix while keeping
exp085 as the dominant source. Use baseline LR (3e-7) and only 2 epochs.
"""

import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

sys_path = str(Path(__file__).resolve().parent.parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from play import ChessTransformer200M
from chess_features import batch_boards_to_fused_token_ids as _batch_fused
from move_vocab import UCI_TO_IDX, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent

# ── Data ──────────────────────────────────────────────────────
DATASET_DIRS = [
    # Original exp085 data (opening-heavy, what baseline was trained on)
    ROOT / "outputs/exp085_hf_data/dataset",
    # Diverse middlegame/endgame harvests
    ROOT / "outputs/exp110_diverse_harvest/dataset",
    ROOT / "outputs/exp110_diverse_harvest_v2/dataset",
    ROOT / "outputs/exp110_diverse_harvest_v3/dataset",
    ROOT / "outputs/exp110_diverse_harvest_v4/dataset",
    # Endgame data
    ROOT / "outputs/exp110_syzygy/dataset",
    # Tactical puzzles
    ROOT / "outputs/exp110_puzzle_harvest/dataset",
    # Deep endgame
    ROOT / "outputs/exp110_tablebase/dataset",
]
INIT_CHECKPOINT = ROOT / "outputs/hf_checkpoint/best_model.pt"
OUTPUT_DIR = ROOT / "outputs/exp111_conservative_continuation"

# ── Hyperparams (EXACTLY matching baseline) ───────────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

BATCH_SIZE = 4
ACCUM_STEPS = 16  # effective batch = 64
EPOCHS = 2
LR = 3e-7          # SAME as baseline
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5

# Loss weights (SAME as baseline)
HARD_CE_WEIGHT = 0.25       # 25% hard CE + 75% KL
VALUE_LOSS_WEIGHT = 0.10    # 10% value (NOT 50% like exp110!)
TEACHER_TEMP = 1.0
SOFT_TOP_K = 0              # use all soft targets
KL_CONF_SCALE = 0.0         # no confidence weighting (uniform)

# Eval/logging
EVAL_FRACTION = 0.01
MAX_EVAL_RECORDS = 2048
LOG_INTERVAL = 25
SAVE_INTERVAL = 500
EVAL_INTERVAL = 100

LOG_FILE = None


def log(message: str) -> None:
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a") as f:
            f.write(stamped + "\n")


# ── Data loading ──────────────────────────────────────────────
def load_dataset(dirs: list[Path]) -> list[dict]:
    """Load all JSONL files from given directories."""
    records = []
    for d in dirs:
        if not d.exists():
            log(f"  SKIP (not found): {d}")
            continue
        files = sorted(d.glob("positions_*.jsonl"))
        dir_count = 0
        for f in files:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        # Require minimum fields
                        if "fen" in item and "best_move" in item and "soft_targets" in item and "value_target" in item:
                            if item["best_move"] in UCI_TO_IDX:
                                records.append(item)
                                dir_count += 1
                    except json.JSONDecodeError:
                        pass
        log(f"  {d.name}: {dir_count:,} records from {len(files)} files")
    return records


def stable_train_eval_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic split based on FEN hash."""
    eval_target = min(MAX_EVAL_RECORDS, max(512, int(len(records) * EVAL_FRACTION)))
    keyed = []
    for item in records:
        digest = hashlib.blake2b(item["fen"].encode(), digest_size=8).digest()
        keyed.append((int.from_bytes(digest, "big"), item))
    keyed.sort(key=lambda p: p[0])
    eval_records = [item for _, item in keyed[:eval_target]]
    train_records = [item for _, item in keyed[eval_target:]]
    return train_records, eval_records


# ── Board encoding ────────────────────────────────────────────
def batch_boards_to_fused_token_ids(boards: list[chess.Board], device: torch.device) -> torch.Tensor:
    """Convert boards to fused token IDs for ChessTransformer200M."""
    return _batch_fused(boards, device)


# ── Soft target helpers ───────────────────────────────────────
def sparse_soft_targets_to_dense(batch: list[dict]) -> torch.Tensor:
    """Convert sparse soft targets to dense probability vectors."""
    dense = torch.zeros(len(batch), VOCAB_SIZE, dtype=torch.float32)
    for i, item in enumerate(batch):
        targets = item["soft_targets"]
        if SOFT_TOP_K > 0:
            targets = targets[:SOFT_TOP_K]
        probs = [max(float(t["prob"]), 1e-12) for t in targets]
        if abs(TEACHER_TEMP - 1.0) > 1e-9:
            probs = [p ** (1.0 / TEACHER_TEMP) for p in probs]
        total = sum(probs)
        if total > 0:
            for t, p in zip(targets, probs):
                if t["uci"] in UCI_TO_IDX:
                    dense[i, UCI_TO_IDX[t["uci"]]] = p / total
    return dense


# ── Loss ──────────────────────────────────────────────────────
def compute_loss(logits, value_logits, best_moves, value_targets, soft_targets):
    """Exact same loss as baseline (exp084)."""
    hard_ce = F.cross_entropy(logits, best_moves)
    log_probs = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(log_probs, soft_targets, reduction="batchmean")
    value_loss = F.cross_entropy(value_logits, value_targets)

    total = (1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_LOSS_WEIGHT * value_loss
    return total, hard_ce, kl, value_loss


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, eval_records):
    model.eval()
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    total = correct = top3_count = 0

    for idx in range(0, len(eval_records), BATCH_SIZE):
        batch = eval_records[idx:idx + BATCH_SIZE]
        boards = [chess.Board(item["fen"]) for item in batch]
        best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
        value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
        soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)

        out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
        logits = out["policy_logits"].float()
        value_logits = out["value_logits"].float()

        loss, hard_ce, kl, value_loss = compute_loss(logits, value_logits, best_moves, value_targets, soft_targets)

        n = len(batch)
        loss_sum += loss.item() * n
        ce_sum += hard_ce.item() * n
        kl_sum += kl.item() * n
        val_sum += value_loss.item() * n

        preds = logits.argmax(dim=-1)
        correct += (preds == best_moves).sum().item()
        top3_preds = logits.topk(3, dim=-1).indices
        top3_count += (top3_preds == best_moves.unsqueeze(1)).any(dim=1).sum().item()
        total += n

    model.train()
    n = max(total, 1)
    return {
        "loss": loss_sum / n,
        "ce": ce_sum / n,
        "kl": kl_sum / n,
        "value_loss": val_sum / n,
        "acc": correct / n,
        "top3": top3_count / n,
        "n": total,
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    global LOG_FILE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = OUTPUT_DIR / "exp111.log"

    random.seed(SEED)
    torch.manual_seed(SEED)

    log("=" * 60)
    log("exp111: Conservative continuation training")
    log(f"  Loss: {1-HARD_CE_WEIGHT:.0%} KL + {HARD_CE_WEIGHT:.0%} CE + {VALUE_LOSS_WEIGHT:.0%} value")
    log(f"  LR={LR}, batch={BATCH_SIZE}x{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}, epochs={EPOCHS}")
    log("=" * 60)

    # Load data
    log(f"Loading datasets from {len(DATASET_DIRS)} dirs...")
    records = load_dataset(DATASET_DIRS)
    log(f"Total: {len(records):,} records")

    train_records, eval_records = stable_train_eval_split(records)
    log(f"Train: {len(train_records):,}, Eval: {len(eval_records):,}")

    # Load model
    log(f"Loading model from {INIT_CHECKPOINT}...")
    model = ChessTransformer200M()
    state = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model loaded ({param_count / 1e6:.0f}M params) on {DEVICE}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP_ENABLED)

    # Training
    steps_per_epoch = math.ceil(len(train_records) / (BATCH_SIZE * ACCUM_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    log(f"Steps per epoch: {steps_per_epoch}, Total: {total_steps}")

    # Initial eval
    init_eval = evaluate_model(model, eval_records)
    log(f"INIT eval: acc={init_eval['acc']:.4f} top3={init_eval['top3']:.4f} "
        f"loss={init_eval['loss']:.4f} ce={init_eval['ce']:.4f} value={init_eval['value_loss']:.4f}")

    best_acc = init_eval["acc"]
    best_step = 0
    global_step = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        log(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        random.shuffle(train_records)
        cursor = 0

        for step_idx in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            step_loss = step_ce = step_kl = step_val = 0.0

            for _ in range(ACCUM_STEPS):
                batch = train_records[cursor:cursor + BATCH_SIZE]
                cursor += BATCH_SIZE
                if len(batch) < BATCH_SIZE:
                    needed = BATCH_SIZE - len(batch)
                    batch = batch + train_records[:needed]
                    cursor = needed

                boards = [chess.Board(item["fen"]) for item in batch]
                best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
                value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
                soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)

                with autocast(enabled=AMP_ENABLED):
                    out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
                    logits = out["policy_logits"]
                    value_logits = out["value_logits"]
                    loss, hard_ce, kl, value_loss = compute_loss(logits, value_logits, best_moves, value_targets, soft_targets)
                    scaled_loss = loss / ACCUM_STEPS

                scaler.scale(scaled_loss).backward()
                step_loss += loss.item()
                step_ce += hard_ce.item()
                step_kl += kl.item()
                step_val += value_loss.item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            elapsed = (time.time() - start_time) / 60.0

            if global_step % LOG_INTERVAL == 0:
                log(f"ep={epoch+1} step={global_step} "
                    f"loss={step_loss/ACCUM_STEPS:.4f} "
                    f"ce={step_ce/ACCUM_STEPS:.4f} "
                    f"kl={step_kl/ACCUM_STEPS:.4f} "
                    f"val={step_val/ACCUM_STEPS:.4f} "
                    f"gnorm={grad_norm:.2f} lr={LR:.2e} "
                    f"elapsed={elapsed:.1f}m")

            if global_step % EVAL_INTERVAL == 0:
                ev = evaluate_model(model, eval_records)
                log(f"EVAL step={global_step}: acc={ev['acc']:.4f} top3={ev['top3']:.4f} "
                    f"loss={ev['loss']:.4f} value={ev['value_loss']:.4f}")
                if ev["acc"] > best_acc:
                    best_acc = ev["acc"]
                    best_step = global_step
                    log(f"  NEW BEST: acc={best_acc:.4f} at step {best_step}")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step": global_step,
                        "acc": best_acc,
                    }, OUTPUT_DIR / "best_model.pt")

            if global_step % SAVE_INTERVAL == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": global_step,
                }, OUTPUT_DIR / "latest_model.pt")

    # Final eval
    final_eval = evaluate_model(model, eval_records)
    elapsed = (time.time() - start_time) / 60.0
    log(f"\n{'='*60}")
    log(f"Training complete: {global_step} steps in {elapsed:.1f} min")
    log(f"Best acc: {best_acc:.4f} at step {best_step}")
    log(f"Final: acc={final_eval['acc']:.4f} top3={final_eval['top3']:.4f} value={final_eval['value_loss']:.4f}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": global_step,
    }, OUTPUT_DIR / "final_model.pt")

    # Config
    config = {
        "experiment": "exp111_conservative_continuation",
        "hypothesis": "Same loss as baseline + diverse data = no regression",
        "init_checkpoint": str(INIT_CHECKPOINT),
        "dataset_dirs": [str(d) for d in DATASET_DIRS],
        "total_records": len(records),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "hard_ce_weight": HARD_CE_WEIGHT,
        "value_loss_weight": VALUE_LOSS_WEIGHT,
        "teacher_temp": TEACHER_TEMP,
        "best_acc": best_acc,
        "best_step": best_step,
        "final_acc": final_eval["acc"],
        "total_steps": global_step,
        "training_minutes": elapsed,
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log("Done.")


if __name__ == "__main__":
    main()
