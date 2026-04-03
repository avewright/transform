"""exp116: Fine-tune with CORRECT White-absolute value_target mapping.

The model was pre-trained on 832M positions with White-absolute WDL convention:
  idx0 = P(White wins), idx1 = P(draw), idx2 = P(White loses)

But exp085 data uses side-to-move (STM) convention:
  value_target: 0=STM loses, 1=draw, 2=STM wins

For White-to-move positions (~50%), these conventions CONFLICT:
  STM target=2 (White wins) should be model idx=0 → training pushed idx2=W_wins (WRONG)
  STM target=0 (White loses) should be model idx=2 → training pushed idx0=W_loses (WRONG)

Fix: For White-to-move, remap value_target: 0→2, 2→0, 1→1
For Black-to-move, keep as-is (STM=Black, target=2=B_wins=W_loses=model_idx2 ✓)

Based on exp084 training pipeline (proven to work). Key changes:
1. value_target remapping for White-to-move positions
2. No HF upload (local only)
3. Save weights-only checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
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

OUTPUT_DIR = Path("outputs/exp116_correct_value_finetune")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "exp116.log"
CONFIG_PATH = OUTPUT_DIR / "config.json"
STATUS_PATH = OUTPUT_DIR / "status.json"
BEST_PATH = OUTPUT_DIR / "best_model.pt"
LATEST_PATH = OUTPUT_DIR / "latest_model.pt"

INIT_CHECKPOINT = Path("outputs/hf_checkpoint/best_model.pt")
DEFAULT_DATASET_DIR = Path("outputs/exp085_hf_data/dataset")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

BATCH_SIZE = 4
ACCUM_STEPS = 16
EPOCHS = 1
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
HARD_CE_WEIGHT = 0.25
VALUE_LOSS_WEIGHT = 0.10
TEACHER_TEMP = 1.0
EVAL_FRACTION = 0.05
MAX_EVAL_RECORDS = 2048
LOG_INTERVAL = 25
SAVE_INTERVAL = 200


def log(message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if LOG_PATH.exists() or OUTPUT_DIR.exists():
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(stamped + "\n")


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
    sd = {k.replace("_orig_mod.", ""): v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": sd}, tmp)
    os.replace(tmp, path)


def remap_value_target_white_absolute(value_target: int, fen: str) -> int:
    """Convert STM value_target to White-absolute convention.

    STM convention: 0=STM loses, 1=draw, 2=STM wins
    White-absolute: 0=White wins, 1=draw, 2=White loses

    For White-to-move: STM=White, so swap 0↔2
    For Black-to-move: STM=Black, target already matches (2=B wins=W loses=idx2)
    """
    turn = fen.split()[1]
    if turn == "w":
        # White to move: STM wins (2) → White wins (0), STM loses (0) → White loses (2)
        if value_target == 0:
            return 2
        elif value_target == 2:
            return 0
    return value_target


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
                        log(f"skipping malformed jsonl row path={path} line={line_no}")
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


def sparse_soft_targets_to_dense(batch: list[dict]) -> torch.Tensor:
    dense = torch.zeros(len(batch), VOCAB_SIZE, dtype=torch.float32)
    for row_idx, item in enumerate(batch):
        targets = item["soft_targets"]
        probs = [max(float(t["prob"]), 1e-12) for t in targets]
        total = sum(probs)
        for t, p in zip(targets, probs):
            dense[row_idx, UCI_TO_IDX[t["uci"]]] = p / total
    return dense


def compute_policy_losses(
    logits: torch.Tensor,
    best_moves: torch.Tensor,
    soft_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hard_ce = F.cross_entropy(logits, best_moves)
    log_probs = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(log_probs, soft_targets, reduction="batchmean")
    return hard_ce, kl


def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer200M:
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device)


def evaluate(model: ChessTransformer200M, eval_records: list[dict]) -> dict:
    model.eval()
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    total = correct = top3 = 0
    white_val_correct = white_val_total = 0
    black_val_correct = black_val_total = 0

    with torch.no_grad():
        for idx in range(0, len(eval_records), BATCH_SIZE):
            batch = eval_records[idx : idx + BATCH_SIZE]
            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor(
                [UCI_TO_IDX[item["best_move"]] for item in batch],
                dtype=torch.long, device=DEVICE,
            )
            # Remap value targets to White-absolute
            value_targets = torch.tensor(
                [remap_value_target_white_absolute(item["value_target"], item["fen"]) for item in batch],
                dtype=torch.long, device=DEVICE,
            )
            soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)

            out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
            logits = out["policy_logits"].float()
            value_logits = out["value_logits"].float()

            hard_ce, kl = compute_policy_losses(logits, best_moves, soft_targets)
            value_loss = F.cross_entropy(value_logits, value_targets)
            total_loss = (1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_LOSS_WEIGHT * value_loss

            loss_sum += total_loss.item() * len(batch)
            ce_sum += hard_ce.item() * len(batch)
            kl_sum += kl.item() * len(batch)
            val_sum += value_loss.item() * len(batch)

            # Per-sample accuracy
            value_preds = value_logits.argmax(dim=-1)
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

                # Track value accuracy by side
                vt = value_targets[row_idx].item()
                vp = value_preds[row_idx].item()
                if board.turn == chess.WHITE:
                    white_val_total += 1
                    if vp == vt:
                        white_val_correct += 1
                else:
                    black_val_total += 1
                    if vp == vt:
                        black_val_correct += 1

    n = max(total, 1)
    return {
        "loss": loss_sum / n,
        "ce": ce_sum / n,
        "kl": kl_sum / n,
        "value": val_sum / n,
        "acc": correct / n,
        "top3": top3 / n,
        "val_acc_white": white_val_correct / max(white_val_total, 1),
        "val_acc_black": black_val_correct / max(black_val_total, 1),
        "n": total,
    }


def train_epoch(
    model: ChessTransformer200M,
    optimizer: AdamW,
    scaler: GradScaler,
    train_records: list[dict],
    eval_records: list[dict],
    state: dict,
) -> dict:
    model.train()
    random.shuffle(train_records)
    steps_per_epoch = math.ceil(len(train_records) / (BATCH_SIZE * ACCUM_STEPS))
    cursor = 0
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    best_eval = state.get("best_eval_loss", float("inf"))

    for step_idx in range(steps_per_epoch):
        optimizer.zero_grad(set_to_none=True)
        step_loss = step_ce = step_kl = step_val = 0.0

        for _ in range(ACCUM_STEPS):
            batch = train_records[cursor : cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            if len(batch) < BATCH_SIZE:
                needed = BATCH_SIZE - len(batch)
                batch = batch + train_records[:needed]
                cursor = needed

            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor(
                [UCI_TO_IDX[item["best_move"]] for item in batch],
                dtype=torch.long, device=DEVICE,
            )
            # Remap value targets to White-absolute
            value_targets = torch.tensor(
                [remap_value_target_white_absolute(item["value_target"], item["fen"]) for item in batch],
                dtype=torch.long, device=DEVICE,
            )
            soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
                out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
                logits = out["policy_logits"]
                value_logits = out["value_logits"]
                hard_ce, kl = compute_policy_losses(logits, best_moves, soft_targets)
                value_loss = F.cross_entropy(value_logits, value_targets)
                total_loss = (
                    (1.0 - HARD_CE_WEIGHT) * kl
                    + HARD_CE_WEIGHT * hard_ce
                    + VALUE_LOSS_WEIGHT * value_loss
                ) / ACCUM_STEPS

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

        state["train_steps"] += 1
        loss_sum += step_loss
        ce_sum += step_ce / ACCUM_STEPS
        kl_sum += step_kl / ACCUM_STEPS
        val_sum += step_val / ACCUM_STEPS

        if state["train_steps"] % LOG_INTERVAL == 0:
            done = step_idx + 1
            log(
                f"step={state['train_steps']} epoch={state['epoch']} "
                f"loss={loss_sum / done:.4f} ce={ce_sum / done:.4f} "
                f"kl={kl_sum / done:.4f} value={val_sum / done:.4f} "
                f"gnorm={float(grad_norm):.2f}"
            )

        if state["train_steps"] % SAVE_INTERVAL == 0 or step_idx == steps_per_epoch - 1:
            eval_metrics = evaluate(model, eval_records)
            log(
                f"eval step={state['train_steps']} loss={eval_metrics['loss']:.4f} "
                f"ce={eval_metrics['ce']:.4f} kl={eval_metrics['kl']:.4f} "
                f"value={eval_metrics['value']:.4f} acc={eval_metrics['acc']:.4f} "
                f"top3={eval_metrics['top3']:.4f} "
                f"val_acc_w={eval_metrics['val_acc_white']:.4f} "
                f"val_acc_b={eval_metrics['val_acc_black']:.4f} n={eval_metrics['n']}"
            )
            is_best = eval_metrics["loss"] < best_eval
            if is_best:
                best_eval = eval_metrics["loss"]
                save_model_weights(BEST_PATH, model)
                log(f"new best model saved (loss={best_eval:.4f})")
            save_model_weights(LATEST_PATH, model)
            state["best_eval_loss"] = best_eval
            state["last_eval"] = eval_metrics
            atomic_write_json(
                STATUS_PATH,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "epoch": state["epoch"],
                    "train_steps": state["train_steps"],
                    "best_eval_loss": best_eval,
                    "last_eval": eval_metrics,
                },
            )

    state["best_eval_loss"] = best_eval
    return state


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_paths = sorted(DEFAULT_DATASET_DIR.glob("positions_*.jsonl"))
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset files in {DEFAULT_DATASET_DIR}")
    records = load_jsonl_dataset(dataset_paths)
    train_records, eval_records = stable_train_eval_split(records)

    # Log value_target remapping stats
    white_remap = sum(1 for r in records if r["fen"].split()[1] == "w" and r["value_target"] != 1)
    total_white = sum(1 for r in records if r["fen"].split()[1] == "w")
    log(f"value_target remapping: {white_remap}/{total_white} White-to-move non-draw records remapped")

    # Load model
    if not INIT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {INIT_CHECKPOINT}")
    model = load_model(INIT_CHECKPOINT, DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=AMP_ENABLED)

    state = {
        "epoch": 1,
        "train_steps": 0,
        "best_eval_loss": float("inf"),
        "last_eval": None,
    }

    # Config
    atomic_write_json(
        CONFIG_PATH,
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "init_checkpoint": str(INIT_CHECKPOINT),
            "dataset_files": len(dataset_paths),
            "total_records": len(records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "white_to_move_remapped": white_remap,
            "batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * ACCUM_STEPS,
            "epochs": EPOCHS,
            "lr": LR,
            "hard_ce_weight": HARD_CE_WEIGHT,
            "value_loss_weight": VALUE_LOSS_WEIGHT,
            "grad_clip": GRAD_CLIP,
            "weight_decay": WEIGHT_DECAY,
            "fix": "value_target remapped to White-absolute for White-to-move positions",
        },
    )

    log("=" * 72)
    log("exp116: Fine-tune with CORRECT White-absolute value_target mapping")
    log("=" * 72)
    log(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    log(f"records={len(records)} train={len(train_records)} eval={len(eval_records)}")
    log(f"effective_batch={BATCH_SIZE * ACCUM_STEPS} epochs={EPOCHS} lr={LR}")
    log(f"hard_ce_weight={HARD_CE_WEIGHT} value_loss_weight={VALUE_LOSS_WEIGHT}")
    log(f"init_checkpoint={INIT_CHECKPOINT}")

    # Initial eval with correct value targets
    initial_eval = evaluate(model, eval_records)
    log(
        f"initial_eval loss={initial_eval['loss']:.4f} ce={initial_eval['ce']:.4f} "
        f"kl={initial_eval['kl']:.4f} value={initial_eval['value']:.4f} "
        f"acc={initial_eval['acc']:.4f} top3={initial_eval['top3']:.4f} "
        f"val_acc_w={initial_eval['val_acc_white']:.4f} "
        f"val_acc_b={initial_eval['val_acc_black']:.4f}"
    )
    state["best_eval_loss"] = initial_eval["loss"]
    state["last_eval"] = initial_eval

    while state["epoch"] <= EPOCHS:
        log(f"=== epoch {state['epoch']}/{EPOCHS} ===")
        train_epoch(model, optimizer, scaler, train_records, eval_records, state)
        state["epoch"] += 1

    # Final eval
    final_eval = evaluate(model, eval_records)
    log(
        f"final_eval loss={final_eval['loss']:.4f} ce={final_eval['ce']:.4f} "
        f"kl={final_eval['kl']:.4f} value={final_eval['value']:.4f} "
        f"acc={final_eval['acc']:.4f} top3={final_eval['top3']:.4f} "
        f"val_acc_w={final_eval['val_acc_white']:.4f} "
        f"val_acc_b={final_eval['val_acc_black']:.4f}"
    )
    save_model_weights(LATEST_PATH, model)
    log("exp116 training complete")
    log(f"best model: {BEST_PATH}")
    log(f"latest model: {LATEST_PATH}")


if __name__ == "__main__":
    main()
