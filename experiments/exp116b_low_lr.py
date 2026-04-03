"""exp116b: Fine-tune with correct value_target at ULTRA-LOW LR.

exp116 at LR=5e-6 showed fast KL divergence growth — policy degraded after ~200 steps.
This version uses LR=5e-7 (10x lower) to preserve policy while still improving value.
Also only trains for 500 steps (not full epoch) as early stopping measure.

Additionally, uses a SEPARATE loss tracking approach:
  - Best model saved by POLICY CE loss (not total loss including KL)
  - This ensures we keep the best policy while still training the value head
"""

from __future__ import annotations

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

OUTPUT_DIR = Path("outputs/exp116b_low_lr")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "exp116b.log"
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
MAX_STEPS = 5000
RESUME_FROM = Path("outputs/exp116b_low_lr/best_model.pt")
LR = 5e-7
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
HARD_CE_WEIGHT = 0.25
VALUE_LOSS_WEIGHT = 0.10
EVAL_FRACTION = 0.05
MAX_EVAL_RECORDS = 2048
LOG_INTERVAL = 25
SAVE_INTERVAL = 100


def log(message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if OUTPUT_DIR.exists():
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


def remap_value_target(value_target: int, fen: str) -> int:
    """Convert STM value_target to White-absolute convention."""
    if fen.split()[1] == "w":
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
                        pass
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
    return [item for _, item in keyed[eval_target:]], [item for _, item in keyed[:eval_target]]


def sparse_soft_targets_to_dense(batch: list[dict]) -> torch.Tensor:
    dense = torch.zeros(len(batch), VOCAB_SIZE, dtype=torch.float32)
    for row_idx, item in enumerate(batch):
        targets = item["soft_targets"]
        probs = [max(float(t["prob"]), 1e-12) for t in targets]
        total = sum(probs)
        for t, p in zip(targets, probs):
            dense[row_idx, UCI_TO_IDX[t["uci"]]] = p / total
    return dense


def load_model(checkpoint_path: Path) -> ChessTransformer200M:
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE)


def evaluate(model: ChessTransformer200M, eval_records: list[dict]) -> dict:
    model.eval()
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    total = correct = top3 = 0
    w_val_ok = w_val_n = b_val_ok = b_val_n = 0

    with torch.no_grad():
        for idx in range(0, len(eval_records), BATCH_SIZE):
            batch = eval_records[idx:idx + BATCH_SIZE]
            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
            value_targets = torch.tensor([remap_value_target(item["value_target"], item["fen"]) for item in batch], dtype=torch.long, device=DEVICE)
            soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)

            out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
            logits = out["policy_logits"].float()
            value_logits = out["value_logits"].float()

            hard_ce = F.cross_entropy(logits, best_moves)
            log_probs = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(log_probs, soft_targets, reduction="batchmean")
            value_loss = F.cross_entropy(value_logits, value_targets)
            total_loss = (1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_LOSS_WEIGHT * value_loss

            loss_sum += total_loss.item() * len(batch)
            ce_sum += hard_ce.item() * len(batch)
            kl_sum += kl.item() * len(batch)
            val_sum += value_loss.item() * len(batch)

            vp = value_logits.argmax(dim=-1)
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
                if board.turn == chess.WHITE:
                    w_val_n += 1
                    if vp[row_idx].item() == value_targets[row_idx].item():
                        w_val_ok += 1
                else:
                    b_val_n += 1
                    if vp[row_idx].item() == value_targets[row_idx].item():
                        b_val_ok += 1

    n = max(total, 1)
    return {
        "loss": loss_sum / n, "ce": ce_sum / n, "kl": kl_sum / n, "value": val_sum / n,
        "acc": correct / n, "top3": top3 / n,
        "val_w": w_val_ok / max(w_val_n, 1), "val_b": b_val_ok / max(b_val_n, 1),
        "n": total,
    }


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(DEFAULT_DATASET_DIR.glob("positions_*.jsonl"))
    if not dataset_paths:
        raise FileNotFoundError(f"No data in {DEFAULT_DATASET_DIR}")
    records = load_jsonl_dataset(dataset_paths)
    train_records, eval_records = stable_train_eval_split(records)

    white_remap = sum(1 for r in records if r["fen"].split()[1] == "w" and r["value_target"] != 1)
    log(f"remapped {white_remap} White-to-move non-draw value_targets")

    # Resume from previous run if available
    init_ckpt = RESUME_FROM if RESUME_FROM.exists() else INIT_CHECKPOINT
    model = load_model(init_ckpt)
    log(f"loaded from {init_ckpt}")
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=AMP_ENABLED)

    atomic_write_json(CONFIG_PATH, {
        "init_checkpoint": str(init_ckpt),
        "total_records": len(records), "train": len(train_records), "eval": len(eval_records),
        "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS, "lr": LR,
        "max_steps": MAX_STEPS, "hard_ce_weight": HARD_CE_WEIGHT, "value_loss_weight": VALUE_LOSS_WEIGHT,
        "fix": "value_target White-absolute remap + ultra-low LR",
    })

    log("=" * 60)
    log("exp116b: Correct value targets, ultra-low LR=5e-7")
    log("=" * 60)
    log(f"records={len(records)} train={len(train_records)} eval={len(eval_records)}")
    log(f"eff_batch={BATCH_SIZE * ACCUM_STEPS} max_steps={MAX_STEPS} lr={LR}")

    initial = evaluate(model, eval_records)
    log(f"initial: loss={initial['loss']:.4f} ce={initial['ce']:.4f} kl={initial['kl']:.4f} "
        f"val={initial['value']:.4f} acc={initial['acc']:.4f} top3={initial['top3']:.4f} "
        f"val_w={initial['val_w']:.4f} val_b={initial['val_b']:.4f}")

    best_ce = initial["ce"]
    best_val_loss = initial["value"]
    save_model_weights(BEST_PATH, model)
    log(f"saved initial as best (ce={best_ce:.4f})")

    model.train()
    random.shuffle(train_records)
    cursor = 0
    loss_accum = ce_accum = kl_accum = val_accum = 0.0

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(ACCUM_STEPS):
            batch = train_records[cursor:cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            if len(batch) < BATCH_SIZE:
                needed = BATCH_SIZE - len(batch)
                batch = batch + train_records[:needed]
                cursor = needed

            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
            value_targets = torch.tensor([remap_value_target(item["value_target"], item["fen"]) for item in batch], dtype=torch.long, device=DEVICE)
            soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
                out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
                logits = out["policy_logits"]
                value_logits = out["value_logits"]
                hard_ce = F.cross_entropy(logits, best_moves)
                log_probs = F.log_softmax(logits, dim=-1)
                kl = F.kl_div(log_probs, soft_targets, reduction="batchmean")
                value_loss = F.cross_entropy(value_logits, value_targets)
                total_loss = ((1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_LOSS_WEIGHT * value_loss) / ACCUM_STEPS

            scaler.scale(total_loss).backward()
            loss_accum += total_loss.item() * ACCUM_STEPS
            ce_accum += hard_ce.item()
            kl_accum += kl.item()
            val_accum += value_loss.item()

        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % LOG_INTERVAL == 0:
            log(f"step={step} loss={loss_accum/step:.4f} ce={ce_accum/(step*ACCUM_STEPS):.4f} "
                f"kl={kl_accum/(step*ACCUM_STEPS):.4f} val={val_accum/(step*ACCUM_STEPS):.4f} "
                f"gnorm={float(gnorm):.2f}")

        if step % SAVE_INTERVAL == 0 or step == MAX_STEPS:
            ev = evaluate(model, eval_records)
            log(f"eval step={step}: loss={ev['loss']:.4f} ce={ev['ce']:.4f} kl={ev['kl']:.4f} "
                f"val={ev['value']:.4f} acc={ev['acc']:.4f} top3={ev['top3']:.4f} "
                f"val_w={ev['val_w']:.4f} val_b={ev['val_b']:.4f}")

            # Save best by POLICY CE (preserves policy quality)
            if ev["ce"] < best_ce:
                best_ce = ev["ce"]
                save_model_weights(BEST_PATH, model)
                log(f"  NEW BEST by ce={best_ce:.4f}")
            save_model_weights(LATEST_PATH, model)

    log(f"training complete. best ce={best_ce:.4f}")
    log(f"best: {BEST_PATH}")
    log(f"latest: {LATEST_PATH}")


if __name__ == "__main__":
    main()
