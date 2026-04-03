"""exp114: Retrain only the value head with correct White-absolute WDL convention.

Hypothesis: The value head was degraded by exp084/085 fine-tuning which used
STM-perspective value labels. These conflicted with the pre-training (White-absolute)
convention on ~50% of data. Retraining ONLY the value head with correct labels should
produce a much better value signal for blend-based search.

Approach:
  1. Load baseline checkpoint (outputs/hf_checkpoint/best_model.pt)
  2. Freeze ALL parameters except value_head (Linear(512,256) + ReLU + Linear(256,3))
  3. Load exp085 JSONL data and convert value targets to White-absolute soft WDL
     using best_cp + FEN turn
  4. Train with KL divergence loss (matching pre-training) for multiple epochs
  5. Save checkpoints and evaluate with blend strategy

Key fix: best_cp in data is STM-relative.
  white_cp = best_cp if turn == WHITE else -best_cp
Then compute soft WDL using compute_wdl() in White-absolute convention.
"""

import argparse
import glob
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_features import batch_boards_to_fused_token_ids
from play import ChessTransformer200M
from move_vocab import UCI_TO_IDX, VOCAB_SIZE


def log(msg):
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)


def compute_wdl_scalar(cp: float) -> list[float]:
    """Compute soft WDL from White-absolute centipawn evaluation.
    Returns [P(W wins), P(draw), P(W loses)].
    """
    if cp > 10000:  # Mate for White
        return [1.0, 0.0, 0.0]
    if cp < -10000:  # Mate for Black
        return [0.0, 0.0, 1.0]
    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return [win / total, draw / total, loss / total]


def load_data(data_dirs: list[Path], max_records: int = 0) -> list[dict]:
    """Load JSONL records from all data directories."""
    records = []
    for d in data_dirs:
        files = sorted(glob.glob(str(d / "positions_*.jsonl")))
        for f in files:
            with open(f) as fp:
                for line in fp:
                    rec = json.loads(line)
                    records.append(rec)
                    if max_records and len(records) >= max_records:
                        return records
    return records


def prepare_value_targets(records: list[dict]) -> list[dict]:
    """Convert STM-relative best_cp to White-absolute soft WDL targets."""
    converted = []
    stats = {"white_turn": 0, "black_turn": 0, "no_cp": 0}
    for rec in records:
        fen = rec.get("fen", rec.get("position_fen", ""))
        if not fen:
            continue

        best_cp = rec.get("best_cp")
        if best_cp is None:
            stats["no_cp"] += 1
            continue

        # Determine side to move
        parts = fen.split()
        turn_str = parts[1] if len(parts) > 1 else "w"
        is_white = turn_str == "w"

        if is_white:
            stats["white_turn"] += 1
        else:
            stats["black_turn"] += 1

        # Convert STM cp to White-absolute cp
        white_cp = best_cp if is_white else -best_cp

        # Compute soft WDL in White-absolute convention
        wdl = compute_wdl_scalar(float(white_cp))

        converted.append({
            "fen": fen,
            "wdl_target": wdl,
            "white_cp": white_cp,
        })

    log(f"Data: {len(converted)}/{len(records)} records converted. "
        f"W-turn: {stats['white_turn']}, B-turn: {stats['black_turn']}, no_cp: {stats['no_cp']}")
    return converted


class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        board = chess.Board(rec["fen"])

        # Encode board using FusedBoardEncoder format
        from chess_features import board_to_fused_token_ids
        token_dict = board_to_fused_token_ids(board)

        wdl = torch.tensor(rec["wdl_target"], dtype=torch.float32)

        return {
            "fused_ids": token_dict["fused_ids"],
            "turn": token_dict["turn"].squeeze(0),
            "castling": token_dict["castling"].squeeze(0),
            "ep_file": token_dict["ep_file"].squeeze(0),
            "wdl": wdl,
        }


def collate_fn(batch):
    board_input = {
        "fused_ids": torch.stack([b["fused_ids"] for b in batch]),
        "turn": torch.stack([b["turn"] for b in batch]).squeeze(-1),
        "castling": torch.stack([b["castling"] for b in batch]).squeeze(-1),
        "ep_file": torch.stack([b["ep_file"] for b in batch]).squeeze(-1),
    }
    wdl = torch.stack([b["wdl"] for b in batch])
    return board_input, wdl


def load_model(checkpoint_path: Path) -> ChessTransformer200M:
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def freeze_all_except_value_head(model: ChessTransformer200M):
    """Freeze everything except the value head."""
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if "value_head" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    log(f"Frozen: {frozen/1e6:.1f}M params, Trainable (value head): {trainable} params")
    return trainable


def train_value_head(model, train_loader, val_loader, epochs, lr, output_dir, device):
    """Train value head with KL divergence loss (soft WDL targets)."""

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    # Cosine schedule
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for step, (board_input, wdl_target) in enumerate(train_loader):
            board_input = {k: v.to(device) for k, v in board_input.items()}
            wdl_target = wdl_target.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                result = model(board_input)
                value_logits = result["value_logits"].float()

                # KL divergence loss: KL(target || predicted)
                # = sum(target * log(target / predicted))
                # Using PyTorch: KLDivLoss expects log-probs as input, probs as target
                log_pred = F.log_softmax(value_logits, dim=-1)
                loss = F.kl_div(log_pred, wdl_target, reduction="batchmean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * wdl_target.size(0)
            train_count += wdl_target.size(0)

            if (step + 1) % 200 == 0:
                avg = train_loss_sum / train_count
                lr_now = scheduler.get_last_lr()[0]
                log(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} "
                    f"loss={avg:.4f} lr={lr_now:.2e}")

        train_avg = train_loss_sum / train_count if train_count else 0.0

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_correct = 0  # Argmax accuracy

        with torch.no_grad():
            for board_input, wdl_target in val_loader:
                board_input = {k: v.to(device) for k, v in board_input.items()}
                wdl_target = wdl_target.to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    result = model(board_input)
                    value_logits = result["value_logits"].float()
                    log_pred = F.log_softmax(value_logits, dim=-1)
                    loss = F.kl_div(log_pred, wdl_target, reduction="batchmean")

                val_loss_sum += loss.item() * wdl_target.size(0)
                val_count += wdl_target.size(0)

                # Argmax accuracy (hard label)
                pred_class = value_logits.argmax(dim=-1)
                true_class = wdl_target.argmax(dim=-1)
                val_correct += (pred_class == true_class).sum().item()

        val_avg = val_loss_sum / val_count if val_count else 0.0
        val_acc = val_correct / val_count if val_count else 0.0

        log(f"Epoch {epoch+1}/{epochs}: train_loss={train_avg:.4f}, "
            f"val_loss={val_avg:.4f}, val_acc={val_acc:.3f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        ckpt_path = output_dir / f"value_head_epoch{epoch+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_avg,
            "val_acc": val_acc,
            "train_loss": train_avg,
        }, ckpt_path)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_epoch = epoch + 1
            best_path = output_dir / "best_value_head.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_avg,
                "val_acc": val_acc,
                "train_loss": train_avg,
            }, best_path)
            log(f"  NEW BEST val_loss={val_avg:.4f} at epoch {epoch+1}")

    log(f"Training complete. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs" / "hf_checkpoint" / "best_model.pt")
    parser.add_argument("--data-dirs", type=Path, nargs="+",
                        default=[ROOT / "outputs" / "exp085_hf_data" / "dataset"])
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "exp114_value_head")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"exp114: Value head retraining with correct White-absolute WDL convention")
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Data dirs: {args.data_dirs}")
    log(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # Load and convert data
    log("Loading data...")
    records = load_data(args.data_dirs, args.max_records)
    log(f"Loaded {len(records)} raw records")

    converted = prepare_value_targets(records)
    random.shuffle(converted)

    # Split
    val_size = int(len(converted) * args.val_split)
    val_records = converted[:val_size]
    train_records = converted[val_size:]
    log(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # Datasets
    train_ds = ValueDataset(train_records)
    val_ds = ValueDataset(val_records)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # Load model with frozen backbone
    log("Loading model...")
    model = load_model(args.checkpoint)
    freeze_all_except_value_head(model)
    model = model.to(DEVICE)

    # Train
    best_path = train_value_head(
        model, train_loader, val_loader,
        args.epochs, args.lr, args.output_dir, DEVICE,
    )

    log(f"Best checkpoint: {best_path}")
    log("Done!")


if __name__ == "__main__":
    main()
