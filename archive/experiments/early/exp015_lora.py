"""exp015: LoRA Backbone Adaptation — Unlock frozen Qwen3 for chess.

Hypothesis: Low-Rank Adaptation (LoRA) of the Qwen3-0.6B attention layers
will significantly improve accuracy compared to a fully frozen backbone,
without the catastrophic forgetting risk of full unfreezing.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
(2021). LoRA adds ~2M trainable params to adapt a 603M backbone.

Context: exp010 showed unfreezing doesn't help with 500 positions — data
volume was the bottleneck. With 5-20K positions and LoRA, the backbone's
attention patterns can adapt to chess structure.

Approach:
  1. Apply LoRA (rank=16) to q_proj and v_proj in all Qwen3 attention layers
  2. Train encoder + LoRA + heads on Stockfish-labeled data
  3. Compare against frozen-backbone baseline (same data, same epochs)
  4. Measure: accuracy, top-3 accuracy, training loss convergence

Memory estimate:
  - Qwen3-0.6B bf16: ~1.2 GB
  - LoRA adapters (r=16, 28 layers × 2 matrices): ~2M params = ~4 MB
  - Gradients for LoRA + encoder + heads: ~500 MB
  - Batch of 32, seq_len 67: ~200 MB activations
  - Total: ~3-4 GB. Fits 8GB, trivial for 18GB.

Time: ~8 min (train 2 models × 10 epochs on 5K positions).
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, legal_move_mask
from config import Config

# --- Configuration ---
OUTPUT_DIR = Path("outputs/exp015_lora")
CACHE_FILE = Path("outputs/exp012_stockfish_supervised/labeled_data.json")

# Data
NUM_TRAIN = 5000
NUM_EVAL = 500

# Training
EPOCHS = 10
BATCH_SIZE = 32            # 8GB: use 16. 18GB: use 64
LR = 1e-3
LR_LORA = 5e-5            # Lower LR for LoRA params (pretrained weights)
ENCODER_DIM = 256
SEED = 42

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]  # Which attention modules to adapt


class LoRALinear(nn.Module):
    """LoRA adapter: wraps an existing nn.Linear with low-rank update.

    output = original(x) + (x @ A^T @ B^T) * (alpha / rank)
    Only A and B are trainable. Original weights are frozen.
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original(x)
        # LoRA delta
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out.to(result.dtype)


def apply_lora(model: nn.Module, rank: int = LORA_RANK, alpha: float = LORA_ALPHA,
               dropout: float = LORA_DROPOUT, targets: list[str] = None) -> int:
    """Apply LoRA adapters to target modules in the model.

    Returns the number of LoRA parameters added.
    """
    targets = targets or LORA_TARGETS
    lora_params = 0
    replaced = 0

    for name, module in list(model.named_modules()):
        for target in targets:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name

                # Replace with LoRA wrapper
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)

                n_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                lora_params += n_params
                replaced += 1

    return lora_params


def cp_to_wdl(cp, ply=30):
    k = 1.0 / (111.7 + 0.5 * max(0, ply))
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0, 0.5 - 0.5 * abs(win - 0.5) * 2)
    total = win + draw + loss
    return win / total, draw / total, loss / total


def prepare_data(labeled):
    data = []
    for e in labeled:
        board = chess.Board(e["fen"])
        move = chess.Move.from_uci(e["uci"])
        if e["eval_type"] == "mate":
            v = e["eval_value"]
            wdl_idx = 0 if v > 0 else 2 if v < 0 else 1
        else:
            cp = e["eval_value"]
            if not board.turn:
                cp = -cp
            wdl_idx = 0 if cp > 100 else 2 if cp < -100 else 1
        data.append({"board": board, "move": move, "wdl_idx": wdl_idx})
    return data


def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], dtype=torch.long, device=device)
        value_targets = torch.tensor([d["wdl_idx"] for d in chunk], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_model(chess_model, eval_data, device, n=None):
    chess_model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for entry in subset:
            board, target = entry["board"], entry["move"]
            pred, probs = chess_model.predict_move(board)
            total += 1
            if pred == target:
                correct += 1
            top3 = probs.topk(min(3, probs.shape[0])).indices.cpu().tolist()
            if move_to_index(target) in top3:
                top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def train_model(chess_model, train_data, eval_data, device, epochs, param_groups):
    """Train with given parameter groups and return history."""
    optimizer = AdamW(param_groups, weight_decay=0.01)
    total_steps = epochs * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    for epoch in range(epochs):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0

        for batch_input, move_targets, value_targets in batches:
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in param_groups for p in group["params"]], 1.0
            )
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            steps += 1

        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_model(chess_model, eval_data, device, n=200)
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%}")

    return history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading cached Stockfish labels...")
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    train_data = prepare_data(cache["train"][:NUM_TRAIN])
    eval_data = prepare_data(cache["eval"][:NUM_EVAL])
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Load backbone (shared between both experiments)
    print("Loading Qwen3-0.6B backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # ---- Variant A: Frozen backbone (baseline) ----
    print(f"\n{'=' * 60}")
    print(f" BASELINE: Frozen backbone ({EPOCHS} epochs)")
    print(f"{'=' * 60}")

    encoder_frozen = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_frozen = ChessModel(full_model, encoder=encoder_frozen, freeze_backbone=True).to(device)
    trainable_frozen = model_frozen.trainable_params()
    print(f"  Trainable: {trainable_frozen:,}")

    pre_eval_frozen = evaluate_model(model_frozen, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_eval_frozen['accuracy']:.1%}")

    frozen_params = [{"params": [p for p in model_frozen.parameters() if p.requires_grad], "lr": LR}]
    frozen_history = train_model(model_frozen, train_data, eval_data, device, EPOCHS, frozen_params)

    # Free memory
    del model_frozen, encoder_frozen
    torch.cuda.empty_cache()

    # ---- Variant B: LoRA backbone ----
    print(f"\n{'=' * 60}")
    print(f" LoRA: Adapted backbone (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    print(f"{'=' * 60}")

    # Need to reload backbone since we modified it
    full_model_lora, _ = load_base_model(cfg)
    full_model_lora = full_model_lora.to(device)

    encoder_lora = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model_lora = ChessModel(full_model_lora, encoder=encoder_lora, freeze_backbone=True).to(device)

    # Apply LoRA to the backbone's attention layers
    lora_params_count = apply_lora(
        model_lora.backbone,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        targets=LORA_TARGETS,
    )
    print(f"  LoRA params added: {lora_params_count:,}")

    # Count all trainable
    total_trainable_lora = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    print(f"  Total trainable: {total_trainable_lora:,} (encoder+heads+LoRA)")

    pre_eval_lora = evaluate_model(model_lora, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_eval_lora['accuracy']:.1%}")

    # Two param groups: higher LR for encoder/heads, lower for LoRA
    lora_param_list = []
    other_param_list = []
    for name, param in model_lora.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_param_list.append(param)
            else:
                other_param_list.append(param)

    print(f"  LoRA params: {sum(p.numel() for p in lora_param_list):,}")
    print(f"  Other params: {sum(p.numel() for p in other_param_list):,}")

    param_groups = [
        {"params": other_param_list, "lr": LR},
        {"params": lora_param_list, "lr": LR_LORA},
    ]
    lora_history = train_model(model_lora, train_data, eval_data, device, EPOCHS, param_groups)

    # ---- Compare ----
    print(f"\n{'=' * 60}")
    print(f" RESULTS COMPARISON")
    print(f"{'=' * 60}")

    fr_final = frozen_history[-1]
    lo_final = lora_history[-1]
    fr_best = max(h["accuracy"] for h in frozen_history)
    lo_best = max(h["accuracy"] for h in lora_history)

    print(f"  Frozen:  acc={fr_final['accuracy']:.1%} top3={fr_final['top3_accuracy']:.1%} "
          f"loss={fr_final['loss']:.4f} (best: {fr_best:.1%})")
    print(f"  LoRA:    acc={lo_final['accuracy']:.1%} top3={lo_final['top3_accuracy']:.1%} "
          f"loss={lo_final['loss']:.4f} (best: {lo_best:.1%})")

    diff = lo_best - fr_best
    winner = "LoRA" if diff > 0.01 else "Frozen" if diff < -0.01 else "TIE"
    print(f"\n  Winner: {winner} (delta: {diff:+.1%})")
    print(f"  Params: Frozen={trainable_frozen:,}, LoRA={total_trainable_lora:,}")

    # Save results
    elapsed = time.time() - t0
    results = {
        "experiment": "exp015_lora",
        "hypothesis": "LoRA adapts frozen backbone to chess, improving accuracy",
        "lora_config": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "targets": LORA_TARGETS,
            "lora_params": lora_params_count,
        },
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "frozen": {
            "trainable_params": trainable_frozen,
            "best_accuracy": fr_best,
            "final": fr_final,
            "history": frozen_history,
        },
        "lora": {
            "trainable_params": total_trainable_lora,
            "best_accuracy": lo_best,
            "final": lo_final,
            "history": lora_history,
        },
        "winner": winner,
        "delta": round(diff, 4),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
