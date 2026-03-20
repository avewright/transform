"""exp_av_comparison_v2.py — Hardened A/B: policy+value CE vs action-value Q(s,a).

Both variants use identical forward paths through the same ChessModel internals.
The ONLY difference is the loss function:

  A) policy_CE(best_move) + 0.5 * value_CE(WDL)           — 1 gradient signal/pos
  B) AV_MSE(Q for all legal) + 0.5 * policy_CE + 0.5 * value_CE  — ~30 signals/pos

Both are evaluated on the SAME policy_logits (argmax of policy head with legal masking).

Fairness controls:
  - Same encoder/backbone/heads, same init seed, same optimizer config
  - Same forward path (manual backbone call in both)
  - Same data split (deterministic by seed), same eval set
  - Same param count per variant (AV head is extra but only feeds auxiliary loss)
  - Config artifact logged for reproducibility
"""

import json
import math
import os
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
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp_av_comparison_v2")
DATA_PATH = Path("data/sf_labels_5k_d8.jsonl")

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
EVAL_FRAC = 0.1  # 10% eval → ~500 positions for tighter confidence


# === Helpers ===

def cp_to_q(cp: int, eval_type: str = "cp") -> float:
    """Convert centipawn to win probability Q in [0,1]."""
    if eval_type == "mate":
        if cp > 0:
            return 1.0 - 0.001 * min(abs(cp), 50)
        elif cp < 0:
            return 0.001 * min(abs(cp), 50)
        return 0.5
    return 1.0 / (1.0 + math.exp(-cp / 111.7))


class ActionValueHead(nn.Module):
    """Q(s,a) head: predicts win probability per move in vocab."""
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


# === Data ===

def load_data(path: Path, eval_frac: float, seed: int):
    data = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["best_uci"] in UCI_TO_IDX:
                data.append(e)
    rng = random.Random(seed)
    rng.shuffle(data)
    n_eval = max(int(len(data) * eval_frac), 100)
    return data[n_eval:], data[:n_eval]


def make_batches(data, batch_size, device, include_av=False):
    """Unified batch builder for both variants.

    When include_av=True, also builds Q-target and mask tensors.
    """
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        best_idx = torch.tensor(
            [UCI_TO_IDX[d["best_uci"]] for d in chunk],
            dtype=torch.long, device=device,
        )
        wdl_targets = torch.tensor(
            [max(range(3), key=lambda x: d.get("wdl", [0.33, 0.34, 0.33])[x])
             for d in chunk],
            dtype=torch.long, device=device,
        )

        batch = {"input": batch_input, "best_idx": best_idx, "wdl": wdl_targets}

        if include_av:
            q_targets = torch.zeros(len(chunk), VOCAB_SIZE, device=device)
            q_mask = torch.zeros(len(chunk), VOCAB_SIZE, dtype=torch.bool, device=device)
            for j, d in enumerate(chunk):
                for mv in d["move_values"]:
                    if mv["uci"] in UCI_TO_IDX:
                        idx = UCI_TO_IDX[mv["uci"]]
                        q_targets[j, idx] = cp_to_q(mv["cp"], mv["type"])
                        q_mask[j, idx] = True
            batch["q_targets"] = q_targets
            batch["q_mask"] = q_mask

        batches.append(batch)
    return batches


# === Shared forward pass ===

def forward_pass(model, batch_input):
    """Identical forward through encoder → backbone → heads for both variants."""
    tokens = model.encoder(batch_input)
    embeds = model.input_proj(tokens)
    backbone_dtype = next(model.backbone.parameters()).dtype
    embeds = embeds.to(backbone_dtype)
    outputs = model.backbone(inputs_embeds=embeds, use_cache=False)
    hidden = outputs.last_hidden_state.float()
    global_hidden = hidden[:, 0, :]
    policy_logits = model.policy_head(global_hidden)
    value_logits = model.value_head(global_hidden)
    return global_hidden, policy_logits, value_logits


# === Evaluation ===

def evaluate(chess_model, eval_data, device, batch_size=64):
    chess_model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [chess.Board(d["fen"]) for d in chunk]
            targets = [UCI_TO_IDX.get(d["best_uci"], 0) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            _, policy_logits, _ = forward_pass(chess_model, batch_input)
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                policy_logits[j, ~mask] = float("-inf")
            preds = policy_logits.argmax(dim=-1).cpu().tolist()
            top3s = policy_logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, t in enumerate(targets):
                total += 1
                if preds[j] == t:
                    correct += 1
                if t in top3s[j]:
                    top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


# === Training loops ===

def train_variant_a(train_data, eval_data, device, full_model):
    """Variant A: policy CE + value CE. No action-value signal."""
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        batches = make_batches(train_data, BATCH_SIZE, device, include_av=False)
        ep_p = ep_v = steps = 0

        for batch in batches:
            _, policy_logits, value_logits = forward_pass(model, batch["input"])
            p_loss = F.cross_entropy(policy_logits, batch["best_idx"])
            v_loss = F.cross_entropy(value_logits, batch["wdl"])
            loss = p_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_p += p_loss.item()
            ep_v += v_loss.item()
            steps += 1

        ev = evaluate(model, eval_data, device)
        elapsed = time.time() - t0
        epoch_metrics = {
            **ev,
            "p_loss": ep_p / max(steps, 1),
            "v_loss": ep_v / max(steps, 1),
            "epoch": epoch + 1,
        }
        history.append(epoch_metrics)
        print(f"  [A:Policy] Ep {epoch+1}/{EPOCHS}: p={epoch_metrics['p_loss']:.4f} "
              f"v={epoch_metrics['v_loss']:.4f} | "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")

    return model, history


def train_variant_b(train_data, eval_data, device, full_model):
    """Variant B: action-value MSE + policy CE + value CE.

    AV head provides auxiliary loss on all labeled legal moves.
    Policy head and value head get same loss as variant A.
    """
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    av_head = ActionValueHead(model.hidden_size, VOCAB_SIZE).to(device)

    # Collect all trainable params (same encoder/proj/heads + extra AV head)
    model_params = [p for p in model.parameters() if p.requires_grad]
    av_params = list(av_head.parameters())
    all_params = model_params + av_params
    optimizer = AdamW(all_params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        av_head.train()
        batches = make_batches(train_data, BATCH_SIZE, device, include_av=True)
        ep_av = ep_p = ep_v = steps = 0

        for batch in batches:
            global_hidden, policy_logits, value_logits = forward_pass(model, batch["input"])

            # Same losses as variant A
            p_loss = F.cross_entropy(policy_logits, batch["best_idx"])
            v_loss = F.cross_entropy(value_logits, batch["wdl"])

            # Additional AV loss on all labeled legal moves
            q_pred = av_head(global_hidden)
            q_mask = batch["q_mask"]
            q_targets = batch["q_targets"]
            mask_sum = q_mask.float().sum().clamp(min=1.0)  # guard against empty
            av_loss = ((q_pred - q_targets) ** 2 * q_mask.float()).sum() / mask_sum

            # Combined: AV is additive auxiliary signal
            loss = av_loss + p_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_av += av_loss.item()
            ep_p += p_loss.item()
            ep_v += v_loss.item()
            steps += 1

        ev = evaluate(model, eval_data, device)
        elapsed = time.time() - t0
        epoch_metrics = {
            **ev,
            "av_loss": ep_av / max(steps, 1),
            "p_loss": ep_p / max(steps, 1),
            "v_loss": ep_v / max(steps, 1),
            "epoch": epoch + 1,
        }
        history.append(epoch_metrics)
        print(f"  [B:AV]     Ep {epoch+1}/{EPOCHS}: av={epoch_metrics['av_loss']:.4f} "
              f"p={epoch_metrics['p_loss']:.4f} | "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")

    return model, av_head, history


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    t0_total = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log config for reproducibility
    config = {
        "experiment": "exp_av_comparison_v2",
        "command": " ".join(sys.argv),
        "seed": SEED,
        "data_path": str(DATA_PATH),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "encoder_dim": ENCODER_DIM,
        "eval_frac": EVAL_FRAC,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        config["gpu"] = torch.cuda.get_device_name(0)
        config["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    train_data, eval_data = load_data(DATA_PATH, EVAL_FRAC, SEED)
    avg_moves = sum(d["num_legal"] for d in train_data + eval_data) / max(len(train_data) + len(eval_data), 1)
    print(f"\nData: {len(train_data)} train, {len(eval_data)} eval, ~{avg_moves:.0f} legal moves/pos")

    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # --- Variant A ---
    print(f"\n{'='*60}")
    print(f" VARIANT A: policy CE + value CE (baseline)")
    print(f" Loss = cross_entropy(policy, best_move) + 0.5 * cross_entropy(value, WDL)")
    print(f" Gradient signals per position: 1 (best move)")
    print(f"{'='*60}")
    model_a, hist_a = train_variant_a(train_data, eval_data, device, full_model)
    final_a = evaluate(model_a, eval_data, device)
    best_a_acc = max(h["accuracy"] for h in hist_a)
    best_a_top3 = max(h["top3_accuracy"] for h in hist_a)
    params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    print(f"  Final: acc={final_a['accuracy']:.1%} top3={final_a['top3_accuracy']:.1%}")
    print(f"  Best:  acc={best_a_acc:.1%} top3={best_a_top3:.1%}")
    print(f"  Trainable params: {params_a:,}")

    del model_a
    torch.cuda.empty_cache()

    # --- Variant B ---
    print(f"\n{'='*60}")
    print(f" VARIANT B: action-value MSE + policy CE + value CE")
    print(f" Loss = MSE(Q_pred, Q_sf) on all legal + CE(policy, best) + 0.5*CE(value, WDL)")
    print(f" Gradient signals per position: ~{avg_moves:.0f} (all legal moves)")
    print(f"{'='*60}")
    model_b, av_head_b, hist_b = train_variant_b(train_data, eval_data, device, full_model)
    final_b = evaluate(model_b, eval_data, device)
    best_b_acc = max(h["accuracy"] for h in hist_b)
    best_b_top3 = max(h["top3_accuracy"] for h in hist_b)
    params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    params_b += sum(p.numel() for p in av_head_b.parameters())
    print(f"  Final: acc={final_b['accuracy']:.1%} top3={final_b['top3_accuracy']:.1%}")
    print(f"  Best:  acc={best_b_acc:.1%} top3={best_b_top3:.1%}")
    print(f"  Trainable params: {params_b:,} (includes AV head)")

    # --- Comparison ---
    diff_acc = best_b_acc - best_a_acc
    diff_top3 = best_b_top3 - best_a_top3
    winner = "B:ACTION-VALUE" if diff_acc > 0.005 else "A:POLICY-ONLY" if diff_acc < -0.005 else "TIE"

    print(f"\n{'='*60}")
    print(f" COMPARISON (evaluated on same {len(eval_data)} held-out positions)")
    print(f"{'='*60}")
    print(f"  A (policy+value CE):    best_acc={best_a_acc:.1%}  best_top3={best_a_top3:.1%}  params={params_a:,}")
    print(f"  B (AV + policy + value): best_acc={best_b_acc:.1%}  best_top3={best_b_top3:.1%}  params={params_b:,}")
    print(f"  Delta: acc={diff_acc:+.1%}  top3={diff_top3:+.1%}")
    print(f"  Winner: {winner}")

    total_time = time.time() - t0_total
    print(f"\n  Total time: {total_time:.0f}s")

    # --- Save ---
    results = {
        "config": config,
        "data": {
            "path": str(DATA_PATH),
            "train": len(train_data),
            "eval": len(eval_data),
            "avg_legal_moves": round(avg_moves, 1),
        },
        "variant_a": {
            "description": "policy CE + value CE",
            "trainable_params": params_a,
            "best_accuracy": best_a_acc,
            "best_top3": best_a_top3,
            "final": final_a,
            "history": hist_a,
        },
        "variant_b": {
            "description": "AV MSE (all legal) + policy CE + value CE",
            "trainable_params": params_b,
            "best_accuracy": best_b_acc,
            "best_top3": best_b_top3,
            "final": final_b,
            "history": hist_b,
        },
        "comparison": {
            "winner": winner,
            "delta_accuracy": round(diff_acc, 4),
            "delta_top3": round(diff_top3, 4),
        },
        "total_time_s": round(total_time, 1),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
