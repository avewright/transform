"""exp015c: Pooling Strategy — Cached Hidden States Probe

Same hypothesis as exp015: first-token pooling is broken in causal models.
But instead of full backprop through backbone (OOM), we:
1. Forward all data through encoder + frozen backbone with torch.no_grad()
2. Cache hidden states at positions [0] (first) and [-1] (last)  
3. Train ONLY the policy/value heads on cached features
4. Compare first-token vs last-token accuracy

This is valid because: the backbone is frozen, so the encoder+backbone
produce fixed features. We're purely testing which position carries better
information through causal attention.

Much faster and ~0.1 GB memory for training (vs ~6 GB for full backprop).
"""

import json
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

OUTPUT_DIR = Path("outputs/exp015_pooling_strategy")

NUM_TRAIN = 3_000
NUM_EVAL = 300
HEAD_EPOCHS = 10  # more epochs since we're only training small heads
BATCH_SIZE = 256  # can use large batches since features are cached
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
EXTRACT_BATCH = 16  # batch size for feature extraction (no grad)


# --- Data utilities ---
def build_old_move_mapping():
    ROOK_DIRS = (+8, -8, +1, -1)
    BISHOP_DIRS = (+9, -9, +7, -7)
    KNIGHT_OFFS = (+17, +15, +10, +6, -6, -10, -15, -17)
    PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    def inside(board_idx, file_from):
        return 0 <= board_idx < 64 and abs((board_idx % 8) - file_from) <= 2
    ucis = set()
    for f in range(64):
        f_file = f % 8
        for d in ROOK_DIRS:
            t = f + d
            while inside(t, f_file):
                ucis.add(chess.Move(f, t).uci()); t += d
        for d in BISHOP_DIRS:
            t = f + d
            while inside(t, f_file):
                ucis.add(chess.Move(f, t).uci()); t += d
        for off in KNIGHT_OFFS:
            t = f + off
            if inside(t, f_file):
                ucis.add(chess.Move(f, t).uci())
    for f in range(64):
        file_ = f % 8; rank = f // 8
        if rank == 6:
            for df in (-9, -8, -7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES:
                        ucis.add(chess.Move(f, t, promotion=p).uci())
        if rank == 1:
            for df in (+9, +8, +7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES:
                        ucis.add(chess.Move(f, t, promotion=p).uci())
    return sorted(ucis)

_PIECE2CHESS = {
    1: (chess.PAWN, chess.WHITE), 2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE), 4: (chess.ROOK, chess.WHITE),
    5: (chess.QUEEN, chess.WHITE), 6: (chess.KING, chess.WHITE),
    7: (chess.PAWN, chess.BLACK), 8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK), 10: (chess.ROOK, chess.BLACK),
    11: (chess.QUEEN, chess.BLACK), 12: (chess.KING, chess.BLACK),
}

def hf_sample_to_board(board_flat, turn_raw):
    board = chess.Board(fen=None); board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8; col = idx % 8; rank = 7 - row; sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = bool(turn_raw)
    return board

def prepare_hf_data(dataset, old_sorted_uci, n, offset=0):
    data = []; skipped = 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n: break
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX: skipped += 1; continue
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError: skipped += 1; continue
        if move not in board.legal_moves: skipped += 1; continue
        winner = s["winner"]
        if winner == 0: vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK): vt = 2
        else: vt = 0
        data.append({"board": board, "move": move, "value_target": vt})
    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


@torch.no_grad()
def extract_features(chess_model, data, device, batch_size=32):
    """Extract hidden states from the backbone for all positions.
    
    Returns dict with:
        first_hidden: (N, hidden_size) — token 0 features
        last_hidden:  (N, hidden_size) — token -1 features
        mean_hidden:  (N, hidden_size) — mean-pooled features
        move_targets: (N,) — target move indices
        value_targets: (N,) — target value classes
        boards: list of boards (for legal mask during eval)
    """
    chess_model.eval()
    all_first = []
    all_last = []
    all_mean = []
    all_moves = []
    all_values = []

    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)

        # Forward through encoder + backbone (no grad)
        tokens = chess_model.encoder(batch_input)
        embeds = chess_model.input_proj(tokens)
        backbone_dtype = next(chess_model.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = chess_model.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()

        # Store features on CPU to save GPU memory
        all_first.append(hidden[:, 0, :].cpu())
        all_last.append(hidden[:, -1, :].cpu())
        all_mean.append(hidden.mean(dim=1).cpu())

        move_idxs = [move_to_index(d["move"]) for d in chunk]
        all_moves.extend(move_idxs)
        all_values.extend([d["value_target"] for d in chunk])

        if (i // batch_size + 1) % 50 == 0:
            print(f"    Extracted {min(i + batch_size, len(data))}/{len(data)}")

    return {
        "first_hidden": torch.cat(all_first),
        "last_hidden": torch.cat(all_last),
        "mean_hidden": torch.cat(all_mean),
        "move_targets": torch.tensor(all_moves, dtype=torch.long),
        "value_targets": torch.tensor(all_values, dtype=torch.long),
        "boards": [d["board"] for d in data],
    }


class PolicyValueHead(nn.Module):
    """Standalone policy + value heads for training on cached features."""
    def __init__(self, hidden_size):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, VOCAB_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, hidden, move_targets=None, value_targets=None):
        policy_logits = self.policy_head(hidden)
        value_logits = self.value_head(hidden)

        result = {"policy_logits": policy_logits, "value_logits": value_logits}
        total_loss = torch.tensor(0.0, device=hidden.device)
        if move_targets is not None:
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            result["policy_loss"] = policy_loss
            total_loss = total_loss + policy_loss
        if value_targets is not None:
            value_loss = F.cross_entropy(value_logits, value_targets)
            result["value_loss"] = value_loss
            total_loss = total_loss + 0.5 * value_loss
        result["loss"] = total_loss
        return result


def train_head(features, move_targets, value_targets, boards_eval, features_eval,
               moves_eval, values_eval, boards_eval_list, hidden_size, device, label):
    """Train a fresh head on cached features and evaluate."""
    print(f"\n  --- Training head: {label} ---")
    torch.manual_seed(SEED)
    head = PolicyValueHead(hidden_size).to(device)
    optimizer = AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    n = features.shape[0]
    total_steps = HEAD_EPOCHS * (n // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()

    for epoch in range(HEAD_EPOCHS):
        head.train()
        perm = torch.randperm(n)
        ep_ploss = 0.0
        steps = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            h = features[idx].to(device)
            mt = move_targets[idx].to(device)
            vt = value_targets[idx].to(device)

            optimizer.zero_grad()
            result = head(h, mt, vt)
            result["loss"].backward()
            optimizer.step()
            scheduler.step()
            ep_ploss += result["policy_loss"].item()
            steps += 1

        avg_ploss = ep_ploss / max(steps, 1)

        # Eval
        head.eval()
        correct = top3_correct = total = 0
        with torch.no_grad():
            for i in range(0, features_eval.shape[0], BATCH_SIZE):
                h = features_eval[i:i + BATCH_SIZE].to(device)
                result = head(h)
                logits = result["policy_logits"]
                # Apply legal move masks
                for j in range(logits.shape[0]):
                    board_idx = i + j
                    if board_idx < len(boards_eval_list):
                        mask = legal_move_mask(boards_eval_list[board_idx]).to(device)
                        logits[j, ~mask] = float("-inf")
                preds = logits.argmax(dim=-1).cpu()
                top3 = logits.topk(3, dim=-1).indices.cpu()
                targets = moves_eval[i:i + BATCH_SIZE]
                for j in range(preds.shape[0]):
                    total += 1
                    if preds[j].item() == targets[j].item():
                        correct += 1
                    if targets[j].item() in top3[j].tolist():
                        top3_correct += 1

        acc = correct / max(total, 1)
        top3_acc = top3_correct / max(total, 1)
        history.append({"accuracy": acc, "top3_accuracy": top3_acc, "policy_loss": avg_ploss})
        elapsed = time.time() - t0
        print(f"    Epoch {epoch+1}/{HEAD_EPOCHS}: p_loss={avg_ploss:.4f} acc={acc:.1%} top3={top3_acc:.1%} [{elapsed:.0f}s]")

    train_time = time.time() - t0
    best_acc = max(h["accuracy"] for h in history)
    best_top3 = max(h["top3_accuracy"] for h in history)

    return {
        "label": label,
        "final_acc": history[-1]["accuracy"],
        "final_top3": history[-1]["top3_accuracy"],
        "best_acc": best_acc,
        "best_top3": best_top3,
        "history": history,
        "train_time_s": train_time,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[1/4] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    print(f"  Dataset: {len(ds):,} samples")

    old_sorted_uci = build_old_move_mapping()
    random.seed(SEED)
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
    del ds  # free RAM
    import gc; gc.collect()

    # --- Load model and extract features ---
    print("\n[2/4] Loading model and extracting features...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    hidden_size = chess_model.hidden_size
    print(f"  Hidden size: {hidden_size}")

    print("  Extracting train features...")
    train_feats = extract_features(chess_model, train_data, device, EXTRACT_BATCH)
    print("  Extracting eval features...")
    eval_feats = extract_features(chess_model, eval_data, device, EXTRACT_BATCH)

    # Free model from GPU
    del chess_model, full_model, encoder
    torch.cuda.empty_cache()
    print("  Model freed. Features cached on CPU.")

    # --- Train heads for each pooling mode ---
    print(f"\n[3/4] Training heads ({HEAD_EPOCHS} epochs each)")

    results = {}
    for mode in ["first", "last", "mean"]:
        feat_key = f"{mode}_hidden"
        r = train_head(
            train_feats[feat_key], train_feats["move_targets"], train_feats["value_targets"],
            eval_feats[feat_key], eval_feats[feat_key],
            eval_feats["move_targets"], eval_feats["value_targets"],
            eval_feats["boards"],
            hidden_size, device, mode,
        )
        results[mode] = r

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" SUMMARY — Pooling Strategy (Cached Feature Probe)")
    print(f"{'='*60}")
    print(f"  {'Mode':<8} {'Best Acc':>10} {'Best Top3':>10} {'Final Acc':>10} {'Time':>6}")
    print(f"  {'-'*48}")
    for mode in ["first", "last", "mean"]:
        r = results[mode]
        print(f"  {mode:<8} {r['best_acc']:>9.1%} {r['best_top3']:>9.1%} {r['final_acc']:>9.1%} {r['train_time_s']:>5.0f}s")

    best_mode = max(results, key=lambda m: results[m]["best_acc"])
    worst_mode = min(results, key=lambda m: results[m]["best_acc"])
    delta = results[best_mode]["best_acc"] - results[worst_mode]["best_acc"]

    print(f"\n  Best: {best_mode} ({results[best_mode]['best_acc']:.1%})")
    print(f"  Worst: {worst_mode} ({results[worst_mode]['best_acc']:.1%})")
    print(f"  Delta: {delta:.1%}")
    print(f"  Total time: {total_time:.0f}s")

    hypothesis_confirmed = best_mode != "first" and delta > 0.02
    print(f"\n  Hypothesis (first-token is suboptimal): {'CONFIRMED' if hypothesis_confirmed else 'NOT CONFIRMED'}")
    if hypothesis_confirmed:
        print(f"  Recommendation: Change chess_model.py to use '{best_mode}' pooling")

    output = {
        "experiment": "exp015c_pooling_probe",
        "hypothesis": "First-token pooling is suboptimal in causal transformer",
        "method": "Train heads on cached backbone features (no backprop through backbone)",
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "head_training": {"epochs": HEAD_EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "results": results,
        "best_mode": best_mode,
        "worst_mode": worst_mode,
        "delta": delta,
        "hypothesis_confirmed": hypothesis_confirmed,
        "total_time_s": total_time,
    }
    with open(OUTPUT_DIR / "results_probe.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results_probe.json'}")


if __name__ == "__main__":
    main()
