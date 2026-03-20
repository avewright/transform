"""exp015b: Last-Token Pooling Test (single variant)

Runs ONLY the "last-token" pooling variant to compare against the "first-token"
baseline already measured:
  first-token:  18.8% acc, 35.4% top3 (10K samples, 3 epochs, batch=64)

In causal attention, last token attends to ALL previous tokens while first token
only sees itself. This should be a substantial improvement.
"""

import json
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp015_pooling_strategy")

NUM_TRAIN = 10_000
NUM_EVAL = 500
EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 8  # effective batch = 64
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
POOLING_MODE = "last"  # test last-token pooling


# --- Data utilities (from exp013) ---
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

def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        moves = [d["move"] for d in chunk]
        values = [d["value_target"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(m) for m in moves], dtype=torch.long, device=device)
        value_targets = torch.tensor(values, dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches

def forward_with_pooling(chess_model, board_input, pooling_mode,
                          move_targets=None, value_targets=None):
    """Forward pass with configurable pooling strategy."""
    tokens = chess_model.encoder(board_input)
    embeds = chess_model.input_proj(tokens)
    backbone_dtype = next(chess_model.backbone.parameters()).dtype
    embeds = embeds.to(backbone_dtype)
    outputs = chess_model.backbone(inputs_embeds=embeds, use_cache=False)
    hidden = outputs.last_hidden_state.float()

    if pooling_mode == "first":
        pooled = hidden[:, 0, :]
    elif pooling_mode == "last":
        pooled = hidden[:, -1, :]
    elif pooling_mode == "mean":
        pooled = hidden.mean(dim=1)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling_mode}")

    policy_logits = chess_model.policy_head(pooled)
    value_logits = chess_model.value_head(pooled)
    result = {"policy_logits": policy_logits, "value_logits": value_logits}

    device = board_input["piece_ids"].device if isinstance(board_input, dict) else board_input.device
    total_loss = torch.tensor(0.0, device=device)
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

def evaluate_accuracy(chess_model, eval_data, device, pooling_mode, n=None, batch_size=128):
    chess_model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = forward_with_pooling(chess_model, batch_input, pooling_mode)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, target_idx in enumerate(targets):
                total += 1
                if preds[j] == target_idx: correct += 1
                if target_idx in top3s[j]: top3_correct += 1
    return {"accuracy": correct / max(total, 1), "top3_accuracy": top3_correct / max(total, 1), "total": total}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pooling mode: {POOLING_MODE}")

    print("\n[1/4] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    print(f"  Dataset: {len(ds):,} samples")

    old_sorted_uci = build_old_move_mapping()
    random.seed(SEED)
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Free dataset from RAM
    del ds

    print("\n[2/4] Loading model...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    trainable = chess_model.trainable_params()
    print(f"  Trainable: {trainable:,}")

    pre_eval = evaluate_accuracy(chess_model, eval_data, device, POOLING_MODE, n=200)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    print(f"\n[3/4] Training: {len(train_data)} positions, {EPOCHS} epochs, batch={BATCH_SIZE}")
    optimizer = AdamW([p for p in chess_model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    train_start = time.time()

    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_ploss = ep_vloss = 0.0
        steps = 0

        optimizer.zero_grad()
        for bi, (batch_input, move_targets, value_targets) in enumerate(batches):
            result = forward_with_pooling(
                chess_model, batch_input, POOLING_MODE,
                move_targets=move_targets, value_targets=value_targets,
            )
            (result["loss"] / GRAD_ACCUM).backward()
            ep_ploss += result.get("policy_loss", torch.tensor(0)).item()
            ep_vloss += result.get("value_loss", torch.tensor(0)).item()
            steps += 1

            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in chess_model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_p = ep_ploss / max(steps, 1)
        avg_v = ep_vloss / max(steps, 1)
        metrics = evaluate_accuracy(chess_model, eval_data, device, POOLING_MODE, n=200)
        metrics["policy_loss"] = avg_p
        metrics["value_loss"] = avg_v
        history.append(metrics)

        elapsed = time.time() - train_start
        print(f"  Epoch {epoch+1}/{EPOCHS}: p_loss={avg_p:.4f} "
              f"acc={metrics['accuracy']:.1%} top3={metrics['top3_accuracy']:.1%} [{elapsed:.0f}s]")

    train_time = time.time() - train_start

    print(f"\n[4/4] Final Eval ({len(eval_data)} positions)")
    final_eval = evaluate_accuracy(chess_model, eval_data, device, POOLING_MODE)
    best_acc = max(h["accuracy"] for h in history)
    print(f"  Accuracy: {final_eval['accuracy']:.1%} (top3: {final_eval['top3_accuracy']:.1%})")
    print(f"  Best epoch: {best_acc:.1%}")

    total_time = time.time() - t0

    # Compare to first-token baseline (from earlier run)
    baseline_acc = 0.188
    baseline_top3 = 0.354
    delta = final_eval["accuracy"] - baseline_acc
    print(f"\n{'='*60}")
    print(f" COMPARISON: {POOLING_MODE}-token vs first-token")
    print(f"{'='*60}")
    print(f"  first-token:  {baseline_acc:.1%} acc, {baseline_top3:.1%} top3 (10K, 3ep)")
    print(f"  {POOLING_MODE}-token:  {final_eval['accuracy']:.1%} acc, {final_eval['top3_accuracy']:.1%} top3 (10K, 3ep)")
    print(f"  Delta: {delta:+.1%}")
    print(f"  Hypothesis confirmed: {delta > 0.03}")
    print(f"  Total time: {total_time:.0f}s")

    results = {
        "experiment": "exp015b_last_token_pooling",
        "pooling_mode": POOLING_MODE,
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "best_epoch_acc": best_acc,
        "history": history,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "baseline_comparison": {
            "first_token_acc": baseline_acc,
            "first_token_top3": baseline_top3,
            "delta_acc": delta,
        },
    }
    with open(OUTPUT_DIR / f"results_{POOLING_MODE}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / f'results_{POOLING_MODE}.json'}")


if __name__ == "__main__":
    main()
