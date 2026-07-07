"""exp015: Causal Pooling Fix — First-Token vs Last-Token vs Mean-Pool

Hypothesis: The current model takes hidden[:, 0, :] (first token) for policy/value
predictions. But Qwen3 is a **causal decoder-only transformer** — token 0 can only
attend to itself! The input order is [turn, castling, ep, sq0...sq63], so token 0
(turn) has zero access to any board square information through causal attention.

The last token (sq63) attends to ALL previous tokens and should carry far richer
information. Mean-pooling mixes all representations equally.

Test: Train 3 models (same init seed) with 10K HF positions, 3 epochs:
  A) first-token pooling (current architecture) — baseline
  B) last-token pooling — causal-aware fix
  C) mean pooling — full aggregation

Target: B or C should substantially outperform A (>5% absolute accuracy gain).
Time budget: ~8 min total.
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

NUM_TRAIN = 5_000
NUM_EVAL = 500
EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-3
ENCODER_DIM = 256
SEED = 42

POOLING_MODES = ["first", "last"]


# --- Reuse data utilities from exp013 ---

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
    board = chess.Board(fen=None)
    board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8; col = idx % 8
            rank = 7 - row; sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = bool(turn_raw)
    return board


def prepare_hf_data(dataset, old_sorted_uci, n, offset=0):
    data = []
    skipped = 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n:
            break
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX:
            skipped += 1; continue
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError:
            skipped += 1; continue
        if move not in board.legal_moves:
            skipped += 1; continue
        winner = s["winner"]
        if winner == 0:
            vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK):
            vt = 2
        else:
            vt = 0
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
        move_targets = torch.tensor(
            [move_to_index(m) for m in moves], dtype=torch.long, device=device
        )
        value_targets = torch.tensor(values, dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


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
                if preds[j] == target_idx:
                    correct += 1
                if target_idx in top3s[j]:
                    top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def forward_with_pooling(chess_model, board_input, pooling_mode,
                          move_targets=None, value_targets=None):
    """Forward pass with configurable pooling strategy."""
    if isinstance(board_input, dict):
        B = board_input["piece_ids"].shape[0]
    else:
        B = board_input.shape[0]

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


def train_one_variant(pooling_mode, train_data, eval_data, device):
    """Train a fresh model with the given pooling strategy."""
    print(f"\n{'='*60}")
    print(f"  VARIANT: pooling={pooling_mode}")
    print(f"{'='*60}")

    # Reproducible init
    torch.manual_seed(SEED)
    random.seed(SEED)

    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # Same seed for encoder/head init
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)

    trainable = chess_model.trainable_params()
    print(f"  Trainable: {trainable:,}")

    # Pre-train eval
    pre_eval = evaluate_accuracy(chess_model, eval_data, device, pooling_mode, n=200)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    optimizer = AdamW(
        [p for p in chess_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    train_start = time.time()

    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_ploss = ep_vloss = 0.0
        steps = 0

        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = forward_with_pooling(
                chess_model, batch_input, pooling_mode,
                move_targets=move_targets, value_targets=value_targets,
            )
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in chess_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            ep_ploss += result.get("policy_loss", torch.tensor(0)).item()
            ep_vloss += result.get("value_loss", torch.tensor(0)).item()
            steps += 1

        avg_p = ep_ploss / max(steps, 1)
        avg_v = ep_vloss / max(steps, 1)
        metrics = evaluate_accuracy(chess_model, eval_data, device, pooling_mode, n=200)
        metrics["policy_loss"] = avg_p
        metrics["value_loss"] = avg_v
        history.append(metrics)

        elapsed = time.time() - train_start
        print(
            f"    Epoch {epoch+1}/{EPOCHS}: p_loss={avg_p:.4f} "
            f"acc={metrics['accuracy']:.1%} top3={metrics['top3_accuracy']:.1%} "
            f"[{elapsed:.0f}s]"
        )

    train_time = time.time() - train_start
    final_eval = evaluate_accuracy(chess_model, eval_data, device, pooling_mode)
    best_acc = max(h["accuracy"] for h in history)

    print(f"  Final: acc={final_eval['accuracy']:.1%} top3={final_eval['top3_accuracy']:.1%}")
    print(f"  Best epoch acc: {best_acc:.1%}  Training: {train_time:.0f}s")

    # Free model memory
    del chess_model, full_model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "pooling_mode": pooling_mode,
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "best_epoch_acc": best_acc,
        "history": history,
        "train_time_s": train_time,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[1] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    print(f"  Dataset: {len(ds):,} samples")

    old_sorted_uci = build_old_move_mapping()

    random.seed(SEED)
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # --- Train each variant ---
    all_results = {}
    for mode in POOLING_MODES:
        result = train_one_variant(mode, train_data, eval_data, device)
        all_results[mode] = result

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" SUMMARY — Pooling Strategy Comparison")
    print(f"{'='*60}")
    print(f"  {'Mode':<8} {'Final Acc':>10} {'Top3':>8} {'Best Ep':>8} {'Time':>6}")
    print(f"  {'-'*44}")

    for mode in POOLING_MODES:
        r = all_results[mode]
        print(
            f"  {mode:<8} {r['final_eval']['accuracy']:>9.1%} "
            f"{r['final_eval']['top3_accuracy']:>7.1%} "
            f"{r['best_epoch_acc']:>7.1%} "
            f"{r['train_time_s']:>5.0f}s"
        )

    best_mode = max(POOLING_MODES, key=lambda m: all_results[m]["final_eval"]["accuracy"])
    worst_mode = min(POOLING_MODES, key=lambda m: all_results[m]["final_eval"]["accuracy"])
    delta = all_results[best_mode]["final_eval"]["accuracy"] - all_results[worst_mode]["final_eval"]["accuracy"]

    print(f"\n  Best: {best_mode} ({all_results[best_mode]['final_eval']['accuracy']:.1%})")
    print(f"  Worst: {worst_mode} ({all_results[worst_mode]['final_eval']['accuracy']:.1%})")
    print(f"  Delta: {delta:.1%}")
    print(f"  Total time: {total_time:.0f}s")

    hypothesis_confirmed = best_mode != "first" and delta > 0.03
    print(f"\n  Hypothesis (first-token is suboptimal): {'CONFIRMED' if hypothesis_confirmed else 'NOT CONFIRMED'}")

    results = {
        "experiment": "exp015_pooling_strategy",
        "hypothesis": "First-token pooling is suboptimal in causal transformer; last-token or mean should win",
        "rationale": "In causal attention, token 0 only sees itself. Last token sees all tokens.",
        "data": {"train": len(train_data), "eval": len(eval_data), "source": "HF game-play"},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "variants": all_results,
        "best_mode": best_mode,
        "worst_mode": worst_mode,
        "accuracy_delta": delta,
        "hypothesis_confirmed": hypothesis_confirmed,
        "total_time_s": total_time,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
