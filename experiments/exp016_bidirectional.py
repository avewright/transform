"""exp016: Bidirectional Attention — Causal vs Non-Causal

Hypothesis: Qwen3 uses causal attention by default, meaning early tokens can't
see later tokens. For chess, there's no natural left-to-right ordering — every
square should attend to every other square. Enabling bidirectional attention
(config.is_causal = False) should improve feature quality.

Combined with exp015's last-token fix, we test 4 variants:
  A) causal + first-token   (original architecture)
  B) causal + last-token    (exp015 fix)
  C) bidirectional + first-token
  D) bidirectional + last-token
  E) bidirectional + mean-pool

Method: Cached feature probe (same as exp015c) — extract features once per
attention mode, train heads on each pooling variant.

Target: Bidirectional should improve all pooling modes, and the gap between
first/last-token should shrink (since all tokens can see everything).
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

OUTPUT_DIR = Path("outputs/exp016_bidirectional")

NUM_TRAIN = 3_000
NUM_EVAL = 300
HEAD_EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
EXTRACT_BATCH = 32


# --- Data utilities (reused from exp015c) ---
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
    """Extract hidden states for all positions. Returns dict of features on CPU."""
    chess_model.eval()
    all_first, all_last, all_mean = [], [], []
    all_moves, all_values = [], []

    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)

        tokens = chess_model.encoder(batch_input)
        embeds = chess_model.input_proj(tokens)
        backbone_dtype = next(chess_model.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = chess_model.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()

        all_first.append(hidden[:, 0, :].cpu())
        all_last.append(hidden[:, -1, :].cpu())
        all_mean.append(hidden.mean(dim=1).cpu())
        all_moves.extend([move_to_index(d["move"]) for d in chunk])
        all_values.extend([d["value_target"] for d in chunk])

    return {
        "first_hidden": torch.cat(all_first),
        "last_hidden": torch.cat(all_last),
        "mean_hidden": torch.cat(all_mean),
        "move_targets": torch.tensor(all_moves, dtype=torch.long),
        "value_targets": torch.tensor(all_values, dtype=torch.long),
        "boards": [d["board"] for d in data],
    }


class PolicyValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, VOCAB_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Linear(256, 3),
        )

    def forward(self, hidden, move_targets=None, value_targets=None):
        policy_logits = self.policy_head(hidden)
        value_logits = self.value_head(hidden)
        result = {"policy_logits": policy_logits, "value_logits": value_logits}
        total_loss = torch.tensor(0.0, device=hidden.device)
        if move_targets is not None:
            pl = F.cross_entropy(policy_logits, move_targets)
            result["policy_loss"] = pl; total_loss = total_loss + pl
        if value_targets is not None:
            vl = F.cross_entropy(value_logits, value_targets)
            result["value_loss"] = vl; total_loss = total_loss + 0.5 * vl
        result["loss"] = total_loss
        return result


def train_head(features, move_targets, value_targets,
               features_eval, moves_eval, values_eval, boards_eval,
               hidden_size, device, label):
    torch.manual_seed(SEED)
    head = PolicyValueHead(hidden_size).to(device)
    optimizer = AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    n = features.shape[0]
    total_steps = HEAD_EPOCHS * (n // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_acc = 0
    best_top3 = 0
    t0 = time.time()

    for epoch in range(HEAD_EPOCHS):
        head.train()
        perm = torch.randperm(n)
        ep_ploss = steps = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            result = head(features[idx].to(device), move_targets[idx].to(device), value_targets[idx].to(device))
            result["loss"].backward()
            optimizer.step(); scheduler.step()
            ep_ploss += result["policy_loss"].item(); steps += 1

        # Eval
        head.eval()
        correct = top3_correct = total = 0
        with torch.no_grad():
            for i in range(0, features_eval.shape[0], BATCH_SIZE):
                h = features_eval[i:i + BATCH_SIZE].to(device)
                result = head(h)
                logits = result["policy_logits"]
                for j in range(logits.shape[0]):
                    bi = i + j
                    if bi < len(boards_eval):
                        mask = legal_move_mask(boards_eval[bi]).to(device)
                        logits[j, ~mask] = float("-inf")
                preds = logits.argmax(dim=-1).cpu()
                top3 = logits.topk(3, dim=-1).indices.cpu()
                targets = moves_eval[i:i + BATCH_SIZE]
                for j in range(preds.shape[0]):
                    total += 1
                    if preds[j].item() == targets[j].item(): correct += 1
                    if targets[j].item() in top3[j].tolist(): top3_correct += 1

        acc = correct / max(total, 1)
        top3_acc = top3_correct / max(total, 1)
        best_acc = max(best_acc, acc)
        best_top3 = max(best_top3, top3_acc)

    return {"best_acc": best_acc, "best_top3": best_top3, "train_time_s": time.time() - t0}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[1/5] Loading data...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    random.seed(SEED)
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
    del ds
    import gc; gc.collect()

    # --- Extract features for CAUSAL mode ---
    print("\n[2/5] Extracting features (CAUSAL)...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    hidden_size = chess_model.hidden_size
    print(f"  is_causal: {getattr(chess_model.backbone.config, 'is_causal', True)}")

    print("  Train features...")
    causal_train = extract_features(chess_model, train_data, device, EXTRACT_BATCH)
    print("  Eval features...")
    causal_eval = extract_features(chess_model, eval_data, device, EXTRACT_BATCH)

    del chess_model, full_model, encoder
    torch.cuda.empty_cache(); gc.collect()
    print("  Freed causal model.")

    # --- Extract features for BIDIRECTIONAL mode ---
    print("\n[3/5] Extracting features (BIDIRECTIONAL)...")
    full_model2, _ = load_base_model(cfg)
    full_model2 = full_model2.to(device)
    torch.manual_seed(SEED)
    encoder2 = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model2 = ChessModel(full_model2, encoder=encoder2, freeze_backbone=True).to(device)

    # Enable bidirectional attention
    chess_model2.backbone.config.is_causal = False
    print(f"  is_causal: {chess_model2.backbone.config.is_causal}")

    print("  Train features...")
    bidir_train = extract_features(chess_model2, train_data, device, EXTRACT_BATCH)
    print("  Eval features...")
    bidir_eval = extract_features(chess_model2, eval_data, device, EXTRACT_BATCH)

    del chess_model2, full_model2, encoder2
    torch.cuda.empty_cache(); gc.collect()
    print("  Freed bidirectional model.")

    # --- Train heads for all variants ---
    print(f"\n[4/5] Training heads ({HEAD_EPOCHS} epochs each)")

    variants = [
        ("causal_first",  causal_train["first_hidden"], causal_eval["first_hidden"]),
        ("causal_last",   causal_train["last_hidden"],  causal_eval["last_hidden"]),
        ("causal_mean",   causal_train["mean_hidden"],  causal_eval["mean_hidden"]),
        ("bidir_first",   bidir_train["first_hidden"],  bidir_eval["first_hidden"]),
        ("bidir_last",    bidir_train["last_hidden"],   bidir_eval["last_hidden"]),
        ("bidir_mean",    bidir_train["mean_hidden"],   bidir_eval["mean_hidden"]),
    ]

    results = {}
    for label, train_feat, eval_feat in variants:
        r = train_head(
            train_feat, causal_train["move_targets"], causal_train["value_targets"],
            eval_feat, causal_eval["move_targets"], causal_eval["value_targets"],
            causal_eval["boards"], hidden_size, device, label,
        )
        results[label] = r
        print(f"  {label:<16} acc={r['best_acc']:.1%} top3={r['best_top3']:.1%} [{r['train_time_s']:.0f}s]")

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f" SUMMARY — Causal vs Bidirectional × Pooling Strategy")
    print(f"{'='*60}")
    print(f"  {'Variant':<18} {'Best Acc':>10} {'Best Top3':>10}")
    print(f"  {'-'*40}")
    for label in [v[0] for v in variants]:
        r = results[label]
        print(f"  {label:<18} {r['best_acc']:>9.1%} {r['best_top3']:>9.1%}")

    # Analysis
    causal_best = max(r["best_acc"] for label, r in results.items() if "causal" in label)
    bidir_best = max(r["best_acc"] for label, r in results.items() if "bidir" in label)
    delta = bidir_best - causal_best

    print(f"\n  Best causal:  {causal_best:.1%}")
    print(f"  Best bidir:   {bidir_best:.1%}")
    print(f"  Delta:        {delta:+.1%}")
    print(f"  Total time:   {total_time:.0f}s")

    bidir_helps = delta > 0.02
    print(f"\n  Bidirectional helps: {'YES' if bidir_helps else 'NO'}")
    if bidir_helps:
        best_overall = max(results.items(), key=lambda x: x[1]["best_acc"])
        print(f"  Best overall: {best_overall[0]} ({best_overall[1]['best_acc']:.1%})")
        print(f"  Recommendation: Set backbone.config.is_causal = False in chess_model.py")

    output = {
        "experiment": "exp016_bidirectional",
        "hypothesis": "Bidirectional attention improves chess feature quality over causal",
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "results": {k: v for k, v in results.items()},
        "causal_best": causal_best,
        "bidir_best": bidir_best,
        "delta": delta,
        "bidir_helps": bidir_helps,
        "total_time_s": total_time,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
