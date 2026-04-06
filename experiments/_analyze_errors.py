"""Analyze model error patterns on 20K eval set.

Produces a breakdown of errors by:
  - Game phase (opening/middlegame/endgame via piece count)
  - Material balance (ahead/equal/behind)
  - Position complexity (number of legal moves)
  - Move type (captures, checks, promotions, castling, quiet)
  - Piece moved
  - Board tension (number of hanging/attacked pieces)

Runs model inference on GPU (~70s), then analysis is pure CPU.

Usage:
  python experiments/_analyze_errors.py --checkpoint outputs/exp149_scratch_204m/best_model.pt
  python experiments/_analyze_errors.py --checkpoint outputs/exp100_diverse_training/best_model.pt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
    PIECE_CHARS = ".PNBRQKpnbrqk"
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file_idx in range(8):
            sq = rank * 8 + file_idx
            p = int(ba_row[sq])
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += PIECE_CHARS[p]
        if empty > 0:
            row += str(empty)
        fen_rows.append(row)
    board_str = "/".join(fen_rows)
    turn_str = "w" if int(turn_val) == 0 else "b"
    castle_str = ""
    cv = int(castling_val)
    if cv & 8: castle_str += "K"
    if cv & 4: castle_str += "Q"
    if cv & 2: castle_str += "k"
    if cv & 1: castle_str += "q"
    if not castle_str: castle_str = "-"
    ev = int(ep_val)
    if 0 <= ev < 64:
        ep_str = chr(ord('a') + ev % 8) + str(ev // 8 + 1)
    else:
        ep_str = "-"
    return f"{board_str} {turn_str} {castle_str} {ep_str} 0 1"


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def classify_phase(board):
    """Classify game phase by piece count."""
    piece_count = len(board.piece_map())
    if piece_count >= 28:
        return "opening"
    elif piece_count >= 14:
        return "middlegame"
    else:
        return "endgame"


def material_balance(board):
    """Material balance from side-to-move perspective."""
    white_mat = sum(PIECE_VALUES.get(p.piece_type, 0)
                    for p in board.piece_map().values() if p.color == chess.WHITE)
    black_mat = sum(PIECE_VALUES.get(p.piece_type, 0)
                    for p in board.piece_map().values() if p.color == chess.BLACK)
    balance = white_mat - black_mat
    if board.turn == chess.BLACK:
        balance = -balance
    if balance > 1:
        return "ahead"
    elif balance < -1:
        return "behind"
    return "equal"


def classify_move(board, move):
    """Classify a chess move into categories."""
    tags = []
    if board.is_capture(move):
        tags.append("capture")
    if board.gives_check(move):
        tags.append("check")
    if move.promotion:
        tags.append("promotion")
    if board.is_castling(move):
        tags.append("castling")
    if not tags:
        tags.append("quiet")
    return tags


def piece_moved(board, move):
    """Return the name of the piece making the move."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "unknown"
    return chess.piece_name(piece.piece_type)


def legal_move_count(board):
    """Bucket legal move count."""
    n = board.legal_moves.count()
    if n <= 10:
        return "few (≤10)"
    elif n <= 25:
        return "medium (11-25)"
    else:
        return "many (26+)"


def sf_rank_of_model_move(board, model_move_idx, true_move_idx, logits_masked):
    """Where does the model's chosen move rank in the policy distribution?"""
    sorted_indices = logits_masked.argsort(descending=True)
    for rank, idx in enumerate(sorted_indices):
        if idx.item() == true_move_idx:
            return rank + 1  # 1-indexed
        if rank > 50:
            break
    return 999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-set", default=None,
                    help="Path to eval .pt file (default: eval_20k.pt or eval.pt)")
    args = ap.parse_args()

    # Find eval set
    shard_dir = ROOT / "outputs" / "exp139_massive_train" / "shards"
    if args.eval_set:
        eval_path = Path(args.eval_set)
    elif (shard_dir / "eval_20k.pt").exists():
        eval_path = shard_dir / "eval_20k.pt"
    else:
        eval_path = shard_dir / "eval.pt"

    print(f"Loading eval: {eval_path}", flush=True)
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)

    # Load model
    print(f"Loading model: {args.checkpoint}", flush=True)
    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    # Prepare eval data
    wdl = compute_wdl(raw["cp"], raw["mate"])
    n_total = raw["board_array"].shape[0]
    fused = board_array_to_fused(raw["board_array"])
    turn = raw["turn"].long()
    castling = raw["castling"].long()
    ep_file = ep_square_to_file(raw["ep_square"].long())
    move_idx = raw["move_idx"].long()

    print(f"Running inference on {n_total} positions...", flush=True)

    # Run model in batches, collect predictions
    all_logits = []
    all_value_logits = []
    bs = 64

    with torch.no_grad():
        for start in range(0, n_total, bs):
            end = min(start + bs, n_total)
            batch = {
                "fused_ids": fused[start:end].to(DEVICE),
                "turn": turn[start:end].to(DEVICE),
                "castling": castling[start:end].to(DEVICE),
                "ep_file": ep_file[start:end].to(DEVICE),
            }
            with autocast('cuda', dtype=torch.float16):
                result = model(batch)
            all_logits.append(result["policy_logits"].float().cpu())
            all_value_logits.append(result["value_logits"].float().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_value_logits = torch.cat(all_value_logits, dim=0)
    print(f"Inference complete. Analyzing errors...", flush=True)

    # Analysis accumulators
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "top3": 0, "top5": 0,
                                  "rank_sum": 0})
    overall = {"correct": 0, "total": 0, "top3": 0, "top5": 0}
    skipped = 0
    error_examples = []  # collect worst errors for inspection

    for i in range(n_total):
        try:
            fen = _board_array_to_fen(
                raw["board_array"][i], raw["turn"][i],
                raw["castling"][i], raw["ep_square"][i])
            board = chess.Board(fen)
            true_uci = IDX_TO_UCI[move_idx[i].item()]
            true_move = chess.Move.from_uci(true_uci)
            if true_move not in board.legal_moves:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Mask illegal moves
        logits = all_logits[i].clone()
        mask = legal_move_mask(board)
        logits[~mask] = float("-inf")

        pred_idx = logits.argmax().item()
        true_idx = move_idx[i].item()
        correct = pred_idx == true_idx

        topk = logits.topk(min(5, logits.shape[0])).indices.tolist()
        in_top3 = true_idx in topk[:3]
        in_top5 = true_idx in topk

        # SF rank of true move in model's distribution
        sf_rank = sf_rank_of_model_move(board, pred_idx, true_idx, logits)

        overall["correct"] += int(correct)
        overall["top3"] += int(in_top3)
        overall["top5"] += int(in_top5)
        overall["total"] += 1

        # Classify position
        phase = classify_phase(board)
        mat = material_balance(board)
        complexity = legal_move_count(board)
        move_tags = classify_move(board, true_move)
        piece = piece_moved(board, true_move)

        # Update stats by category
        for cat_name, cat_val in [("phase", phase), ("material", mat),
                                   ("complexity", complexity), ("piece", piece)]:
            key = f"{cat_name}:{cat_val}"
            stats[key]["correct"] += int(correct)
            stats[key]["top3"] += int(in_top3)
            stats[key]["top5"] += int(in_top5)
            stats[key]["total"] += 1
            stats[key]["rank_sum"] += sf_rank

        for tag in move_tags:
            key = f"move_type:{tag}"
            stats[key]["correct"] += int(correct)
            stats[key]["top3"] += int(in_top3)
            stats[key]["top5"] += int(in_top5)
            stats[key]["total"] += 1
            stats[key]["rank_sum"] += sf_rank

        # Collect error examples (positions where model was most wrong)
        if not correct and sf_rank >= 5:
            pred_uci = IDX_TO_UCI[pred_idx] if pred_idx < len(IDX_TO_UCI) else "???"
            error_examples.append({
                "fen": fen,
                "true": true_uci,
                "pred": pred_uci,
                "rank": sf_rank,
                "phase": phase,
                "piece": piece,
                "tags": move_tags,
            })

    # Print results
    t = overall["total"]
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS: {args.checkpoint}")
    print(f"{'='*70}")
    print(f"Overall: {overall['correct']}/{t} = {overall['correct']/t:.2%} top-1, "
          f"{overall['top3']/t:.2%} top-3, {overall['top5']/t:.2%} top-5")
    print(f"Skipped: {skipped}/{n_total}")

    # Print by category
    for category in ["phase", "material", "complexity", "piece", "move_type"]:
        print(f"\n--- By {category.upper()} ---")
        cat_items = [(k, v) for k, v in stats.items() if k.startswith(f"{category}:")]
        cat_items.sort(key=lambda x: -x[1]["total"])
        for key, s in cat_items:
            label = key.split(":", 1)[1]
            n = s["total"]
            if n < 10:
                continue
            acc = s["correct"] / n
            t3 = s["top3"] / n
            t5 = s["top5"] / n
            avg_rank = s["rank_sum"] / n
            print(f"  {label:20s}: {acc:6.2%} top-1  {t3:6.2%} top-3  {t5:6.2%} top-5  "
                  f"avg_rank={avg_rank:5.1f}  (n={n:,})")

    # Worst errors
    error_examples.sort(key=lambda x: -x["rank"])
    print(f"\n--- WORST ERRORS (rank ≥ 5, showing top 20) ---")
    for ex in error_examples[:20]:
        print(f"  rank={ex['rank']:3d} phase={ex['phase']:10s} "
              f"piece={ex['piece']:6s} tags={ex['tags']} "
              f"true={ex['true']} pred={ex['pred']}")
        print(f"         FEN: {ex['fen']}")

    # Value accuracy by phase
    print(f"\n--- VALUE ACCURACY BY PHASE ---")
    for phase_label in ["opening", "middlegame", "endgame"]:
        phase_indices = []
        for i in range(n_total):
            try:
                fen = _board_array_to_fen(
                    raw["board_array"][i], raw["turn"][i],
                    raw["castling"][i], raw["ep_square"][i])
                board = chess.Board(fen)
                if classify_phase(board) == phase_label:
                    phase_indices.append(i)
            except Exception:
                continue

        if not phase_indices:
            continue
        idx = torch.tensor(phase_indices)
        v_logits = all_value_logits[idx]
        v_targets = wdl[idx]
        v_pred = v_logits.argmax(dim=-1)
        v_true = v_targets.argmax(dim=-1)
        v_acc = (v_pred == v_true).float().mean().item()
        print(f"  {phase_label:12s}: {v_acc:.2%} (n={len(phase_indices):,})")


if __name__ == "__main__":
    main()
