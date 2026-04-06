"""Quick eval on 20K positions for reliable checkpoint comparison.

Usage:
  python experiments/_eval_20k.py --checkpoint outputs/exp149_scratch_204m/best_model.pt
  python experiments/_eval_20k.py --checkpoint outputs/exp100_diverse_training/best_model.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl

ROOT = Path(__file__).resolve().parent.parent


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


def load_eval(eval_path):
    import chess
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    wdl = compute_wdl(raw["cp"], raw["mate"])
    eval_data = []
    surviving = []
    for i in range(raw["board_array"].shape[0]):
        try:
            fen = _board_array_to_fen(
                raw["board_array"][i], raw["turn"][i],
                raw["castling"][i], raw["ep_square"][i])
            board = chess.Board(fen)
            uci = IDX_TO_UCI[raw["move_idx"][i].item()]
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            eval_data.append({"board": board, "move": move,
                              "wdl": (wdl[i, 0].item(), wdl[i, 1].item(), wdl[i, 2].item())})
            surviving.append(i)
        except Exception:
            continue
    idx = torch.tensor(surviving, dtype=torch.long)
    eval_tensors = {
        "turn": raw["turn"][idx].long(),
        "castling": raw["castling"][idx].long(),
        "ep_file": ep_square_to_file(raw["ep_square"][idx].long()),
        "fused_ids": board_array_to_fused(raw["board_array"][idx]),
    }
    return eval_data, eval_tensors


def run_eval(model, eval_data, eval_tensors, device, batch_size=64):
    model.eval()
    correct = top3 = top5 = val_correct = total = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            n = len(chunk)
            idx = slice(i, i + n)
            batch_input = {
                "fused_ids": eval_tensors["fused_ids"][idx].to(device),
                "turn": eval_tensors["turn"][idx].to(device),
                "castling": eval_tensors["castling"][idx].to(device),
                "ep_file": eval_tensors["ep_file"][idx].to(device),
            }
            if use_amp:
                with autocast('cuda', dtype=torch.float16):
                    result = model(batch_input)
            else:
                result = model(batch_input)
            logits = result["policy_logits"].float()
            value_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                board, true_move = d["board"], d["move"]
                l = logits[j].clone()
                mask = legal_move_mask(board).to(device)
                l[~mask] = float("-inf")
                true_idx = move_to_index(true_move)
                if l.argmax().item() == true_idx:
                    correct += 1
                topk3 = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk3:
                    top3 += 1
                topk5 = l.topk(min(5, l.shape[0])).indices.tolist()
                if true_idx in topk5:
                    top5 += 1
                vp = F.softmax(value_logits[j], dim=-1)
                pred_class = vp.argmax().item()
                true_class = max(range(3), key=lambda k: d["wdl"][k])
                if pred_class == true_class:
                    val_correct += 1
                total += 1
    return {
        "top1": correct / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "val": val_correct / max(total, 1),
        "total": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-path", default=None,
                    help="Path to eval .pt file (default: eval_20k.pt)")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU mode (useful when GPU is busy)")
    args = ap.parse_args()

    DEVICE = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_path = args.eval_path or str(
        ROOT / "outputs" / "exp139_massive_train" / "shards" / "eval_20k.pt")
    if not Path(eval_path).exists():
        eval_path = str(ROOT / "outputs" / "exp139_massive_train" / "shards" / "eval.pt")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}")

    print(f"Loading eval: {eval_path}")
    eval_data, eval_tensors = load_eval(eval_path)
    print(f"Eval positions: {len(eval_data)}")

    t0 = time.time()
    r = run_eval(model, eval_data, eval_tensors, DEVICE)
    elapsed = time.time() - t0

    print(f"\nResults ({r['total']} positions, {elapsed:.1f}s):")
    print(f"  top-1: {r['top1']:.2%}")
    print(f"  top-3: {r['top3']:.2%}")
    print(f"  top-5: {r['top5']:.2%}")
    print(f"  value: {r['val']:.2%}")


if __name__ == "__main__":
    main()
