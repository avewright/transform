#!/usr/bin/env python3
"""Interactive policy inference for exp186 soft-finetuned model.

Commands:
  <fen>           score position (top-8 legal)
  move <uci>      push move on current board
  undo            pop last move
  reset           start position
  new <fen>       set board from FEN
  quit
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_inference import load_checkpoint, get_model_move
from chess_features import batch_boards_to_fused_token_ids
from move_vocab import IDX_TO_UCI, legal_move_mask

CKPT = os.environ.get(
    "INFER_CHECKPOINT",
    str(ROOT / "outputs/exp186_finetune_soft/best.pt"),
)

def top_moves(model, board, device, k=8):
    board_input = batch_boards_to_fused_token_ids([board], device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out = model(board_input)
    logits = out["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    kk = min(k, int(mask.sum().item()))
    top = torch.topk(probs, kk)
    rows = [(IDX_TO_UCI[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]
    wdl = F.softmax(out["value_logits"][0].float(), dim=-1)
    return rows, {"win": wdl[0].item(), "draw": wdl[1].item(), "loss": wdl[2].item()}

def main():
    print(f"Loading {CKPT} ...", flush=True)
    model = load_checkpoint(CKPT)
    device = next(model.parameters()).device
    print(f"Ready on {device} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
    print("Enter FEN, or: move/undo/reset/new/quit", flush=True)
    board = chess.Board()
    while True:
        try:
            line = input("infer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "reset":
            board = chess.Board(); print(board); continue
        if line == "undo":
            if board.move_stack: board.pop()
            print(board); continue
        if line.startswith("new "):
            try:
                board = chess.Board(line[4:].strip())
            except Exception as e:
                print("bad fen:", e); continue
            print(board); continue
        if line.startswith("move "):
            try:
                board.push_uci(line.split(None, 1)[1])
            except Exception as e:
                print("bad move:", e); continue
            print(board); continue
        # treat as FEN or empty = score current
        if line.count("/") >= 7:
            try:
                board = chess.Board(line)
            except Exception as e:
                print("bad fen:", e); continue
        rows, wdl = top_moves(model, board, device)
        print(board)
        print(f"side={'w' if board.turn else 'b'}  WDL w={wdl['win']:.3f} d={wdl['draw']:.3f} l={wdl['loss']:.3f}")
        for i, (u, p) in enumerate(rows, 1):
            print(f"  {i}. {u:6s} {p*100:5.1f}%")
        best, _ = get_model_move(model, board, device)
        print(f"argmax → {best.uci()}")

if __name__ == "__main__":
    main()
