#!/usr/bin/env python3
"""Web GUI to play chess against ChessQwen (exp181 checkpoints).

Usage:
    python play_qwen_gui.py -c outputs/exp181_qwen_unsloth/step_004000
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_qwen_factory import load_chess_qwen_checkpoint, value_logits_to_white_win
from move_vocab import IDX_TO_UCI, index_to_move, legal_move_mask

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

MODEL = None
DEVICE = None

# Reuse play_gui HTML + piece SVGs — you play White, model plays Black.
from play_gui import HTML_PAGE, serve_piece  # noqa: E402


@app.route("/")
def index():
    return HTML_PAGE.replace("ChessTransformer200M", "ChessQwen")


@app.route("/pieces/<piece>.svg")
def pieces(piece):
    return serve_piece(piece)


@app.route("/api/move", methods=["POST"])
@torch.no_grad()
def api_move():
    data = request.get_json(force=True)
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "missing fen"}), 400

    board = chess.Board(fen)
    if board.is_game_over():
        return jsonify({"error": "game over"}), 400

    board_input = batch_boards_to_fused_token_ids([board], DEVICE)
    result = MODEL(board_input)

    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(DEVICE)
    logits[~mask] = float("-inf")

    move_idx = logits.argmax().item()
    move = index_to_move(move_idx)

    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, mask.sum().item()))
    top_moves = [[IDX_TO_UCI[i], f"{p * 100:.1f}%"]
                 for i, p in zip(topk.indices.tolist(), topk.values.tolist())]

    white_win = value_logits_to_white_win(result["value_logits"][0])
    wdl = {"win": white_win, "draw": 0.0, "loss": 1.0 - white_win}

    return jsonify({"move": move.uci(), "top_moves": top_moves, "wdl": wdl})


def main():
    global MODEL, DEVICE

    parser = argparse.ArgumentParser(description="Play vs ChessQwen in browser")
    parser.add_argument(
        "--checkpoint", "-c",
        default="outputs/exp181_qwen_unsloth/step_004000",
        help="Checkpoint directory (contains chess_modules.pt + backbone/)",
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--port", "-p", type=int, default=5000)
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parent / ckpt

    print(f"Loading ChessQwen from {ckpt}...")
    MODEL = load_chess_qwen_checkpoint(ckpt, device=DEVICE)
    print(f"Ready on {DEVICE}. You play White.\n")
    print(f"  Open http://localhost:{args.port} in your browser\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
