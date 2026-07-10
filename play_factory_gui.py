#!/usr/bin/env python3
"""Browser GUI to play against ChessTransformer + MCTS (no Flask).

Usage:
    python play_factory_gui.py
    python play_factory_gui.py -c outputs/exp186_finetune_soft/best.pt --sims 256
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_inference import load_checkpoint
from move_vocab import IDX_TO_UCI, index_to_move, legal_move_mask
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
_spec = importlib.util.spec_from_file_location("play_gui_assets", STATIC_DIR / "play_gui_assets.py")
_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_assets)
HTML_PAGE = _assets.HTML_PAGE
PIECE_SVGS = _assets.PIECE_SVGS

MODEL = None
DEVICE = None
SEARCH = None
SIMS = 256
SEARCH_LOCK = threading.Lock()
TITLE = "ChessTransformer 25M + MCTS"

MIME = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


@torch.no_grad()
def policy_topk(board: chess.Board, k: int = 5):
    board_input = batch_boards_to_fused_token_ids([board], DEVICE)
    with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
        result = MODEL(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(DEVICE)
    logits[~mask] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(k, int(mask.sum().item())))
    top_moves = [
        [IDX_TO_UCI[i], f"{p * 100:.1f}%"]
        for i, p in zip(topk.indices.tolist(), topk.values.tolist())
    ]
    wdl_probs = F.softmax(result["value_logits"][0].float(), dim=-1).tolist()
    wdl = {"win": wdl_probs[0], "draw": wdl_probs[1], "loss": wdl_probs[2]}
    return top_moves, wdl


@torch.no_grad()
def model_reply(fen: str) -> dict:
    board = chess.Board(fen)
    if board.is_game_over():
        raise ValueError("game over")

    top_moves, wdl = policy_topk(board)

    if SEARCH is None:
        return {
            "move": top_moves[0][0],
            "top_moves": top_moves,
            "wdl": wdl,
            "search": {"sims": 0, "source": "policy"},
        }

    with SEARCH_LOCK:
        old_frac = SEARCH.root_noise_frac
        SEARCH.root_noise_frac = 0.0
        SEARCH.new_game()
        move, info = SEARCH.search(board, max_sims=SIMS)
        SEARCH.root_noise_frac = old_frac

        visit_top = []
        root = getattr(SEARCH, "root", None)
        if root is not None and root.children:
            total = sum(c.visit_count for c in root.children.values()) or 1
            ranked = sorted(root.children.items(), key=lambda x: -x[1].visit_count)[:5]
            visit_top = [[m.uci(), f"{100.0 * c.visit_count / total:.1f}%"] for m, c in ranked]

    return {
        "move": move.uci(),
        "top_moves": visit_top or top_moves,
        "wdl": wdl,
        "search": {
            "sims": int(info.get("sims", SIMS)),
            "source": info.get("source", "mcts"),
            "nn_evals": int(info.get("nn_evals", 0)),
            "elapsed": round(float(info.get("elapsed", 0.0)), 2),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path in ("/", "/index.html"):
            html = HTML_PAGE.replace("ChessTransformer200M", TITLE).encode("utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if path.startswith("/pieces/") and path.endswith(".svg"):
            piece = path.rsplit("/", 1)[-1].removesuffix(".svg")
            svg = PIECE_SVGS.get(piece)
            if not svg:
                return self._send(404, b"not found", "text/plain")
            return self._send(200, svg.encode("utf-8"), "image/svg+xml")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            fp = (STATIC_DIR / rel).resolve()
            if not str(fp).startswith(str(STATIC_DIR.resolve())) or not fp.is_file():
                return self._send(404, b"not found", "text/plain")
            data = fp.read_bytes()
            ctype = MIME.get(fp.suffix.lower(), "application/octet-stream")
            return self._send(200, data, ctype)
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/move":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
            fen = data.get("fen")
            if not fen:
                raise ValueError("missing fen")
            out = model_reply(fen)
            body = json.dumps(out).encode("utf-8")
            return self._send(200, body, "application/json")
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode("utf-8")
            return self._send(400, body, "application/json")


def main():
    global MODEL, DEVICE, SEARCH, SIMS, TITLE
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default="outputs/exp186_finetune_soft/best.pt")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("--sims", type=int, default=256, help="MCTS simulations per move")
    parser.add_argument("--policy-only", action="store_true", help="Disable MCTS (raw argmax)")
    args = parser.parse_args()

    SIMS = args.sims
    DEVICE = torch.device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt

    print(f"Loading {ckpt} ...", flush=True)
    MODEL = load_checkpoint(ckpt, device=DEVICE)
    n = sum(p.numel() for p in MODEL.parameters()) / 1e6

    if args.policy_only:
        TITLE = "ChessTransformer 25M (policy-only)"
        print(f"Ready on {DEVICE} ({n:.1f}M) POLICY-ONLY. You play White.", flush=True)
    else:
        TITLE = f"ChessTransformer 25M + MCTS ({SIMS} sims)"
        SEARCH = MCTSSearch(
            MODEL, DEVICE, SyzygyProbe(),
            c_puct=2.5, batch_size=16,
            root_noise_frac=0.0,  # exploit vs human
        )
        print(f"Ready on {DEVICE} ({n:.1f}M) MCTS sims={SIMS}. You play White.", flush=True)
        # Warmup one search so first move isn't cold
        try:
            SEARCH.search(chess.Board(), max_sims=min(32, SIMS))
            SEARCH.new_game()
            print("MCTS warmup done.", flush=True)
        except Exception as e:
            print(f"warmup warn: {e}", flush=True)

    print(f"\n  Open http://localhost:{args.port}\n", flush=True)
    # Single-threaded: MCTS isn't re-entrant across requests cleanly enough
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye", flush=True)


if __name__ == "__main__":
    main()
