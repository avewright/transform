#!/usr/bin/env python3
"""Web GUI to play chess against ChessTransformer200M.

Usage:
    python play_gui.py                          # defaults
    python play_gui.py --port 8080              # custom port
    python play_gui.py -c path/to/checkpoint.pt # custom checkpoint
    python play_gui.py --device cpu             # force CPU
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response

sys.path.insert(0, str(Path(__file__).resolve().parent))

from play import ChessTransformer200M, load_model, encode_board
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, index_to_move, legal_move_mask

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# Globals set in main()
MODEL = None
DEVICE = None

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Play vs ChessTransformer200M</title>
<link rel="stylesheet" href="/static/chessboard-1.0.0.min.css">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    justify-content: center;
    padding: 20px;
    min-height: 100vh;
  }
  .container {
    display: flex;
    gap: 24px;
    max-width: 1000px;
    align-items: flex-start;
  }
  .board-wrap {
    flex-shrink: 0;
    width: 480px;
  }
  #board { width: 480px; }
  .side-panel {
    width: 280px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .panel-box {
    background: #16213e;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #0f3460;
  }
  h1 {
    font-size: 18px;
    color: #e94560;
    margin-bottom: 8px;
    text-align: center;
  }
  h2 {
    font-size: 14px;
    color: #999;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .status {
    font-size: 16px;
    font-weight: 600;
    padding: 10px;
    text-align: center;
    border-radius: 8px;
    background: #0f3460;
  }
  .status.your-turn { color: #4ecca3; }
  .status.thinking { color: #f0c040; }
  .status.game-over { color: #e94560; }
  .wdl-bar {
    display: flex;
    height: 24px;
    border-radius: 6px;
    overflow: hidden;
    margin: 8px 0;
    font-size: 11px;
    font-weight: 600;
  }
  .wdl-w { background: #f0f0f0; color: #222; display: flex; align-items: center; justify-content: center; }
  .wdl-d { background: #888; color: #fff; display: flex; align-items: center; justify-content: center; }
  .wdl-l { background: #333; color: #ccc; display: flex; align-items: center; justify-content: center; }
  .top-moves {
    list-style: none;
    font-size: 13px;
    font-family: 'Courier New', monospace;
  }
  .top-moves li {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px solid #0f3460;
  }
  .top-moves li:last-child { border-bottom: none; }
  .move-prob { color: #4ecca3; }
  .move-list {
    max-height: 260px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    padding-right: 4px;
  }
  .move-list::-webkit-scrollbar { width: 4px; }
  .move-list::-webkit-scrollbar-thumb { background: #0f3460; border-radius: 2px; }
  .move-num { color: #666; }
  .white-move { color: #f0f0f0; }
  .black-move { color: #aaa; }
  .buttons {
    display: flex;
    gap: 8px;
  }
  button {
    flex: 1;
    padding: 10px;
    border: none;
    border-radius: 8px;
    background: #0f3460;
    color: #e0e0e0;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }
  button:hover { background: #e94560; }
  button:disabled { opacity: 0.4; cursor: default; background: #0f3460; }
  .highlight-move {
    box-shadow: inset 0 0 0 4px rgba(78, 204, 163, 0.7);
  }
  /* chessboard.js piece sizing */
  .board-b72b1 .piece-417db { cursor: grab; }
</style>
</head>
<body>
<div class="container">
  <div class="board-wrap">
    <h1>You vs ChessTransformer200M (204M)</h1>
    <div id="board"></div>
  </div>
  <div class="side-panel">
    <div class="panel-box">
      <div id="status" class="status your-turn">Your turn (White)</div>
    </div>

    <div class="panel-box">
      <h2>Model Evaluation</h2>
      <div class="wdl-bar" id="wdl-bar">
        <div class="wdl-w" id="wdl-w" style="width:33%">33%</div>
        <div class="wdl-d" id="wdl-d" style="width:34%">34%</div>
        <div class="wdl-l" id="wdl-l" style="width:33%">33%</div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#888">
        <span>White</span><span>Draw</span><span>Black</span>
      </div>
    </div>

    <div class="panel-box">
      <h2>Model Top Moves</h2>
      <ul class="top-moves" id="top-moves">
        <li><span>-</span><span class="move-prob">-</span></li>
      </ul>
    </div>

    <div class="panel-box">
      <h2>Moves</h2>
      <div class="move-list" id="move-list"></div>
    </div>

    <div class="buttons">
      <button id="btn-undo" onclick="undoMove()">↩ Undo</button>
      <button id="btn-new" onclick="newGame()">⟳ New Game</button>
    </div>
  </div>
</div>

<script src="/static/jquery-3.7.1.min.js"></script>
<script src="/static/chessboard-1.0.0.min.js"></script>
<script src="/static/chess.min.js"></script>
<script>
const game = new Chess();
let board = null;
let gameOver = false;

function setStatus(text, cls) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status ' + (cls || '');
}

function updateWDL(wdl) {
  if (!wdl) return;
  // wdl is from model's perspective (side to move). Convert to white/draw/black.
  const w = (wdl.win * 100).toFixed(0);
  const d = (wdl.draw * 100).toFixed(0);
  const l = (wdl.loss * 100).toFixed(0);
  document.getElementById('wdl-w').style.width = w + '%';
  document.getElementById('wdl-w').textContent = w > 8 ? w + '%' : '';
  document.getElementById('wdl-d').style.width = d + '%';
  document.getElementById('wdl-d').textContent = d > 8 ? d + '%' : '';
  document.getElementById('wdl-l').style.width = l + '%';
  document.getElementById('wdl-l').textContent = l > 8 ? l + '%' : '';
}

function updateTopMoves(moves) {
  const ul = document.getElementById('top-moves');
  if (!moves || moves.length === 0) {
    ul.innerHTML = '<li><span>-</span><span class="move-prob">-</span></li>';
    return;
  }
  ul.innerHTML = moves.map(m =>
    `<li><span>${m[0]}</span><span class="move-prob">${m[1]}</span></li>`
  ).join('');
}

function updateMoveList() {
  const el = document.getElementById('move-list');
  const history = game.history();
  let html = '';
  for (let i = 0; i < history.length; i += 2) {
    const num = Math.floor(i / 2) + 1;
    html += `<span class="move-num">${num}.</span> `;
    html += `<span class="white-move">${history[i]}</span> `;
    if (i + 1 < history.length) {
      html += `<span class="black-move">${history[i+1]}</span> `;
    }
  }
  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
}

function checkGameOver() {
  if (game.game_over()) {
    gameOver = true;
    let msg = 'Game Over: ';
    if (game.in_checkmate()) {
      msg += game.turn() === 'w' ? 'Black wins!' : 'White wins!';
    } else if (game.in_draw()) {
      msg += 'Draw';
      if (game.in_stalemate()) msg += ' (stalemate)';
      else if (game.in_threefold_repetition()) msg += ' (repetition)';
      else if (game.insufficient_material()) msg += ' (insufficient material)';
      else msg += ' (50-move rule)';
    }
    setStatus(msg, 'game-over');
    return true;
  }
  return false;
}

function highlightMove(from, to) {
  // Remove old highlights
  document.querySelectorAll('.highlight-move').forEach(el => el.classList.remove('highlight-move'));
  // Add new ones
  const sq1 = document.querySelector(`.square-${from}`);
  const sq2 = document.querySelector(`.square-${to}`);
  if (sq1) sq1.classList.add('highlight-move');
  if (sq2) sq2.classList.add('highlight-move');
}

async function requestModelMove() {
  if (gameOver) return;
  setStatus('Model thinking...', 'thinking');

  try {
    const resp = await fetch('/api/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({fen: game.fen()})
    });
    const data = await resp.json();

    if (data.error) {
      setStatus('Error: ' + data.error, 'game-over');
      return;
    }

    const move = game.move({from: data.move.slice(0,2), to: data.move.slice(2,4),
                            promotion: data.move.length > 4 ? data.move[4] : undefined});
    if (!move) {
      setStatus('Model returned illegal move: ' + data.move, 'game-over');
      return;
    }

    board.position(game.fen());
    highlightMove(data.move.slice(0,2), data.move.slice(2,4));
    updateMoveList();
    updateWDL(data.wdl);
    updateTopMoves(data.top_moves);

    if (!checkGameOver()) {
      setStatus('Your turn (White)', 'your-turn');
    }
  } catch (e) {
    setStatus('Network error: ' + e.message, 'game-over');
  }
}

// --- chessboard.js callbacks ---

function onDragStart(source, piece, position, orientation) {
  if (gameOver) return false;
  if (game.turn() !== 'w') return false;  // only white
  if (piece.search(/^b/) !== -1) return false;  // can't move black pieces
}

function onDrop(source, target, piece, newPos, oldPos, orientation) {
  // Check promotion
  let promotion = undefined;
  if (piece === 'wP' && target[1] === '8') {
    promotion = 'q';  // auto-promote to queen
  }

  const move = game.move({from: source, to: target, promotion: promotion});
  if (move === null) return 'snapback';

  highlightMove(source, target);
  updateMoveList();

  if (!checkGameOver()) {
    setTimeout(requestModelMove, 100);
  }
}

function onSnapEnd() {
  board.position(game.fen());
}

function undoMove() {
  if (game.history().length < 2) return;
  game.undo(); // undo model move
  game.undo(); // undo your move
  board.position(game.fen());
  gameOver = false;
  updateMoveList();
  setStatus('Your turn (White)', 'your-turn');
  document.querySelectorAll('.highlight-move').forEach(el => el.classList.remove('highlight-move'));
}

function newGame() {
  game.reset();
  board.start();
  gameOver = false;
  updateMoveList();
  updateWDL({win: 0.33, draw: 0.34, loss: 0.33});
  updateTopMoves(null);
  setStatus('Your turn (White)', 'your-turn');
  document.querySelectorAll('.highlight-move').forEach(el => el.classList.remove('highlight-move'));
}

// Initialize board
board = Chessboard('board', {
  draggable: true,
  position: 'start',
  orientation: 'white',
  pieceTheme: '/pieces/{piece}.svg',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
});

$(window).on('resize', () => board.resize());
</script>
</body>
</html>
"""


PIECE_SVGS = {}

def _init_piece_svgs():
    """Generate simple SVG chess pieces."""
    # Unicode chess symbols rendered as SVG text
    symbols = {
        "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
        "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
    }
    for piece, sym in symbols.items():
        is_white = piece[0] == "w"
        fill = "#fff" if is_white else "#333"
        stroke = "#333" if is_white else "#fff"
        PIECE_SVGS[piece] = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            f'<text x="50" y="78" font-size="80" text-anchor="middle" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" '
            f'font-family="serif">{sym}</text></svg>'
        )

_init_piece_svgs()


@app.route("/pieces/<piece>.svg")
def serve_piece(piece):
    svg = PIECE_SVGS.get(piece)
    if not svg:
        return "Not found", 404
    return Response(svg, mimetype="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json()
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "Missing fen"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    if board.is_game_over():
        return jsonify({"error": "Game is over"}), 400

    board_input = encode_board(board, DEVICE)
    with torch.no_grad():
        result = MODEL(board_input)

    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(DEVICE)
    logits[~mask] = float("-inf")

    move_idx = logits.argmax().item()
    move = index_to_move(move_idx)

    # Top 5
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, mask.sum().item()))
    top_moves = [[IDX_TO_UCI[i], f"{p*100:.1f}%"]
                 for i, p in zip(topk.indices.tolist(), topk.values.tolist())]

    # WDL is White-absolute: idx0=P(W wins), idx1=P(draw), idx2=P(W loses)
    wdl_logits = result["value_logits"][0].float()
    wdl_probs = F.softmax(wdl_logits, dim=-1).tolist()
    wdl = {"win": wdl_probs[0], "draw": wdl_probs[1], "loss": wdl_probs[2]}

    return jsonify({
        "move": move.uci(),
        "top_moves": top_moves,
        "wdl": wdl,
    })


def main():
    global MODEL, DEVICE

    parser = argparse.ArgumentParser(description="Chess GUI — play against ChessTransformer200M")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="outputs/exp073_200m_full_epoch/best_model.pt")
    parser.add_argument("--device", "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", "-p", type=int, default=5000)
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    MODEL = load_model(args.checkpoint, DEVICE)

    print(f"\n  Open http://localhost:{args.port} in your browser\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
