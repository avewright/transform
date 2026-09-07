#!/usr/bin/env python3
"""Search-free head-to-head: two frozen checkpoints, fixed openings, swapped colors.

No book, tablebase, or search during play. Policy argmax after a legal mask.
Internal match Elo only — not FIDE / Lichess / UCI_Elo.

Usage:
  MOVE_VOCAB_VERSION=compact python scripts/match_two_ckpts.py --go \\
    --a outputs/hf100m_lapse_ft/init_from_hf.pt \\
    --b outputs/hf100m_lapse_ft/latest.pt \\
    --out outputs/hf100m_lapse_ft/match_init_vs_latest.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import chess

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_inference import get_model_move, load_checkpoint  # noqa: E402

# Fixed set used for baseline vs checkpoint. Same lines, both colors.
OPENINGS: list[list[str]] = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "g8f6"],
    ["d2d4", "f7f5"],
]


def _apply_opening(board: chess.Board, ucis: list[str]) -> None:
    for uci in ucis:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci} after {board.fen()}")
        board.push(mv)


def _termination(board: chess.Board, *, ply_cap: int) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient"
    if board.can_claim_fifty_moves() or board.is_fifty_moves():
        return "fifty_move"
    if board.is_repetition() or board.can_claim_threefold_repetition():
        return "repetition_draw"
    if board.ply() >= ply_cap:
        return "ply_cap"
    return "unknown"


def _score_for_white(board: chess.Board, *, ply_cap: int) -> tuple[float, str]:
    term = _termination(board, ply_cap=ply_cap)
    if term == "checkmate":
        return (0.0 if board.turn == chess.WHITE else 1.0), term
    return 0.5, term


def play_game(white, black, device, opening: list[str], ply_cap: int) -> dict:
    board = chess.Board()
    _apply_opening(board, opening)
    while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
        model = white if board.turn == chess.WHITE else black
        move, _ = get_model_move(model, board, device, temperature=0.0)
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)
    white_score, term = _score_for_white(board, ply_cap=ply_cap)
    return {
        "opening": " ".join(opening) if opening else "startpos",
        "plies": board.ply(),
        "termination": term,
        "white_score": white_score,
        "fen": board.fen(),
        "pgn": " ".join(m.uci() for m in board.move_stack),
    }


def match_elo(score: float, n: int) -> float | None:
    """Logistic match Elo of B vs A. 0.5 → 0. None if n=0 or score on the rail."""
    if n <= 0 or score <= 0.0 or score >= 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--a", required=True, help="Baseline checkpoint")
    ap.add_argument("--b", required=True, help="Challenger checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--ply-cap", type=int, default=160)
    ap.add_argument("--max-openings", type=int, default=0, help="0 = all fixed openings")
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")

    import torch

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    openings = OPENINGS if args.max_openings <= 0 else OPENINGS[: args.max_openings]
    print(f"loading A={args.a}", flush=True)
    model_a = load_checkpoint(args.a, device)
    print(f"loading B={args.b}", flush=True)
    model_b = load_checkpoint(args.b, device)

    games = []
    t0 = time.time()
    for i, opening in enumerate(openings):
        for a_white in (True, False):
            white, black = (model_a, model_b) if a_white else (model_b, model_a)
            g = play_game(white, black, device, opening, args.ply_cap)
            g["a_color"] = "white" if a_white else "black"
            g["a_score"] = g["white_score"] if a_white else 1.0 - g["white_score"]
            games.append(g)
            print(
                f"  {len(games):2d}/{len(openings)*2} {g['opening']:<16} "
                f"A={g['a_color']:<5} {g['termination']:<16} "
                f"a_score={g['a_score']} plies={g['plies']}",
                flush=True,
            )

    a_pts = sum(g["a_score"] for g in games)
    n = len(games)
    b_pts = n - a_pts
    genuine = [g for g in games if g["termination"] != "ply_cap"]
    ply_caps = n - len(genuine)
    summary = {
        "a": str(Path(args.a).resolve()),
        "b": str(Path(args.b).resolve()),
        "device": str(device),
        "n_games": n,
        "openings": [" ".join(o) if o else "startpos" for o in openings],
        "a_points": a_pts,
        "b_points": b_pts,
        "b_score": b_pts / n if n else None,
        "b_match_elo_vs_a": match_elo(b_pts / n, n) if n else None,
        "ply_cap_games": ply_caps,
        "terminations": {},
        "elapsed_s": time.time() - t0,
        "games": games,
    }
    for g in games:
        summary["terminations"][g["termination"]] = summary["terminations"].get(g["termination"], 0) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "n_games", "a_points", "b_points", "b_score", "b_match_elo_vs_a",
        "ply_cap_games", "terminations", "elapsed_s",
    )}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
