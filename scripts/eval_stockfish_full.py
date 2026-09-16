#!/usr/bin/env python3
"""Greedy recurrent policy vs full-strength Stockfish under an explicit budget.

No UCI_Elo handicap. Unfinished games remain unknown, never scored as draws.
This is a budget-specific benchmark, not a claim about unlimited Stockfish.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import chess.engine
import chess.pgn
import torch
from chess_inference import load_checkpoint
from move_vocab import index_to_move
from rl_selfplay.ppo import check_model, decisions


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def full_strength_options(engine, threads=1, hash_mb=128):
    # Require these options to verify that an Elo handicap is actually disabled.
    if "UCI_LimitStrength" not in engine.options or "Skill Level" not in engine.options:
        raise ValueError("Expected Stockfish UCI strength-control options")
    settings = {"UCI_LimitStrength": False, "Skill Level": 20,
                "Threads": threads, "Hash": hash_mb}
    engine.configure(settings)
    if "SyzygyPath" in engine.options:
        engine.configure({"SyzygyPath": ""})
        settings["SyzygyPath"] = ""
    return settings


def paired_score_summary(games, seed=285, bootstrap_samples=2000):
    """Paired-opening bootstrap, with pessimistic/optimistic truncation bounds."""
    if not games:
        raise ValueError("No evaluation games")
    grouped = {}
    for game in games:
        grouped.setdefault(game["pair"], []).append(game["score"])
    if any(len(pair) != 2 for pair in grouped.values()):
        raise ValueError("Need both colors for every opening")
    pairs = list(grouped.values())
    lower = [sum(0. if s is None else s for s in pair) / 2 for pair in pairs]
    upper = [sum(1. if s is None else s for s in pair) / 2 for pair in pairs]
    lo_score, hi_score = sum(lower) / len(pairs), sum(upper) / len(pairs)
    interval = None
    if len(pairs) >= 2:
        rng = random.Random(seed)
        samples_lo, samples_hi = [], []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(len(pairs)) for _ in pairs]
            samples_lo.append(sum(lower[i] for i in indices) / len(pairs))
            samples_hi.append(sum(upper[i] for i in indices) / len(pairs))
        samples_lo.sort(); samples_hi.sort()
        interval = [samples_lo[int(.025 * bootstrap_samples)],
                    samples_hi[min(bootstrap_samples - 1, int(.975 * bootstrap_samples))]]
    scores = [g["score"] for g in games]
    return dict(games=len(games), opening_pairs=len(pairs), wins=scores.count(1.),
                draws=scores.count(.5), losses=scores.count(0.), unfinished=scores.count(None),
                score_bounds=[lo_score, hi_score], paired_bootstrap_interval_95=interval,
                note="Descriptive paired bootstrap, not a sequential promotion test; no automatic superiority claim.")


def play_game(model, engine, opening, actor_white, device, depth, limit, ply_cap, pair):
    board = chess.Board()
    for move in opening:
        board.push_uci(move)
    if board.is_game_over(claim_draw=True) or len(board.move_stack) >= ply_cap:
        raise ValueError("Invalid benchmark opening")
    rng = torch.Generator().manual_seed(285)
    sf_game = object()  # python-chess sends ucinewgame for each game.
    model_seconds, sf_seconds, actor_moves = 0., 0., 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        t0 = time.perf_counter()
        if board.turn == actor_white:
            row = decisions(model, [board], device, depth, 0., rng)[0]
            move = index_to_move(row["action"])
            model_seconds += time.perf_counter() - t0
            actor_moves += 1
        else:
            move = engine.play(board, limit, game=sf_game).move
            sf_seconds += time.perf_counter() - t0
        if move is None or move not in board.legal_moves:
            raise ValueError("Invalid move; benchmark will not substitute a fallback")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    score = None if outcome is None else .5 if outcome.winner is None else float(outcome.winner == actor_white)
    game = chess.pgn.Game.from_board(board)
    game.headers.update(White="PPO model" if actor_white else "Stockfish", Black="Stockfish" if actor_white else "PPO model",
                        Result=board.result(claim_draw=True))
    return dict(pair=pair, actor_white=actor_white, score=score, plies=len(board.move_stack),
                termination=outcome.termination.name if outcome else "PLY_CAP", final_fen=board.fen(),
                model_seconds=model_seconds, stockfish_seconds=sf_seconds, model_moves=actor_moves), str(game)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/stockfish_full_policy.json")
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH"))
    p.add_argument("--openings", type=Path, help="JSON list of unique UCI move lists; overrides protocol openings")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()
    if not a.stockfish:
        p.error("Pin the engine with --stockfish or STOCKFISH_PATH")
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Output must be new or empty")
    cfg = json.loads(a.protocol.read_text())
    openings = json.loads(a.openings.read_text()) if a.openings else cfg["openings"]
    if not openings or len({tuple(x) for x in openings}) != len(openings):
        raise ValueError("Need nonempty unique opening list")
    nodes, seconds = cfg.get("nodes", 0), cfg.get("movetime", 0)
    if (nodes > 0) == (seconds > 0) or min(cfg["threads"], cfg["hash_mb"], cfg["ply_cap"], cfg["depth"]) < 1:
        raise ValueError("Specify exactly one positive engine budget and valid protocol settings")
    limit = chess.engine.Limit(nodes=nodes) if nodes > 0 else chess.engine.Limit(time=seconds)
    device = torch.device(a.device)
    model = load_checkpoint(a.ckpt, device).eval()
    n = check_model(model)
    with chess.engine.SimpleEngine.popen_uci(a.stockfish) as engine:
        options = full_strength_options(engine, cfg["threads"], cfg["hash_mb"])
        protocol = dict(config=cfg, openings=openings, engine_id=engine.id, engine_options=options,
                        stockfish_sha256=sha256(a.stockfish), checkpoint_sha256=sha256(a.ckpt),
                        parameters=n, device=str(device), torch_version=torch.__version__,
                        comparison="Greedy search-free model vs full-strength Stockfish at stated per-move budget; not equal-compute")
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
        games = []
        for pair, opening in enumerate(openings):
            for white in (True, False):
                result, pgn = play_game(model, engine, opening, white, device, cfg["depth"], limit, cfg["ply_cap"], pair)
                games.append(result)
                with (a.out / "games.pgn").open("a") as f:
                    f.write(pgn + "\n\n")
                (a.out / "games.json").write_text(json.dumps(games, indent=2))
                print(json.dumps(result), flush=True)
        report = paired_score_summary(games)
        (a.out / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
