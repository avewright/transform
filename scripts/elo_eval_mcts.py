#!/usr/bin/env python3
"""MCTS Elo gauntlet vs Stockfish UCI_Elo brackets (path-to-2500 Phase 0/3).

Uses production MCTS from uci_engine (PUCT / Gumbel / auto) with MPS/CUDA.

Usage:
  python scripts/elo_eval_mcts.py outputs/hf_437m/best_model.pt hf437m_mcts \
    --sims 200 --elos 1750 1900 2050 --games-per-opening-per-color 1
  python scripts/elo_eval_mcts.py ... --sims 800 --search-mode puct \
    --elos 2200 2350 2500 --games-per-opening-per-color 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.engine
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_inference import load_checkpoint  # noqa: E402
from uci_engine import MCTSSearch, SyzygyProbe, _pick_device  # noqa: E402

OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
]


def log(msg: str, log_path: Path | None = None) -> None:
    print(msg, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def resolve_sf() -> str:
    for c in (
        shutil.which("stockfish"),
        str(ROOT / "stockfish" / "stockfish-native-arm64"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("Stockfish not found")


def play_one(engine, mcts, sf_elo, model_color, opening, sims, movetime, ply_cap):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    mcts.new_game()
    t_search = nn = sims_tot = 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move, info = mcts.search(board, max_sims=sims)
            t_search += info.get("elapsed", 0)
            nn += info.get("nn_evals", 0)
            sims_tot += info.get("sims", 0)
            mcts.new_game()
            if move not in board.legal_moves:
                move = next(iter(board.legal_moves))
            board.push(move)
        else:
            mv = engine.play(board, chess.engine.Limit(time=movetime)).move
            if mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            board.push(mv)
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        score = 0.5
    elif o.winner == model_color:
        score = 1.0
    else:
        score = 0.0
    return {
        "sf_elo": sf_elo,
        "color": "white" if model_color == chess.WHITE else "black",
        "opening": "startpos" if not opening else " ".join(opening),
        "score": score,
        "plies": len(board.move_stack),
        "t_search": t_search,
        "nn": nn,
        "sims": sims_tot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_prefix", nargs="?", default=None)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--elos", type=int, nargs="+", default=[1750, 1900, 2050])
    ap.add_argument("--games-per-opening-per-color", type=int, default=1)
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--ply-cap", type=int, default=160)
    ap.add_argument("--search-mode", choices=("auto", "puct", "gumbel"), default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--syzygy", default=str(ROOT / "syzygy"))
    ap.add_argument("--stop-after-bracket", action="store_true")
    args = ap.parse_args()

    device = _pick_device(args.device)
    prefix = args.out_prefix or Path(args.checkpoint).parent.name + f"_mcts{args.sims}"
    out_json = ROOT / "outputs" / f"elo_eval_{prefix}.json"
    out_log = ROOT / "outputs" / f"elo_eval_{prefix}.log"
    out_log.write_text("")

    def L(msg: str) -> None:
        log(msg, out_log)

    L(f"checkpoint={args.checkpoint} device={device} sims={args.sims} "
      f"mode={args.search_mode} batch={args.batch_size}")
    model = load_checkpoint(args.checkpoint, device)
    model.eval()
    syzygy = SyzygyProbe(args.syzygy if Path(args.syzygy).exists() else None)
    mcts = MCTSSearch(
        model, device, syzygy,
        c_puct=2.5, batch_size=args.batch_size,
        root_noise_frac=0.0,
        search_mode=args.search_mode,
    )

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    summaries = []
    estimated = None
    try:
        for elo in args.elos:
            engine.configure({
                "UCI_LimitStrength": True, "UCI_Elo": elo,
                "Threads": 1, "Hash": 32,
            })
            L(f"begin sf_elo={elo}")
            results = []
            tot = 0.0
            n_rep = args.games_per_opening_per_color
            for op in OPENINGS:
                for color in (chess.WHITE, chess.BLACK):
                    for _ in range(n_rep):
                        t0 = time.time()
                        r = play_one(
                            engine, mcts, elo, color, op,
                            args.sims, args.movetime, args.ply_cap,
                        )
                        results.append(r)
                        tot += r["score"]
                        n = len(results)
                        ci = wilson(tot, n)
                        tag = "W" if r["score"] == 1 else ("D" if r["score"] == 0.5 else "L")
                        L(f"game {json.dumps({**r, 'tag': tag, 'wall': round(time.time()-t0,1)})}")
                        L(f"  running {tot/n:.3f} [{ci[0]:.3f},{ci[1]:.3f}] n={n}")
            score = tot / len(results)
            ci = wilson(tot, len(results))
            summary = {
                "sf_elo": elo,
                "games": len(results),
                "score": score,
                "score_ci95": list(ci),
                "w": sum(1 for r in results if r["score"] == 1),
                "d": sum(1 for r in results if r["score"] == 0.5),
                "l": sum(1 for r in results if r["score"] == 0),
                "avg_nn": sum(r["nn"] for r in results) / len(results),
                "avg_search_t": sum(r["t_search"] for r in results) / len(results),
            }
            summaries.append(summary)
            L(f"summary {json.dumps(summary)}")
            if score >= 0.5:
                estimated = elo
            if args.stop_after_bracket and score < 0.35 and estimated is None:
                L("stop-after-bracket: score too low, stopping climb")
                break
            if args.stop_after_bracket and estimated is not None and score < 0.45:
                break
    finally:
        engine.quit()

    payload = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "sims": args.sims,
        "search_mode": args.search_mode,
        "summaries": summaries,
        "estimated_elo": estimated,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    L(f"estimate {json.dumps({'estimated_elo': estimated})}")
    L(f"wrote {out_json}")


if __name__ == "__main__":
    main()
