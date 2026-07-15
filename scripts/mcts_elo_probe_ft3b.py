#!/usr/bin/env python3
"""MCGS Elo probe + game dumps for pitfall analysis (exp191 FT3b).

Plays vs limited SF with MCTS search; writes PGN + JSON per game so we can
study short losses, color asymmetry, and opening holes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.engine
import chess.pgn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_inference import load_checkpoint  # noqa: E402
from uci_engine import MCTSSearch, SyzygyProbe  # noqa: E402

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


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def wilson_ci(s: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, c - m), min(1.0, c + m)


def elo_diff(score: float) -> float:
    if score <= 0:
        return -400.0
    if score >= 1:
        return 400.0
    return -400.0 * math.log10(1 / score - 1)


def resolve_sf() -> Path:
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
        ROOT / "stockfish" / "stockfish-latest",
    ]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Stockfish not found")


def opening_name(op: list[str]) -> str:
    return "startpos" if not op else " ".join(op)


def play_game(
    engine: chess.engine.SimpleEngine,
    mcts: MCTSSearch,
    *,
    model_color: chess.Color,
    opening: list[str],
    sims: int,
    ply_cap: int,
    sf_movetime: float,
) -> dict:
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    game = chess.pgn.Game()
    game.headers["Event"] = "MCGS Elo probe"
    game.headers["Opening"] = opening_name(opening)
    node = game
    for mv in board.move_stack:
        node = node.add_variation(mv)

    mcts.new_game()
    t_search = nn_total = sims_total = 0.0
    move_infos: list[dict] = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move, info = mcts.search(board, max_sims=sims)
            t_search += float(info.get("elapsed", 0))
            nn_total += int(info.get("nn_evals", 0))
            sims_total += int(info.get("sims", 0))
            move_infos.append(
                {
                    "ply": len(board.move_stack),
                    "uci": move.uci(),
                    "source": info.get("source", "mcts"),
                    "sims": info.get("sims", 0),
                    "score_cp": info.get("score_cp"),
                    "pv": info.get("pv", [])[:6],
                    "elapsed": round(float(info.get("elapsed", 0)), 3),
                }
            )
            # Keep tree reuse off between our moves if SF replies (simple + stable)
            mcts.new_game()
        else:
            move = engine.play(board, chess.engine.Limit(time=sf_movetime)).move
            if move not in board.legal_moves:
                move = next(iter(board.legal_moves))
        board.push(move)
        node = node.add_variation(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
        result = "1/2-1/2"
    elif outcome.winner == model_color:
        score = 1.0
        result = "1-0" if model_color == chess.WHITE else "0-1"
    else:
        score = 0.0
        result = "0-1" if model_color == chess.WHITE else "1-0"

    game.headers["Result"] = result
    game.headers["White"] = "Model" if model_color == chess.WHITE else "SF"
    game.headers["Black"] = "SF" if model_color == chess.WHITE else "Model"

    return {
        "score": score,
        "result": result,
        "plies": len(board.move_stack),
        "termination": outcome.termination.name if outcome else "PLY_CAP",
        "model_color": "white" if model_color == chess.WHITE else "black",
        "opening": opening_name(opening),
        "final_fen": board.fen(),
        "t_search": round(t_search, 2),
        "nn_evals": int(nn_total),
        "sims_total": int(sims_total),
        "pgn": str(game),
        "move_infos": move_infos,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/exp191_soft_ft3b_unseen/best.pt")
    ap.add_argument("--out-dir", default="outputs/exp191_ft3b_mcts_probe")
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--elos", type=int, nargs="+", default=[1750, 1900, 2050])
    ap.add_argument("--games-per-elo", type=int, default=16, help="8 openings × 2 colors")
    ap.add_argument("--sf-movetime", type=float, default=0.05)
    ap.add_argument("--ply-cap", type=int, default=160)
    ap.add_argument("--c-puct", type=float, default=2.5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    if log_path.exists():
        log_path.unlink()

    log(f"checkpoint={args.checkpoint}", log_path)
    log(f"sims={args.sims} c_puct={args.c_puct} elos={args.elos}", log_path)
    model = load_checkpoint(args.checkpoint)
    model.eval()
    log(f"model on {next(model.parameters()).device}", log_path)

    syzygy = SyzygyProbe()
    log(f"syzygy={'yes' if syzygy.available else 'no'}", log_path)

    mcts = MCTSSearch(
        model,
        next(model.parameters()).device,
        syzygy,
        c_puct=args.c_puct,
        batch_size=8,
        fpu_reduction=0.25,
        root_noise_alpha=0.3,
        root_noise_frac=0.0,
        use_fp16=True,
        use_transpositions=True,
    )

    sf = resolve_sf()
    all_games: list[dict] = []
    summaries: list[dict] = []

    for sf_elo in args.elos:
        engine = chess.engine.SimpleEngine.popen_uci(str(sf))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
        log(f"\n=== vs SF{sf_elo} ===", log_path)
        bucket: list[dict] = []
        n = args.games_per_elo
        for i in range(n):
            op = OPENINGS[i % len(OPENINGS)]
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            t0 = time.time()
            g = play_game(
                engine,
                mcts,
                model_color=color,
                opening=op,
                sims=args.sims,
                ply_cap=args.ply_cap,
                sf_movetime=args.sf_movetime,
            )
            g["sf_elo"] = sf_elo
            g["game_idx"] = i
            g["wall_s"] = round(time.time() - t0, 1)
            bucket.append(g)
            all_games.append(g)

            # persist PGN for losses / short games
            tag = f"sf{sf_elo}_g{i:02d}_{g['model_color']}_{g['score']}"
            if g["score"] < 1.0 or g["plies"] < 70:
                (out / "games").mkdir(exist_ok=True)
                (out / "games" / f"{tag}.pgn").write_text(g["pgn"], encoding="utf-8")

            tot = sum(x["score"] for x in bucket)
            sc = tot / len(bucket)
            ci = wilson_ci(tot, len(bucket))
            rs = {1.0: "W", 0.5: "D", 0.0: "L"}[g["score"]]
            log(
                f"  g{i+1:>2}/{n} {g['model_color'][0]} {rs} "
                f"{g['opening'][:20]:<20} ply={g['plies']:<3} "
                f"{g['wall_s']:>5.0f}s | {sc:.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
                f"~{sf_elo + elo_diff(sc):.0f}",
                log_path,
            )

        engine.quit()
        tot = sum(x["score"] for x in bucket)
        sc = tot / len(bucket)
        ci = wilson_ci(tot, len(bucket))
        summ = {
            "sf_elo": sf_elo,
            "games": len(bucket),
            "score": sc,
            "ci95": list(ci),
            "w": sum(1 for x in bucket if x["score"] == 1),
            "d": sum(1 for x in bucket if x["score"] == 0.5),
            "l": sum(1 for x in bucket if x["score"] == 0),
            "est_elo": round(sf_elo + elo_diff(sc)),
            "by_color": {
                c: round(
                    sum(x["score"] for x in bucket if x["model_color"] == c)
                    / max(1, sum(1 for x in bucket if x["model_color"] == c)),
                    3,
                )
                for c in ("white", "black")
            },
        }
        summaries.append(summ)
        log(
            f"  FINAL SF{sf_elo}: {sc:.3f} ({summ['w']}W-{summ['d']}D-{summ['l']}L) "
            f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ~{summ['est_elo']} "
            f"W={summ['by_color']['white']} B={summ['by_color']['black']}",
            log_path,
        )

        # stop climbing if crushed
        if sc < 0.35:
            log(f"score<{0.35} at SF{sf_elo} — stopping ladder", log_path)
            break

    # pitfall digest
    losses = [g for g in all_games if g["score"] == 0.0]
    short = [g for g in losses if g["plies"] < 60]
    by_open: dict[str, list[float]] = defaultdict(list)
    for g in all_games:
        by_open[g["opening"]].append(g["score"])
    worst_open = sorted(
        ((sum(v) / len(v), k, len(v)) for k, v in by_open.items()),
        key=lambda x: x[0],
    )[:5]

    digest = {
        "checkpoint": args.checkpoint,
        "sims": args.sims,
        "summaries": summaries,
        "n_losses": len(losses),
        "n_short_losses": len(short),
        "worst_openings": [
            {"opening": k, "score": round(s, 3), "n": n} for s, k, n in worst_open
        ],
        "short_loss_fens": [
            {
                "sf_elo": g["sf_elo"],
                "color": g["model_color"],
                "opening": g["opening"],
                "plies": g["plies"],
                "term": g["termination"],
                "fen": g["final_fen"],
            }
            for g in short
        ],
    }
    (out / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    # strip bulky pgn from full dump optional — keep move_infos for losses only
    slim = []
    for g in all_games:
        item = {k: v for k, v in g.items() if k != "pgn"}
        if g["score"] == 1.0:
            item.pop("move_infos", None)
        slim.append(item)
    (out / "games.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    log(f"\nwrote {out}/digest.json games.json ({len(all_games)} games)", log_path)
    log("DIGEST " + json.dumps(digest, indent=2), log_path)


if __name__ == "__main__":
    main()
