#!/usr/bin/env python3
"""exp281: search-free challenger vs frozen 99M incumbent.

  1. Frozen 99M samples at T=1.0. Student samples at T=0.7. Paired colors.
  2. Train CE only on the student's moves from games the student won.
  3. Repeat until the student beats the incumbent. Eval every ~10 min.

Does not write the public 99M incumbent repo.

  MOVE_VOCAB_VERSION=compact python experiments/exp281_searchfree_selfplay.py --go --smoke
  MOVE_VOCAB_VERSION=compact python experiments/exp281_searchfree_selfplay.py --go

Watch:
  tail -f outputs/exp281_searchfree_selfplay/selfplay.log
  tail -f outputs/exp281_searchfree_selfplay/metrics.jsonl
  python scripts/exp201_loss_server.py --log outputs/exp281_searchfree_selfplay/selfplay.log -p 8091
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_inference import get_model_move, load_checkpoint
from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64, count_parameters
from exp273_puzzle_finetune import pull_99m
from move_vocab import VOCAB_SIZE
from rl_selfplay.config import SelfPlayConfig, searchfree_99m_config
from rl_selfplay.generate import generate_positions
from rl_selfplay.storage import append_dataset, load_positions, save_positions
from rl_selfplay.train import train_on_positions

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp281_searchfree_selfplay"
INCUMBENT = ROOT / "outputs" / "hf_models" / "99m" / "latest.pt"
EXPECTED_99M_PARAMS = 98_971_224
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

LOG_PATH: Path | None = None


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit("Expected compact vocab 1968. Export MOVE_VOCAB_VERSION=compact.")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def save_rl_checkpoint(model, path: Path, step: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    cfg = model.config.to_dict() if hasattr(model, "config") and hasattr(model.config, "to_dict") else meta.get("config")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "arch": "squares64",
        "step": step,
        "meta": meta,
        "note": "exp281 search-free self-play; not the public 99M incumbent",
    }, tmp)
    os.replace(str(tmp), str(path))


def resolve_incumbent(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"checkpoint missing: {p}")
        return p
    if INCUMBENT.exists():
        return INCUMBENT
    return pull_99m(INCUMBENT.parent)


@torch.no_grad()
def eval_vs_incumbent(
    student,
    incumbent,
    n_games: int,
    ply_cap: int,
    student_temp: float = 0.7,
    incumbent_temp: float = 1.0,
) -> dict:
    """Score from the student. Same temps as training games unless overridden."""
    student.eval()
    incumbent.eval()
    scores: list[float] = []
    from rl_selfplay.config import OPENINGS
    from rl_selfplay.searchfree import _play_opening

    for i in range(n_games):
        board = chess.Board()
        _play_opening(board, OPENINGS[i % len(OPENINGS)])
        student_white = i % 2 == 0
        while len(board.move_stack) < ply_cap and not board.is_game_over(claim_draw=True):
            is_student = (board.turn == chess.WHITE) == student_white
            active = student if is_student else incumbent
            temp = student_temp if is_student else incumbent_temp
            move, _ = get_model_move(active, board, DEVICE, temperature=temp)
            if move not in board.legal_moves:
                move = next(iter(board.legal_moves))
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            scores.append(0.5)
        elif outcome.winner == (chess.WHITE if student_white else chess.BLACK):
            scores.append(1.0)
        else:
            scores.append(0.0)
    score = sum(scores) / max(1, len(scores))
    return {"n": n_games, "score": score, "games": scores}


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def cheap_elo_screen(model, iter_idx: int, output_dir: Path) -> dict | None:
    """8+8 greedy games vs SF 1900/2050. Skip if Stockfish is missing."""
    try:
        from rl_selfplay.utils import resolve_stockfish

        sf_path = resolve_stockfish()
    except FileNotFoundError as exc:
        log(f"  elo skip: {exc}")
        return None

    from harness.elo import estimate_elo, play_one_policy, summarize_results

    openings = [
        [],
        ["e2e4", "e7e5"],
        ["d2d4", "d7d5"],
        ["e2e4", "c7c5"],
    ]
    elos = [1900, 2050]
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    summaries = []
    try:
        for elo in elos:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
            games = []
            for i, opening in enumerate(openings):
                for color in (chess.WHITE, chess.BLACK):
                    games.append(play_one_policy(
                        engine, model, get_model_move, DEVICE, elo, color, opening,
                        movetime=0.05, ply_cap=120, use_book=False,
                        get_book_move=None, get_syzygy_move=None,
                    ))
            summaries.append(summarize_results(elo, games))
            log(
                f"  elo screen SF{elo}: {summaries[-1]['score']:.3f} "
                f"({summaries[-1]['games']} games)"
            )
    finally:
        engine.quit()

    estimate = estimate_elo(summaries)
    est = estimate.get("estimated_elo")
    log(f"elo@{iter_idx} estimate={est} rc=0")
    row = {
        "step": iter_idx,
        "elo": est,
        "estimate": estimate,
        "summaries": summaries,
        "rc": 0,
    }
    _append_jsonl(output_dir / "elo_gauntlet.jsonl", row)
    (output_dir / f"iter_{iter_idx:03d}" / "elo.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8",
    )
    return row


def run_iteration(
    model,
    incumbent,
    cfg: SelfPlayConfig,
    iter_idx: int,
    output_dir: Path,
    checkpoint_path: str,
    *,
    total_iters: int,
    due_elo: bool,
) -> dict:
    iter_dir = output_dir / f"iter_{iter_idx:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    n_games = cfg.games_per_iter or cfg.n_games

    log(
        f"--- Iteration {iter_idx}: {n_games} games  "
        f"student T={cfg.sample_temp} vs frozen T={cfg.incumbent_temp} ---"
    )
    t0 = time.time()
    positions, results = generate_positions(
        model, DEVICE, cfg, n_games=n_games, prior_model=incumbent, log_fn=log,
    )
    gen_s = time.time() - t0
    n_win = sum(1 for r in results if r > 0.5)
    n_loss = sum(1 for r in results if r < 0.5)
    n_draw = len(results) - n_win - n_loss
    score = sum(results) / max(1, len(results))
    log(
        f"  generated {len(positions)} train positions in {gen_s:.0f}s "
        f"student W/D/L={n_win}/{n_draw}/{n_loss} score={score:.3f}"
    )
    if score > 0.5:
        log(f"  BEAT incumbent this iter score={score:.3f}")

    meta = {
        "iteration": iter_idx,
        "checkpoint": checkpoint_path,
        "mode": "searchfree",
        "sample_temp": cfg.sample_temp,
        "winner_only": cfg.winner_only,
        "n_games": n_games,
        "n_positions": len(positions),
        "student_score": score,
        "n_student_wins": n_win,
        "n_incumbent_wins": n_loss,
        "n_draws": n_draw,
        "config": cfg.to_dict(),
    }
    save_positions(iter_dir / "data.pt", positions, meta)
    if cfg.dataset_dir:
        shard = append_dataset(Path(cfg.dataset_dir), positions, meta)
        log(f"  dataset += {len(positions)} → {shard}")

    log(f"--- Iteration {iter_idx}: train ({cfg.train_epochs} epoch(s)) ---")
    n_value = model.config.n_value_classes if hasattr(model, "config") else 3
    metrics = train_on_positions(model, positions, DEVICE, cfg, n_value, log_fn=log)

    save_rl_checkpoint(model, iter_dir / "model.pt", iter_idx, meta)
    save_rl_checkpoint(model, output_dir / "latest.pt", iter_idx, meta)
    log(f"  saved {iter_dir / 'model.pt'}")

    eval_info = None
    if cfg.eval_games > 0 and incumbent is not None:
        eval_info = eval_vs_incumbent(
            model, incumbent, cfg.eval_games, min(cfg.ply_cap, 160),
            student_temp=cfg.sample_temp, incumbent_temp=cfg.incumbent_temp,
        )
        log(
            f"  eval vs frozen 99M T={cfg.sample_temp}/{cfg.incumbent_temp}: "
            f"{eval_info['score']:.3f} ({eval_info['n']} games)"
        )
        (iter_dir / "eval.json").write_text(json.dumps(eval_info, indent=2), encoding="utf-8")

    elo_info = None
    if due_elo:
        elo_info = cheap_elo_screen(model, iter_idx, output_dir)
    else:
        log("  elo gauntlet skipped (time gate)")

    pos_s = len(positions) / max(gen_s, 1e-6)
    log(
        f"step {iter_idx}/{total_iters} | loss={metrics['loss']:.4f} "
        f"p={metrics['policy']:.4f} {pos_s:.1f} pos/s"
    )
    _append_jsonl(output_dir / "metrics.jsonl", {
        "iteration": iter_idx,
        "loss": metrics["loss"],
        "policy": metrics["policy"],
        "value": metrics["value"],
        "n_positions": len(positions),
        "n_student_wins": n_win,
        "n_incumbent_wins": n_loss,
        "n_draws": n_draw,
        "student_score": score,
        "incumbent_score": None if eval_info is None else eval_info["score"],
        "estimated_elo": None if elo_info is None else elo_info.get("elo"),
        "gen_s": gen_s,
    })

    return {
        "metrics": metrics,
        "n_positions": len(positions),
        "wbd": [n_win, n_loss, n_draw],
        "student_score": score,
        "eval": eval_info,
        "elo": elo_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="99M latest.pt (default outputs/hf_models/99m/latest.pt)")
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--play-batch", type=int, default=None)
    ap.add_argument("--eval-games", type=int, default=None)
    ap.add_argument(
        "--eval-every-sec", type=int, default=600,
        help="SF Elo gauntlet at most this often (0=off). Default 10 min. Incumbent eval is every iter.",
    )
    ap.add_argument("--elo-every", type=int, default=1, help="Deprecated; use --eval-every-sec. 0=no SF Elo.")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument(
        "--from-incumbent",
        action="store_true",
        help="Re-init student from frozen 99M (ignore latest.pt weights).",
    )
    args = ap.parse_args()
    _assert_compact()

    cfg = searchfree_99m_config(output_dir=args.output_dir)
    if args.iterations is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "iterations": args.iterations})
    if args.games is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "n_games": args.games, "games_per_iter": args.games})
    if args.temp is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "sample_temp": args.temp})
    if args.lr is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "train_lr": args.lr})
    if args.batch_size is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "train_batch_size": args.batch_size})
    if args.play_batch is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "play_batch_size": args.play_batch})
    if args.eval_games is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "eval_games": args.eval_games})
    if args.smoke:
        cfg = SelfPlayConfig(**{
            **cfg.to_dict(),
            "n_games": 2,
            "games_per_iter": 2,
            "play_batch_size": 2,
            "iterations": 1,
            "eval_games": 0,
            "train_epochs": 1,
            "train_batch_size": 8,
            "ply_cap": 40,
        })
        args.elo_every = 0
        args.eval_every_sec = 0

    n = count_parameters(build_squares64(DEFAULT_100M_SQUARES64_CONFIG))
    print(
        f"exp281 search-free self-play  params={n:,} expected={EXPECTED_99M_PARAMS:,}  "
        f"student T={cfg.sample_temp} vs inc T={cfg.incumbent_temp}  device={DEVICE}",
        flush=True,
    )
    if args.pull:
        pull_99m(INCUMBENT.parent)
    if not args.go:
        print(
            f"  games={cfg.n_games} play_bs={cfg.play_batch_size} "
            f"train_bs={cfg.train_batch_size} iters={cfg.iterations}\n"
            f"  output={cfg.output_dir}\n"
            f"  pass --go to run, --smoke for a 2-game check",
            flush=True,
        )
        return

    global LOG_PATH
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = output_dir / "selfplay.log"

    ckpt = resolve_incumbent(args.checkpoint)
    log("=" * 60)
    log(f"exp281 search-free | incumbent={ckpt} device={DEVICE}")
    log(
        f"  student T={cfg.sample_temp} vs frozen T={cfg.incumbent_temp}  "
        f"games={cfg.n_games}  train=student wins only"
    )
    log(f"  incumbent eval every iter; elo gauntlet every {int(args.eval_every_sec)}s")
    log(f"  teacher={TEACHER_REPO}")
    log("=" * 60)

    model = load_checkpoint(ckpt, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  loaded {n_params:,} params")
    if n_params != EXPECTED_99M_PARAMS:
        log(f"  warn: expected {EXPECTED_99M_PARAMS:,}")

    incumbent = load_checkpoint(ckpt, DEVICE)
    incumbent.eval()
    for p in incumbent.parameters():
        p.requires_grad_(False)

    if args.train_only:
        if not args.data:
            ap.error("--train-only requires --data PATH")
        positions, meta = load_positions(Path(args.data))
        log(f"training on {len(positions)} positions from {args.data}")
        n_value = model.config.n_value_classes if hasattr(model, "config") else 3
        train_on_positions(model, positions, DEVICE, cfg, n_value, log_fn=log)
        save_rl_checkpoint(model, output_dir / "latest.pt", 0, meta)
        return

    if args.generate_only:
        positions, results = generate_positions(
            model, DEVICE, cfg, n_games=cfg.games_per_iter or cfg.n_games,
            prior_model=incumbent, log_fn=log,
        )
        save_positions(output_dir / "generated_data.pt", positions, {
            "mode": "searchfree", "n_games": len(results), "checkpoint": str(ckpt),
        })
        log(f"saved {len(positions)} positions")
        return

    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    start_iter = 1
    latest = output_dir / "latest.pt"
    if latest.exists() and not args.smoke:
        prev = torch.load(latest, map_location="cpu", weights_only=False)
        start_iter = int(prev.get("step", 0)) + 1
        log(f"  continuing from iteration {start_iter}")
        if args.from_incumbent:
            log(f"  student reset from frozen incumbent {ckpt}")
        else:
            model = load_checkpoint(latest, DEVICE)

    summary = []
    elo_every_s = max(0, int(args.eval_every_sec))
    elo_enabled = int(args.elo_every) != 0 and elo_every_s > 0
    last_elo_t = time.time() if elo_enabled else 0.0
    end_iter = start_iter + cfg.iterations
    for it in range(start_iter, end_iter):
        due_elo = False
        if elo_enabled:
            wait = elo_every_s - (time.time() - last_elo_t)
            if wait <= 0:
                due_elo = True
            else:
                log(f"  next elo gauntlet in {wait / 60:.1f}m")
        info = run_iteration(
            model, incumbent, cfg, it, output_dir, str(ckpt),
            total_iters=end_iter - 1,
            due_elo=due_elo,
        )
        if due_elo:
            last_elo_t = time.time()
        summary.append(info)

    log("=" * 60)
    log(f"Done. {cfg.iterations} iteration(s).")
    for i, info in enumerate(summary, start=start_iter):
        m = info["metrics"]
        ev = info.get("eval")
        ev_s = f" eval={ev['score']:.3f}" if ev else ""
        log(f"  iter {i}: {info['n_positions']} pos, loss={m['loss']:.4f}{ev_s}")
    log("=" * 60)


if __name__ == "__main__":
    main()
