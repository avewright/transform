"""exp183: Expert-iteration self-play loop (8GB laptop → A100 scalable).

AlphaZero-style RL without raw REINFORCE:
  1. Play games with MCTS (self, vs SF, or vs frozen prior)
  2. Record visit distributions + root Q at each search position
  3. Fine-tune policy (KL on visits) + value (WDL / HL-Gauss)
  4. Repeat

Presets:
  --preset laptop   RTX 4060 8GB: 4 games, 32 sims, batch 4
  --preset a100     A100 80GB:   64 games, 200 sims, batch 32

Usage:
  python experiments/exp183_selfplay.py --preset laptop --go --smoke
  python experiments/exp183_selfplay.py --preset laptop --go --mode sf
  python experiments/exp183_selfplay.py --preset a100 --go --iterations 5
  python experiments/exp183_selfplay.py --generate-only --games 8 --checkpoint PATH
  python experiments/exp183_selfplay.py --train-only --data outputs/rl_selfplay/iter_001/data.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import chess
import torch

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_inference import load_checkpoint, resolve_checkpoint
from rl_selfplay.config import (
    SelfPlayConfig, a100_80gb_config, a40_45gb_config, laptop_8gb_config,
)
from rl_selfplay.generate import build_mcts, generate_positions
from rl_selfplay.storage import append_dataset, load_positions, save_positions
from rl_selfplay.train import train_on_positions
from rl_selfplay.utils import resolve_stockfish

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH: Path | None = None


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
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict() if hasattr(model, "config") else meta.get("config"),
        "step": step,
        "meta": meta,
    }, tmp)
    os.replace(str(tmp), str(path))


def quick_eval(model, cfg: SelfPlayConfig, checkpoint_path: str) -> float:
    """Short SF match using MCTS for a sanity score."""
    from rl_selfplay.generate import build_mcts, play_sf_game
    import chess.engine

    sf_path = resolve_stockfish()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    if cfg.sf_full_strength:
        engine.configure({"Threads": 2, "Hash": 256})
        label = f"SFfull/d{cfg.sf_depth}"
    else:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": cfg.sf_elo, "Threads": 1})
        label = f"SF{cfg.sf_elo}"
    mcts = build_mcts(model, DEVICE, cfg)
    scores = []
    try:
        for i in range(cfg.eval_games):
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            _pos, res = play_sf_game(mcts, cfg, engine, i, color, log_fn=lambda _: None)
            scores.append(res)
    finally:
        engine.quit()
    score = sum(scores) / max(1, len(scores))
    log(f"  eval vs {label}: {score:.3f} ({cfg.eval_games} games, {cfg.eval_sims} sims)")
    return score


def run_iteration(
    model,
    cfg: SelfPlayConfig,
    iter_idx: int,
    output_dir: Path,
    prior_model=None,
    checkpoint_path: str = "",
) -> dict:
    iter_dir = output_dir / f"iter_{iter_idx:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    data_path = iter_dir / "data.pt"
    n_games = cfg.games_per_iter or cfg.n_games

    log(f"--- Iteration {iter_idx}: generate {n_games} games ({cfg.mode}, {cfg.mcts_sims} sims) ---")
    t0 = time.time()
    positions, results = generate_positions(
        model, DEVICE, cfg, n_games=n_games, prior_model=prior_model, log_fn=log,
    )
    gen_s = time.time() - t0
    avg_result = sum(results) / max(1, len(results))
    n_mcts = sum(1 for p in positions if p.get("source", "mcts") != "sf")
    n_sf = sum(1 for p in positions if p.get("source") == "sf")
    log(f"  generated {len(positions)} positions in {gen_s:.0f}s "
        f"(mcts={n_mcts} sf={n_sf}, {len(positions)/max(gen_s,1):.1f} pos/s), "
        f"avg_result={avg_result:.3f}")

    meta = {
        "iteration": iter_idx,
        "checkpoint": checkpoint_path,
        "mode": cfg.mode,
        "n_games": n_games,
        "mcts_sims": cfg.mcts_sims,
        "sf_full_strength": cfg.sf_full_strength,
        "sf_depth": cfg.sf_depth,
        "n_positions": len(positions),
        "n_mcts": n_mcts,
        "n_sf": n_sf,
        "avg_result": avg_result,
        "config": cfg.to_dict(),
    }
    save_positions(data_path, positions, meta)
    if cfg.dataset_dir:
        shard = append_dataset(Path(cfg.dataset_dir), positions, meta)
        log(f"  dataset += {len(positions)} → {shard} "
            f"(total tracked in {Path(cfg.dataset_dir) / 'manifest.json'})")

    log(f"--- Iteration {iter_idx}: train ({cfg.train_epochs} epoch(s)) ---")
    n_value = model.config.n_value_classes if hasattr(model, "config") else 128
    metrics = train_on_positions(model, positions, DEVICE, cfg, n_value, log_fn=log)

    ckpt_path = iter_dir / "model.pt"
    save_rl_checkpoint(model, ckpt_path, iter_idx, meta)
    latest = output_dir / "latest.pt"
    save_rl_checkpoint(model, latest, iter_idx, meta)
    log(f"  saved {ckpt_path} and {latest}")

    if cfg.eval_games > 0:
        eval_cfg = SelfPlayConfig(**{**cfg.to_dict(), "mcts_sims": cfg.eval_sims})
        quick_eval(model, eval_cfg, checkpoint_path)

    return {"metrics": metrics, "n_positions": len(positions), "data_path": str(data_path)}


def main():
    parser = argparse.ArgumentParser(description="Expert-iteration self-play loop")
    parser.add_argument("--go", action="store_true", help="Run (default is dry-run)")
    parser.add_argument("--preset", choices=["laptop", "a40", "a100"], default="laptop")
    parser.add_argument("--mode", choices=["self", "sf", "prior"], default=None)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--prior-checkpoint", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--sf-elo", type=int, default=None,
                        help="Stockfish UCI_Elo when mode=sf (e.g. 1500–2800)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny 1-game smoke test")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--data", type=str, default=None, help="Path to data.pt for --train-only")
    args = parser.parse_args()

    if args.preset == "a100":
        cfg = a100_80gb_config()
    elif args.preset == "a40":
        cfg = a40_45gb_config()
    else:
        cfg = laptop_8gb_config()
    if args.mode:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "mode": args.mode})
    if args.iterations is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "iterations": args.iterations})
    if args.games is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "n_games": args.games, "games_per_iter": args.games})
    if args.sims is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "mcts_sims": args.sims})
    if args.sf_elo is not None:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "sf_elo": args.sf_elo})
    if args.output_dir:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "output_dir": args.output_dir})
    if args.prior_checkpoint:
        cfg = SelfPlayConfig(**{**cfg.to_dict(), "prior_checkpoint": args.prior_checkpoint, "mode": "prior"})

    if args.smoke:
        cfg = SelfPlayConfig(**{
            **cfg.to_dict(),
            "n_games": 1,
            "games_per_iter": 1,
            "mcts_sims": min(cfg.mcts_sims, 16),
            "iterations": 1,
            "eval_games": 0,
            "train_epochs": 1,
            "ply_cap": 60,
        })

    output_dir = Path(cfg.output_dir)

    if not args.go:
        print("DRY RUN — expert-iteration self-play")
        print(f"  preset: {args.preset}")
        print(f"  mode: {cfg.mode} | games: {cfg.n_games} | sims: {cfg.mcts_sims}")
        print(f"  mcts_batch: {cfg.mcts_batch_size} | train_batch: {cfg.train_batch_size}")
        print(f"  output: {output_dir}")
        print()
        print("  Laptop smoke:  python experiments/exp183_selfplay.py --preset laptop --go --smoke")
        print("  Laptop SF:     python experiments/exp183_selfplay.py --preset laptop --go --mode sf")
        print("  A40 loop:      python experiments/exp183_selfplay.py --preset a40 --go")
        print("  A100 loop:     python experiments/exp183_selfplay.py --preset a100 --go")
        return

    global LOG_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = output_dir / "selfplay.log"

    ckpt_path = str(resolve_checkpoint(args.checkpoint or cfg.checkpoint))
    log("=" * 60)
    log(f"exp183 self-play | preset={args.preset} mode={cfg.mode} device={DEVICE}")
    log(f"  checkpoint: {ckpt_path}")
    log(f"  games={cfg.n_games} sims={cfg.mcts_sims} mcts_bs={cfg.mcts_batch_size}")
    log("=" * 60)

    model = load_checkpoint(ckpt_path, DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"  loaded {n_params:.0f}M params")

    prior_model = None
    if cfg.mode == "prior":
        prior_path = args.prior_checkpoint or cfg.prior_checkpoint or ckpt_path
        log(f"  prior opponent: {prior_path}")
        prior_model = load_checkpoint(prior_path, DEVICE)
        prior_model.eval()

    if args.train_only:
        if not args.data:
            parser.error("--train-only requires --data PATH")
        positions, meta = load_positions(Path(args.data))
        log(f"training on {len(positions)} positions from {args.data}")
        n_value = model.config.n_value_classes if hasattr(model, "config") else 128
        train_on_positions(model, positions, DEVICE, cfg, n_value, log_fn=log)
        save_rl_checkpoint(model, output_dir / "latest.pt", 0, meta)
        return

    if args.generate_only:
        n_games = cfg.games_per_iter or cfg.n_games
        positions, results = generate_positions(
            model, DEVICE, cfg, n_games=n_games, prior_model=prior_model, log_fn=log,
        )
        data_path = output_dir / "generated_data.pt"
        save_positions(data_path, positions, {
            "mode": cfg.mode, "n_games": n_games, "checkpoint": ckpt_path,
        })
        log(f"saved {len(positions)} positions to {data_path}")
        return

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    start_iter = 1
    latest = output_dir / "latest.pt"
    if latest.exists():
        prev = torch.load(latest, map_location="cpu", weights_only=False)
        start_iter = prev.get("step", 0) + 1
        log(f"  continuing from iteration {start_iter}")

    summary = []
    for it in range(start_iter, start_iter + cfg.iterations):
        info = run_iteration(
            model, cfg, it, output_dir, prior_model=prior_model, checkpoint_path=ckpt_path,
        )
        summary.append(info)
        ckpt_path = str(output_dir / "latest.pt")

    log("=" * 60)
    log(f"Done. {cfg.iterations} iteration(s).")
    for i, info in enumerate(summary, start=start_iter):
        m = info["metrics"]
        log(f"  iter {i}: {info['n_positions']} pos, loss={m['loss']:.4f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
