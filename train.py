"""Main training script for chess-transformer.

Supports two modes:
  1. RandOpt: Sample N perturbations, evaluate on Stockfish-labeled positions, ensemble top K
  2. Self-play: Evolve model by playing itself — two perturbed copies compete, winner survives

Usage:
    # === Self-play (new default) ===
    # Quick test:
    python train.py selfplay --generations 10 --games 4 --device cpu

    # Full run, 1v1 mode (champion vs single challenger):
    python train.py selfplay --generations 500 --games 10 --noise-std 0.01

    # Tournament mode (population of 4, round-robin):
    python train.py selfplay --mode tournament --population 4 --generations 100

    # === RandOpt (original mode) ===
    python train.py randopt --n-perturbations 5000 --top-k 16
    python train.py randopt --pgn data/lichess.pgn --stockfish /path/to/stockfish

    # Disable AttnRes in either mode:
    python train.py selfplay --no-attnres --generations 50
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import Config, ModelConfig, AttnResConfig, RandOptConfig, DataConfig, SelfPlayConfig
from data import load_chess_data, tokenize_positions
from model import load_base_model, wrap_with_attnres, save_randopt_results
from randopt import randopt
from evaluate import evaluate_single_model, evaluate_ensemble, print_eval_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chess-Transformer Training")
    sub = p.add_subparsers(dest="command", help="Training mode")

    # ── Shared args (added to both subparsers) ─────────────────
    def add_common_args(parser):
        parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name or path")
        parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--no-attnres", action="store_true", help="Disable AttnRes wrapper")
        parser.add_argument("--block-size", type=int, default=4, help="AttnRes block size")
        parser.add_argument("--head-dim", type=int, default=64, help="AttnRes head dimension")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--output-dir", default="outputs", help="Output directory")

    # ── selfplay subcommand ────────────────────────────────────
    sp = sub.add_parser("selfplay", help="Self-play evolutionary training")
    add_common_args(sp)
    sp.add_argument("--mode", default="1v1", choices=["1v1", "tournament"],
                    help="1v1 (champion vs challenger) or tournament (population round-robin)")
    sp.add_argument("--generations", type=int, default=100, help="Number of evolution generations")
    sp.add_argument("--games", type=int, default=10, help="Games per matchup")
    sp.add_argument("--max-moves", type=int, default=150, help="Max moves per game")
    sp.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std")
    sp.add_argument("--noise-decay", type=float, default=0.999, help="Noise decay per generation")
    sp.add_argument("--noise-floor", type=float, default=0.001, help="Minimum noise std")
    sp.add_argument("--temperature", type=float, default=0.8, help="Move generation temperature")
    sp.add_argument("--population", type=int, default=4, help="Population size (tournament mode)")
    sp.add_argument("--challengers", type=int, default=1,
                    help="Challengers per generation (1v1 mode)")
    sp.add_argument("--save-every", type=int, default=10, help="Checkpoint interval")
    sp.add_argument("--log-games", action="store_true", help="Log full game PGNs")
    sp.add_argument("--no-constrained", action="store_true",
                    help="Disable constrained decoding (allow illegal moves with retry fallback)")
    sp.add_argument("--board-encoding", default="fen",
                    choices=["fen", "grid", "grid_compact", "squares"],
                    help="Board encoding strategy for prompts")
    sp.add_argument("--adjudicate-material", type=int, default=5,
                    help="Material advantage threshold to adjudicate win at max_moves (0=disabled)")

    # ── randopt subcommand ─────────────────────────────────────
    ro = sub.add_parser("randopt", help="RandOpt post-training (original mode)")
    add_common_args(ro)
    ro.add_argument("--n-perturbations", type=int, default=5000)
    ro.add_argument("--top-k", type=int, default=16)
    ro.add_argument("--noise-std", type=float, default=0.01)
    ro.add_argument("--eval-positions", type=int, default=1024)
    ro.add_argument("--eval-batch-size", type=int, default=64)
    ro.add_argument("--pgn", default=None, help="Path to PGN file")
    ro.add_argument("--stockfish", default=None, help="Path to Stockfish binary")
    ro.add_argument("--stockfish-depth", type=int, default=12)
    ro.add_argument("--max-positions", type=int, default=500_000)
    ro.add_argument("--eval-only", default=None, help="Path to saved perturbations for eval-only")

    args = p.parse_args()

    # Default to selfplay if no subcommand given
    if args.command is None:
        p.print_help()
        sys.exit(1)

    return args


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        model=ModelConfig(
            name_or_path=args.model,
            torch_dtype=args.dtype,
        ),
        attnres=AttnResConfig(
            enabled=not args.no_attnres,
            block_size=args.block_size,
            head_dim=args.head_dim,
        ),
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.command == "randopt":
        cfg.randopt = RandOptConfig(
            n_perturbations=args.n_perturbations,
            top_k=args.top_k,
            noise_std=args.noise_std,
            eval_positions=args.eval_positions,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
        )
        cfg.data = DataConfig(
            pgn_path=args.pgn,
            stockfish_path=args.stockfish,
            stockfish_depth=args.stockfish_depth,
            max_positions=args.max_positions,
        )

    return cfg


def build_selfplay_config(args: argparse.Namespace) -> SelfPlayConfig:
    return SelfPlayConfig(
        generations=args.generations,
        games_per_matchup=args.games,
        max_moves=args.max_moves,
        noise_std=args.noise_std,
        noise_decay=args.noise_decay,
        noise_floor=args.noise_floor,
        temperature=args.temperature,
        population_size=args.population,
        challengers_per_gen=args.challengers,
        mode=args.mode,
        seed=args.seed,
        save_every=args.save_every,
        log_games=args.log_games,
        constrained_decoding=not args.no_constrained,
        board_encoding=args.board_encoding,
        adjudicate_material=args.adjudicate_material,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(args.seed)

    if args.command == "selfplay":
        main_selfplay(args, cfg)
    elif args.command == "randopt":
        main_randopt(args, cfg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-play evolutionary mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main_selfplay(args, cfg: Config):
    from selfplay import selfplay_1v1, selfplay_evolve

    sp_cfg = build_selfplay_config(args)

    print("=" * 60)
    print(" Chess-Transformer: Self-Play Evolution")
    print("=" * 60)
    print(f" Model:       {cfg.model.name_or_path}")
    print(f" AttnRes:     {'enabled' if cfg.attnres.enabled else 'disabled'}")
    print(f" Mode:        {sp_cfg.mode}")
    print(f" Generations: {sp_cfg.generations}")
    print(f" Games/match: {sp_cfg.games_per_matchup}")
    print(f" Noise:       σ={sp_cfg.noise_std}, decay={sp_cfg.noise_decay}")
    print(f" Temperature: {sp_cfg.temperature}")
    print(f" Device:      {cfg.device}")
    if sp_cfg.mode == "tournament":
        print(f" Population:  {sp_cfg.population_size}")
    print("=" * 60)

    # Load model
    print("\n[1/2] Loading model...")
    model, tokenizer = load_base_model(cfg)

    if cfg.attnres.enabled:
        model = wrap_with_attnres(model, cfg)

    base_model = model.model if cfg.attnres.enabled else model

    # Run self-play
    print("\n[2/2] Starting self-play evolution...")
    output_dir = Path(cfg.output_dir) / f"selfplay_{sp_cfg.mode}"

    if sp_cfg.mode == "1v1":
        evolved = selfplay_1v1(
            model=base_model,
            tokenizer=tokenizer,
            cfg=sp_cfg,
            output_dir=output_dir,
            device=cfg.device,
        )
    else:
        evolved = selfplay_evolve(
            model=base_model,
            tokenizer=tokenizer,
            cfg=sp_cfg,
            randopt_cfg=cfg.randopt,
            output_dir=output_dir,
            device=cfg.device,
        )

    print(f"\nEvolved model saved to: {output_dir}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RandOpt mode (original)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main_randopt(args, cfg: Config):
    print("=" * 60)
    print(" Chess-Transformer: RandOpt Post-Training")
    print("=" * 60)
    print(f" Model:       {cfg.model.name_or_path}")
    print(f" AttnRes:     {'enabled' if cfg.attnres.enabled else 'disabled'}")
    print(f" RandOpt:     N={cfg.randopt.n_perturbations}, K={cfg.randopt.top_k}, σ={cfg.randopt.noise_std}")
    print(f" Device:      {cfg.device}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading chess data...")
    train_positions, val_positions = load_chess_data(cfg.data)
    print(f"  Train: {len(train_positions)} positions")
    print(f"  Val:   {len(val_positions)} positions")

    # ── Load model ─────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model, tokenizer = load_base_model(cfg)

    if cfg.attnres.enabled:
        model = wrap_with_attnres(model, cfg)

    print("  Tokenizing evaluation data...")
    train_tokenized = tokenize_positions(train_positions[:cfg.randopt.eval_positions], tokenizer)

    # ── Eval-only mode ─────────────────────────────────────────
    if args.eval_only:
        print(f"\n[Eval-only] Loading perturbations from {args.eval_only}")
        from model import load_randopt_results
        top_k = load_randopt_results(args.eval_only, device=cfg.device)

        base_model = model.model if cfg.attnres.enabled else model
        baseline_results = evaluate_single_model(
            base_model, tokenizer, val_positions[:500],
        )
        print_eval_results(baseline_results, "Baseline (no perturbation)")

        ensemble_results = evaluate_ensemble(
            base_model, tokenizer, val_positions[:500], top_k,
        )
        print_eval_results(ensemble_results, f"Ensemble (top {cfg.randopt.top_k})")
        return

    # ── Baseline evaluation ────────────────────────────────────
    print("\n[3/4] Baseline evaluation...")
    base_model = model.model if cfg.attnres.enabled else model
    baseline_results = evaluate_single_model(
        base_model, tokenizer, val_positions[:200],
    )
    print_eval_results(baseline_results, "Baseline (pretrained, no perturbation)")

    # ── RandOpt ────────────────────────────────────────────────
    print("\n[4/4] Running RandOpt post-training...")
    t0 = time.time()
    top_k = randopt(
        model=base_model,
        eval_data=train_tokenized,
        tokenizer=tokenizer,
        cfg=cfg.randopt,
        device=torch.device(cfg.device),
    )
    elapsed = time.time() - t0
    print(f"RandOpt completed in {elapsed:.1f}s")

    # ── Save results ───────────────────────────────────────────
    output_dir = Path(cfg.output_dir) / f"randopt_N{cfg.randopt.n_perturbations}_K{cfg.randopt.top_k}"
    save_randopt_results(top_k, output_dir, cfg)

    # ── Final evaluation ───────────────────────────────────────
    print("\nFinal evaluation on validation set...")

    best = top_k[0]
    from randopt import apply_perturbation, remove_perturbation
    apply_perturbation(base_model, best.noise_vectors)
    best_results = evaluate_single_model(
        base_model, tokenizer, val_positions[:200],
    )
    remove_perturbation(base_model, best.noise_vectors)
    print_eval_results(best_results, "Best single perturbation")

    ensemble_results = evaluate_ensemble(
        base_model, tokenizer, val_positions[:200], top_k[:8],
    )
    print_eval_results(ensemble_results, "Ensemble (top 8, majority vote)")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f" Baseline accuracy:         {baseline_results['accuracy']:.4f}")
    print(f" Best perturbation:         {best_results['accuracy']:.4f}")
    print(f" Ensemble (top 8):          {ensemble_results['accuracy']:.4f}")
    print(f" Baseline legal-move rate:  {baseline_results['legal_rate']:.4f}")
    print(f" Ensemble legal-move rate:  {ensemble_results['legal_rate']:.4f}")
    print(f" Results saved to:          {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
