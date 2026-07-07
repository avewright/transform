"""exp007: Sigma sweep diagnostic.

Hypothesis: The σ value determines whether random perturbations can compete
with the base model. At σ=0.01 (exp006), all challengers lose. Sweeping σ
reveals the competitive threshold.

Design:
  - σ ∈ {0.005, 0.01, 0.02, 0.05, 0.1}
  - For each σ: 2 challengers, 2 games each vs champion
  - Total: 5 × 2 × 2 = 20 games ≈ 9-10 min
  - Measure: win/draw/loss, margin, game terminations
"""

import json
import sys
import time
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, SelfPlayConfig, RandOptConfig
from constrained import build_token_text_map
from model import load_base_model
from randopt import get_perturbable_params, sample_noise
from selfplay import Candidate, play_match

SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.1]
CHALLENGERS_PER_SIGMA = 2
GAMES_PER_MATCH = 2
MAX_MOVES = 80
BOARD_ENCODING = "grid_compact"
ADJUDICATE_MATERIAL = 5
TEMPERATURE = 0.8

OUTPUT_DIR = Path("outputs/exp007_sigma_sweep")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    print(f"Loading model: {cfg.model.name_or_path}")
    model, tokenizer = load_base_model(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dtype = next(model.parameters()).dtype
    perturbable = get_perturbable_params(
        model,
        SelfPlayConfig().perturb_patterns,
        SelfPlayConfig().skip_patterns,
    )
    param_shapes = {name: p.shape for name, p in perturbable.items()}
    n_params = sum(p.numel() for p in perturbable.values())
    print(f"Perturbable: {len(perturbable)} tensors, {n_params:,} params")

    print("Building token-text map...")
    token_texts = build_token_text_map(tokenizer)
    print(f"Token map: {len(token_texts)} usable tokens")

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    results = []
    total_start = time.time()

    for sigma in SIGMAS:
        print(f"\n{'='*50}")
        print(f"  σ = {sigma}")
        print(f"{'='*50}")

        for c_idx in range(CHALLENGERS_PER_SIGMA):
            noise = sample_noise(param_shapes, sigma, device, dtype, generator)

            champion = Candidate(name="champion", noise=None)
            challenger = Candidate(name=f"chall_{c_idx}", noise=noise)

            match_start = time.time()
            score_champ, score_chall, games = play_match(
                model, tokenizer, champion, challenger,
                games=GAMES_PER_MATCH,
                max_moves=MAX_MOVES,
                temperature=TEMPERATURE,
                log_games=True,
                constrained=True,
                token_texts=token_texts,
                board_encoding=BOARD_ENCODING,
                adjudicate_material=ADJUDICATE_MATERIAL,
            )
            match_elapsed = time.time() - match_start

            # Tally W/D/L
            for g_idx, game in enumerate(games):
                champ_is_white = g_idx % 2 == 0
                if game.outcome == "1-0":
                    if champ_is_white:
                        champion.wins += 1; challenger.losses += 1
                    else:
                        challenger.wins += 1; champion.losses += 1
                elif game.outcome == "0-1":
                    if champ_is_white:
                        champion.losses += 1; challenger.wins += 1
                    else:
                        challenger.losses += 1; champion.wins += 1
                else:
                    champion.draws += 1; challenger.draws += 1

            margin = score_chall - score_champ
            result_label = "CHALL WIN" if margin > 0 else ("TIE" if margin == 0 else "CHAMP WIN")

            print(
                f"  σ={sigma} chall_{c_idx}: "
                f"champ={score_champ:.1f} chall={score_chall:.1f} margin={margin:+.1f} "
                f"→ {result_label} | "
                f"W/D/L(ch)={champion.wins}/{champion.draws}/{champion.losses} | "
                f"{match_elapsed:.1f}s"
            )

            game_details = []
            for g_idx, game in enumerate(games):
                game_details.append({
                    "outcome": game.outcome,
                    "termination": game.termination,
                    "num_moves": game.num_moves,
                })

            results.append({
                "sigma": sigma,
                "challenger_idx": c_idx,
                "champion_score": score_champ,
                "challenger_score": score_chall,
                "margin": margin,
                "champion_wdl": [champion.wins, champion.draws, champion.losses],
                "elapsed_s": match_elapsed,
                "games": game_details,
            })

            del noise
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  SIGMA SWEEP SUMMARY  ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")
    print(f"{'σ':>8} | {'Chall':>5} | {'Margin':>7} | {'Result':>10} | Games")
    print("-" * 60)
    for r in results:
        result_label = "CHALL WIN" if r["margin"] > 0 else ("TIE" if r["margin"] == 0 else "CHAMP WIN")
        game_str = ", ".join(
            f"{g['outcome']}({g['termination'][:3]},{g['num_moves']}m)"
            for g in r["games"]
        )
        print(f"{r['sigma']:>8.3f} | {r['challenger_idx']:>5d} | {r['margin']:>+7.1f} | {result_label:>10} | {game_str}")

    # Aggregate by sigma
    print(f"\n{'σ':>8} | {'Avg Margin':>10} | {'Chall Wins':>10} | {'Ties':>5} | {'Champ Wins':>10}")
    print("-" * 60)
    for sigma in SIGMAS:
        sigma_results = [r for r in results if r["sigma"] == sigma]
        avg_margin = sum(r["margin"] for r in sigma_results) / len(sigma_results)
        chall_wins = sum(1 for r in sigma_results if r["margin"] > 0)
        ties = sum(1 for r in sigma_results if r["margin"] == 0)
        champ_wins = sum(1 for r in sigma_results if r["margin"] < 0)
        print(f"{sigma:>8.3f} | {avg_margin:>+10.2f} | {chall_wins:>10d} | {ties:>5d} | {champ_wins:>10d}")

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
