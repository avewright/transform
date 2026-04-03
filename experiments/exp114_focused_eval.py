"""Quick focused eval: Test blend_k10_w30 on both baseline and retrained value head.
Compare against SF1750, SF1900, SF2050 with 32 games each."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.exp113_blend_sweep import (
    init_syzygy, load_model, log, run_strategy_eval,
    strategy_greedy, strategy_blend_batched, DEFAULT_OPENINGS, DEVICE
)

import json
from datetime import datetime

LOG_FILE = ROOT / "outputs" / "elo_eval_exp114_focused.log"
JSON_OUT = ROOT / "outputs" / "elo_eval_exp114_focused.json"

# Monkey-patch the module's LOG_FILE
import experiments.exp113_blend_sweep as sweep_mod
sweep_mod.LOG_FILE = LOG_FILE

SF_ELOS = [1750, 1900, 2050]

def make_blend(top_k, value_weight):
    def fn(model, board, device):
        return strategy_blend_batched(model, board, device,
                                      top_k=top_k, value_weight=value_weight,
                                      repetition_penalty=0.0)
    return fn


def main():
    init_syzygy()

    checkpoints = {
        "baseline": ROOT / "outputs" / "hf_checkpoint" / "best_model.pt",
        "retrained_vh": ROOT / "outputs" / "exp114_value_head" / "best_value_head.pt",
    }

    strategies = {
        "greedy": strategy_greedy,
        "blend_k10_w30": make_blend(10, 0.30),
        "blend_k10_w50": make_blend(10, 0.50),
    }

    all_results = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        log(f"\n{'#'*60}")
        log(f"CHECKPOINT: {ckpt_name} ({ckpt_path})")
        log(f"{'#'*60}")

        model = load_model(ckpt_path)

        for strat_name, strat_fn in strategies.items():
            key = f"{ckpt_name}_{strat_name}"
            result = run_strategy_eval(
                model, strat_fn, key, SF_ELOS, DEFAULT_OPENINGS,
                games_per_opening_per_color=2, use_syzygy=True,
            )
            all_results[key] = result

            with open(JSON_OUT, "w") as f:
                json.dump(all_results, f, indent=2)

    log(f"\n{'='*60}")
    log("FINAL COMPARISON")
    log(f"{'='*60}")

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]["elo_estimate"]["estimated_elo"], reverse=True)
    for name, r in ranked:
        est = r["elo_estimate"]
        scores = " | ".join(f"SF{s['sf_elo']}:{s['score']:.3f}" for s in r["summaries"])
        log(f"  {name:40s} ELO~{est['estimated_elo']:4d}  {scores}")


if __name__ == "__main__":
    main()
