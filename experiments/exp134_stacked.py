"""exp134: Composable MCTS improvements (stacked).

KEY FINDING from exp133: no-noise is WORSE (~0.36 vs 0.69 with noise).
Dirichlet noise HELPS this model because the policy is weak — noise forces
exploration of moves the policy underrates, which MCTS then corrects via value.

Tests composable improvements that KEEP noise:
  1. Inner tree temperature 0.7 (sharpen policy at depth >= 1 for exploitation)
  2. Higher noise (noise=0.35) to test if even MORE noise helps
  3. Combined inner_temp + noise at 200 sims

Config matrix (32 games each vs SF1900 for statistical power):
  A: baseline      c=2.5, 100s, noise=0.25, inner=1.0, fpu=0.25 (confirm exp125)
  B: inner_temp    c=2.5, 100s, noise=0.25, inner=0.7, fpu=0.25
  C: high_noise    c=2.5, 100s, noise=0.35, inner=1.0, fpu=0.25
  D: combo_200     c=2.5, 200s, noise=0.25, inner=0.7, fpu=0.25
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, index_to_move, legal_move_mask, move_to_index
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = None


def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")


def wilson_ci(s, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0, c - m), min(1, c + m)


def elo_diff(score):
    if score <= 0:
        return -400
    if score >= 1:
        return 400
    return -400 * math.log10(1 / score - 1)


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

SF_PATH = None


def resolve_sf():
    global SF_PATH
    if SF_PATH:
        return SF_PATH
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
    ]:
        if p and p.exists() and p.is_file():
            SF_PATH = p
            return p
    raise FileNotFoundError("Stockfish not found")


def play_game(engine, model, mcts: MCTSSearch, sf_elo, model_color,
              opening, sims=100, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    mcts.new_game()
    t_search = 0.0
    nn_total = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            tb = mcts.syzygy.get_move(board)
            if tb:
                move = tb
                mcts.new_game()
            else:
                move, info = mcts.search(board, max_sims=sims)
                t_search += info.get("elapsed", 0)
                nn_total += info.get("nn_evals", 0)
                mcts.new_game()
            board.push(move)
        else:
            sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            board.push(sf_move)

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        sc = 0.5
    elif o.winner == model_color:
        sc = 1.0
    else:
        sc = 0.0

    return {"score": sc, "plies": len(board.move_stack),
            "color": "W" if model_color == chess.WHITE else "B",
            "t_search": t_search, "nn": nn_total}


def run_config(model, syzygy, sf_elo, n_games, sims, label,
               c_puct=2.5, fpu_reduction=0.25,
               noise_frac=0.25, use_fp16=True,
               policy_temp=1.0, inner_temp=1.0, root_widening=0):
    mcts = MCTSSearch(model, DEVICE, syzygy,
                      c_puct=c_puct, batch_size=8,
                      fpu_reduction=fpu_reduction,
                      root_noise_alpha=0.3, root_noise_frac=noise_frac,
                      use_fp16=use_fp16, policy_temp=policy_temp,
                      inner_temp=inner_temp, root_widening=root_widening)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    log(f"\n{'=' * 60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims, c={c_puct}, "
        f"noise={noise_frac}, fpu={fpu_reduction}, "
        f"inner_t={inner_temp}, widen={root_widening})")
    log(f"{'=' * 60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, model, mcts, sf_elo, mc, op, sims=sims)
        el = time.time() - t0
        results.append(r)
        tot += r["score"]
        w = sum(1 for x in results if x["score"] == 1.0)
        d = sum(1 for x in results if x["score"] == 0.5)
        l = sum(1 for x in results if x["score"] == 0.0)
        sc = tot / len(results)
        ci = wilson_ci(tot, len(results))
        nn_s = f" nn={r['nn']}" if r['nn'] > 0 else ""
        rs = "WIN" if r["score"] == 1 else ("DRAW" if r["score"] == 0.5 else "LOSS")
        log(f"  G{i + 1:>3}/{n_games}: {r['color']} {rs} "
            f"({r['plies']}ply {el:.0f}s){nn_s}"
            f" | {sc:.3f} ({w}W-{d}D-{l}L) [{ci[0]:.3f},{ci[1]:.3f}]")

    engine.quit()
    sc = tot / n_games
    w = sum(1 for x in results if x["score"] == 1.0)
    d = sum(1 for x in results if x["score"] == 0.5)
    l = sum(1 for x in results if x["score"] == 0.0)
    ci = wilson_ci(tot, n_games)
    ed = elo_diff(sc)
    avg_nn = sum(r["nn"] for r in results) / n_games

    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo + ed:.0f}")
    log(f"  avg nn={avg_nn:.0f}/g")

    return {
        "name": label, "sf_elo": sf_elo, "games": n_games,
        "score": sc, "w": w, "d": d, "l": l,
        "ci95": list(ci), "elo_diff": round(ed),
        "est_elo": round(sf_elo + ed), "avg_nn": round(avg_nn),
        "sims": sims, "c_puct": c_puct, "fpu_reduction": fpu_reduction,
        "noise_frac": noise_frac, "policy_temp": policy_temp,
        "inner_temp": inner_temp, "root_widening": root_widening,
    }


def find_checkpoint():
    candidates = [
        ROOT / "outputs" / "hf" / "chess-transformer-200m-latest" / "best_model.pt",
        ROOT / "outputs" / "hf_checkpoint" / "best_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("avewright/chess-transformer-200m-latest",
                               "best_model.pt")
    except Exception:
        pass
    raise FileNotFoundError("Checkpoint not found")


CONFIGS = {
    "A": {
        "label": "baseline_noise",
        "sims": 100, "c_puct": 2.5, "noise_frac": 0.25,
        "fpu_reduction": 0.25, "inner_temp": 1.0, "root_widening": 0,
    },
    "B": {
        "label": "inner_temp_07",
        "sims": 100, "c_puct": 2.5, "noise_frac": 0.25,
        "fpu_reduction": 0.25, "inner_temp": 0.7, "root_widening": 0,
    },
    "C": {
        "label": "high_noise_35",
        "sims": 100, "c_puct": 2.5, "noise_frac": 0.35,
        "fpu_reduction": 0.25, "inner_temp": 1.0, "root_widening": 0,
    },
    "D": {
        "label": "combo_200",
        "sims": 200, "c_puct": 2.5, "noise_frac": 0.25,
        "fpu_reduction": 0.25, "inner_temp": 0.7, "root_widening": 0,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--config", type=str, default="all",
                    help="A, B, C, D, or 'all'")
    args = ap.parse_args()

    n_games = args.games
    sf_elo = args.sf_elo

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp134_stacked.log"
    json_path = ROOT / "outputs" / "exp134_stacked.json"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    ckpt_path = args.checkpoint or find_checkpoint()
    log(f"Loading checkpoint: {ckpt_path}")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    log(f"Model loaded on {DEVICE}")

    syzygy = SyzygyProbe()
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    all_results = []

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [c.upper() for c in args.config.split(",")]

    for cfg_key in configs_to_run:
        if cfg_key not in CONFIGS:
            log(f"Unknown config: {cfg_key}")
            continue
        cfg = CONFIGS[cfg_key]
        r = run_config(model, syzygy, sf_elo, n_games, cfg["sims"],
                       label=cfg["label"],
                       c_puct=cfg["c_puct"],
                       noise_frac=cfg["noise_frac"],
                       fpu_reduction=cfg["fpu_reduction"],
                       inner_temp=cfg["inner_temp"],
                       root_widening=cfg["root_widening"])
        all_results.append(r)

        with open(json_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"{'Config':<20} {'Score':>7} {'W-D-L':>12} {'ELO':>6} {'CI95':>18}")
    log("-" * 70)
    for r in all_results:
        log(f"{r['name']:<20} {r['score']:>7.3f} "
            f"{r['w']:>3}W-{r['d']:>2}D-{r['l']:>3}L "
            f"{r['est_elo']:>6} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]")

    log("\nReferences:")
    log("  exp125 c=2.5 100s noise=.25: 0.688 ~2037 (8 games)")

    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
