"""exp127: Sim count scaling + cPUCT sweep + tree reuse fix.

Hypothesis 1: Higher sim counts scale roughly log-linearly with ELO.
              fixed_100 = ~2037 → fixed_200 ~2100? fixed_400 ~2150?
Hypothesis 2: cPUCT=2.5 may be suboptimal at 100 sims.  Lower cPUCT (1.5)
              may be better at high sim counts, higher cPUCT (4.0) may help at low.
Hypothesis 3: Tree reuse with visit count decay (0.5-0.75) recovers the
              tree reuse benefit without the stale-visit problem seen in exp125.

Test matrix (each config: 8 games vs SF1900, alternating colors):
  Phase 1 — Sim scaling (most impactful):
    fixed_200          c=2.5, 200 sims, no tree reuse
    fixed_400          c=2.5, 400 sims, no tree reuse

  Phase 2 — cPUCT sweep at 200 sims:
    cpuct_1.5_200      c=1.5, 200 sims, no tree reuse
    cpuct_4.0_200      c=4.0, 200 sims, no tree reuse

  Phase 3 — Tree reuse with decay at 200 sims:
    reuse_d50_200      c=2.5, 200 sims, tree reuse, decay=0.5
    reuse_d75_200      c=2.5, 200 sims, tree reuse, decay=0.75

Value convention: White-absolute
  wdl[0] = P(White wins), wdl[1] = P(draw), wdl[2] = P(White loses)
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
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ]:
        if p and p.exists() and p.is_file():
            SF_PATH = p
            return p
    raise FileNotFoundError("Stockfish not found")


def play_game(engine, model, mcts: MCTSSearch, sf_elo, model_color,
              opening, sims=200, tree_reuse=False, decay=0.0, ply_cap=300):
    """Play one game against Stockfish."""
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    mcts.new_game()
    t_search = 0.0
    nn_total = 0
    sims_total = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            # Syzygy at game level
            tb = mcts.syzygy.get_move(board)
            if tb:
                move = tb
                if tree_reuse:
                    mcts.advance_tree(move, decay=decay)
                else:
                    mcts.new_game()
            else:
                move, info = mcts.search(board, max_sims=sims)
                t_search += info.get("elapsed", 0)
                nn_total += info.get("nn_evals", 0)
                sims_total += info.get("sims", 0)

                if not tree_reuse:
                    mcts.new_game()
            board.push(move)
        else:
            sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            if tree_reuse:
                mcts.advance_tree(sf_move, decay=decay)
            board.push(sf_move)

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        sc = 0.5
    elif o.winner == model_color:
        sc = 1.0
    else:
        sc = 0.0

    return {
        "score": sc,
        "plies": len(board.move_stack),
        "result": board.result(claim_draw=True),
        "color": "W" if model_color == chess.WHITE else "B",
        "t_search": t_search,
        "nn": nn_total,
        "sims": sims_total,
    }


def run_config(model, syzygy, sf_elo, n_games, sims, c_puct,
               tree_reuse, decay, label):
    """Run a test configuration (games against SF)."""
    mcts = MCTSSearch(model, DEVICE, syzygy,
                      c_puct=c_puct, batch_size=8,
                      fpu_reduction=0.25,
                      root_noise_alpha=0.3, root_noise_frac=0.25)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    log(f"\n{'=' * 60}")
    reuse_str = f", reuse+decay={decay}" if tree_reuse else ", no reuse"
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims, c={c_puct}{reuse_str})")
    log(f"{'=' * 60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, model, mcts, sf_elo, mc, op,
                      sims=sims, tree_reuse=tree_reuse, decay=decay)
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
    avg_t = sum(r["t_search"] for r in results) / n_games
    avg_sims = sum(r["sims"] for r in results) / n_games

    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo + ed:.0f}")
    log(f"  avg nn={avg_nn:.0f}/g sims={avg_sims:.0f}/g "
        f"search_t={avg_t:.1f}s/g")

    return {
        "name": label,
        "sf_elo": sf_elo,
        "games": n_games,
        "score": sc,
        "w": w, "d": d, "l": l,
        "ci95": list(ci),
        "elo_diff": round(ed),
        "est_elo": round(sf_elo + ed),
        "avg_nn": round(avg_nn),
        "avg_sims": round(avg_sims),
        "avg_search_t": round(avg_t, 1),
        "sims_per_move": sims,
        "c_puct": c_puct,
        "tree_reuse": tree_reuse,
        "decay": decay,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--games", type=int, default=16,
                    help="Games per configuration")
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 8 games per config")
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all, 1=sim scaling, 2=cpuct sweep, 3=tree reuse fix")
    args = ap.parse_args()

    n_games = 8 if args.quick else args.games
    sf_elo = args.sf_elo

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp127_sim_scaling.log"
    json_path = ROOT / "outputs" / "exp127_sim_scaling.json"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Load model
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

    # Phase 1: Sim count scaling (highest priority)
    if args.phase in (0, 1):
        log("\n" + "=" * 60)
        log("PHASE 1: SIM COUNT SCALING")
        log("=" * 60)

        for sims in [200, 400]:
            r = run_config(model, syzygy, sf_elo, n_games,
                           sims=sims, c_puct=2.5,
                           tree_reuse=False, decay=0.0,
                           label=f"fixed_{sims}")
            all_results.append(r)

            # Early stop: if 200 sims already dominates, skip 400
            # (save compute for more productive tests)

    # Phase 2: cPUCT sweep at 200 sims
    if args.phase in (0, 2):
        log("\n" + "=" * 60)
        log("PHASE 2: cPUCT SWEEP AT 200 SIMS")
        log("=" * 60)

        for cpuct in [1.5, 4.0]:
            r = run_config(model, syzygy, sf_elo, n_games,
                           sims=200, c_puct=cpuct,
                           tree_reuse=False, decay=0.0,
                           label=f"cpuct_{cpuct}_200")
            all_results.append(r)

    # Phase 3: Tree reuse with visit decay at 200 sims
    if args.phase in (0, 3):
        log("\n" + "=" * 60)
        log("PHASE 3: TREE REUSE WITH VISIT DECAY")
        log("=" * 60)

        for decay in [0.5, 0.75]:
            r = run_config(model, syzygy, sf_elo, n_games,
                           sims=200, c_puct=2.5,
                           tree_reuse=True, decay=decay,
                           label=f"reuse_d{int(decay*100)}_200")
            all_results.append(r)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"{'Config':<25} {'Score':>7} {'W-D-L':>9} {'ELO':>6} {'NN/g':>7} {'t/g':>6}")
    log("-" * 60)
    for r in all_results:
        log(f"{r['name']:<25} {r['score']:>7.3f} "
            f"{r['w']}W-{r['d']}D-{r['l']}L "
            f"{r['est_elo']:>6} {r['avg_nn']:>7} {r['avg_search_t']:>5.1f}s")

    # Reference from exp125
    log("\nReference from exp125:")
    log(f"  fixed_100: 0.688 → ELO ~2037 (5W-1D-2L, nn=4582/g, 63.9s/g)")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
