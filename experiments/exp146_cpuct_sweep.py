"""exp146: c_puct sweep to find optimal exploration constant.

Codex shows:
- c_puct=2.5 at 100 sims: 2037 ELO (8g) → 1845 (32g)
- c_puct=2.5 at 200 sims: WORSE (exploration grows with sqrt(N))
- c_puct=1.0 at 100 sims: 1763 (catastrophic)
- c_puct=1.25 at 100 sims: ~1650 (catastrophic)
- c_puct=1.5 at 100 sims: NEVER TESTED
- c_puct=2.0 at 100 sims: NEVER TESTED

Hypothesis: c_puct=1.5-2.0 may be the sweet spot. At 200 sims, lower
c_puct prevents exploration explosion that killed exp127.

Test matrix:
  1. c_puct=1.5 at 100 sims
  2. c_puct=2.0 at 100 sims
  3. c_puct=1.5 at 200 sims
  4. c_puct=2.0 at 200 sims
  5. c_puct=2.5 at 100 sims (reference)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from opening_book import get_book_move
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None


def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")


def resolve_sf():
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Stockfish not found")


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


def play_game(engine, mcts, sf_elo, model_color, opening, sims, ply_cap=300):
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
            if tb is not None:
                move = tb
                mcts.new_game()
            elif (bm := get_book_move(board)) is not None:
                move = bm
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

    return {
        "score": sc,
        "plies": len(board.move_stack),
        "color": "W" if model_color == chess.WHITE else "B",
        "t_search": t_search,
        "nn": nn_total,
    }


def run_test(model, device, syzygy, sf_elo, n_games, sims, c_puct, label):
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    mcts = MCTSSearch(
        model, device, syzygy,
        c_puct=c_puct, batch_size=8,
        fpu_reduction=0.25,
        root_noise_alpha=0.3, root_noise_frac=0.0,
        use_fp16=True,
    )

    results = []
    total = 0.0

    log(f"\n{'='*60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims} sims, c_puct={c_puct})")
    log(f"{'='*60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, mcts, sf_elo, mc, op, sims=sims)
        el = time.time() - t0
        results.append(r)
        total += r["score"]
        w = sum(1 for x in results if x["score"] == 1.0)
        d = sum(1 for x in results if x["score"] == 0.5)
        l = sum(1 for x in results if x["score"] == 0.0)
        sc = total / len(results)
        ci = wilson_ci(total, len(results))
        rs = "WIN" if r["score"] == 1 else ("DRAW" if r["score"] == 0.5 else "LOSS")
        log(f"  G{i+1:>3}/{n_games}: {r['color']} {rs} "
            f"({r['plies']}ply {el:.0f}s nn={r['nn']}) | "
            f"{sc:.3f} ({w}W-{d}D-{l}L) [{ci[0]:.3f},{ci[1]:.3f}]")

    engine.quit()
    sc = total / n_games
    w = sum(1 for x in results if x["score"] == 1.0)
    d = sum(1 for x in results if x["score"] == 0.5)
    l = sum(1 for x in results if x["score"] == 0.0)
    ci = wilson_ci(total, n_games)
    ed = elo_diff(sc)
    avg_nn = sum(r["nn"] for r in results) / n_games
    avg_t = sum(r["t_search"] for r in results) / n_games

    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo + ed:.0f}")
    log(f"  avg nn={avg_nn:.0f}/g search_t={avg_t:.1f}s/g")

    return {
        "name": label,
        "sf_elo": sf_elo,
        "games": n_games,
        "sims": sims,
        "c_puct": c_puct,
        "score": sc,
        "w": w, "d": d, "l": l,
        "ci95": list(ci),
        "elo_diff": round(ed),
        "est_elo": round(sf_elo + ed),
        "avg_nn": round(avg_nn),
        "avg_search_t": round(avg_t, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_games = 8 if args.quick else args.games

    global LOG_PATH
    out_dir = ROOT / "outputs" / "exp146_cpuct_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / "eval.log"
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    syzygy = SyzygyProbe()
    log(f"Device: {DEVICE}")
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    ckpt_path = args.checkpoint or str(
        ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt")
    log(f"Loading model from {ckpt_path}...")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    log("Model loaded (204M params)")

    # Test matrix: c_puct × sims combinations
    configs = [
        (1.5, 100, "cpuct1.5_100sim"),
        (2.0, 100, "cpuct2.0_100sim"),
        (1.5, 200, "cpuct1.5_200sim"),
        (2.0, 200, "cpuct2.0_200sim"),
        (2.5, 100, "cpuct2.5_100sim_ref"),
    ]

    all_results = []
    for c_puct, sims, label in configs:
        r = run_test(model, DEVICE, syzygy, args.sf_elo, n_games,
                     sims, c_puct, label)
        all_results.append(r)

    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"{'Config':<25} {'cPUCT':>6} {'Sims':>5} {'Score':>7} {'W-D-L':>9} {'ELO':>6}")
    log("-" * 70)
    for r in all_results:
        log(f"{r['name']:<25} {r['c_puct']:>6.1f} {r['sims']:>5} {r['score']:>7.3f} "
            f"{r['w']}W-{r['d']}D-{r['l']}L "
            f"{r['est_elo']:>6}")

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
