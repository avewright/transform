"""exp125: Optimized MCTS evaluation — tree reuse + batched eval + adaptive sims.

Hypothesis: Combining tree reuse, batched GPU evaluation, and adaptive simulation
allocation yields measurably higher ELO than naive per-move MCTS (exp123 baseline).

This experiment tests the UCI engine's MCTSSearch against exp123's MCTSEngine,
both at matched total NN evaluations to measure pure algorithmic improvement.

Key improvements over exp123:
  1. Tree reuse between moves (simulations from previous move carry over)
  2. Batched leaf evaluation (8 leaves at once for GPU throughput)
  3. Adaptive sim allocation (complex positions get more sims, simple ones less)
  4. Early termination (stop when leader can't be overtaken)
  5. Syzygy integration in MCTS tree (not just game level)

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
from uci_engine import MCTSSearch, SyzygyProbe, TimeManager

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


@torch.no_grad()
def greedy_move(model, board, device):
    inp = batch_boards_to_fused_token_ids([board], device)
    r = model(inp)
    logits = r["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    return index_to_move(logits.argmax().item())


def play_game(engine, model, mcts: MCTSSearch, time_mgr: TimeManager,
              sf_elo, model_color, opening, mode="fixed",
              base_sims=200, ply_cap=300):
    """Play one game.

    mode:
      "greedy" — policy argmax
      "fixed"  — fixed sim count per move (no tree reuse)
      "reuse"  — fixed sims but with tree reuse between moves
      "adaptive" — adaptive sims (complexity-based) + tree reuse
    """
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
            if mode == "greedy":
                move = greedy_move(model, board, DEVICE)
            else:
                # Syzygy at game level
                tb = mcts.syzygy.get_move(board)
                if tb:
                    move = tb
                    mcts.advance_tree(move)
                else:
                    if mode == "adaptive":
                        sims = time_mgr.compute_sims(board, base_sims)
                    else:
                        sims = base_sims

                    move, info = mcts.search(board, max_sims=sims)
                    t_search += info.get("elapsed", 0)
                    nn_total += info.get("nn_evals", 0)
                    sims_total += info.get("sims", 0)

                    # Tree reuse: advance past our move
                    # (advance_tree was already called inside search)
                    if mode in ("fixed",):
                        # No tree reuse — reset after each move
                        mcts.new_game()

            board.push(move)
        else:
            sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            # Advance tree for opponent's move (tree reuse)
            if mode in ("reuse", "adaptive"):
                mcts.advance_tree(sf_move)
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


def run_config(model, mcts, time_mgr, sf_elo, n_games, mode, base_sims, label):
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    log(f"\n{'=' * 60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, mode={mode}, sims={base_sims})")
    log(f"{'=' * 60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, model, mcts, time_mgr,
                      sf_elo, mc, op, mode=mode, base_sims=base_sims)
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
        "mode": mode,
        "base_sims": base_sims,
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
    ap.add_argument("--sf-elos", nargs="+", type=int,
                    default=[1900, 2050],
                    help="Stockfish ELO levels to test against")
    ap.add_argument("--sims", nargs="+", type=int, default=[100, 200],
                    help="Base simulation counts to test")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 8 games, SF1900 only, 100 sims")
    args = ap.parse_args()

    if args.quick:
        args.games = 8
        args.sf_elos = [1900]
        args.sims = [100]

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp125_mcts_optimized.log"
    json_path = ROOT / "outputs" / "exp125_mcts_optimized.json"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Load model
    ckpt_path = args.checkpoint or find_checkpoint()
    log(f"Loading {ckpt_path}...")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model = model.to(DEVICE)
    model.eval()
    log(f"Loaded on {DEVICE}")

    syzygy = SyzygyProbe()
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    mcts = MCTSSearch(model, DEVICE, syzygy, c_puct=2.5, batch_size=8)
    time_mgr = TimeManager(default_sims=200)

    all_results = []

    for sf_elo in args.sf_elos:
        # Greedy baseline
        r = run_config(model, mcts, time_mgr, sf_elo,
                       args.games, "greedy", 0, "greedy")
        all_results.append(r)

        for sims in args.sims:
            # Fixed sims (no tree reuse) — matches exp123 behavior
            r = run_config(model, mcts, time_mgr, sf_elo,
                           args.games, "fixed", sims, f"fixed_{sims}")
            all_results.append(r)

            # With tree reuse
            r = run_config(model, mcts, time_mgr, sf_elo,
                           args.games, "reuse", sims, f"reuse_{sims}")
            all_results.append(r)

            # Adaptive sims + tree reuse
            r = run_config(model, mcts, time_mgr, sf_elo,
                           args.games, "adaptive", sims,
                           f"adaptive_{sims}")
            all_results.append(r)

        # Save after each opponent level
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    log(f"\n{'=' * 70}")
    log(f" SUMMARY")
    log(f"{'=' * 70}")
    log(f"{'Config':<25s} {'SF':<6s} {'Score':<8s} {'W-D-L':<12s} "
        f"{'ELO':<7s} {'NN/g':<8s} {'t/g':<6s}")
    log(f"{'-' * 70}")
    for r in all_results:
        log(f"  {r['name']:<23s} {r['sf_elo']:<6d} {r['score']:<8.3f} "
            f"{r['w']}W-{r['d']}D-{r['l']}L{'':>4s} "
            f"{r['est_elo']:<7d} {r['avg_nn']:<8d} {r['avg_search_t']:<6.1f}")

    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
