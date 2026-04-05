"""exp128: MCTS search refinements — noise, FPU, widening.

Based on wiki research findings (puct-and-alphazero-search, chess-engine-architecture):

Hypothesis 1: Dirichlet noise at root HURTS during evaluation play.
              Noise is designed for training diversity, not playing strength.
              At 200 sims, 25% noise wastes ~50 sims on random bad moves.

Hypothesis 2: Lower FPU reduction (0.05-0.10) improves exploitation when sim
              budget is limited. Default 0.25 is too pessimistic about unvisited moves.

Hypothesis 3: Progressive widening (only expand top-K moves initially) at root
              concentrates sims on promising moves. Many positions have 30+ legal
              moves; spending sims on clearly bad moves is wasteful.

Test matrix (8 games each vs SF1900, uses best sim count from exp127):
  Phase 1 — Noise:
    no_noise           root_noise_frac=0.0 (disable Dirichlet noise)
    low_noise          root_noise_frac=0.10 (10% instead of 25%)

  Phase 2 — FPU reduction:
    fpu_005            fpu_reduction=0.05 (exploit harder)
    fpu_010            fpu_reduction=0.10
    fpu_050            fpu_reduction=0.50 (more pessimistic)

  Phase 3 — Progressive widening:
    wide_top10         Only expand top 10 moves at root, add more as visits grow
    wide_sqrt          Expand sqrt(n_legal) moves, add more per PUCT

Value convention: White-absolute
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
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, index_to_move, legal_move_mask, move_to_index
from uci_engine import MCTSNode, MCTSSearch, SyzygyProbe

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


class ProgressiveWidenMCTS(MCTSSearch):
    """MCTSSearch with progressive widening: only expand top-K moves initially."""

    def __init__(self, *args, widen_mode="top10", **kwargs):
        super().__init__(*args, **kwargs)
        self.widen_mode = widen_mode

    def _expand_node(self, node, board):
        """Override: expand with progressive widening."""
        sv = self.syzygy.probe_value(board)
        if sv is not None:
            legal = list(board.legal_moves)
            if legal:
                prior = 1.0 / len(legal)
                for m in legal:
                    node.children[m] = MCTSNode(prior=prior)
            node.is_expanded = True
            return sv

        policy, value = self._batch_evaluate([board])[0]

        # Sort moves by prior probability
        sorted_moves = sorted(policy.items(), key=lambda x: -x[1])
        n_legal = len(sorted_moves)

        if self.widen_mode == "top10":
            k = min(10, n_legal)
        elif self.widen_mode == "sqrt":
            k = min(max(3, int(math.sqrt(n_legal))), n_legal)
        else:
            k = n_legal  # No widening

        # Expand top-K moves, redistribute probability mass
        top_moves = sorted_moves[:k]
        total_prob = sum(p for _, p in top_moves)
        if total_prob > 0:
            for m, p in top_moves:
                node.children[m] = MCTSNode(prior=p / total_prob)
        else:
            # Fallback: expand all with uniform prior
            for m, p in sorted_moves:
                node.children[m] = MCTSNode(prior=1.0 / n_legal)

        # Store remaining moves for later expansion
        node._remaining_moves = sorted_moves[k:]
        node.is_expanded = True
        return value


def play_game(engine, model, mcts, sf_elo, model_color,
              opening, sims=200, ply_cap=300):
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
            tb = mcts.syzygy.get_move(board)
            if tb:
                move = tb
                mcts.new_game()
            else:
                move, info = mcts.search(board, max_sims=sims)
                t_search += info.get("elapsed", 0)
                nn_total += info.get("nn_evals", 0)
                sims_total += info.get("sims", 0)
                mcts.new_game()  # No tree reuse (clean test)
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
        "sims": sims_total,
    }


def run_config(model, syzygy, sf_elo, n_games, sims, label,
               c_puct=2.5, fpu_reduction=0.25,
               noise_alpha=0.3, noise_frac=0.25,
               widen_mode=None):
    """Run a config."""
    if widen_mode:
        mcts = ProgressiveWidenMCTS(
            model, DEVICE, syzygy,
            c_puct=c_puct, batch_size=8,
            fpu_reduction=fpu_reduction,
            root_noise_alpha=noise_alpha,
            root_noise_frac=noise_frac,
            widen_mode=widen_mode)
    else:
        mcts = MCTSSearch(
            model, DEVICE, syzygy,
            c_puct=c_puct, batch_size=8,
            fpu_reduction=fpu_reduction,
            root_noise_alpha=noise_alpha,
            root_noise_frac=noise_frac)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    log(f"\n{'=' * 60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims, c={c_puct}, "
        f"fpu={fpu_reduction}, noise={noise_frac}"
        f"{', widen=' + widen_mode if widen_mode else ''})")
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
        "fpu_reduction": fpu_reduction,
        "noise_frac": noise_frac,
        "widen_mode": widen_mode,
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
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--sims", type=int, default=200,
                    help="Base sim count (use best from exp127)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all, 1=noise, 2=fpu, 3=widening")
    args = ap.parse_args()

    n_games = 8 if args.quick else args.games
    sf_elo = args.sf_elo
    sims = args.sims

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp128_search_refinements.log"
    json_path = ROOT / "outputs" / "exp128_search_refinements.json"
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
    log(f"Model loaded on {DEVICE}, sims={sims}")

    syzygy = SyzygyProbe()
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    all_results = []

    # Phase 1: Noise ablation
    if args.phase in (0, 1):
        log("\n" + "=" * 60)
        log("PHASE 1: DIRICHLET NOISE ABLATION")
        log("=" * 60)

        for noise_frac, label in [(0.0, "no_noise"), (0.10, "low_noise")]:
            r = run_config(model, syzygy, sf_elo, n_games, sims, label,
                           noise_frac=noise_frac)
            all_results.append(r)

    # Phase 2: FPU reduction sweep
    if args.phase in (0, 2):
        log("\n" + "=" * 60)
        log("PHASE 2: FPU REDUCTION SWEEP")
        log("=" * 60)

        for fpu, label in [(0.05, "fpu_005"), (0.10, "fpu_010"), (0.50, "fpu_050")]:
            r = run_config(model, syzygy, sf_elo, n_games, sims, label,
                           fpu_reduction=fpu)
            all_results.append(r)

    # Phase 3: Progressive widening
    if args.phase in (0, 3):
        log("\n" + "=" * 60)
        log("PHASE 3: PROGRESSIVE WIDENING")
        log("=" * 60)

        for wm, label in [("top10", "wide_top10"), ("sqrt", "wide_sqrt")]:
            r = run_config(model, syzygy, sf_elo, n_games, sims, label,
                           widen_mode=wm)
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

    log("\nReference: fixed_100 = 0.688, ELO ~2037 (exp125)")

    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
