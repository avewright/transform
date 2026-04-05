"""exp138: MCGS (Monte-Carlo Graph Search) transposition test.

HYPOTHESIS: Sharing nodes for identical positions across the search DAG
(Czech et al. 2020) will improve ELO at the same simulation budget by
avoiding redundant evaluations.

This is a ZERO-TRAINING improvement — uses existing best checkpoint.

Test: MCGS vs tree-only at 100 sims, 32 games each vs SF1900.
"""

import argparse
import math
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

import chess
import chess.engine
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["e2e4", "c7c5"],
    ["d2d4", "g8f6"], ["e2e4", "e7e6"], ["c2c4", "e7e5"], ["g1f3", "d7d5"],
]

SHUTDOWN = False
def _sig(s, f):
    global SHUTDOWN
    SHUTDOWN = True
    print("\n[SIGNAL] Shutting down...", flush=True)
signal.signal(signal.SIGINT, _sig)

LOG_FILE = None
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


def wilson_ci(s, n, z=1.96):
    if n <= 0: return 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0, c - m), min(1, c + m)

def elo_diff(score):
    if score <= 0: return -400
    if score >= 1: return 400
    return -400 * math.log10(1 / score - 1)


def find_checkpoint():
    candidates = [
        ROOT / "outputs" / "hf" / "chess-transformer-200m-latest" / "best_model.pt",
        ROOT / "outputs" / "hf_checkpoint" / "best_model.pt",
        ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("avewright/chess-transformer-200m-latest", "best_model.pt"))


def resolve_sf():
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
    ]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Stockfish not found")


def play_game(sf_engine, model, mcts, sf_elo, model_color, opening,
              sims=100, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    mcts.new_game()

    total_tt_hits = 0
    total_nn_evals = 0
    move_count = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            tb = mcts.syzygy.get_move(board)
            if tb:
                move = tb
            else:
                move, info = mcts.search(board, max_sims=sims)
                total_tt_hits += info.get("tt_hits", 0)
                total_nn_evals += info.get("nn_evals", 0)
                move_count += 1
            # Don't reset entire game — keep TT across moves for MCGS benefit
            # Just clear root for new position
            mcts.root = None
            board.push(move)
        else:
            sf_move = sf_engine.play(board, chess.engine.Limit(time=0.05)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            board.push(sf_move)

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        result = 0.5
    else:
        result = 1.0 if o.winner == model_color else 0.0

    return result, {
        "tt_hits": total_tt_hits,
        "nn_evals": total_nn_evals,
        "moves": move_count,
        "tt_size": len(mcts._tt),
        "plies": len(board.move_stack),
    }


def run_config(label, model, sf_elo, n_games, sims, use_transpositions,
               c_puct=2.5, root_noise_frac=0.0):
    syzygy = SyzygyProbe()
    mcts = MCTSSearch(model, DEVICE, syzygy, c_puct=c_puct, batch_size=8,
                      root_noise_alpha=0.3, root_noise_frac=root_noise_frac,
                      use_fp16=True, use_transpositions=use_transpositions)
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    total_score = 0.0
    total_tt_hits = 0
    total_nn_evals = 0

    log(f"\n{'='*60}")
    log(f"Config: {label} | MCGS={use_transpositions} | sims={sims} | c_puct={c_puct}")

    for gi in range(n_games):
        if SHUTDOWN:
            break
        color = chess.WHITE if gi % 2 == 0 else chess.BLACK
        opening = OPENINGS[gi % len(OPENINGS)]
        score, stats = play_game(engine, model, mcts, sf_elo, color, opening,
                                 sims=sims)
        total_score += score
        total_tt_hits += stats["tt_hits"]
        total_nn_evals += stats["nn_evals"]
        avg = total_score / (gi + 1)
        lo, hi = wilson_ci(total_score, gi + 1)
        tag = "W" if score == 1.0 else ("D" if score == 0.5 else "L")
        est_elo = sf_elo + elo_diff(avg)
        log(f"  [{label}] G{gi+1}/{n_games}: {tag} | {avg:.3f} [{lo:.3f},{hi:.3f}] "
            f"~{est_elo:.0f} | tt_hits={stats['tt_hits']} nn={stats['nn_evals']} "
            f"tt_sz={stats['tt_size']} plies={stats['plies']}")

    engine.quit()

    avg_score = total_score / max(n_games, 1)
    return {
        "label": label,
        "score": avg_score,
        "elo": sf_elo + elo_diff(avg_score),
        "ci": list(wilson_ci(total_score, n_games)),
        "n": n_games,
        "total_tt_hits": total_tt_hits,
        "total_nn_evals": total_nn_evals,
        "avg_tt_hits_per_game": total_tt_hits / max(n_games, 1),
        "avg_nn_evals_per_game": total_nn_evals / max(n_games, 1),
    }


def main():
    global LOG_FILE

    ap = argparse.ArgumentParser(description="exp138: MCGS transposition test")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--quick", action="store_true", help="8 games per config")
    args = ap.parse_args()

    n_games = 8 if args.quick else args.games

    out_dir = ROOT / "outputs" / "exp138_mcgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(out_dir / "exp138.log", "w")

    log("=" * 60)
    log(f"exp138: MCGS Transposition Test")
    log(f"  {n_games} games/config vs SF{args.sf_elo}, {args.sims} sims")
    log(f"  device: {DEVICE}")

    # Load model
    ckpt_path = find_checkpoint()
    log(f"  checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = build_model(None)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(DEVICE).eval()
    log("  model loaded")

    results = []

    # Config A: Baseline (no transpositions)
    r = run_config("baseline", model, args.sf_elo, n_games, args.sims,
                   use_transpositions=False, c_puct=2.5)
    results.append(r)
    log(f"\n  [{r['label']}] FINAL: score={r['score']:.3f} elo={r['elo']:.0f} "
        f"nn/g={r['avg_nn_evals_per_game']:.0f}")

    if not SHUTDOWN:
        # Config B: MCGS (with transpositions)
        r = run_config("mcgs", model, args.sf_elo, n_games, args.sims,
                       use_transpositions=True, c_puct=2.5)
        results.append(r)
        log(f"\n  [{r['label']}] FINAL: score={r['score']:.3f} elo={r['elo']:.0f} "
            f"nn/g={r['avg_nn_evals_per_game']:.0f} tt_hits/g={r['avg_tt_hits_per_game']:.0f}")

    if not SHUTDOWN:
        # Config C: MCGS + lower c_puct (1.5) — better exploitation with transpositions
        r = run_config("mcgs_c1.5", model, args.sf_elo, n_games, args.sims,
                       use_transpositions=True, c_puct=1.5)
        results.append(r)
        log(f"\n  [{r['label']}] FINAL: score={r['score']:.3f} elo={r['elo']:.0f} "
            f"nn/g={r['avg_nn_evals_per_game']:.0f} tt_hits/g={r['avg_tt_hits_per_game']:.0f}")

    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY:")
    log(f"{'Config':<15} {'Score':<8} {'ELO':<8} {'CI':<20} {'NN/g':<8} {'TT/g':<8}")
    log("-" * 70)
    for r in results:
        ci_str = f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
        log(f"{r['label']:<15} {r['score']:<8.3f} {r['elo']:<8.0f} {ci_str:<20} "
            f"{r['avg_nn_evals_per_game']:<8.0f} {r['avg_tt_hits_per_game']:<8.0f}")

    import json
    with open(out_dir / "exp138_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if LOG_FILE:
        LOG_FILE.close()


if __name__ == "__main__":
    main()
