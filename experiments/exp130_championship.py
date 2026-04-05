"""exp130: Championship eval — best search settings at scale.

Takes the winning search configuration from exp127/128/129 and runs a
definitive evaluation with more games and against stronger opponents.

Default best-known config: 200 sims, c_puct=2.5, no noise, FP16.
Override with --config flags based on exp127-129 results.

Test matrix:
  32 games vs SF1900  — confirm ELO with tight CI
  32 games vs SF2050  — bracket upper bound
  16 games vs SF2200  — find ceiling
  8 games vs SF2400   — stretch test

Uses Wilson CI for statistical significance. Stops early if
clearly losing (SPRT-style).

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
    ["e2e4", "c7c6"],
    ["d2d4", "e7e6"],
    ["e2e4", "d7d5"],
    ["g1f3", "g8f6"],
    ["c2c4", "g8f6"],
    ["e2e4", "g7g6"],
    ["d2d4", "c7c5"],
    ["b1c3", "d7d5"],
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
              opening, sims=200, ply_cap=300):
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
        "sims": sims_total,
    }


def run_gauntlet(model, syzygy, sf_elo, n_games, sims, c_puct,
                 fpu_reduction, noise_frac, use_fp16, label):
    mcts = MCTSSearch(model, DEVICE, syzygy,
                      c_puct=c_puct, batch_size=8,
                      fpu_reduction=fpu_reduction,
                      root_noise_alpha=0.3, root_noise_frac=noise_frac,
                      use_fp16=use_fp16)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    prec = "fp16" if use_fp16 else "fp32"
    noise = "noise" if noise_frac > 0 else "no_noise"
    log(f"\n{'=' * 70}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims, c={c_puct}, "
        f"fpu={fpu_reduction}, {noise}, {prec})")
    log(f"{'=' * 70}")

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

        # SPRT-style early stop: if clearly losing after 8+ games
        if len(results) >= 8:
            lo, hi = ci
            if hi < 0.30:  # Upper CI < 0.30 → clearly weaker
                log(f"  EARLY STOP: upper CI {hi:.3f} < 0.30 after {len(results)} games")
                break

    engine.quit()
    sc = tot / len(results)
    n = len(results)
    w = sum(1 for x in results if x["score"] == 1.0)
    d = sum(1 for x in results if x["score"] == 0.5)
    l = sum(1 for x in results if x["score"] == 0.0)
    ci = wilson_ci(tot, n)
    ed = elo_diff(sc)
    avg_nn = sum(r["nn"] for r in results) / n
    avg_t = sum(r["t_search"] for r in results) / n

    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo + ed:.0f}")
    log(f"  avg nn={avg_nn:.0f}/g search_t={avg_t:.1f}s/g")

    return {
        "name": label,
        "sf_elo": sf_elo,
        "games": n,
        "score": sc,
        "w": w, "d": d, "l": l,
        "ci95": list(ci),
        "elo_diff": round(ed),
        "est_elo": round(sf_elo + ed),
        "avg_nn": round(avg_nn),
        "avg_search_t": round(avg_t, 1),
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
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=2.5)
    ap.add_argument("--fpu-reduction", type=float, default=0.25)
    ap.add_argument("--noise-frac", type=float, default=0.0,
                    help="Dirichlet noise fraction (0=disabled for eval)")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--quick", action="store_true",
                    help="Quick: 8g at SF1900 only")
    ap.add_argument("--sf-elos", nargs="+", type=int,
                    default=[1900, 2050, 2200])
    ap.add_argument("--games-per-level", nargs="+", type=int,
                    default=[32, 32, 16])
    args = ap.parse_args()

    if args.quick:
        args.sf_elos = [1900]
        args.games_per_level = [8]

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp130_championship.log"
    json_path = ROOT / "outputs" / "exp130_championship.json"
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

    prec = "FP16" if args.fp16 else "FP32"
    log(f"Model: {DEVICE}, {prec}")
    log(f"Config: sims={args.sims}, c_puct={args.c_puct}, "
        f"fpu={args.fpu_reduction}, noise={args.noise_frac}")

    syzygy = SyzygyProbe()
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    all_results = []

    for sf_elo, n_games in zip(args.sf_elos, args.games_per_level):
        label = f"best_{args.sims}_sf{sf_elo}"
        r = run_gauntlet(model, syzygy, sf_elo, n_games,
                         sims=args.sims, c_puct=args.c_puct,
                         fpu_reduction=args.fpu_reduction,
                         noise_frac=args.noise_frac,
                         use_fp16=args.fp16,
                         label=label)
        all_results.append(r)

    # Summary
    log("\n" + "=" * 70)
    log("CHAMPIONSHIP SUMMARY")
    log("=" * 70)
    log(f"Config: {args.sims} sims, c_puct={args.c_puct}, "
        f"fpu={args.fpu_reduction}, noise={args.noise_frac}, {prec}")
    log(f"\n{'Opponent':<20} {'Score':>7} {'W-D-L':>12} {'ELO':>7} {'CI':>15}")
    log("-" * 70)
    for r in all_results:
        log(f"SF{r['sf_elo']:<16} {r['score']:>7.3f} "
            f"{r['w']:>3}W-{r['d']}D-{r['l']}L "
            f"{r['est_elo']:>7} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]")

    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "sims": args.sims,
                "c_puct": args.c_puct,
                "fpu_reduction": args.fpu_reduction,
                "noise_frac": args.noise_frac,
                "fp16": args.fp16,
            },
            "results": all_results,
        }, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
