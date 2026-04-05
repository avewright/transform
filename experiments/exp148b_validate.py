"""exp148b: 32-game validation of 200-sim PUCT baseline.

Validates the 0.750 (2091 ELO) result from exp148 phase 1 with 32 games.
Also tests 400 sims after the 200-sim phase.

Config: c_puct=2.5, MCGS (transpositions), FP16, batch_size=8, no root noise.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model
from opening_book import get_book_move
from uci_engine import MCTSSearch, SyzygyProbe

import torch

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
    ["g1f3", "g8f6"],
    ["e2e4", "g7g6"],
    ["d2d4", "c7c5"],
    ["c2c4", "c7c5"],
    ["e2e4", "d7d6"],
    ["d2d4", "g7g6"],
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


def play_game(engine, mcts, sf_elo, model_color, opening, sims=200, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    mcts.new_game()
    t_search = 0.0
    nn_total = 0
    sims_total = 0
    tt_hits_total = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move, info = mcts.search(board, max_sims=sims)
            t_search += info.get("elapsed", 0)
            nn_total += info.get("nn_evals", 0)
            sims_total += info.get("sims", 0)
            tt_hits_total += info.get("tt_hits", 0)
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
        "tt_hits": tt_hits_total,
    }


def run_phase(model, syzygy, sf_elo, n_games, sims, label):
    mcts = MCTSSearch(model, DEVICE, syzygy,
                      c_puct=2.5, batch_size=8,
                      fpu_reduction=0.25,
                      root_noise_alpha=0.3, root_noise_frac=0.0,
                      use_fp16=True, use_transpositions=True)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    log(f"\n{'=' * 60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims)")
    log(f"{'=' * 60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, mcts, sf_elo, mc, op, sims=sims)
        el = time.time() - t0
        results.append(r)
        tot += r["score"]
        w = sum(1 for x in results if x["score"] == 1.0)
        d = sum(1 for x in results if x["score"] == 0.5)
        l = sum(1 for x in results if x["score"] == 0.0)
        sc = tot / len(results)
        ci = wilson_ci(tot, len(results))
        ed = elo_diff(sc)
        rs = "WIN" if r["score"] == 1 else ("DRAW" if r["score"] == 0.5 else "LOSS")
        tt_str = f" tt={r['tt_hits']}" if r['tt_hits'] > 0 else ""
        log(f"  G{i + 1:>3}/{n_games}: {r['color']} {rs} "
            f"({r['plies']}ply {el:.0f}s nn={r['nn']}{tt_str})"
            f" | {sc:.3f} ({w}W-{d}D-{l}L) [{ci[0]:.3f},{ci[1]:.3f}] ~{sf_elo + ed:.0f}")

    engine.quit()
    sc = tot / n_games
    w = sum(1 for x in results if x["score"] == 1.0)
    d = sum(1 for x in results if x["score"] == 0.5)
    l = sum(1 for x in results if x["score"] == 0.0)
    ci = wilson_ci(tot, n_games)
    ed = elo_diff(sc)

    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo + ed:.0f}")

    return {"name": label, "score": sc, "w": w, "d": d, "l": l,
            "ci95": list(ci), "est_elo": round(sf_elo + ed), "games": n_games}


def find_checkpoint():
    candidates = [
        ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt",
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
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--also-400", action="store_true",
                    help="Also run 400 sims after the main test")
    args = ap.parse_args()

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp148b_validate.log"
    json_path = ROOT / "outputs" / "exp148b_validate.json"
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

    # Main: 32 games at target sims
    r = run_phase(model, syzygy, args.sf_elo, args.games, args.sims,
                  f"puct_{args.sims}sims")
    all_results.append(r)

    # Optional: 400 sims
    if args.also_400:
        r = run_phase(model, syzygy, args.sf_elo, args.games, 400,
                      "puct_400sims")
        all_results.append(r)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for r in all_results:
        log(f"  {r['name']}: {r['score']:.3f} ({r['w']}W-{r['d']}D-{r['l']}L) "
            f"CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] ELO~{r['est_elo']}")

    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
