"""exp132: Policy temperature sweep + combined optimization.

MOTIVATION from exp131:
- c_puct=1.0 at 100 sims is WORSE than c_puct=2.5 (0.250 vs 0.688)
- Model's policy is weak (c2c3 as top move from starting position)
- More exploration (higher c_puct) helps because MCTS can correct wrong policy rankings
- But at 200 sims, c_puct=2.5 wastes sims on excessive exploration

The policy temperature (T) applied BEFORE softmax changes how peaked the prior is:
- T < 1: sharpen → top moves get more prior → MCTS visits them more  
- T > 1: flatten → spread out prior → more moves explored
- T = 1: default

For a weak policy, T > 1 might help by giving more chances to correct moves.
For a strong policy, T < 1 concentrates on already-good moves.

Test matrix (8 games each vs SF1900):
  Phase 1 — Temperature sweep at c_puct=2.5 (known best for 100 sims):
    temp_0.5           ultra-sharp policy
    temp_0.75          moderately sharp
    temp_1.0           default (reference = exp125 fixed_100)
    temp_1.5           flattened policy
    temp_2.0           very flat policy

  Phase 2 — Best temp at 200 sims with c_puct=2.0 (slightly lowered for more sims):
    besttemp_200_c2.0  best temp from Phase 1
    besttemp_200_c1.5  best temp, lower c_puct

  Phase 3 — No-noise + best temp at 100 and 200 sims:
    nn_100             no noise + best temp + best c_puct
    nn_200             same at 200 sims
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
               noise_frac=0.25, use_fp16=True, policy_temp=1.0):
    mcts = MCTSSearch(model, DEVICE, syzygy,
                      c_puct=c_puct, batch_size=8,
                      fpu_reduction=fpu_reduction,
                      root_noise_alpha=0.3, root_noise_frac=noise_frac,
                      use_fp16=use_fp16, policy_temp=policy_temp)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0

    log(f"\n{'=' * 60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims, c={c_puct}, "
        f"fpu={fpu_reduction}, noise={noise_frac}, temp={policy_temp})")
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
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all, 1=temp sweep, 2=200 sims, 3=no-noise")
    ap.add_argument("--best-temp", type=float, default=1.0)
    ap.add_argument("--best-cpuct", type=float, default=2.5)
    args = ap.parse_args()

    n_games = args.games
    sf_elo = args.sf_elo

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp132_policy_temp.log"
    json_path = ROOT / "outputs" / "exp132_policy_temp.json"
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

    # Phase 1: Temperature sweep at c_puct=2.5, 100 sims
    if args.phase in (0, 1):
        log("\n" + "=" * 60)
        log("PHASE 1: POLICY TEMPERATURE SWEEP (c=2.5, 100 sims)")
        log("=" * 60)

        for temp in [0.5, 0.75, 1.5, 2.0]:
            r = run_config(model, syzygy, sf_elo, n_games, 100,
                           label=f"temp_{temp}",
                           c_puct=2.5, noise_frac=0.25, policy_temp=temp)
            all_results.append(r)

    # Phase 2: Best temp at 200 sims with lowered c_puct
    if args.phase in (0, 2):
        log("\n" + "=" * 60)
        log(f"PHASE 2: 200 SIMS WITH TEMP={args.best_temp}")
        log("=" * 60)

        for cpuct in [2.0, 1.5]:
            r = run_config(model, syzygy, sf_elo, n_games, 200,
                           label=f"t{args.best_temp}_200_c{cpuct}",
                           c_puct=cpuct, noise_frac=0.25,
                           policy_temp=args.best_temp)
            all_results.append(r)

    # Phase 3: No-noise combos
    if args.phase in (0, 3):
        log("\n" + "=" * 60)
        log(f"PHASE 3: NO NOISE (temp={args.best_temp}, c={args.best_cpuct})")
        log("=" * 60)

        for sims in [100, 200]:
            r = run_config(model, syzygy, sf_elo, n_games, sims,
                           label=f"nn_t{args.best_temp}_c{args.best_cpuct}_{sims}",
                           c_puct=args.best_cpuct, noise_frac=0.0,
                           policy_temp=args.best_temp)
            all_results.append(r)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"{'Config':<35} {'Score':>7} {'W-D-L':>9} {'ELO':>6} {'NN/g':>7}")
    log("-" * 65)
    for r in all_results:
        log(f"{r['name']:<35} {r['score']:>7.3f} "
            f"{r['w']}W-{r['d']}D-{r['l']}L "
            f"{r['est_elo']:>6} {r['avg_nn']:>7}")

    log("\nReferences:")
    log("  exp125 fixed_100 (c=2.5, T=1.0): 0.688 → ELO ~2037")
    log("  exp127 fixed_200 (c=2.5, T=1.0): ~0.50 → ELO ~1900")

    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
