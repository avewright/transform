"""exp144: Test NNUE-MCTS at high simulation counts.

Hypothesis: NNUE (60x faster than transformer) at 2000-6000 sims should
exceed transformer-MCTS at 100 sims (~1845 ELO), breaking the search ceiling.

The NNUE model provides both policy and value outputs, enabling MCTS with
dramatically more simulations per move within the same wall-clock budget.

Test matrix:
  1. NNUE-MCTS at 1000 sims (10x transformer budget)
  2. NNUE-MCTS at 2000 sims (20x)
  3. NNUE-MCTS at 4000 sims (40x)
  4. Transformer-MCTS at 100 sims (baseline reference)
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

from chess_features import batch_boards_to_fused_token_ids, batch_boards_to_planes
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, move_to_index, legal_move_mask
from nnue_model import NNUEModel, batch_boards_to_halfka_sparse
from opening_book import get_book_move
from uci_engine import MCTSNode, MCTSSearch, SyzygyProbe

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


class NNUEMCTSSearch(MCTSSearch):
    """MCTSSearch that uses NNUE instead of transformer for evaluation.
    
    Overrides _batch_evaluate to use the NNUE model's faster inference.
    """
    
    def __init__(self, nnue_model, device, syzygy, **kwargs):
        # Create a dummy transformer model reference (not used)
        # We override _batch_evaluate entirely
        super().__init__(model=None, device=device, syzygy=syzygy, **kwargs)
        self.nnue = nnue_model
        self.model = None  # Explicitly None — we don't use transformer
    
    @torch.no_grad()
    def _batch_evaluate(self, boards, is_root=False):
        """Evaluate boards using NNUE model."""
        if not boards:
            return []
        
        halfka = batch_boards_to_halfka_sparse(boards, self.device)
        planes = batch_boards_to_planes(boards).to(self.device)
        r = self.nnue(halfka, planes)
        
        temp = self.policy_temp if is_root else self.inner_temp
        
        results = []
        for i, board in enumerate(boards):
            logits = r["policy_logits"][i].float()
            mask = legal_move_mask(board).to(self.device)
            logits[~mask] = float("-inf")
            if temp != 1.0:
                logits = logits / temp
            probs = F.softmax(logits, dim=-1)
            policy = {}
            for m in board.legal_moves:
                idx = move_to_index(m)
                policy[m] = probs[idx].item()
            # Value: White-absolute → side-to-move
            wdl = F.softmax(r["value_logits"][i].float(), dim=-1)
            white_val = (wdl[0] - wdl[2]).item()
            stm_val = white_val if board.turn == chess.WHITE else -white_val
            results.append((policy, stm_val))
        self.nn_evals += len(boards)
        return results


def play_game(engine, mcts, sf_elo, model_color, opening, sims, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    
    t_search = 0.0
    nn_total = 0
    
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            # Syzygy
            tb = mcts.syzygy.get_move(board)
            if tb is not None:
                move = tb
            # Opening book
            elif (bm := get_book_move(board)) is not None:
                move = bm
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


def run_test(mcts, sf_elo, n_games, sims, label):
    """Run n_games against SF and report results."""
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
    
    results = []
    total = 0.0
    
    log(f"\n{'='*60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims} sims)")
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
        "score": sc,
        "w": w, "d": d, "l": l,
        "ci95": list(ci),
        "elo_diff": round(ed),
        "est_elo": round(sf_elo + ed),
        "avg_nn": round(avg_nn),
        "avg_search_t": round(avg_t, 1),
        "sims": sims,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnue-checkpoint", default=None,
                    help="Path to NNUE distilled model")
    ap.add_argument("--transformer-checkpoint", default=None,
                    help="Path to transformer model (for baseline)")
    ap.add_argument("--games", type=int, default=16,
                    help="Games per configuration")
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--quick", action="store_true",
                    help="Quick test: 8 games, fewer sim counts")
    ap.add_argument("--nnue-sims", type=int, nargs="+",
                    default=[1000, 2000, 4000],
                    help="NNUE sim counts to test")
    ap.add_argument("--transformer-sims", type=int, default=100,
                    help="Transformer baseline sim count")
    ap.add_argument("--skip-transformer", action="store_true",
                    help="Skip transformer baseline test")
    args = ap.parse_args()
    
    n_games = 8 if args.quick else args.games
    if args.quick:
        args.nnue_sims = [1000, 2000]
    
    global LOG_PATH
    out_dir = ROOT / "outputs" / "exp144_nnue_mcts"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / "eval.log"
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    
    syzygy = SyzygyProbe()
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")
    
    all_results = []
    
    # Load NNUE
    nnue_path = args.nnue_checkpoint or str(
        ROOT / "outputs" / "exp126_nnue_distill" / "best_nnue.pt")
    log(f"Loading NNUE from {nnue_path}...")
    nnue = NNUEModel(accumulator_size=512, hidden1=32, hidden2=32,
                     policy_channels=32)
    ckpt = torch.load(nnue_path, map_location="cpu", weights_only=False)
    nnue.load_state_dict(ckpt["model_state_dict"])
    nnue = nnue.to(DEVICE).eval()
    log(f"NNUE loaded ({sum(p.numel() for p in nnue.parameters())/1e6:.2f}M params)")
    
    # Test NNUE at each sim count
    for sims in args.nnue_sims:
        mcts = NNUEMCTSSearch(
            nnue, DEVICE, syzygy,
            c_puct=2.5, batch_size=8,
            fpu_reduction=0.25,
            root_noise_alpha=0.3, root_noise_frac=0.0,
            use_fp16=False,  # NNUE doesn't need FP16 (already fast)
        )
        r = run_test(mcts, args.sf_elo, n_games, sims,
                     label=f"nnue_{sims}")
        all_results.append(r)
    
    # Transformer baseline
    if not args.skip_transformer:
        tf_path = args.transformer_checkpoint or str(
            ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt")
        log(f"\nLoading transformer from {tf_path}...")
        transformer = build_model()
        tf_ckpt = torch.load(tf_path, map_location="cpu", weights_only=False)
        sd = tf_ckpt.get("model_state_dict", tf_ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        transformer.load_state_dict(sd)
        transformer = transformer.to(DEVICE).eval()
        
        mcts = MCTSSearch(
            transformer, DEVICE, syzygy,
            c_puct=2.5, batch_size=8,
            fpu_reduction=0.25,
            root_noise_alpha=0.3, root_noise_frac=0.0,
            use_fp16=True,
        )
        r = run_test(mcts, args.sf_elo, n_games, args.transformer_sims,
                     label=f"transformer_{args.transformer_sims}")
        all_results.append(r)
    
    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"{'Config':<25} {'Score':>7} {'W-D-L':>9} {'ELO':>6} {'NN/g':>8} {'t/g':>6}")
    log("-" * 70)
    for r in all_results:
        log(f"{r['name']:<25} {r['score']:>7.3f} "
            f"{r['w']}W-{r['d']}D-{r['l']}L "
            f"{r['est_elo']:>6} {r['avg_nn']:>8} {r['avg_search_t']:>5.1f}s")
    
    # Save JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
