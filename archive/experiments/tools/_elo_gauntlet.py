"""Automated MCTS ELO gauntlet — configurable checkpoint, sims, SF ELO.

Usage:
  python experiments/_elo_gauntlet.py                                     # exp100 baseline
  python experiments/_elo_gauntlet.py --checkpoint outputs/exp149_scratch_204m/best_model.pt --sims 100
  python experiments/_elo_gauntlet.py --sims 800 --sf-elo 2100 --games 32
  python experiments/_elo_gauntlet.py --quick                             # 8 games at 100 sims (fast screening)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must set compact vocab BEFORE any chess imports
if '--compact' in sys.argv:
    os.environ['MOVE_VOCAB_VERSION'] = 'compact'
else:
    # Auto-detect from checkpoint
    _ckpt_idx = None
    for i, a in enumerate(sys.argv):
        if a == '--checkpoint' and i + 1 < len(sys.argv):
            _ckpt_idx = i + 1
            break
    if _ckpt_idx:
        import torch as _torch
        try:
            _ck = _torch.load(sys.argv[_ckpt_idx], map_location='cpu', weights_only=False)
            if _ck.get('vocab_version') == 'compact':
                os.environ['MOVE_VOCAB_VERSION'] = 'compact'
                print("Auto-detected compact vocab from checkpoint")
            del _ck
        except Exception:
            pass

import chess
import chess.engine
import torch
from torch import nn

from chess_transformer_factory import build_model, ChessTransformerConfig
from uci_engine import MCTSSearch, SyzygyProbe
from opening_book import get_book_move

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OPENINGS = [
    ['e2e4', 'e7e5'], ['d2d4', 'd7d5'], ['c2c4', 'e7e5'], ['g1f3', 'd7d5'],
    ['e2e4', 'c7c5'], ['d2d4', 'g8f6'], ['e2e4', 'e7e6'], ['d2d4', 'd7d6'],
    ['e2e4', 'c7c6'], ['g1f3', 'g8f6'], ['c2c4', 'c7c5'], ['e2e4', 'g7g6'],
    ['d2d4', 'e7e6'], ['c2c4', 'g8f6'], ['b1c3', 'd7d5'], ['g2g3', 'd7d5'],
]

SF_PATH = ROOT / 'stockfish' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe'


def elo_diff(s):
    if s <= 0: return -400
    if s >= 1: return 400
    return -400 * math.log10(1 / s - 1)


def wilson_ci(score, n, z=1.96):
    p = score / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0, c - m), min(1, c + m)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_model(checkpoint_path, config_path=None):
    """Load model from checkpoint, auto-detecting config."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    # Try to detect config from checkpoint
    if 'config' in ckpt:
        cfg = ChessTransformerConfig(**ckpt['config'])
    elif config_path:
        cfg = ChessTransformerConfig.from_json(config_path)
    else:
        cfg = None  # default

    model = build_model(cfg)

    # Auto-detect distributional value head (128-bin HL-Gauss vs 3-class WDL)
    ckpt_vbias = sd.get('value_head.2.bias')
    if ckpt_vbias is not None and ckpt_vbias.shape[0] != model.value_head[2].out_features:
        n_bins = ckpt_vbias.shape[0]
        old_head = model.value_head
        model.value_head = nn.Sequential(
            old_head[0], old_head[1],
            nn.Linear(old_head[0].out_features, n_bins),
        )

    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model


def run_gauntlet(model, sf_elo, sims, games, c_puct=2.5, use_book=True,
                 dynamic_cpuct=False, batch_size=16, policy_temp=1.0,
                 fpu_reduction=0.25, inner_temp=1.0):
    """Run MCTS gauntlet. Returns summary dict."""
    sf = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo, 'Threads': 1})

    syzygy = SyzygyProbe()
    mcts = MCTSSearch(
        model, DEVICE, syzygy,
        c_puct=c_puct, batch_size=batch_size,
        root_noise_alpha=0.3, root_noise_frac=0.25,
        use_fp16=True, use_transpositions=True,
        dynamic_cpuct=dynamic_cpuct,
        policy_temp=policy_temp,
        fpu_reduction=fpu_reduction,
        inner_temp=inner_temp,
    )

    results = []
    total_score = 0.0

    log(f"Starting {games}-game gauntlet vs SF{sf_elo} @ {sims} sims")

    for gi in range(games):
        color = chess.WHITE if gi % 2 == 0 else chess.BLACK
        opening = OPENINGS[gi % len(OPENINGS)]

        board = chess.Board()
        for uci in opening:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)
        mcts.new_game()

        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 300:
            if board.turn == color:
                # Syzygy
                tb = mcts.syzygy.get_move(board)
                if tb:
                    move = tb
                # Opening book
                elif use_book and (bm := get_book_move(board)) is not None:
                    move = bm
                else:
                    move, info = mcts.search(board, max_sims=sims)
                # Advance tree past our move so opponent's subtree is at root
                mcts.advance_tree(move)
                board.push(move)
            else:
                sf_move = sf.play(board, chess.engine.Limit(time=0.05)).move
                # Advance tree past opponent's move for inter-move tree reuse
                mcts.advance_tree(sf_move)
                board.push(sf_move)

        o = board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            result = 0.5
            rs = 'D'
        elif o.winner == color:
            result = 1.0
            rs = 'W'
        else:
            result = 0.0
            rs = 'L'

        total_score += result
        lo, hi = wilson_ci(total_score, gi + 1)
        elo = sf_elo + elo_diff(total_score / (gi + 1))

        results.append({
            'game': gi + 1,
            'color': 'W' if color == chess.WHITE else 'B',
            'opening': ' '.join(opening),
            'result': rs,
            'plies': len(board.move_stack),
            'score': result,
        })

        log(f"G{gi+1:2d}/{games}: {rs} ({results[-1]['color']}) "
            f"| {total_score/(gi+1):.3f} [{lo:.3f},{hi:.3f}] "
            f"~{elo:.0f} ELO | {results[-1]['plies']} plies")

    sf.quit()

    lo, hi = wilson_ci(total_score, games)
    elo = sf_elo + elo_diff(total_score / games)
    w = sum(1 for r in results if r['result'] == 'W')
    d = sum(1 for r in results if r['result'] == 'D')
    l = sum(1 for r in results if r['result'] == 'L')

    summary = {
        'sf_elo': sf_elo,
        'sims': sims,
        'games': games,
        'score': total_score / games,
        'ci95': [round(lo, 4), round(hi, 4)],
        'elo_estimate': round(elo),
        'w': w, 'd': d, 'l': l,
        'avg_plies': round(sum(r['plies'] for r in results) / games, 1),
    }

    log(f"\nFINAL: {w}W-{d}D-{l}L = {total_score/games:.3f} "
        f"[{lo:.3f},{hi:.3f}] ELO={elo:.0f}")

    return summary, results


def main():
    ap = argparse.ArgumentParser(description="Automated MCTS ELO gauntlet")
    ap.add_argument('--checkpoint', type=str,
                    default=str(ROOT / 'outputs' / 'exp100_diverse_training' / 'best_model.pt'))
    ap.add_argument('--config', type=str, default=None,
                    help="Model config JSON (auto-detected from checkpoint if possible)")
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--sf-elo', type=int, default=1900)
    ap.add_argument('--games', type=int, default=16)
    ap.add_argument('--c-puct', type=float, default=2.5)
    ap.add_argument('--policy-temp', type=float, default=1.0,
                    help="Policy temperature at root (lower=sharper, default: 1.0)")
    ap.add_argument('--fpu-reduction', type=float, default=0.25,
                    help="First play urgency reduction (default: 0.25)")
    ap.add_argument('--inner-temp', type=float, default=1.0,
                    help="Policy temperature at non-root nodes (default: 1.0)")
    ap.add_argument('--no-book', action='store_true')
    ap.add_argument('--dynamic-cpuct', action='store_true',
                    help="Enable KataGo-style dynamic variance-scaled cPUCT")
    ap.add_argument('--batch-size', type=int, default=16,
                    help="NN batch size for MCTS leaf eval (default: 16)")
    ap.add_argument('--quick', action='store_true',
                    help="Quick screening: 8 games at 100 sims")
    ap.add_argument('--output', type=str, default=None,
                    help="Save results to JSON file")
    ap.add_argument('--compact', action='store_true',
                    help="Use compact vocab (1968 moves, auto-detected if possible)")
    args = ap.parse_args()

    if args.quick:
        args.games = 8
        args.sims = 100

    log(f"Checkpoint: {args.checkpoint}")
    log(f"Config: sims={args.sims}, SF_ELO={args.sf_elo}, games={args.games}")
    log(f"Device: {DEVICE}")

    model = load_model(args.checkpoint, args.config)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params/1e6:.1f}M params")

    t0 = time.time()
    summary, results = run_gauntlet(
        model, args.sf_elo, args.sims, args.games,
        c_puct=args.c_puct, use_book=not args.no_book,
        dynamic_cpuct=args.dynamic_cpuct,
        batch_size=args.batch_size,
        policy_temp=args.policy_temp,
        fpu_reduction=args.fpu_reduction,
        inner_temp=args.inner_temp,
    )
    elapsed = time.time() - t0
    summary['elapsed_seconds'] = round(elapsed, 1)
    summary['checkpoint'] = str(args.checkpoint)
    log(f"Total time: {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'summary': summary, 'games': results}, f, indent=2)
        log(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
