"""Verified 32-game gauntlet: MCGS (transposition table) vs SF1900."""
import sys, math, os, time
from pathlib import Path
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.engine
import torch
from chess_transformer_factory import build_model
from uci_engine import MCTSSearch, SyzygyProbe
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = build_model()
ckpt = torch.load(ROOT / 'outputs/exp100_diverse_training/best_model.pt',
                  map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)
sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(DEVICE).eval()
print('Model loaded', flush=True)

OPENINGS = [
    ['e2e4','e7e5'], ['d2d4','d7d5'], ['c2c4','e7e5'], ['g1f3','d7d5'],
    ['e2e4','c7c5'], ['d2d4','g8f6'], ['e2e4','e7e6'], ['d2d4','d7d6'],
    ['e2e4','c7c6'], ['g1f3','g8f6'], ['c2c4','c7c5'], ['e2e4','g7g6'],
    ['d2d4','e7e6'], ['c2c4','g8f6'], ['b1c3','d7d5'], ['g2g3','d7d5'],
]

def elo_diff(s):
    if s <= 0: return -400
    if s >= 1: return 400
    return -400 * math.log10(1/s - 1)

def wci(s, n, z=1.96):
    p = s / n
    d = 1 + z*z / n
    c = (p + z*z / (2*n)) / d
    m = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / d
    return max(0, c-m), min(1, c+m)

# Stockfish
sf_path = ROOT / 'stockfish' / 'stockfish' / 'stockfish-windows-x86-64-avx2.exe'
sf = chess.engine.SimpleEngine.popen_uci(str(sf_path))
sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': 1900, 'Threads': 1})

# MCGS search (transposition table enabled)
syzygy = SyzygyProbe()
mcts = MCTSSearch(model, DEVICE, syzygy, c_puct=2.5, batch_size=8,
                  root_noise_alpha=0.3, root_noise_frac=0.25,
                  use_fp16=True, use_transpositions=True)

N_GAMES = 32
SIMS = 100
total_score = 0.0
total_tt = 0

ts = datetime.now().strftime("%H:%M:%S")
print(f"[{ts}] Starting 32-game MCGS gauntlet vs SF1900 @ {SIMS} sims", flush=True)

for gi in range(N_GAMES):
    color = chess.WHITE if gi % 2 == 0 else chess.BLACK
    opening = OPENINGS[gi % len(OPENINGS)]

    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    mcts.new_game()
    game_tt = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 300:
        if board.turn == color:
            tb = mcts.syzygy.get_move(board)
            if tb:
                move = tb
            else:
                move, info = mcts.search(board, max_sims=SIMS)
                game_tt += info.get('tt_hits', 0)
            mcts.root = None
            board.push(move)
        else:
            sf_move = sf.play(board, chess.engine.Limit(time=0.05)).move
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
    total_tt += game_tt
    lo, hi = wci(total_score, gi + 1)
    elo = 1900 + elo_diff(total_score / (gi + 1))
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] G{gi+1:2d}/32: {rs} | {total_score/(gi+1):.3f} "
          f"[{lo:.3f},{hi:.3f}] ~{elo:.0f} | tt={game_tt}", flush=True)

sf.quit()
lo, hi = wci(total_score, N_GAMES)
elo = 1900 + elo_diff(total_score / N_GAMES)
print(f"\nFINAL: score={total_score/N_GAMES:.3f} [{lo:.3f},{hi:.3f}] ELO={elo:.0f}")
print(f"TT hits: total={total_tt} avg={total_tt/N_GAMES:.0f}/game")
