"""Quick test: Alpha-Beta d=2 + exp105 Mirror vs Greedy at SF 1320."""
import os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import torch
import torch.nn.functional as F
from play import load_model
from move_vocab import index_to_move, legal_move_mask
from chess_features import batch_boards_to_fused_token_ids

DEVICE = torch.device('cuda')
SF_PATH = 'stockfish/stockfish/stockfish-windows-x86-64-avx2.exe'
CHECKPOINT = 'outputs/exp093_ema_curriculum_d8/ema_model.pt'
N_GAMES = 10
SF_ELO = 1320


def play_games(move_fn, label, n=N_GAMES, sf_elo=SF_ELO):
    engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    w = d = l = 0
    total_time = 0
    for gi in range(n):
        board = chess.Board()
        mc = chess.WHITE if gi % 2 == 0 else chess.BLACK
        moves = 0
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
            if board.turn == mc:
                t0 = time.time()
                m, info = move_fn(board)
                total_time += time.time() - t0
                if m is None:
                    break
            else:
                m = engine.play(board, chess.engine.Limit(time=0.05)).move
            board.push(m)
            moves += 1
        result = board.result(claim_draw=True)
        side = "W" if mc == chess.WHITE else "B"
        if result == '1-0':
            if mc == chess.WHITE: w += 1
            else: l += 1
        elif result == '0-1':
            if mc == chess.BLACK: w += 1
            else: l += 1
        else:
            d += 1
        print(f'  {label} game {gi+1}: {result} ({side}) {moves} moves', flush=True)
    engine.quit()
    score = (w + 0.5*d) / n
    avg_t = total_time / max(n, 1)
    print(f'{label} vs SF {sf_elo}: +{w}={d}-{l} ({score:.0%}) avg={avg_t:.1f}s/game', flush=True)
    return w, d, l, score


def main():
    print(f"Loading model from {CHECKPOINT}...", flush=True)
    model = load_model(CHECKPOINT, DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}", flush=True)

    # --- Alpha-Beta depth 2 ---
    from experiments.exp104_policy_guided_search import PolicyAlphaBetaSearcher
    ab = PolicyAlphaBetaSearcher(model, DEVICE, max_depth=2, root_k=8, child_k=5)

    def ab_fn(board):
        return ab.search(board)

    print('\n=== ALPHA-BETA (depth=2, root_k=8) vs SF 1320 ===', flush=True)
    t0 = time.time()
    aw, ad, al, ascore = play_games(ab_fn, 'AB-d2')
    print(f'  Total time: {time.time()-t0:.0f}s\n', flush=True)

    # --- Mirror search ---
    from experiments.exp105_batched_policy_lookahead import PolicyMirrorSearcher
    mirror = PolicyMirrorSearcher(model, DEVICE, top_k=8, alpha=0.5, beta=0.3, gamma=0.2)

    def mirror_fn(board):
        return mirror.search(board)

    print('=== MIRROR (k=8) vs SF 1320 ===', flush=True)
    t0 = time.time()
    mw, md, ml, mscore = play_games(mirror_fn, 'Mirror-8')
    print(f'  Total time: {time.time()-t0:.0f}s\n', flush=True)

    # Summary (including greedy baseline from previous test)
    print('=' * 50)
    print('FULL RESULTS vs SF 1320 (10 games each):')
    print(f'  Greedy (prev):        +7=3-0 (85%)')
    print(f'  Gumbel-policy (prev): +5=2-3 (60%)')
    print(f'  Gumbel-VH (prev):     +6=3-1 (75%)')
    print(f'  Alpha-Beta d=2:       +{aw}={ad}-{al} ({ascore:.0%})')
    print(f'  Mirror-8:             +{mw}={md}-{ml} ({mscore:.0%})')
    print('=' * 50)


if __name__ == '__main__':
    main()
