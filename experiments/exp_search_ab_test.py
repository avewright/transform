"""Quick A/B test: Greedy vs Gumbel search vs Alpha-beta at SF 1320."""
import os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import torch
import torch.nn.functional as F
from play import load_model
from move_vocab import IDX_TO_UCI, index_to_move, legal_move_mask
from chess_features import batch_boards_to_fused_token_ids

DEVICE = torch.device('cuda')
SF_PATH = 'stockfish/stockfish/stockfish-windows-x86-64-avx2.exe'
CHECKPOINT = 'outputs/exp093_ema_curriculum_d8/ema_model.pt'
N_GAMES = 10
SF_ELO = 1320


def play_games(move_fn, label, model, n=N_GAMES, sf_elo=SF_ELO):
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
                m = move_fn(board)
                total_time += time.time() - t0
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

    # --- Greedy ---
    @torch.no_grad()
    def greedy_fn(board):
        bi = batch_boards_to_fused_token_ids([board], DEVICE)
        r = model(bi)
        logits = r['policy_logits'][0].float()
        mask = legal_move_mask(board).to(DEVICE)
        logits[~mask] = float('-inf')
        return index_to_move(logits.argmax().item())

    print('\n=== GREEDY (baseline) vs SF 1320 ===', flush=True)
    t0 = time.time()
    gw, gd, gl, gscore = play_games(greedy_fn, 'Greedy', model)
    print(f'  Total time: {time.time()-t0:.0f}s\n', flush=True)

    # --- Gumbel (8 sims, policy_consistency) ---
    from experiments.exp103_gumbel_search import GumbelSearcher
    gumbel = GumbelSearcher(model, DEVICE, n_simulations=8, value_mode='policy_consistency', c_scale=1.0)

    def gumbel_fn(board):
        m, _ = gumbel.search(board)
        return m

    print('=== GUMBEL (8 sims, policy_consistency) vs SF 1320 ===', flush=True)
    t0 = time.time()
    sw, sd, sl, sscore = play_games(gumbel_fn, 'Gumbel-8', model)
    print(f'  Total time: {time.time()-t0:.0f}s\n', flush=True)

    # --- Gumbel (8 sims, value_head) ---
    gumbel_vh = GumbelSearcher(model, DEVICE, n_simulations=8, value_mode='value_head', c_scale=1.0)

    def gumbel_vh_fn(board):
        m, _ = gumbel_vh.search(board)
        return m

    print('=== GUMBEL (8 sims, value_head) vs SF 1320 ===', flush=True)
    t0 = time.time()
    vw, vd, vl, vscore = play_games(gumbel_vh_fn, 'Gumbel-VH', model)
    print(f'  Total time: {time.time()-t0:.0f}s\n', flush=True)

    # --- Alpha-Beta depth 2 ---
    from experiments.exp104_policy_guided_search import PolicyAlphaBetaSearcher
    ab = PolicyAlphaBetaSearcher(model, DEVICE, max_depth=2, root_k=8, child_k=5)

    def ab_fn(board):
        m, _ = ab.search(board)
        return m

    print('=== ALPHA-BETA (depth=2, root_k=8) vs SF 1320 ===', flush=True)
    t0 = time.time()
    aw, ad, al, ascore = play_games(ab_fn, 'AB-d2', model)
    print(f'  Total time: {time.time()-t0:.0f}s\n', flush=True)

    # --- Summary ---
    print('=' * 50)
    print('SUMMARY vs SF 1320:')
    print(f'  Greedy:              +{gw}={gd}-{gl} ({gscore:.0%})')
    print(f'  Gumbel-8 (policy):   +{sw}={sd}-{sl} ({sscore:.0%}) delta={sscore-gscore:+.0%}')
    print(f'  Gumbel-8 (value):    +{vw}={vd}-{vl} ({vscore:.0%}) delta={vscore-gscore:+.0%}')
    print(f'  Alpha-Beta d=2:      +{aw}={ad}-{al} ({ascore:.0%}) delta={ascore-gscore:+.0%}')
    print('=' * 50)


if __name__ == '__main__':
    main()
