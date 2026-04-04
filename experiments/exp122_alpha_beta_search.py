"""exp122: Alpha-beta search with NN leaf evaluation.

Optimized for speed: only calls the NN at leaf nodes, NOT at interior nodes
for move ordering. Uses simple heuristic ordering (captures/promotions first)
at interior nodes.

Key optimizations:
1. NN only at leaves (not for interior move ordering)
2. Limited quiescence (captures only, max depth 2)
3. TT with alpha/beta/exact flags
4. Policy-ordered root moves (single NN call at root)
5. Iterative deepening with best-move-first reordering

Value convention: White-absolute
  wdl[0] = P(White wins), wdl[1] = P(draw), wdl[2] = P(White loses)
  Score = wdl[0] - wdl[2] in [-1, +1]
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
import chess.syzygy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, index_to_move, legal_move_mask, move_to_index

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYZYGY_DIR = ROOT / "syzygy"
SYZYGY_TB = None

def init_syzygy():
    global SYZYGY_TB
    if SYZYGY_DIR.exists() and any(SYZYGY_DIR.glob("*.rtbw")):
        try:
            SYZYGY_TB = chess.syzygy.open_tablebase(str(SYZYGY_DIR))
        except Exception:
            SYZYGY_TB = None

def get_syzygy_move(board):
    if SYZYGY_TB is None or len(board.piece_map()) > 5:
        return None
    try:
        best_move, best_wdl, best_dtz = None, -3, 0
        for move in board.legal_moves:
            board.push(move)
            try:
                wdl = -SYZYGY_TB.probe_wdl(board)
                dtz = -SYZYGY_TB.probe_dtz(board)
                if wdl > best_wdl or (wdl == best_wdl and (
                    (wdl > 0 and dtz < best_dtz) or
                    (wdl < 0 and dtz > best_dtz) or
                    (wdl == 0 and abs(dtz) < abs(best_dtz)))):
                    best_move, best_wdl, best_dtz = move, wdl, dtz
            except Exception:
                pass
            board.pop()
        return best_move
    except Exception:
        return None

# ── Move ordering ──

PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
      chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

def order_moves(board):
    moves = list(board.legal_moves)
    def score(m):
        s = 0
        if m.promotion:
            s += 10000
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            vp = PV.get(victim.piece_type, 1) if victim else 1  # en passant = pawn
            ap = PV.get(attacker.piece_type, 1) if attacker else 1
            s += 5000 + vp * 10 - ap
        if board.gives_check(m):
            s += 3000
        return s
    moves.sort(key=score, reverse=True)
    return moves

# ── TT ──

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

class TT:
    def __init__(self, max_size=1_000_000):
        self._d = {}
        self._max = max_size
        self.hits = self.useful = 0

    def probe(self, key, depth, alpha, beta):
        e = self._d.get(key)
        if e is None:
            return None
        self.hits += 1
        sd, sv, sf = e
        if sd >= depth:
            if sf == TT_EXACT:
                self.useful += 1; return sv
            if sf == TT_LOWER and sv >= beta:
                self.useful += 1; return sv
            if sf == TT_UPPER and sv <= alpha:
                self.useful += 1; return sv
        return None

    def store(self, key, depth, value, flag):
        if len(self._d) >= self._max:
            # Evict oldest half
            keys = list(self._d.keys())
            for k in keys[:len(keys)//2]:
                del self._d[k]
        self._d[key] = (depth, value, flag)

# ── Search ──

class ABEngine:
    def __init__(self, model, device, max_depth=3, q_depth=2,
                 root_k=10, inner_k=6):
        self.model = model
        self.device = device
        self.max_depth = max_depth
        self.q_depth = q_depth
        self.root_k = root_k
        self.inner_k = inner_k
        self.tt = TT()
        self.nodes = self.nn_evals = self.q_nodes = 0

    def reset(self):
        self.nodes = self.nn_evals = self.q_nodes = 0

    @torch.no_grad()
    def _eval(self, board):
        inp = batch_boards_to_fused_token_ids([board], self.device)
        r = self.model(inp)
        wdl = F.softmax(r["value_logits"][0].float(), dim=-1)
        self.nn_evals += 1
        return (wdl[0] - wdl[2]).item()  # White perspective

    @torch.no_grad()
    def _root_policy(self, board, top_k=15):
        inp = batch_boards_to_fused_token_ids([board], self.device)
        r = self.model(inp)
        logits = r["policy_logits"][0].float()
        mask = legal_move_mask(board).to(self.device)
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        mp = []
        for m in board.legal_moves:
            mp.append((m, probs[move_to_index(m)].item()))
        mp.sort(key=lambda x: x[1], reverse=True)
        self.nn_evals += 1
        return mp[:top_k]

    def get_move(self, board):
        self.reset()
        t0 = time.time()
        root_mp = self._root_policy(board)
        root_moves = [m for m, _ in root_mp]
        policy_probs = {m: p for m, p in root_mp}

        is_white = board.turn == chess.WHITE
        best_move = root_moves[0]
        best_score = -2.0

        for depth in range(1, self.max_depth + 1):
            a, b = -2.0, 2.0
            d_best_move, d_best_score = root_moves[0], -2.0
            for move in root_moves:
                board.push(move)
                s = -self._negamax(board, depth - 1, -b, -a)
                board.pop()
                self.nodes += 1
                if s > d_best_score:
                    d_best_score = s
                    d_best_move = move
                a = max(a, s)
            best_move, best_score = d_best_move, d_best_score
            root_moves.remove(best_move)
            root_moves.insert(0, best_move)

        elapsed = time.time() - t0
        ws = best_score if is_white else -best_score
        return best_move, {
            "depth": self.max_depth, "score": ws,
            "nodes": self.nodes, "q_nodes": self.q_nodes,
            "nn_evals": self.nn_evals, "tt_hits": self.tt.hits,
            "elapsed": elapsed,
        }

    def _negamax(self, board, depth, alpha, beta):
        if board.is_game_over(claim_draw=True):
            o = board.outcome(claim_draw=True)
            if o is None or o.winner is None:
                return 0.0
            return 1.0 if o.winner == board.turn else -1.0

        key = board.fen()
        tv = self.tt.probe(key, depth, alpha, beta)
        if tv is not None:
            return tv

        if depth <= 0:
            return self._quiescence(board, alpha, beta, self.q_depth)

        moves = order_moves(board)
        # Limit branching at interior nodes
        moves = moves[:self.inner_k]
        best = -2.0
        flag = TT_UPPER

        for move in moves:
            board.push(move)
            s = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            self.nodes += 1
            if s > best:
                best = s
            if s > alpha:
                alpha = s
                flag = TT_EXACT
            if alpha >= beta:
                flag = TT_LOWER
                break

        self.tt.store(key, depth, best, flag)
        return best

    def _quiescence(self, board, alpha, beta, qd):
        self.q_nodes += 1
        wv = self._eval(board)
        sp = wv if board.turn == chess.WHITE else -wv

        if sp >= beta:
            return sp
        alpha = max(alpha, sp)
        if qd <= 0:
            return sp

        caps = [m for m in board.legal_moves
                if board.is_capture(m) or m.promotion is not None]
        if not caps:
            return sp

        def cscore(m):
            v = board.piece_at(m.to_square)
            a = board.piece_at(m.from_square)
            vp = PV.get(v.piece_type, 1) if v else 1
            ap = PV.get(a.piece_type, 1) if a else 1
            return vp * 10 - ap + (1000 if m.promotion else 0)
        caps.sort(key=cscore, reverse=True)

        for m in caps[:4]:
            board.push(m)
            s = -self._quiescence(board, -beta, -alpha, qd - 1)
            board.pop()
            self.q_nodes += 1
            if s > alpha:
                alpha = s
            if alpha >= beta:
                break
        return alpha

# ── Greedy baseline ──

@torch.no_grad()
def greedy_move(model, board, device, temperature=0.0):
    inp = batch_boards_to_fused_token_ids([board], device)
    r = model(inp)
    logits = r["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    move = index_to_move(logits.argmax().item())
    wdl = F.softmax(r["value_logits"][0].float(), dim=-1).tolist()
    return move, {"wdl": {"win": wdl[0], "draw": wdl[1], "loss": wdl[2]}}

# ── Eval framework ──

def resolve_sf():
    for p in [Path(os.environ.get("STOCKFISH_PATH", "")),
              Path(shutil.which("stockfish") or ""),
              ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2"]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Stockfish not found")

LOG_PATH = None
def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")

def wilson_ci(s, n, z=1.96):
    if n <= 0: return 0.0, 1.0
    p = s / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / d
    return max(0, c-m), min(1, c+m)

OPENINGS = [[], ["e2e4","e7e5"], ["d2d4","d7d5"], ["e2e4","c7c5"],
            ["d2d4","g8f6"], ["e2e4","e7e6"], ["c2c4","e7e5"], ["g1f3","d7d5"]]

def play_game(engine, model, ab, sf_elo, model_color, opening, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    t_search = nn_total = nodes_total = 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            tb = get_syzygy_move(board)
            if tb:
                move = tb
            elif ab:
                move, info = ab.get_move(board)
                t_search += info["elapsed"]
                nn_total += info["nn_evals"]
                nodes_total += info["nodes"]
            else:
                move, _ = greedy_move(model, board, DEVICE)
        else:
            move = engine.play(board, chess.engine.Limit(time=0.05)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        sc = 0.5
    elif o.winner == model_color:
        sc = 1.0
    else:
        sc = 0.0
    return {"score": sc, "plies": len(board.move_stack), "result": board.result(claim_draw=True),
            "color": "W" if model_color == chess.WHITE else "B",
            "t_search": t_search, "nn": nn_total, "nodes": nodes_total}

def run_config(model, sf_elo, n_games, depth, label, no_q=False):
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
    init_syzygy()

    ab = ABEngine(model, DEVICE, max_depth=depth, q_depth=0 if no_q else 2) if depth > 0 else None
    results = []
    tot = 0.0

    log(f"\n{'='*60}")
    log(f"{label} vs SF{sf_elo} ({n_games} games, depth={depth})")
    log(f"{'='*60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, model, ab, sf_elo, mc, op)
        el = time.time() - t0
        results.append(r)
        tot += r["score"]
        w = sum(1 for x in results if x["score"] == 1.0)
        d = sum(1 for x in results if x["score"] == 0.5)
        l = sum(1 for x in results if x["score"] == 0.0)
        sc = tot / len(results)
        ci = wilson_ci(tot, len(results))
        nn_s = f" nn={r['nn']} nodes={r['nodes']}" if r['nn'] > 0 else ""
        rs = "WIN" if r["score"]==1 else ("DRAW" if r["score"]==0.5 else "LOSS")
        log(f"  G{i+1:>3}/{n_games}: {r['color']} {rs} ({r['plies']}ply {el:.0f}s){nn_s}"
            f" | {sc:.3f} ({w}W-{d}D-{l}L) [{ci[0]:.3f},{ci[1]:.3f}]")
        if ab:
            ab.tt = TT()  # reset TT between games

    engine.quit()
    sc = tot / n_games
    w = sum(1 for x in results if x["score"] == 1.0)
    d = sum(1 for x in results if x["score"] == 0.5)
    l = sum(1 for x in results if x["score"] == 0.0)
    ci = wilson_ci(tot, n_games)
    ed = -400 * math.log10(1/sc - 1) if 0 < sc < 1 else (400 if sc >= 1 else -400)
    avg_nn = sum(r["nn"] for r in results) / n_games
    avg_t = sum(r["t_search"] for r in results) / n_games
    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo+ed:.0f}")
    log(f"  avg nn={avg_nn:.0f}/game search_t={avg_t:.1f}s/game")
    return {"name": label, "sf_elo": sf_elo, "games": n_games,
            "score": sc, "w": w, "d": d, "l": l, "ci95": list(ci),
            "elo_diff": round(ed), "est_elo": round(sf_elo + ed),
            "avg_nn": avg_nn, "avg_search_t": avg_t}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/hf_checkpoint/best_model.pt")
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--no-q", action="store_true")
    args = ap.parse_args()

    global LOG_PATH
    LOG_PATH = Path("outputs/elo_eval_exp122_search.log")
    json_path = Path("outputs/elo_eval_exp122_search.json")
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log(f"Loading {args.checkpoint}...")
    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model = model.to(DEVICE)
    model.eval()
    log(f"Loaded on {DEVICE}")

    all_results = []
    for d in args.depths:
        label = "greedy" if d == 0 else f"ab_d{d}{'_noq' if args.no_q else ''}"
        r = run_config(model, args.sf_elo, args.games, d, label, no_q=args.no_q)
        all_results.append(r)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    log(f"\n{'='*60}")
    log(f" SUMMARY vs SF{args.sf_elo}")
    log(f"{'='*60}")
    for r in all_results:
        log(f"  {r['name']:15s} {r['score']:.3f} ({r['w']}W-{r['d']}D-{r['l']}L) ELO~{r['est_elo']} nn={r['avg_nn']:.0f}/g")

if __name__ == "__main__":
    main()
