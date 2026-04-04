"""exp124: Optimized MCTS with tree reuse, batched eval, and sim sweep.

exp123 showed MCTS with 100 sims gets ~85% vs SF1900 (est. ~2200 ELO).
This experiment optimizes the MCTS for maximum ELO:

1. Tree reuse: keep subtree after each move pair → effective ~2x sims
2. Batched leaf evaluation: evaluate B leaves at once for GPU efficiency
3. Higher sim counts: 100/200/400/800 sweep
4. Higher opponents: SF2050/SF2200/SF2400 to bracket ceiling
5. Syzygy in MCTS tree: perfect endgame value at ≤5 pieces
6. Early termination: stop sims when one move has >85% visits
7. c_puct tuning: sweep 1.5/2.5/4.0

Value convention: White-absolute
  wdl[0] = P(White wins), wdl[1] = P(draw), wdl[2] = P(White loses)
  Score = wdl[0] - wdl[2] in [-1, +1], converted to side-to-move perspective.
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
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, index_to_move, legal_move_mask, move_to_index

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
    """Get perfect tablebase move for ≤5 piece positions."""
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


def syzygy_value(board):
    """Get exact value from Syzygy tablebase. Returns None if not available."""
    if SYZYGY_TB is None or len(board.piece_map()) > 5:
        return None
    try:
        wdl = SYZYGY_TB.probe_wdl(board)
        # wdl is from side-to-move perspective: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
        if wdl >= 2:
            stm_val = 1.0
        elif wdl == 1:
            stm_val = 0.5  # cursed win (50-move rule may prevent it)
        elif wdl == 0:
            stm_val = 0.0
        elif wdl == -1:
            stm_val = -0.5  # blessed loss
        else:
            stm_val = -1.0
        return stm_val
    except Exception:
        return None


# ── Optimized MCTS ──

class MCTSNode:
    __slots__ = ['prior', 'visit_count', 'value_sum', 'children', 'is_expanded']

    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class BatchedMCTS:
    """MCTS with batched leaf evaluation, tree reuse, and early termination."""

    def __init__(self, model, device, num_simulations=200, c_puct=2.5,
                 batch_size=8, fpu_reduction=0.25, root_noise_frac=0.25,
                 root_noise_alpha=0.3, early_stop_frac=0.85,
                 early_stop_min_sims=40, use_syzygy_in_tree=True):
        self.model = model
        self.device = device
        self.num_sims = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.fpu_reduction = fpu_reduction
        self.root_noise_frac = root_noise_frac
        self.root_noise_alpha = root_noise_alpha
        self.early_stop_frac = early_stop_frac
        self.early_stop_min_sims = early_stop_min_sims
        self.use_syzygy = use_syzygy_in_tree
        self.nn_evals = 0
        self.root = None  # For tree reuse

    def reset_stats(self):
        self.nn_evals = 0

    @torch.no_grad()
    def _batch_evaluate(self, boards):
        """Evaluate multiple boards at once. Returns list of (policy_dict, value)."""
        if not boards:
            return []
        inp = batch_boards_to_fused_token_ids(boards, self.device)
        r = self.model(inp)

        results = []
        for i, board in enumerate(boards):
            logits = r["policy_logits"][i].float()
            mask = legal_move_mask(board).to(self.device)
            logits[~mask] = float("-inf")
            probs = F.softmax(logits, dim=-1)

            policy = {}
            for m in board.legal_moves:
                idx = move_to_index(m)
                policy[m] = probs[idx].item()

            wdl = F.softmax(r["value_logits"][i].float(), dim=-1)
            white_val = (wdl[0] - wdl[2]).item()
            value = white_val if board.turn == chess.WHITE else -white_val

            results.append((policy, value))

        self.nn_evals += len(boards)
        return results

    def _expand(self, node, board):
        """Expand a single node using nn eval or Syzygy."""
        # Try Syzygy first for endgame positions
        if self.use_syzygy:
            sv = syzygy_value(board)
            if sv is not None:
                # Still need to create children with uniform priors for legal moves
                legal = list(board.legal_moves)
                if legal:
                    prior = 1.0 / len(legal)
                    for m in legal:
                        node.children[m] = MCTSNode(prior=prior)
                node.is_expanded = True
                return sv

        policy, value = self._batch_evaluate([board])[0]
        for move, prob in policy.items():
            node.children[move] = MCTSNode(prior=prob)
        node.is_expanded = True
        return value

    def _add_root_noise(self, root):
        if not root.children or self.root_noise_frac <= 0:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.root_noise_alpha] * len(moves))
        frac = self.root_noise_frac
        for i, m in enumerate(moves):
            child = root.children[m]
            child.prior = (1 - frac) * child.prior + frac * noise[i]

    def _select_child(self, node):
        parent_n = node.visit_count
        parent_q = node.q_value()
        fpu_value = -parent_q - self.fpu_reduction

        best_score = -float('inf')
        best_move = None
        best_child = None
        sqrt_parent = math.sqrt(max(1, parent_n))

        for move, child in node.children.items():
            if child.visit_count == 0:
                q = fpu_value
            else:
                q = -child.q_value()
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _should_early_stop(self, root, sims_done):
        """Stop early if one move dominates visits."""
        if sims_done < self.early_stop_min_sims:
            return False
        if not root.children:
            return True
        max_visits = max(c.visit_count for c in root.children.values())
        remaining = self.num_sims - sims_done
        second_best = sorted([c.visit_count for c in root.children.values()], reverse=True)
        if len(second_best) < 2:
            return True
        # Can second-best catch up even if all remaining sims go to it?
        if second_best[1] + remaining < max_visits:
            return True
        return max_visits / max(1, sims_done) > self.early_stop_frac

    def _run_simulations_batched(self, root, base_board):
        """Run all simulations with batched leaf evaluation."""
        VIRTUAL_LOSS = 1  # penalty for in-flight nodes
        
        sims_done = 0
        while sims_done < self.num_sims:
            if self._should_early_stop(root, sims_done):
                break

            # Collect a batch of leaves
            batch_boards = []
            batch_paths = []
            batch_nodes = []

            for _ in range(min(self.batch_size, self.num_sims - sims_done)):
                node = root
                scratch = base_board.copy()
                path = [node]

                # Selection with virtual loss
                while node.is_expanded and node.children:
                    if scratch.is_game_over(claim_draw=True):
                        break
                    move, child = self._select_child(node)
                    if move is None:
                        break
                    scratch.push(move)
                    path.append(child)
                    node = child

                # Terminal check
                if scratch.is_game_over(claim_draw=True):
                    outcome = scratch.outcome(claim_draw=True)
                    if outcome is None or outcome.winner is None:
                        leaf_value = 0.0
                    elif outcome.winner == scratch.turn:
                        leaf_value = 1.0
                    else:
                        leaf_value = -1.0
                    # Immediate backup for terminals
                    value = leaf_value
                    for n in reversed(path):
                        n.visit_count += 1
                        n.value_sum += value
                        value = -value
                    sims_done += 1
                    continue

                # Check Syzygy for leaf
                if self.use_syzygy and not node.is_expanded:
                    sv = syzygy_value(scratch)
                    if sv is not None:
                        # Expand with uniform priors
                        legal = list(scratch.legal_moves)
                        if legal:
                            prior = 1.0 / len(legal)
                            for m in legal:
                                node.children[m] = MCTSNode(prior=prior)
                        node.is_expanded = True
                        # Backup
                        value = sv
                        for n in reversed(path):
                            n.visit_count += 1
                            n.value_sum += value
                            value = -value
                        sims_done += 1
                        continue

                if not node.is_expanded:
                    # Apply virtual loss
                    for n in path:
                        n.visit_count += VIRTUAL_LOSS
                        n.value_sum -= VIRTUAL_LOSS  # pessimistic
                    batch_boards.append(scratch)
                    batch_paths.append(path)
                    batch_nodes.append(node)
                else:
                    # Expanded but no children (shouldn't happen often)
                    value = node.q_value()
                    for n in reversed(path):
                        n.visit_count += 1
                        n.value_sum += value
                        value = -value
                    sims_done += 1

            # Batch evaluate leaves
            if batch_boards:
                results = self._batch_evaluate(batch_boards)
                for i, (board, path, node, (policy, value)) in enumerate(
                        zip(batch_boards, batch_paths, batch_nodes, results)):
                    # Remove virtual loss
                    for n in path:
                        n.visit_count -= VIRTUAL_LOSS
                        n.value_sum += VIRTUAL_LOSS

                    # Expand
                    for move, prob in policy.items():
                        node.children[move] = MCTSNode(prior=prob)
                    node.is_expanded = True

                    # Backup
                    v = value
                    for n in reversed(path):
                        n.visit_count += 1
                        n.value_sum += v
                        v = -v
                    sims_done += 1

        return sims_done

    def get_move(self, board, reuse_tree=True):
        """Run MCTS with optional tree reuse."""
        self.reset_stats()
        t0 = time.time()

        # Try to reuse tree from previous search
        if reuse_tree and self.root is not None and self.root.children:
            # Check if last two moves are in our tree
            # (our move + opponent's response)
            reused = False
            # This is handled by advance_tree() called externally
        
        if self.root is None or not self.root.is_expanded:
            self.root = MCTSNode()
            self._expand(self.root, board)

        if not self.root.children:
            return list(board.legal_moves)[0], {"nn_evals": self.nn_evals, "elapsed": 0}

        # Add Dirichlet noise at root
        self._add_root_noise(self.root)

        # Run simulations
        total_sims = self._run_simulations_batched(self.root, board)

        # Select best move by visit count
        best_move = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]

        elapsed = time.time() - t0

        # Info
        top_moves = {}
        for m, c in sorted(self.root.children.items(),
                           key=lambda x: x[1].visit_count, reverse=True)[:5]:
            top_moves[m.uci()] = {
                'visits': c.visit_count,
                'q': round(-c.q_value(), 4),
                'prior': round(c.prior, 4),
            }

        info = {
            "nn_evals": self.nn_evals,
            "simulations": total_sims,
            "root_q": round(self.root.q_value(), 4),
            "elapsed": elapsed,
            "top_moves": top_moves,
        }
        return best_move, info

    def advance_tree(self, our_move, opp_move):
        """Advance tree after our move and opponent's response (tree reuse)."""
        if self.root is None:
            return
        # After our move
        if our_move in self.root.children:
            self.root = self.root.children[our_move]
            # After opponent's move
            if opp_move and opp_move in self.root.children:
                self.root = self.root.children[opp_move]
            else:
                self.root = None  # Lost the branch
        else:
            self.root = None

    def new_game(self):
        """Reset tree for a new game."""
        self.root = None


# ── Greedy baseline ──

@torch.no_grad()
def greedy_move(model, board, device):
    inp = batch_boards_to_fused_token_ids([board], device)
    r = model(inp)
    logits = r["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    move = index_to_move(logits.argmax().item())
    return move


# ── Evaluation framework ──

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


def play_game(engine, model, mcts, sf_elo, model_color, opening, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    if mcts:
        mcts.new_game()

    t_search = nn_total = 0
    last_model_move = None

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            # Syzygy at game level (override)
            tb = get_syzygy_move(board)
            if tb:
                move = tb
            elif mcts:
                move, info = mcts.get_move(board)
                t_search += info["elapsed"]
                nn_total += info["nn_evals"]
                last_model_move = move
            else:
                move = greedy_move(model, board, DEVICE)
            board.push(move)
        else:
            sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            board.push(sf_move)
            # Tree reuse: advance after opponent's move
            if mcts and last_model_move is not None:
                mcts.advance_tree(last_model_move, sf_move)
                last_model_move = None

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        sc = 0.5
    elif o.winner == model_color:
        sc = 1.0
    else:
        sc = 0.0
    return {"score": sc, "plies": len(board.move_stack),
            "result": board.result(claim_draw=True),
            "color": "W" if model_color == chess.WHITE else "B",
            "t_search": t_search, "nn": nn_total}


def run_config(model, sf_elo, n_games, num_sims, label, c_puct=2.5,
               batch_size=8, no_noise=False):
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
    init_syzygy()

    mcts = None
    if num_sims > 0:
        noise_frac = 0.0 if no_noise else 0.25
        mcts = BatchedMCTS(model, DEVICE, num_simulations=num_sims,
                           c_puct=c_puct, batch_size=batch_size,
                           root_noise_frac=noise_frac)

    results = []
    tot = 0.0

    log(f"\n{'='*60}")
    log(f"{label} vs SF{sf_elo} ({n_games} games, sims={num_sims}, c_puct={c_puct})")
    log(f"{'='*60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK
        t0 = time.time()
        r = play_game(engine, model, mcts, sf_elo, mc, op)
        el = time.time() - t0
        results.append(r)
        tot += r["score"]
        w = sum(1 for x in results if x["score"] == 1.0)
        d = sum(1 for x in results if x["score"] == 0.5)
        l = sum(1 for x in results if x["score"] == 0.0)
        sc = tot / len(results)
        ci = wilson_ci(tot, len(results))
        nn_s = f" nn={r['nn']}" if r['nn'] > 0 else ""
        rs = "WIN" if r["score"]==1 else ("DRAW" if r["score"]==0.5 else "LOSS")
        log(f"  G{i+1:>3}/{n_games}: {r['color']} {rs} ({r['plies']}ply {el:.0f}s){nn_s}"
            f" | {sc:.3f} ({w}W-{d}D-{l}L) [{ci[0]:.3f},{ci[1]:.3f}]")

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
            "avg_nn": avg_nn, "avg_search_t": avg_t,
            "num_sims": num_sims, "c_puct": c_puct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/hf_checkpoint/best_model.pt")
    ap.add_argument("--sf-elos", nargs="+", type=int, default=[2050, 2200])
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", nargs="+", type=int, default=[100, 200, 400])
    ap.add_argument("--c-puct", nargs="+", type=float, default=[2.5])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--output-tag", default="exp124_mcts_opt")
    args = ap.parse_args()

    global LOG_PATH
    LOG_PATH = Path(f"outputs/elo_eval_{args.output_tag}.log")
    json_path = Path(f"outputs/elo_eval_{args.output_tag}.json")

    log(f"Loading {args.checkpoint}...")
    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model = model.to(DEVICE)
    model.eval()

    # Compile for speed if possible
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log("Model compiled with torch.compile")
    except Exception:
        log("torch.compile not available, using eager mode")

    log(f"Loaded on {DEVICE}")
    log(f"Config: sf_elos={args.sf_elos} games={args.games} "
        f"sims={args.sims} c_puct={args.c_puct} batch={args.batch_size}")

    all_results = []
    for sf_elo in args.sf_elos:
        for c_puct in args.c_puct:
            for s in args.sims:
                cp_tag = f"_cp{c_puct}" if len(args.c_puct) > 1 else ""
                label = f"mcts_{s}{cp_tag}" if s > 0 else "greedy"
                r = run_config(model, sf_elo, args.games, s, label,
                               c_puct=c_puct, batch_size=args.batch_size,
                               no_noise=args.no_noise)
                all_results.append(r)
                with open(json_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    log(f"\n{'='*60}")
    log(f" SUMMARY")
    log(f"{'='*60}")
    for r in all_results:
        log(f"  {r['name']:20s} vs SF{r['sf_elo']}: {r['score']:.3f} "
            f"({r['w']}W-{r['d']}D-{r['l']}L) ELO~{r['est_elo']} "
            f"nn={r['avg_nn']:.0f}/g t={r['avg_search_t']:.1f}s/g")


if __name__ == "__main__":
    main()
