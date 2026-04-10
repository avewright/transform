"""exp129: Gumbel MCTS — principled action selection for low sim budgets.

From Danihelka et al. 2022 "Policy improvement by planning with Gumbel."
Referenced in alphazero/possible_improvements.md as the modern replacement for PUCT.

Key insight: Standard PUCT with c_puct hyperparameter works well at 800+ sims but
is suboptimal at low sim counts (100-400). Gumbel MCTS replaces PUCT with a
principled sampling scheme that:
  1. Requires NO c_puct hyperparameter
  2. Works well with very small simulation budgets
  3. Guarantees policy improvement (proven mathematically)

Algorithm:
  1. At root: sample top-K actions using Gumbel noise + log prior
     g_a = log P(a) + Gumbel(0,1)
     Select top-K by g_a
  2. Allocate budget across K actions using Sequential Halving:
     Run sims/2 on K, keep top K/2, repeat
  3. Select final action by completed Q + Gumbel shift (σ formula)

For interior nodes, we still use standard PUCT traversal (Gumbel is root-only).

Also tests: FP16 autocast (2.14x measured speedup), which effectively doubles
sim budget within the same wall-clock time.

Test matrix:
  Phase 1: Gumbel MCTS at 200 sims (K=16 top actions)
  Phase 2: Gumbel MCTS at 200 sims with FP16 (effective ~420 sims in same time)
  Phase 3: Standard MCTS + FP16 at 400 sims

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
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Compact vocab auto-detection (must run BEFORE chess imports)
if '--compact' in sys.argv:
    os.environ['MOVE_VOCAB_VERSION'] = 'compact'
else:
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

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, index_to_move, legal_move_mask, move_to_index
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


class GumbelMCTS:
    """Gumbel MCTS: principled action selection for low simulation budgets.

    At root: uses Gumbel noise + Sequential Halving to allocate sims.
    At interior nodes: standard PUCT traversal.
    """

    def __init__(self, model, device, syzygy: SyzygyProbe,
                 batch_size=8, fpu_reduction=0.25,
                 top_k=16, c_puct_interior=2.5, use_fp16=True):
        self.model = model
        self.device = device
        self.syzygy = syzygy
        self.batch_size = batch_size
        self.fpu_reduction = fpu_reduction
        self.top_k = top_k
        self.c_puct = c_puct_interior  # Only for interior nodes
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.root = None
        self.nn_evals = 0

    def reset_stats(self):
        self.nn_evals = 0

    @torch.no_grad()
    def _batch_evaluate(self, boards):
        """Evaluate boards. Returns list of (policy_dict, stm_value)."""
        if not boards:
            return []
        inp = batch_boards_to_fused_token_ids(boards, self.device)
        if self.use_fp16:
            # FP16-safe: backbone in FP16, heads in FP32
            # Policy logits are ~2000+ magnitude where FP16 precision fails
            m = self.model
            with torch.amp.autocast('cuda'):
                hidden = m.input_proj(m.encoder(inp))
                B = hidden.shape[0]
                hidden = torch.cat([m.cls_token.expand(B, -1, -1), hidden], dim=1)
                if m.pos_embed is not None:
                    hidden = hidden + m.pos_embed
                hidden = m.norm(m.transformer(hidden))
            hidden = hidden.float()
            cls_hidden = hidden[:, 0, :]
            r = {
                "policy_logits": m.policy_head(hidden, cls_hidden),
                "value_logits": m.value_head(cls_hidden),
            }
        else:
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
            # Value: handle both 3-class WDL and N-bin distributional
            val_logits = r["value_logits"][i].float()
            if val_logits.shape[-1] == 3:
                wdl = F.softmax(val_logits, dim=-1)
                white_val = (wdl[0] - wdl[2]).item()
            else:
                n_bins = val_logits.shape[-1]
                bin_centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                                             device=val_logits.device)
                probs_v = F.softmax(val_logits, dim=-1)
                win_pct = (probs_v * bin_centers).sum().item()
                white_val = win_pct * 2 - 1  # map [0,1] → [-1,+1]
            stm_val = white_val if board.turn == chess.WHITE else -white_val
            results.append((policy, stm_val))
        self.nn_evals += len(boards)
        return results

    def _select_child_puct(self, node, c_puct=None):
        """Standard PUCT for interior nodes."""
        if c_puct is None:
            c_puct = self.c_puct
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
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _expand_and_eval(self, node, board):
        """Expand node, return stm_value."""
        sv = self.syzygy.probe_value(board)
        if sv is not None:
            legal = list(board.legal_moves)
            if legal:
                p = 1.0 / len(legal)
                for m in legal:
                    node.children[m] = MCTSNode(prior=p)
            node.is_expanded = True
            return sv
        policy, value = self._batch_evaluate([board])[0]
        for m, p in policy.items():
            node.children[m] = MCTSNode(prior=p)
        node.is_expanded = True
        return value

    def _run_sim_from(self, root, base_board, target_move):
        """Run one MCTS sim from root, forcing target_move at root.

        Traverses from target child using PUCT for interior nodes.
        Returns updated value.
        """
        child = root.children[target_move]
        board = base_board.copy()
        board.push(target_move)
        path = [root, child]
        node = child

        # Traverse to a leaf using PUCT
        while node.is_expanded and node.children:
            if board.is_game_over(claim_draw=True):
                break
            move, next_child = self._select_child_puct(node)
            if move is None:
                break
            board.push(move)
            path.append(next_child)
            node = next_child

        # Terminal
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                value = 0.0
            elif outcome.winner == board.turn:
                value = 1.0
            else:
                value = -1.0
        elif not node.is_expanded:
            value = self._expand_and_eval(node, board)
        else:
            value = node.q_value()

        # Backup
        v = value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += v
            v = -v

        return value

    def _sigma(self, q, v_root, max_n, c_visit=5.0):
        """Completed Q-value transform for Gumbel action selection.

        From Danihelka et al. 2022: saturating sigma that maps Q-values
        to logit-scale corrections. Uses (q - v_root) to center around 0.
        Saturates at c_visit * (q - v_root) as max_n → ∞.
        """
        q_centered = q - v_root
        return c_visit * max_n * q_centered / (c_visit + max_n)

    def search(self, board, max_sims=200):
        """Gumbel MCTS search."""
        self.reset_stats()
        t0 = time.time()

        # Syzygy
        tb = self.syzygy.get_move(board)
        if tb is not None:
            return tb, {"source": "syzygy", "nn_evals": 0, "sims": 0,
                        "elapsed": time.time() - t0}

        # Opening book
        book_move = get_book_move(board)
        if book_move is not None:
            return book_move, {"source": "book", "nn_evals": 0, "sims": 0,
                               "elapsed": time.time() - t0}

        # Expand root
        root = MCTSNode()
        self._expand_and_eval(root, board)

        if not root.children:
            return list(board.legal_moves)[0], {"nn_evals": 0, "sims": 0, "elapsed": 0}
        if len(root.children) == 1:
            move = next(iter(root.children))
            return move, {"nn_evals": self.nn_evals, "sims": 0, "elapsed": time.time() - t0}

        moves = list(root.children.keys())
        n_moves = len(moves)

        # Step 1: Sample top-K actions using Gumbel noise + log prior
        log_priors = np.array([math.log(max(root.children[m].prior, 1e-8))
                               for m in moves])
        gumbel_noise = np.random.gumbel(size=n_moves)
        gumbel_scores = log_priors + gumbel_noise

        K = min(self.top_k, n_moves)
        top_indices = np.argsort(gumbel_scores)[-K:][::-1]
        selected_moves = [moves[i] for i in top_indices]
        selected_gumbels = gumbel_scores[top_indices]

        # Step 2: Sequential Halving to allocate budget
        remaining = list(range(K))
        sims_used = 0
        n_halving_rounds = max(1, int(math.log2(K)))
        sims_per_round = max_sims // max(1, n_halving_rounds)

        for round_idx in range(n_halving_rounds):
            if len(remaining) <= 1:
                break

            # Allocate sims equally among remaining actions
            sims_each = max(1, sims_per_round // len(remaining))

            for idx in remaining:
                move = selected_moves[idx]
                for _ in range(sims_each):
                    self._run_sim_from(root, board, move)
                    sims_used += 1
                    if sims_used >= max_sims:
                        break
                if sims_used >= max_sims:
                    break

            if sims_used >= max_sims:
                break

            # Halve: keep top half by completed Q + Gumbel
            completed_scores = []
            max_n = max((root.children[selected_moves[idx]].visit_count
                         for idx in remaining), default=1)
            v_root = root.q_value() if root.visit_count > 0 else 0.0
            for idx in remaining:
                move = selected_moves[idx]
                child = root.children[move]
                if child.visit_count > 0:
                    q = -child.q_value()  # From root's perspective
                    sigma_q = self._sigma(q, v_root, max_n)
                    score = selected_gumbels[idx] + sigma_q
                else:
                    score = selected_gumbels[idx]
                completed_scores.append((idx, score))

            completed_scores.sort(key=lambda x: -x[1])
            keep = max(1, len(remaining) // 2)
            remaining = [idx for idx, _ in completed_scores[:keep]]

        # Use any remaining budget on the survivors
        while sims_used < max_sims and remaining:
            for idx in remaining:
                if sims_used >= max_sims:
                    break
                move = selected_moves[idx]
                self._run_sim_from(root, board, move)
                sims_used += 1

        # Step 3: Select action by visit count (most robust)
        best_move = max(root.children.items(),
                        key=lambda x: x[1].visit_count)[0]

        elapsed = time.time() - t0

        # Extract PV
        pv = [best_move.uci()]
        node = root.children[best_move]
        for _ in range(10):
            if not node.children:
                break
            bm = max(node.children.items(), key=lambda x: x[1].visit_count)[0]
            pv.append(bm.uci())
            node = node.children[bm]

        score_cp = self._q_to_cp(root.q_value())

        return best_move, {
            "nn_evals": self.nn_evals,
            "sims": sims_used,
            "elapsed": elapsed,
            "score_cp": score_cp,
            "pv": pv,
            "k_selected": K,
            "n_survivors": len(remaining),
        }

    def _q_to_cp(self, q):
        q = max(-0.999, min(0.999, q))
        return int(111.714 * math.tan(1.5620688 * q))

    def new_game(self):
        self.root = None


def play_game(engine, model, mcts, sf_elo, model_color,
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
            tb_move = mcts.syzygy.get_move(board) if hasattr(mcts, 'syzygy') else None
            if tb_move:
                move = tb_move
                mcts.new_game()
            elif (book_move := get_book_move(board)) is not None:
                move = book_move
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


def run_config(model, syzygy, sf_elo, n_games, sims, label,
               use_gumbel=False, use_fp16=True, top_k=16,
               c_puct=2.5, fpu_reduction=0.25):
    if use_gumbel:
        mcts = GumbelMCTS(model, DEVICE, syzygy,
                          batch_size=8, fpu_reduction=fpu_reduction,
                          top_k=top_k, c_puct_interior=c_puct,
                          use_fp16=use_fp16)
    else:
        mcts = MCTSSearch(model, DEVICE, syzygy,
                          c_puct=c_puct, batch_size=8,
                          fpu_reduction=fpu_reduction,
                          root_noise_alpha=0.3, root_noise_frac=0.0,
                          use_fp16=use_fp16)

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    results = []
    tot = 0.0
    mode_str = f"gumbel_k{top_k}" if use_gumbel else "puct"
    prec_str = "fp16" if use_fp16 else "fp32"

    log(f"\n{'=' * 60}")
    log(f"{label} vs SF{sf_elo} ({n_games}g, {sims}sims, {mode_str}, {prec_str})")
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
    avg_t = sum(r["t_search"] for r in results) / n_games
    avg_sims = sum(r["sims"] for r in results) / n_games

    log(f"\n  FINAL {label}: {sc:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO~{sf_elo + ed:.0f}")
    log(f"  avg nn={avg_nn:.0f}/g sims={avg_sims:.0f}/g "
        f"search_t={avg_t:.1f}s/g")

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
        "avg_sims": round(avg_sims),
        "avg_search_t": round(avg_t, 1),
        "sims_per_move": sims,
        "use_gumbel": use_gumbel,
        "use_fp16": use_fp16,
        "top_k": top_k if use_gumbel else None,
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
    ap.add_argument("--compact", action="store_true",
                    help="Use compact vocab (auto-detected from checkpoint if possible)")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all, 1=gumbel, 2=gumbel+fp16, 3=puct+fp16")
    args = ap.parse_args()

    n_games = 8 if args.quick else args.games
    sf_elo = args.sf_elo
    sims = args.sims

    global LOG_PATH
    LOG_PATH = ROOT / "outputs" / "exp129_gumbel_mcts.log"
    json_path = ROOT / "outputs" / "exp129_gumbel_mcts.json"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    ckpt_path = args.checkpoint or find_checkpoint()
    log(f"Loading checkpoint: {ckpt_path}")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    # Auto-detect distributional value head (128-bin HL-Gauss vs 3-class WDL)
    ckpt_vbias = sd.get('value_head.2.bias')
    if ckpt_vbias is not None and ckpt_vbias.shape[0] != model.value_head[2].out_features:
        n_bins = ckpt_vbias.shape[0]
        old_head = model.value_head
        model.value_head = torch.nn.Sequential(
            old_head[0], old_head[1],
            torch.nn.Linear(old_head[0].out_features, n_bins),
        )
        log(f"Rebuilt value head for {n_bins}-bin distributional output")

    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    log(f"Model loaded on {DEVICE}")

    syzygy = SyzygyProbe()
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    all_results = []

    # Phase 1: Gumbel MCTS K=4 (concentrated budget per action)
    if args.phase in (0, 1):
        log("\n" + "=" * 60)
        log("PHASE 1: GUMBEL K=4 (FP16)")
        log("=" * 60)
        r = run_config(model, syzygy, sf_elo, n_games, sims,
                       label=f"gumbel_k4_{sims}",
                       use_gumbel=True, use_fp16=True, top_k=4)
        all_results.append(r)

    # Phase 2: Gumbel MCTS K=8
    if args.phase in (0, 2):
        log("\n" + "=" * 60)
        log("PHASE 2: GUMBEL K=8 (FP16)")
        log("=" * 60)
        r = run_config(model, syzygy, sf_elo, n_games, sims,
                       label=f"gumbel_k8_{sims}",
                       use_gumbel=True, use_fp16=True, top_k=8)
        all_results.append(r)

    # Phase 3: Standard PUCT + FP16 + no noise (baseline comparison)
    if args.phase in (0, 3):
        log("\n" + "=" * 60)
        log("PHASE 3: PUCT BASELINE (FP16)")
        log("=" * 60)
        r = run_config(model, syzygy, sf_elo, n_games, sims,
                       label=f"puct_fp16_{sims}",
                       use_gumbel=False, use_fp16=True)
        all_results.append(r)

    # Phase 4: Gumbel K=4 at 2x sims (test if Gumbel scales better)
    if args.phase in (0, 4):
        log("\n" + "=" * 60)
        log(f"PHASE 4: GUMBEL K=4 at {sims*2} sims (FP16)")
        log("=" * 60)
        r = run_config(model, syzygy, sf_elo, n_games, sims * 2,
                       label=f"gumbel_k4_{sims*2}",
                       use_gumbel=True, use_fp16=True, top_k=4)
        all_results.append(r)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"{'Config':<30} {'Score':>7} {'W-D-L':>9} {'ELO':>6} {'NN/g':>7} {'t/g':>6}")
    log("-" * 70)
    for r in all_results:
        log(f"{r['name']:<30} {r['score']:>7.3f} "
            f"{r['w']}W-{r['d']}D-{r['l']}L "
            f"{r['est_elo']:>6} {r['avg_nn']:>7} {r['avg_search_t']:>5.1f}s")

    log("\nReference: fixed_100 = 0.688, ELO ~2037 (exp125)")

    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
