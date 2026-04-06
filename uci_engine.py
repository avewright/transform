"""UCI-compliant chess engine wrapping the 200M chess-transformer + MCTS.

Features:
  - MCTS search with policy prior + value backup (AlphaZero-style PUCT)
  - Batched leaf evaluation for GPU throughput
  - Tree reuse between moves (acts as pondering)
  - Pondering: continues MCTS during opponent's time
  - Adaptive time management: more time for complex positions
  - Syzygy endgame tablebases (≤5 pieces → perfect play)
  - Early termination when one move dominates

Usage:
  python uci_engine.py [--checkpoint PATH] [--syzygy PATH] [--default-sims N]

  Or configure in cutechess-cli / Arena / etc. as a UCI engine.

Value convention: White-absolute
  wdl[0] = P(White wins), wdl[1] = P(draw), wdl[2] = P(White loses)
"""

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model, ChessTransformerConfig
from move_vocab import (VOCAB_SIZE, index_to_move, legal_move_mask,
                        move_to_index, UCI_TO_IDX,
                        _CASTLE_STD_TO_960)
from opening_book import get_book_move

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Syzygy Tablebase ──

class SyzygyProbe:
    def __init__(self, path: str | Path | None = None):
        self.tb = None
        self.max_pieces = 5
        if path is None:
            path = ROOT / "syzygy"
        path = Path(path)
        if path.exists() and any(path.glob("*.rtbw")):
            try:
                self.tb = chess.syzygy.open_tablebase(str(path))
            except Exception:
                pass

    @property
    def available(self):
        return self.tb is not None

    def get_move(self, board: chess.Board):
        """Get perfect tablebase move for positions with ≤ max_pieces."""
        if self.tb is None or len(board.piece_map()) > self.max_pieces:
            return None
        try:
            best_move, best_wdl, best_dtz = None, -3, 0
            for move in board.legal_moves:
                board.push(move)
                try:
                    wdl = -self.tb.probe_wdl(board)
                    dtz = -self.tb.probe_dtz(board)
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

    def probe_value(self, board: chess.Board):
        """Get exact value from side-to-move perspective. Returns None if unavailable."""
        if self.tb is None or len(board.piece_map()) > self.max_pieces:
            return None
        try:
            wdl = self.tb.probe_wdl(board)
            if wdl >= 2:
                return 1.0
            elif wdl == 1:
                return 0.5
            elif wdl == 0:
                return 0.0
            elif wdl == -1:
                return -0.5
            else:
                return -1.0
        except Exception:
            return None


# ── MCTS Engine ──

class MCTSNode:
    __slots__ = ['prior', 'visit_count', 'value_sum', 'children', 'is_expanded',
                 '_deferred']

    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
        self._deferred = None

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSSearch:
    """MCTS/MCGS with batched NN eval, transpositions, tree reuse, and pondering.

    When use_transpositions=True, converts the search tree into a directed acyclic
    graph (DAG) by sharing nodes for identical positions reached via different move
    orders (Czech et al. 2020 Monte-Carlo Graph Search). This provides ~+50 ELO
    at the same simulation budget by avoiding redundant evaluations.
    """

    def __init__(self, model, device, syzygy: SyzygyProbe,
                 c_puct=2.5, batch_size=8, fpu_reduction=0.25,
                 root_noise_alpha=0.3, root_noise_frac=0.25,
                 use_fp16=True, policy_temp=1.0,
                 root_widening=0, inner_temp=1.0,
                 use_transpositions=True):
        self.model = model
        self.device = device
        self.syzygy = syzygy
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.fpu_reduction = fpu_reduction
        self.root_noise_alpha = root_noise_alpha
        self.root_noise_frac = root_noise_frac
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.policy_temp = policy_temp
        # Progressive widening at root: only expand top-K moves initially.
        # 0 = disabled (expand all). K>0 = start with K, add 1 per K visits.
        self.root_widening = root_widening
        # Inner tree temperature: sharpen policy at non-root depth for exploitation
        self.inner_temp = inner_temp
        self.use_transpositions = use_transpositions
        self.root = None
        self.root_board = None
        self.nn_evals = 0
        # Transposition table: zobrist hash → MCTSNode
        self._tt = {}  # cleared per game via new_game()
        self._tt_hits = 0
        # Pondering control
        self._stop_event = threading.Event()
        self._ponder_thread = None
        self._lock = threading.Lock()

    def reset_stats(self):
        self.nn_evals = 0
        self._tt_hits = 0

    def _fp16_safe_forward(self, inp):
        """Run backbone in FP16 for speed, heads in FP32 for precision.

        Policy logits are ~2000+ magnitude where FP16 precision is ±2,
        causing near-uniform softmax output. Heads must stay FP32.
        """
        m = self.model
        with torch.amp.autocast('cuda'):
            hidden = m.input_proj(m.encoder(inp))
            B = hidden.shape[0]
            hidden = torch.cat([m.cls_token.expand(B, -1, -1), hidden], dim=1)
            if m.pos_embed is not None:
                hidden = hidden + m.pos_embed
            hidden = m.norm(m.transformer(hidden))
        # Cast to FP32 before heads to preserve policy logit precision
        hidden = hidden.float()
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": m.policy_head(hidden, cls_hidden),
            "value_logits": m.value_head(cls_hidden),
        }

    @torch.no_grad()
    def _batch_evaluate(self, boards, is_root=False):
        """Evaluate multiple boards. Returns list of (policy_dict, stm_value).
        
        is_root: if True, use policy_temp; if False, use inner_temp.
        """
        if not boards:
            return []
        inp = batch_boards_to_fused_token_ids(boards, self.device)
        if self.use_fp16:
            r = self._fp16_safe_forward(inp)
        else:
            r = self.model(inp)
        
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
                p = probs[idx].item()
                # Castling: combine probability from both UCI formats
                # (model trained on e1h1 but python-chess uses e1g1)
                uci = m.uci()
                if uci in _CASTLE_STD_TO_960:
                    alt_idx = UCI_TO_IDX[_CASTLE_STD_TO_960[uci]]
                    p += probs[alt_idx].item()
                policy[m] = p
            # Value: White-absolute → side-to-move
            wdl = F.softmax(r["value_logits"][i].float(), dim=-1)
            white_val = (wdl[0] - wdl[2]).item()
            stm_val = white_val if board.turn == chess.WHITE else -white_val
            results.append((policy, stm_val))
        self.nn_evals += len(boards)
        return results

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

    def _add_root_noise(self, root):
        if not root.children or self.root_noise_frac <= 0:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.root_noise_alpha] * len(moves))
        frac = self.root_noise_frac
        for i, m in enumerate(moves):
            child = root.children[m]
            child.prior = (1 - frac) * child.prior + frac * noise[i]

    def _tt_lookup(self, board):
        """Look up a position in the transposition table. Returns node or None."""
        if not self.use_transpositions:
            return None
        zhash = chess.polyglot.zobrist_hash(board)
        return self._tt.get(zhash)

    def _tt_store(self, board, node):
        """Store a node in the transposition table."""
        if self.use_transpositions:
            self._tt[chess.polyglot.zobrist_hash(board)] = node

    def _expand_node(self, node, board, max_children=0, is_root=False):
        """Expand a single node. Returns stm_value.
        
        max_children: if >0, only expand top-K moves by policy prior.
        Remaining moves stored in node._deferred for later widening.
        """
        # Try Syzygy
        sv = self.syzygy.probe_value(board)
        if sv is not None:
            legal = list(board.legal_moves)
            if legal:
                prior = 1.0 / len(legal)
                for m in legal:
                    node.children[m] = MCTSNode(prior=prior)
            node.is_expanded = True
            self._tt_store(board, node)
            return sv
        policy, value = self._batch_evaluate([board], is_root=is_root)[0]
        if max_children > 0 and len(policy) > max_children:
            # Sort by prior descending, expand only top-K
            sorted_moves = sorted(policy.items(), key=lambda x: -x[1])
            for move, prob in sorted_moves[:max_children]:
                node.children[move] = MCTSNode(prior=prob)
            # Store deferred moves for later widening
            node._deferred = sorted_moves[max_children:]
        else:
            for move, prob in policy.items():
                node.children[move] = MCTSNode(prior=prob)
        node.is_expanded = True
        self._tt_store(board, node)
        return value

    def _widen_root(self, root, n_to_add=1):
        """Add next best deferred move(s) to root via progressive widening."""
        deferred = getattr(root, '_deferred', None)
        if not deferred:
            return
        added = 0
        while deferred and added < n_to_add:
            move, prob = deferred.pop(0)
            root.children[move] = MCTSNode(prior=prob)
            added += 1

    def _run_sims(self, root, base_board, max_sims=None, stop_event=None):
        """Run MCTS simulations. Returns number of sims completed.

        Stops when max_sims reached, stop_event is set, or move is stable.
        """
        VIRTUAL_LOSS = 1
        sims_done = 0
        last_widen_at = 0  # Track when we last widened

        while True:
            if max_sims is not None and sims_done >= max_sims:
                break
            if stop_event is not None and stop_event.is_set():
                break

            # Progressive widening at root: add 1 move every root_widening visits
            if (self.root_widening > 0 and root.visit_count > 0
                    and root.visit_count >= last_widen_at + self.root_widening):
                self._widen_root(root, n_to_add=1)
                last_widen_at = root.visit_count

            # Early termination: if one move can't be overtaken
            if sims_done >= 40 and max_sims is not None and root.children:
                visits = sorted([c.visit_count for c in root.children.values()], reverse=True)
                if len(visits) >= 2:
                    remaining = max_sims - sims_done
                    if visits[1] + remaining < visits[0]:
                        break
                    # Stability check: if top move has >70% of visits after
                    # using at least 25% of budget, the position is likely clear
                    total_visits = sum(visits)
                    if (total_visits > 0
                            and sims_done >= max_sims * 0.25
                            and visits[0] / total_visits > 0.70
                            and visits[0] - visits[1] > max_sims * 0.15):
                        break

            # Collect a batch of leaves
            batch_boards = []
            batch_paths = []
            batch_nodes = []

            batch_target = min(self.batch_size,
                               (max_sims - sims_done) if max_sims else self.batch_size)

            for _ in range(batch_target):
                node = root
                scratch = base_board.copy()
                path = [node]

                while node.is_expanded and node.children:
                    if scratch.is_game_over(claim_draw=True):
                        break
                    move, child = self._select_child(node)
                    if move is None:
                        break
                    scratch.push(move)
                    # MCGS: check transposition table for DAG sharing
                    tt_node = self._tt_lookup(scratch)
                    if tt_node is not None and tt_node is not child:
                        # Relink child to shared node (DAG)
                        node.children[move] = tt_node
                        child = tt_node
                        self._tt_hits += 1
                    path.append(child)
                    node = child

                # Terminal
                if scratch.is_game_over(claim_draw=True):
                    outcome = scratch.outcome(claim_draw=True)
                    if outcome is None or outcome.winner is None:
                        leaf_value = 0.0  # draw / stalemate
                    else:
                        # Checkmate: winner is opponent of scratch.turn (the mated side)
                        leaf_value = -1.0
                    value = leaf_value
                    for n in reversed(path):
                        n.visit_count += 1
                        n.value_sum += value
                        value = -value
                    sims_done += 1
                    continue

                # Syzygy leaf
                sv = self.syzygy.probe_value(scratch)
                if sv is not None and not node.is_expanded:
                    legal = list(scratch.legal_moves)
                    if legal:
                        prior = 1.0 / len(legal)
                        for m in legal:
                            node.children[m] = MCTSNode(prior=prior)
                    node.is_expanded = True
                    self._tt_store(scratch, node)
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
                        n.value_sum -= VIRTUAL_LOSS
                    batch_boards.append(scratch)
                    batch_paths.append(path)
                    batch_nodes.append(node)
                else:
                    value = node.q_value()
                    for n in reversed(path):
                        n.visit_count += 1
                        n.value_sum += value
                        value = -value
                    sims_done += 1

            # Batch evaluate
            if batch_boards:
                results = self._batch_evaluate(batch_boards)
                for board_i, path, node, (policy, value) in zip(
                        batch_boards, batch_paths, batch_nodes, results):
                    # Remove virtual loss
                    for n in path:
                        n.visit_count -= VIRTUAL_LOSS
                        n.value_sum += VIRTUAL_LOSS
                    # Expand
                    for move, prob in policy.items():
                        node.children[move] = MCTSNode(prior=prob)
                    node.is_expanded = True
                    # Store in transposition table
                    self._tt_store(board_i, node)
                    # Backup
                    v = value
                    for n in reversed(path):
                        n.visit_count += 1
                        n.value_sum += v
                        v = -v
                    sims_done += 1

        return sims_done

    def _ensure_root(self, board):
        """Ensure root node exists and is expanded for the given board."""
        if self.root is None or not self.root.is_expanded:
            self.root = MCTSNode()
            self._expand_node(self.root, board,
                              max_children=self.root_widening,
                              is_root=True)
            self.root_board = board.copy()

    def search(self, board, max_sims=None, time_limit=None):
        """Run MCTS search. Returns (best_move, info_dict).

        Uses either sim count or time limit (whichever stops first).
        """
        self.reset_stats()
        t0 = time.time()

        # Syzygy at game level — perfect play
        tb_move = self.syzygy.get_move(board)
        if tb_move is not None:
            return tb_move, {"source": "syzygy", "nn_evals": 0, "sims": 0,
                             "elapsed": time.time() - t0}

        # Opening book — principled mainlines for first ~10 moves
        book_move = get_book_move(board)
        if book_move is not None:
            return book_move, {"source": "book", "nn_evals": 0, "sims": 0,
                               "elapsed": time.time() - t0}

        with self._lock:
            self._ensure_root(board)
            root = self.root

            if not root.children:
                return list(board.legal_moves)[0], {
                    "nn_evals": 0, "sims": 0, "elapsed": 0}

            # Only one legal move → return immediately
            if len(root.children) == 1:
                move = next(iter(root.children))
                return move, {"nn_evals": self.nn_evals, "sims": 0,
                              "elapsed": time.time() - t0, "forced": True}

            # Add Dirichlet noise at root for exploration
            self._add_root_noise(root)

        # Run simulations with time limit
        if time_limit is not None and max_sims is None:
            # Time-based search: run batches until time runs out
            stop = threading.Event()

            def _timer():
                time.sleep(time_limit)
                stop.set()

            timer_thread = threading.Thread(target=_timer, daemon=True)
            timer_thread.start()
            with self._lock:
                total_sims = self._run_sims(root, board, max_sims=100000,
                                            stop_event=stop)
            stop.set()
        elif time_limit is not None and max_sims is not None:
            stop = threading.Event()

            def _timer():
                time.sleep(time_limit)
                stop.set()

            timer_thread = threading.Thread(target=_timer, daemon=True)
            timer_thread.start()
            with self._lock:
                total_sims = self._run_sims(root, board, max_sims=max_sims,
                                            stop_event=stop)
            stop.set()
        else:
            max_sims = max_sims or 200
            with self._lock:
                total_sims = self._run_sims(root, board, max_sims=max_sims)

        # Select best move by visit count
        with self._lock:
            best_move = max(root.children.items(),
                            key=lambda x: x[1].visit_count)[0]

        elapsed = time.time() - t0

        # Build PV info
        pv_moves = self._extract_pv(root)
        score_cp = self._q_to_cp(root.q_value())

        info = {
            "nn_evals": self.nn_evals,
            "sims": total_sims,
            "elapsed": elapsed,
            "score_cp": score_cp,
            "pv": pv_moves,
            "root_visits": root.visit_count,
            "tt_size": len(self._tt),
            "tt_hits": self._tt_hits,
        }
        return best_move, info

    def _extract_pv(self, root):
        """Extract principal variation from tree by following most-visited children."""
        pv = []
        node = root
        for _ in range(20):  # max PV length
            if not node.children:
                break
            best_move = max(node.children.items(),
                            key=lambda x: x[1].visit_count)[0]
            pv.append(best_move.uci())
            node = node.children[best_move]
        return pv

    def _q_to_cp(self, q):
        """Convert Q value [-1, +1] to centipawn score."""
        # Clamp to avoid log(0)
        q = max(-0.999, min(0.999, q))
        # Logistic scaling similar to Lc0: cp = 111.714 * tan(1.5620688 * q)
        return int(111.714 * math.tan(1.5620688 * q))

    def advance_tree(self, move, decay=0.0):
        """Advance tree after a move (tree reuse).

        decay: fraction of visit counts to discard (0=keep all, 1=reset).
               At low sim counts (100), decay=0.5-0.75 helps by allowing
               re-exploration of children that inherited stale visit counts.
        """
        with self._lock:
            if self.root is not None and move in self.root.children:
                new_root = self.root.children[move]
                if decay > 0:
                    self._decay_visits(new_root, decay)
                self.root = new_root
            else:
                self.root = None

    def _decay_visits(self, node, decay):
        """Recursively decay visit counts to allow re-exploration."""
        keep = 1.0 - decay
        node.visit_count = max(0, int(node.visit_count * keep))
        node.value_sum *= keep
        for child in node.children.values():
            if child.visit_count > 0:
                self._decay_visits(child, decay)

    def start_pondering(self, board, ponder_move):
        """Start pondering: continue MCTS assuming opponent plays ponder_move."""
        self._stop_event.clear()

        def _ponder_loop():
            # Advance tree to the position after ponder_move
            with self._lock:
                if self.root and ponder_move in self.root.children:
                    ponder_root = self.root.children[ponder_move]
                else:
                    return
                ponder_board = board.copy()
                ponder_board.push(ponder_move)

            # Continue expanding from ponder position
            self._run_sims(ponder_root, ponder_board,
                           max_sims=100000, stop_event=self._stop_event)

        self._ponder_thread = threading.Thread(target=_ponder_loop, daemon=True)
        self._ponder_thread.start()

    def stop_pondering(self):
        """Stop pondering thread."""
        self._stop_event.set()
        if self._ponder_thread is not None:
            self._ponder_thread.join(timeout=2.0)
            self._ponder_thread = None

    def new_game(self):
        """Reset for a new game."""
        self.stop_pondering()
        self.root = None
        self.root_board = None
        self._tt.clear()
        self._tt_hits = 0


# ── Adaptive Time Management ──

class TimeManager:
    """Allocate search time based on position complexity and remaining time."""

    def __init__(self, default_sims=200):
        self.default_sims = default_sims

    def compute_time(self, board: chess.Board, wtime_ms: int, btime_ms: int,
                     winc_ms: int = 0, binc_ms: int = 0,
                     movestogo: int = 0, movetime_ms: int = 0) -> float:
        """Return time in seconds to allocate for this move."""
        if movetime_ms > 0:
            # Fixed time per move
            return movetime_ms / 1000.0 * 0.95  # 5% safety margin

        my_time = wtime_ms if board.turn == chess.WHITE else btime_ms
        my_inc = winc_ms if board.turn == chess.WHITE else binc_ms

        if my_time <= 0:
            return 0.5  # fallback

        # Estimate moves remaining
        if movestogo > 0:
            moves_left = movestogo
        else:
            # Heuristic: game phase estimation
            move_num = board.fullmove_number
            if move_num < 20:
                moves_left = 40
            elif move_num < 40:
                moves_left = 25
            else:
                moves_left = 15

        # Base allocation
        base_time = my_time / moves_left + my_inc * 0.85

        # Complexity multiplier based on position features
        multiplier = self._complexity_multiplier(board)

        # Cap: never use more than 20% of remaining time or 30 seconds
        allocated = base_time * multiplier
        max_time = min(my_time * 0.20, 30000)
        allocated = min(allocated, max_time)

        # Floor: at least 100ms
        allocated = max(allocated, 100)

        return allocated / 1000.0

    def compute_sims(self, board: chess.Board, base_sims: int = None) -> int:
        """Compute adaptive simulation count based on position complexity."""
        if base_sims is None:
            base_sims = self.default_sims
        multiplier = self._complexity_multiplier(board)
        return max(20, int(base_sims * multiplier))

    def _complexity_multiplier(self, board: chess.Board) -> float:
        """Compute position complexity multiplier (0.5 = simple, 2.0 = complex)."""
        mult = 1.0
        n_legal = board.legal_moves.count()

        # More legal moves → more complex
        if n_legal > 40:
            mult *= 1.3
        elif n_legal > 30:
            mult *= 1.1
        elif n_legal < 10:
            mult *= 0.6
        elif n_legal < 20:
            mult *= 0.8

        # Check situations → tactical, need more time
        if board.is_check():
            mult *= 1.4

        # Endgame with few pieces → simpler (especially with tablebases)
        piece_count = len(board.piece_map())
        if piece_count <= 7:
            mult *= 0.5
        elif piece_count <= 12:
            mult *= 0.7

        # Open position (few pawns) → more tactical
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        if white_pawns + black_pawns <= 6:
            mult *= 1.2

        return max(0.3, min(2.5, mult))


# ── UCI Protocol ──

class UCIEngine:
    """UCI protocol handler."""

    def __init__(self, checkpoint_path: str, syzygy_path: str = None,
                 default_sims: int = 800):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.search = None
        self.time_manager = TimeManager(default_sims=default_sims)
        self.syzygy = SyzygyProbe(syzygy_path)
        self.default_sims = default_sims
        self.board = chess.Board()
        self.ponder = False
        self.debug = False
        self._load_model()

    def _load_model(self):
        self.model = build_model()
        ckpt = torch.load(self.checkpoint_path, map_location=DEVICE,
                          weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in state.items()})
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.search = MCTSSearch(
            self.model, DEVICE, self.syzygy,
            c_puct=2.5, batch_size=8,
        )

    def run(self):
        """Main UCI loop."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                self._cmd_uci()
            elif cmd == "isready":
                self._cmd_isready()
            elif cmd == "setoption":
                self._cmd_setoption(tokens)
            elif cmd == "ucinewgame":
                self._cmd_ucinewgame()
            elif cmd == "position":
                self._cmd_position(tokens)
            elif cmd == "go":
                self._cmd_go(tokens)
            elif cmd == "stop":
                self._cmd_stop()
            elif cmd == "ponderhit":
                self._cmd_ponderhit()
            elif cmd == "quit":
                self.search.stop_pondering()
                break
            elif cmd == "debug":
                self.debug = len(tokens) > 1 and tokens[1] == "on"

    def _send(self, msg):
        print(msg, flush=True)

    def _cmd_uci(self):
        self._send("id name Transform-MCTS")
        self._send("id author avewright")
        self._send(f"option name DefaultSims type spin default {self.default_sims} "
                    f"min 1 max 10000")
        self._send("option name CPuct type string default 2.5")
        self._send("option name Ponder type check default true")
        self._send("option name SyzygyPath type string default syzygy")
        self._send("uciok")

    def _cmd_isready(self):
        self._send("readyok")

    def _cmd_setoption(self, tokens):
        # Parse: setoption name <name> value <value>
        try:
            name_idx = tokens.index("name") + 1
            value_idx = tokens.index("value") + 1
            name = " ".join(tokens[name_idx:tokens.index("value")])
            value = " ".join(tokens[value_idx:])

            if name.lower() == "defaultsims":
                self.default_sims = int(value)
                self.time_manager.default_sims = int(value)
            elif name.lower() == "cpuct":
                self.search.c_puct = float(value)
            elif name.lower() == "ponder":
                self.ponder = value.lower() == "true"
            elif name.lower() == "syzygypath":
                self.syzygy = SyzygyProbe(value)
                self.search.syzygy = self.syzygy
        except (ValueError, IndexError):
            pass

    def _cmd_ucinewgame(self):
        self.search.new_game()
        self.board = chess.Board()

    def _cmd_position(self, tokens):
        self.search.stop_pondering()

        idx = 1
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            fen_parts = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))

        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            prev_board = self.board.copy()
            for uci in tokens[idx:]:
                move = chess.Move.from_uci(uci)
                # Advance search tree for tree reuse
                self.search.advance_tree(move)
                self.board.push(move)

    def _cmd_go(self, tokens):
        self.search.stop_pondering()

        # Parse go parameters
        params = {}
        i = 1
        while i < len(tokens):
            key = tokens[i]
            if key in ("wtime", "btime", "winc", "binc", "movestogo",
                       "depth", "nodes", "movetime"):
                if i + 1 < len(tokens):
                    params[key] = int(tokens[i + 1])
                    i += 2
                else:
                    i += 1
            elif key == "ponder":
                params["ponder"] = True
                i += 1
            elif key == "infinite":
                params["infinite"] = True
                i += 1
            else:
                i += 1

        # Determine search limits
        if "movetime" in params:
            time_limit = params["movetime"] / 1000.0 * 0.95
            max_sims = None
        elif "wtime" in params or "btime" in params:
            time_limit = self.time_manager.compute_time(
                self.board,
                wtime_ms=params.get("wtime", 60000),
                btime_ms=params.get("btime", 60000),
                winc_ms=params.get("winc", 0),
                binc_ms=params.get("binc", 0),
                movestogo=params.get("movestogo", 0),
            )
            max_sims = None
        elif "nodes" in params:
            max_sims = params["nodes"]
            time_limit = None
        elif "infinite" in params or "ponder" in params:
            max_sims = None
            time_limit = None  # will be stopped by "stop" command
        else:
            # Default: use fixed sim count (adaptive was ~1645 ELO, fixed 800 validated at 2077)
            max_sims = self.default_sims
            time_limit = None

        # Run search
        if time_limit is not None or "infinite" not in params:
            best_move, info = self.search.search(
                self.board, max_sims=max_sims, time_limit=time_limit)
        else:
            # Infinite search — run until stop
            self.search._stop_event.clear()
            self.search._ensure_root(self.board)
            self.search._add_root_noise(self.search.root)

            def _infinite_search():
                self.search._run_sims(
                    self.search.root, self.board,
                    max_sims=1000000, stop_event=self.search._stop_event)

            t = threading.Thread(target=_infinite_search, daemon=True)
            t.start()
            t.join()  # Will be interrupted by stop command
            best_move = max(self.search.root.children.items(),
                            key=lambda x: x[1].visit_count)[0]
            info = {"sims": self.search.root.visit_count, "pv": [],
                    "score_cp": 0}

        # Send info
        pv_str = " ".join(info.get("pv", [best_move.uci()]))
        score_cp = info.get("score_cp", 0)
        nodes = info.get("sims", 0) + info.get("nn_evals", 0)
        elapsed_ms = max(1, int(info.get("elapsed", 0.001) * 1000))
        nps = int(nodes / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        self._send(f"info depth 1 seldepth {len(info.get('pv', []))} "
                    f"score cp {score_cp} nodes {nodes} "
                    f"nps {nps} time {elapsed_ms} pv {pv_str}")

        # Determine ponder move (most visited child of best move's subtree)
        ponder_uci = ""
        if self.ponder and best_move in self.search.root.children:
            best_child = self.search.root.children[best_move]
            if best_child.children:
                ponder_move = max(best_child.children.items(),
                                   key=lambda x: x[1].visit_count)[0]
                ponder_uci = f" ponder {ponder_move.uci()}"

        self._send(f"bestmove {best_move.uci()}{ponder_uci}")

        # Advance tree for next search
        self.search.advance_tree(best_move)

        # Start pondering on predicted opponent move
        if self.ponder and ponder_uci:
            ponder_move = chess.Move.from_uci(ponder_uci.strip().split()[-1])
            ponder_board = self.board.copy()
            ponder_board.push(best_move)
            self.search.start_pondering(ponder_board, ponder_move)

    def _cmd_stop(self):
        """Stop current search or pondering."""
        self.search._stop_event.set()
        self.search.stop_pondering()

        # If search is still running, the thread will exit and the go handler
        # will send bestmove. If pondering, we just stop it silently.

    def _cmd_ponderhit(self):
        """Opponent played our predicted move — continue searching."""
        # The pondering thread was expanding the right subtree.
        # Stop pondering and let the next "go" reuse the expanded tree.
        self.search.stop_pondering()


# ── Entry point ──

def find_checkpoint():
    """Find the best model checkpoint."""
    candidates = [
        ROOT / "outputs" / "hf" / "chess-transformer-200m-latest" / "best_model.pt",
        ROOT / "outputs" / "hf_checkpoint" / "best_model.pt",
        Path.home() / ".cache" / "huggingface" / "hub" /
        "models--avewright--chess-transformer-200m-latest" /
        "snapshots" / "b8f432d18c54ae56eac4568dc72a5b3f7bb6a288" / "best_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Try huggingface_hub download
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("avewright/chess-transformer-200m-latest",
                               "best_model.pt")
    except Exception:
        pass
    raise FileNotFoundError(
        "No checkpoint found. Place best_model.pt in outputs/hf_checkpoint/ "
        "or install huggingface_hub.")


def main():
    parser = argparse.ArgumentParser(description="Transform-MCTS UCI Engine")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--syzygy", default=None,
                        help="Path to Syzygy tablebase directory")
    parser.add_argument("--default-sims", type=int, default=200,
                        help="Default MCTS simulations per move")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_checkpoint()
    engine = UCIEngine(ckpt, args.syzygy, args.default_sims)
    engine.run()


if __name__ == "__main__":
    main()
