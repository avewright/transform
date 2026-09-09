"""Chess-specific label contracts. Probabilities are never logits."""
from __future__ import annotations

from typing import Any

import numpy as np

from chess_master.identity import reconstruct_board
from chess_master.schema import (
    LICHESS_CP_MATE_SENTINEL,
    LICHESS_DEPTH_SENTINEL_MIN,
    LICHESS_MIN_DEPTH,
    SF19_IMPLAUSIBLE_DEPTH,
    SF19_MIN_BUDGET,
    SF19_MIN_DEPTH,
    SYZYGY_DEPTH_SENTINEL,
    TB_WDL_VALUES,
    UNKNOWN,
    VOCAB_ADAPTER_VERSION,
)

try:
    from move_vocab import VOCAB_SIZE, index_to_move
except Exception:  # pragma: no cover
    VOCAB_SIZE = 1968

    def index_to_move(idx: int):
        raise KeyError(idx)


def as_list(val, n=None) -> list:
    if val is None:
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    out = list(val)
    if n is not None:
        out = out[:n]
    return out


def finite_probs(probs) -> bool:
    p = np.asarray(as_list(probs), dtype=np.float64)
    if p.size == 0 or not np.isfinite(p).all() or (p < -1e-8).any():
        return False
    return True


def probs_normalized(probs, *, tol: float = 0.001) -> bool:
    p = np.asarray(as_list(probs), dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return False
    return abs(float(p.sum()) - 1.0) <= tol


def pad8_indices(indices) -> list[int]:
    out = [-1] * 8
    for i, v in enumerate(as_list(indices)[:8]):
        out[i] = int(v)
    return out


def pad8_probs(probs) -> list[float]:
    out = [0.0] * 8
    for i, v in enumerate(as_list(probs)[:8]):
        out[i] = float(v)
    return out


def depth_sentinel(source_name: str, depth, *, nodes=None, budget=None) -> tuple[bool, str | None]:
    if depth is None:
        return False, None
    d = int(depth)
    if source_name == "syzygy" and d == SYZYGY_DEPTH_SENTINEL:
        return True, "syzygy_label_depth_999"
    if source_name == "lichess" and d >= LICHESS_DEPTH_SENTINEL_MIN:
        return True, "lichess_depth_ge_128"
    if source_name == "sf19" and d > SF19_IMPLAUSIBLE_DEPTH:
        return True, "sf19_implausible_depth"
    if source_name == "sf19" and nodes is not None and budget is not None:
        if int(budget) >= SF19_MIN_BUDGET and int(nodes) < 1000 and d >= 100:
            return True, "sf19_depth_vs_tiny_nodes"
    return False, None


def lichess_cp_is_mate_sentinel(cp, mate) -> bool:
    try:
        return int(mate) == 0 and abs(int(cp)) >= LICHESS_CP_MATE_SENTINEL
    except (TypeError, ValueError):
        return False


def value_wdl_ok(wdl) -> bool:
    w = np.asarray(as_list(wdl), dtype=np.float64).reshape(-1)
    if w.size != 3 or not np.isfinite(w).all() or (w < -1e-8).any():
        return False
    return abs(float(w.sum()) - 1.0) <= 0.001


def syzygy_meta_ok(tb_wdl, dtz) -> tuple[bool, str]:
    try:
        w = int(tb_wdl)
        z = int(dtz)
    except (TypeError, ValueError):
        return False, "meta_unreadable"
    if w not in TB_WDL_VALUES:
        return False, "tb_wdl_range"
    if abs(z) > 1000:
        return False, "dtz_range"
    return True, "ok"


def policy_indices_ok(indices, probs) -> bool:
    si = pad8_indices(indices)
    sp = pad8_probs(probs)
    active = [i for i, p in zip(si, sp) if p > 0]
    if not active:
        return False
    if any(i < 0 or i >= VOCAB_SIZE for i in active):
        return False
    return len(set(active)) == len(active)


def legal_policy(board_array, turn, castling, ep_square, move_idx, indices, probs) -> tuple[bool, str]:
    if not finite_probs(probs) or not probs_normalized(probs) or not policy_indices_ok(indices, probs):
        return False, "invalid_policy"
    board = reconstruct_board(board_array, turn, castling, ep_square)
    if board is None:
        return False, "invalid_position"
    if board.is_game_over(claim_draw=False):
        return False, "terminal_position"
    try:
        moves = [index_to_move(int(move_idx))]
        si, sp = pad8_indices(indices), pad8_probs(probs)
        moves.extend(index_to_move(i) for i, p in zip(si, sp) if p > 0)
    except (IndexError, ValueError, KeyError):
        return False, "move_index_unreadable"
    if not all(mv in board.legal_moves for mv in moves):
        return False, "illegal_move"
    return True, "ok"


def regret_status(model_in_pv, drop_cp) -> tuple[int | None, str]:
    """A move missing from MultiPV has unknown regret, not an assumed penalty."""
    if model_in_pv is None:
        return None, UNKNOWN
    if int(model_in_pv) == 1:
        if drop_cp is None:
            return None, UNKNOWN
        return int(drop_cp), "verified"
    return None, "unknown_off_pv"


def qualify_source(source_name: str, fields: dict[str, Any]) -> str | None:
    """Hard quality gate used by recipes. Never silently relaxed."""
    if source_name == "sf19":
        if int(fields.get("split", 0) or 0) != 0:
            return "upstream_holdout"
        if int(fields.get("policy_mask", 1) or 0) != 1:
            return "teacher_quality"
        budget = int(fields.get("nodes_requested") or fields.get("nodes_budget") or 0)
        depth = int(fields.get("depth") or fields.get("label_depth") or 0)
        sentinel, _ = depth_sentinel("sf19", depth, nodes=fields.get("nodes_achieved"), budget=budget)
        if sentinel:
            return "depth_or_sentinel"
        if budget < SF19_MIN_BUDGET or depth < SF19_MIN_DEPTH:
            return "teacher_quality"
        wdl = fields.get("wdl")
        if wdl is None:
            wdl = fields.get("value_wdl")
        if wdl is not None and not value_wdl_ok(wdl):
            return "invalid_value"
    if source_name == "lichess":
        depth = int(fields.get("depth") or fields.get("label_depth") or 0)
        sentinel, _ = depth_sentinel("lichess", depth)
        if sentinel or not (LICHESS_MIN_DEPTH <= depth < LICHESS_DEPTH_SENTINEL_MIN):
            return "depth_or_sentinel"
    if source_name == "syzygy":
        ok, reason = syzygy_meta_ok(fields.get("tb_wdl"), fields.get("tb_dtz", fields.get("dtz")))
        if not ok:
            return reason
    indices = fields.get("soft_indices")
    if indices is None:
        indices = fields.get("policy_indices")
    probs = fields.get("soft_probs")
    if probs is None:
        probs = fields.get("policy_probs")
    ok, reason = legal_policy(
        fields["board_array"], fields["turn"], fields["castling"], fields["ep_square"],
        fields["move_idx"], indices, probs,
    )
    if not ok:
        return reason
    return None


def vocab_note() -> str:
    return VOCAB_ADAPTER_VERSION
