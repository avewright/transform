"""Normalize source rows into position / annotation / membership records."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]

from chess_master.contracts import (
    depth_sentinel,
    lichess_cp_is_mate_sentinel,
    pad8_indices,
    pad8_probs,
    regret_status,
    value_wdl_ok,
)
from chess_master.identity import namespace_game_id, position_record
from chess_master.io_util import annotation_id, membership_id, search_record_id
from chess_master.schema import (
    ANNOTATION_ENGINE_POLICY,
    ANNOTATION_MODEL,
    ANNOTATION_PUZZLE,
    ANNOTATION_TABLEBASE,
    PERSPECTIVE_STM,
    PERSPECTIVE_UNKNOWN,
    PERSPECTIVE_WHITE,
    POLICY_ONEHOT,
    POLICY_PROBS,
    POLICY_TB_SOFTMAX,
    POLICY_UNKNOWN,
    UNKNOWN,
    VOCAB_ADAPTER_VERSION,
)

try:
    from move_vocab import IDX_TO_UCI
except Exception:  # pragma: no cover
    IDX_TO_UCI = []


def _uci(idx) -> str | None:
    try:
        i = int(idx)
        if 0 <= i < len(IDX_TO_UCI):
            return IDX_TO_UCI[i]
    except (TypeError, ValueError):
        return None
    return None


def _themes(val) -> list[str] | None:
    if val is None:
        return None
    if isinstance(val, str):
        return [t for t in val.split() if t]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return [str(val)]


def _arr(val) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(val, dtype=np.int8)).reshape(64)


def _scalar(val, default=None):
    if val is None:
        return default
    if hasattr(val, "item"):
        return val.item()
    return val


def indices_to_uci(indices, probs) -> tuple[list[str], list[int], list[float]]:
    si = pad8_indices(indices)
    sp = pad8_probs(probs)
    uci, idx, pr = [], [], []
    for i, p in zip(si, sp):
        if p <= 0 or i < 0:
            continue
        move = _uci(i)
        if move is None:
            continue
        uci.append(move)
        idx.append(i)
        pr.append(float(p))
    return uci, idx, pr


def empty_annotation(**overrides) -> dict[str, Any]:
    row = {
        "search_record_id": None,
        "engine": None,
        "engine_version": None,
        "network": None,
        "nodes_requested": None,
        "nodes_achieved": None,
        "depth": None,
        "depth_is_sentinel": False,
        "depth_sentinel_reason": None,
        "multipv": None,
        "bound_skipped": None,
        "flags": None,
        "incomplete_search": None,
        "policy_kind": POLICY_PROBS,
        "policy_transform": POLICY_UNKNOWN,
        "tau": None,
        "vocab_adapter": VOCAB_ADAPTER_VERSION,
        "best_uci": None,
        "policy_uci": None,
        "policy_indices": None,
        "policy_probs": None,
        "policy_cps": None,
        "policy_mates": None,
        "trainer_soft_indices": [-1] * 8,
        "trainer_soft_probs": [0.0] * 8,
        "value_cp": None,
        "value_mate": None,
        "value_wdl": None,
        "value_perspective": PERSPECTIVE_UNKNOWN,
        "value_valid": np.int8(0),
        "value_eligible": np.int8(0),
        "policy_eligible": np.int8(1),
        "original_cp": None,
        "original_mate": None,
        "original_wdl_scalar": None,
        "original_perspective": PERSPECTIVE_UNKNOWN,
        "mate_is_dtz_proxy": np.int8(0),
        "lichess_cp_mate_sentinel": np.int8(0),
        "puzzle_id": None,
        "puzzle_game_id": None,
        "puzzle_rating": None,
        "puzzle_rating_deviation": None,
        "puzzle_themes": None,
        "puzzle_opening_tags": None,
        "puzzle_moves": None,
        "puzzle_setup_fen": None,
        "puzzle_solver_fen": None,
        "tb_wdl": None,
        "tb_dtz": None,
        "tb_n_pieces": None,
        "tb_rule": None,
        "tb_probe": None,
        "model_checkpoint": None,
        "model_checkpoint_hash": None,
        "model_move_uci": None,
        "model_in_pv": None,
        "model_tag": None,
        "model_drop_cp": None,
        "regret_cp": None,
        "regret_status": UNKNOWN,
        "quality_status": "accepted",
        "reject_reason": None,
        "source_phase": None,
        "source_fields_json": None,
    }
    row.update(overrides)
    return row


def base_from_encoded(board_array, turn, castling, ep_square, move_idx, soft_indices, soft_probs,
                      *, cp=None, mate=None, depth=None, phase=None, opening_eco=None,
                      opening_name=None, halfmove=None, fullmove=None) -> tuple[dict, dict]:
    pos = position_record(
        board_array, turn, castling, ep_square,
        halfmove=halfmove, fullmove=fullmove,
        opening_eco=opening_eco, opening_name=opening_name,
    )
    uci, idx, pr = indices_to_uci(soft_indices, soft_probs)
    ann = empty_annotation(
        best_uci=_uci(move_idx) or (uci[0] if uci else None),
        policy_uci=uci or None,
        policy_indices=idx or None,
        policy_probs=pr or None,
        trainer_soft_indices=pad8_indices(soft_indices),
        trainer_soft_probs=pad8_probs(soft_probs),
        original_cp=None if cp is None else int(cp),
        original_mate=None if mate is None else int(mate),
        depth=None if depth is None else int(depth),
        source_phase=None if phase is None else int(phase),
    )
    return pos, ann


def finish_annotation(pos: dict, ann: dict, *, source_name: str, source_revision: str | None,
                      source_path: str, source_row: int, source_license: str | None,
                      annotation_type: str) -> dict:
    sent, why = depth_sentinel(
        source_name, ann.get("depth"),
        nodes=ann.get("nodes_achieved"), budget=ann.get("nodes_requested"),
    )
    ann["depth_is_sentinel"] = bool(sent)
    ann["depth_sentinel_reason"] = why
    ann["annotation_id"] = annotation_id(source_name, source_path, source_row, annotation_type)
    ann["position_id"] = pos["position_id"]
    ann["search_record_id"] = search_record_id(source_name, source_path, source_row)
    ann["annotation_type"] = annotation_type
    ann["source_name"] = source_name
    ann["source_revision"] = source_revision or UNKNOWN
    ann["source_path"] = source_path
    ann["source_row"] = int(source_row)
    ann["source_license"] = source_license or UNKNOWN
    return ann


def membership_record(pos: dict, ann: dict, *, pool: str, split: str, game_raw, source_name: str,
                      mix_row: int | None = None, included: bool = True,
                      exposure: str = "pool_member") -> dict:
    raw, ns = namespace_game_id(source_name, game_raw)
    return {
        "membership_id": membership_id(pool, split, ann["annotation_id"]),
        "position_id": pos["position_id"],
        "annotation_id": ann["annotation_id"],
        "legacy_hash": pos["legacy_hash"],
        "equivalence_key": pos["equivalence_key"],
        "pool_name": pool,
        "split": split,
        "game_id_raw": raw,
        "game_id_ns": ns,
        "group_key": ns if raw else f"pos:{int(pos['legacy_hash'])}",
        "included_in_pool": included,
        "actually_sampled": UNKNOWN,
        "exposure_certainty": exposure,
        "checkpoint_lineage": UNKNOWN,
        "mix_row": mix_row,
    }


def from_sf19(d: dict, i: int, *, source_path: str, revision: str, extra: dict | None = None) -> tuple[dict, dict]:
    extra = extra or {}
    ba, turn, castle, ep = d["board_array"][i], d["turn"][i], d["castling"][i], d["ep_square"][i]
    pos, ann = base_from_encoded(
        ba, turn, castle, ep, d["move_idx"][i], d["soft_indices"][i], d["soft_probs"][i],
        cp=d["cp"][i], mate=d["mate"][i], depth=d.get("label_depth", [None])[i] if "label_depth" in d else extra.get("label_depth"),
        phase=d["phase"][i] if "phase" in d else None,
    )
    wdl = extra.get("wdl")
    if wdl is None and "wdl" in d:
        wdl = d["wdl"][i]
    if wdl is not None:
        w = np.asarray(wdl, dtype=np.float32).reshape(-1).tolist()
        ann["value_wdl"] = w
        ann["value_valid"] = np.int8(1 if value_wdl_ok(w) else 0)
        ann["value_eligible"] = ann["value_valid"]
    ann.update(
        engine="stockfish",
        engine_version="19",
        network=UNKNOWN,
        policy_kind=POLICY_PROBS,
        policy_transform="softmax_stm_rank_score_tau",
        tau=float(extra["tau"]) if extra.get("tau") is not None else (float(d["tau"][i]) if "tau" in d else 120.0),
        nodes_requested=int(extra["nodes_budget"]) if extra.get("nodes_budget") is not None else (int(d["nodes_budget"][i]) if "nodes_budget" in d else None),
        nodes_achieved=int(extra["nodes"]) if extra.get("nodes") is not None else (int(d["nodes"][i]) if "nodes" in d else None),
        multipv=8,
        bound_skipped=int(extra["bound_skipped"]) if extra.get("bound_skipped") is not None else (int(d["bound_skipped"][i]) if "bound_skipped" in d else None),
        flags=int(extra["flags"]) if extra.get("flags") is not None else (int(d["flags"][i]) if "flags" in d else None),
        value_cp=int(d["cp"][i]),
        value_mate=int(d["mate"][i]),
        value_perspective=PERSPECTIVE_WHITE,
        original_perspective=PERSPECTIVE_WHITE,
        policy_cps=list(np.asarray(d["soft_cps"][i], dtype=np.int32)) if "soft_cps" in d else None,
        policy_mates=list(np.asarray(d["soft_mates"][i], dtype=np.int32)) if "soft_mates" in d else None,
    )
    if extra.get("soft_cps") is not None:
        ann["policy_cps"] = [int(x) for x in extra["soft_cps"]]
    if extra.get("soft_mates") is not None:
        ann["policy_mates"] = [int(x) for x in extra["soft_mates"]]
    src_fields = {k: extra.get(k) for k in ("split", "policy_mask", "game_id", "origin", "ply") if k in extra}
    ann["source_fields_json"] = json.dumps(src_fields, default=str) if src_fields else None
    return pos, finish_annotation(
        pos, ann, source_name="sf19", source_revision=revision, source_path=source_path,
        source_row=i, source_license="mit", annotation_type=ANNOTATION_ENGINE_POLICY,
    )


def from_lichess(d: dict, i: int, *, source_path: str, revision: str) -> tuple[dict, dict]:
    pos, ann = base_from_encoded(
        d["board_array"][i], d["turn"][i], d["castling"][i], d["ep_square"][i],
        d["move_idx"][i], d["soft_indices"][i], d["soft_probs"][i],
        cp=d["cp"][i], mate=d["mate"][i],
        depth=d["label_depth"][i] if "label_depth" in d else None,
        phase=d["phase"][i] if "phase" in d else None,
    )
    cp, mate = int(d["cp"][i]), int(d["mate"][i])
    ann.update(
        engine=UNKNOWN,
        engine_version=UNKNOWN,
        network=UNKNOWN,
        policy_kind=POLICY_PROBS,
        policy_transform=POLICY_UNKNOWN,
        value_cp=cp,
        value_mate=mate,
        value_perspective=PERSPECTIVE_UNKNOWN,
        original_perspective=PERSPECTIVE_UNKNOWN,
        value_valid=np.int8(0),
        value_eligible=np.int8(0),
        lichess_cp_mate_sentinel=np.int8(1 if lichess_cp_is_mate_sentinel(cp, mate) else 0),
    )
    return pos, finish_annotation(
        pos, ann, source_name="lichess", source_revision=revision, source_path=source_path,
        source_row=i, source_license="mit", annotation_type=ANNOTATION_ENGINE_POLICY,
    )


def from_syzygy(d: dict, i: int, *, source_path: str, revision: str, extra: dict | None = None) -> tuple[dict, dict]:
    extra = extra or {}
    pos, ann = base_from_encoded(
        d["board_array"][i], d["turn"][i], d["castling"][i], d["ep_square"][i],
        d["move_idx"][i], d["soft_indices"][i], d["soft_probs"][i],
        cp=d["cp"][i], mate=d["mate"][i],
        depth=d["label_depth"][i] if "label_depth" in d else None,
        phase=d["phase"][i] if "phase" in d else None,
    )
    tb = extra.get("tb_wdl", extra.get("wdl"))
    if tb is None and "wdl" in d and not hasattr(d["wdl"][i], "shape"):
        tb = d["wdl"][i]
    if tb is None and "wdl" in d:
        w = np.asarray(d["wdl"][i])
        tb = int(w) if w.ndim == 0 else None
    dtz = extra.get("dtz", d["dtz"][i] if "dtz" in d else None)
    n_pieces = extra.get("n_pieces", d["n_pieces"][i] if "n_pieces" in d else pos["piece_count"])
    mate = int(d["mate"][i])
    ann.update(
        engine=None,
        policy_kind=POLICY_PROBS,
        policy_transform=POLICY_TB_SOFTMAX,
        tau=None,
        value_valid=np.int8(0),
        value_eligible=np.int8(0),
        value_perspective=PERSPECTIVE_STM,
        original_perspective=PERSPECTIVE_STM,
        original_wdl_scalar=None if tb is None else int(tb),
        mate_is_dtz_proxy=np.int8(1 if mate != 0 else 0),
        tb_wdl=None if tb is None else np.int8(int(tb)),
        tb_dtz=None if dtz is None else int(dtz),
        tb_n_pieces=None if n_pieces is None else np.int8(int(n_pieces)),
        tb_rule=UNKNOWN,
        tb_probe="python-chess.syzygy",
    )
    return pos, finish_annotation(
        pos, ann, source_name="syzygy", source_revision=revision, source_path=source_path,
        source_row=i, source_license="mit", annotation_type=ANNOTATION_TABLEBASE,
    )


def from_puzzle_packed(d: dict, i: int, meta: dict, *, source_path: str, revision: str,
                       source_row: int) -> tuple[dict, dict]:
    opening = meta.get("OpeningTags") or meta.get("opening_tags")
    eco = None
    name = None
    tags = _themes(opening)
    if tags:
        eco = tags[0]
        name = " ".join(tags)
    pos, ann = base_from_encoded(
        d["board_array"][i], d["turn"][i], d["castling"][i], d["ep_square"][i],
        d["move_idx"][i], d["soft_indices"][i], d["soft_probs"][i],
        cp=0, mate=0, depth=0, phase=d["phase"][i] if "phase" in d else None,
        opening_eco=eco, opening_name=name,
    )
    # Setup-FEN clocks are not the solver position's rule state.
    pos["halfmove"] = None
    pos["fullmove"] = None
    pos["rule_state_available"] = False
    pos["fen_6"] = None
    ann.update(
        policy_kind=POLICY_PROBS,
        policy_transform=POLICY_ONEHOT,
        tau=None,
        value_valid=np.int8(0),
        value_eligible=np.int8(0),
        value_perspective=PERSPECTIVE_UNKNOWN,
        puzzle_id=meta.get("PuzzleId"),
        puzzle_game_id=meta.get("GameId"),
        puzzle_rating=int(meta["Rating"]) if meta.get("Rating") not in (None, "") else None,
        puzzle_rating_deviation=int(meta["RatingDeviation"]) if meta.get("RatingDeviation") not in (None, "") else None,
        puzzle_themes=_themes(meta.get("Themes")),
        puzzle_opening_tags=tags,
        puzzle_moves=meta.get("Moves"),
        puzzle_setup_fen=meta.get("FEN"),
        puzzle_solver_fen=meta.get("solver_fen"),
        source_fields_json=json.dumps({k: meta.get(k) for k in ("Popularity", "NbPlays") if k in meta}, default=str) or None,
    )
    return pos, finish_annotation(
        pos, ann, source_name="puzzles", source_revision=revision, source_path=source_path,
        source_row=source_row, source_license="cc0-1.0", annotation_type=ANNOTATION_PUZZLE,
    )


def from_swa_scan(d: dict, i: int, *, source_path: str, revision: str) -> list[tuple[dict, dict]]:
    """Teacher policy plus a separate model-prediction annotation."""
    from harvest_swa_mistakes import I_TO_TAG

    pos, teacher = base_from_encoded(
        d["board_array"][i], d["turn"][i], d["castling"][i], d["ep_square"][i],
        d["move_idx"][i], d["soft_indices"][i], d["soft_probs"][i],
        cp=d["cp"][i], mate=d["mate"][i],
        depth=d["label_depth"][i] if "label_depth" in d else None,
        phase=d["phase"][i] if "phase" in d else None,
    )
    teacher.update(
        engine="stockfish",
        engine_version="19",
        network=UNKNOWN,
        policy_kind=POLICY_PROBS,
        policy_transform="softmax_stm_rank_score_tau",
        value_cp=int(d["cp"][i]),
        value_mate=int(d["mate"][i]),
        value_perspective=PERSPECTIVE_WHITE,
        original_perspective=PERSPECTIVE_WHITE,
        policy_cps=list(np.asarray(d["soft_cps"][i], dtype=np.int32)) if "soft_cps" in d else None,
        policy_mates=list(np.asarray(d["soft_mates"][i], dtype=np.int32)) if "soft_mates" in d else None,
    )
    teacher = finish_annotation(
        pos, teacher, source_name="swa_mistakes", source_revision=revision,
        source_path=source_path, source_row=i, source_license="mit",
        annotation_type=ANNOTATION_ENGINE_POLICY,
    )
    model_idx = int(d["model_move_idx"][i]) if "model_move_idx" in d else None
    in_pv = int(d["model_in_pv"][i]) if "model_in_pv" in d else None
    drop = int(d["drop_cp"][i]) if "drop_cp" in d else None
    tag_i = int(d["tag"][i]) if "tag" in d else None
    regret, status = regret_status(in_pv, drop)
    model = empty_annotation(
        best_uci=_uci(model_idx),
        policy_uci=[_uci(model_idx)] if _uci(model_idx) else None,
        policy_indices=[model_idx] if model_idx is not None and model_idx >= 0 else None,
        policy_probs=[1.0] if model_idx is not None and model_idx >= 0 else None,
        trainer_soft_indices=pad8_indices([model_idx] if model_idx is not None else []),
        trainer_soft_probs=pad8_probs([1.0] if model_idx is not None and model_idx >= 0 else []),
        policy_kind=POLICY_PROBS,
        policy_transform="model_greedy",
        policy_eligible=np.int8(0),
        value_eligible=np.int8(0),
        model_checkpoint="avewright/chess-transformer-100m-overnight_20260908",
        model_checkpoint_hash=UNKNOWN,
        model_move_uci=_uci(model_idx),
        model_in_pv=None if in_pv is None else np.int8(in_pv),
        model_tag=I_TO_TAG.get(tag_i, UNKNOWN) if tag_i is not None else UNKNOWN,
        model_drop_cp=drop,
        regret_cp=regret,
        regret_status=status,
    )
    model = finish_annotation(
        pos, model, source_name="swa_mistakes", source_revision=revision,
        source_path=source_path, source_row=i, source_license="mit",
        annotation_type=ANNOTATION_MODEL,
    )
    return [(pos, teacher), (pos, model)]


def from_mix_row(d: dict, i: int, source_name: str, *, source_path: str, revision: str,
                 extra: dict | None = None) -> tuple[dict, dict]:
    extra = extra or {}
    if source_name == "sf19":
        return from_sf19(d, i, source_path=source_path, revision=revision, extra=extra)
    if source_name == "lichess":
        return from_lichess(d, i, source_path=source_path, revision=revision)
    if source_name == "syzygy":
        return from_syzygy(d, i, source_path=source_path, revision=revision, extra=extra)
    if source_name == "puzzles":
        return from_puzzle_packed(d, i, extra, source_path=source_path, revision=revision,
                                  source_row=int(extra.get("input_row", i)))
    raise KeyError(source_name)
