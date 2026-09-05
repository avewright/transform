"""Correctness, checkpoint, validation, and data helpers for exp201.

Imported by train_trial. Safe to unit-test on CPU without touching a live run.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CKPT_FORMAT = 2
SOFT_PAD = -1


def soft_policy_loss(logits, soft_indices, soft_probs):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = (soft_indices >= 0) & (soft_probs > 0)
    safe = soft_indices.clamp(min=0).long()
    gathered = log_probs.gather(1, safe) * valid.float()
    return -(soft_probs.float() * gathered).sum(dim=-1).mean()


def soft_target_valid_mask(soft_indices, soft_probs):
    """Validity from indices and positive probabilities, before any softening."""
    return (soft_indices >= 0) & (soft_probs > 0)


def soften_policy_targets(soft_indices, soft_probs, temperature: float = 4.0):
    """Raise valid teacher mass to 1/T and renormalize. Invalid slots stay 0.

    Rows with no valid target become an all-zero distribution (loss 0, no NaN).
    """
    valid = soft_target_valid_mask(soft_indices, soft_probs)
    p = torch.where(valid, soft_probs.float(), torch.zeros_like(soft_probs, dtype=torch.float32))
    inv_t = 1.0 / max(float(temperature), 1e-6)
    p_t = torch.where(valid, p.clamp_min(0.0).pow(inv_t), torch.zeros_like(p))
    denom = p_t.sum(dim=-1, keepdim=True)
    has = denom > 0
    return torch.where(has, p_t / denom.clamp_min(1e-12), p_t)


def soft_temp_policy_loss(logits, soft_indices, soft_probs, temperature: float = 4.0):
    """Chessformer soft-policy aux with padding-safe temperature."""
    p_t = soften_policy_targets(soft_indices, soft_probs, temperature)
    return soft_policy_loss(logits, soft_indices, p_t)


def teacher_entropy(soft_indices, soft_probs):
    valid = soft_target_valid_mask(soft_indices, soft_probs)
    p = torch.where(valid, soft_probs.float(), torch.zeros_like(soft_probs, dtype=torch.float32))
    z = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = p / z
    ent = -(p.clamp_min(1e-12).log() * p).sum(dim=-1)
    ent = torch.where(z.squeeze(-1) > 0, ent, torch.zeros_like(ent))
    return ent.mean()


def teacher_kl(logits, soft_indices, soft_probs):
    """KL(teacher || model) over valid teacher support."""
    valid = soft_target_valid_mask(soft_indices, soft_probs)
    p = torch.where(valid, soft_probs.float(), torch.zeros_like(soft_probs, dtype=torch.float32))
    z = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = p / z
    log_q = F.log_softmax(logits.float(), dim=-1)
    safe = soft_indices.clamp(min=0).long()
    log_q_s = log_q.gather(1, safe)
    log_p = p.clamp_min(1e-12).log()
    kl = (p * (log_p - log_q_s) * valid.float()).sum(dim=-1)
    kl = torch.where(z.squeeze(-1) > 0, kl, torch.zeros_like(kl))
    return kl.mean()


def pick_mix_source(
    draw: float,
    bonus_mix: float,
    deep_mix: float,
    *,
    has_bonus: bool,
    has_deep: bool,
) -> str:
    """bonus, then deep, else shallow. ``draw`` is Uniform[0, 1)."""
    if has_bonus and draw < bonus_mix:
        return "bonus"
    if has_deep and draw < (bonus_mix + deep_mix):
        return "deep"
    return "shallow"


def policy_soft_temp_weight(
    use_bonus: bool,
    default_weight: float,
    bonus_weight: float | None,
) -> float:
    """Per-pool override for the Chessformer softened-target aux."""
    if use_bonus and bonus_weight is not None:
        return float(bonus_weight)
    return float(default_weight)


def session_throughput(session_positions: int, session_elapsed_s: float) -> float:
    """Positions processed this process / session wall time. Not cumulative resume."""
    return float(session_positions) / max(float(session_elapsed_s), 1e-6)


def concat_soft_tables(chunks: list[dict]) -> dict:
    """Concat cache dicts on shared tensor keys (does not rewrite files)."""
    if not chunks:
        raise ValueError("no chunks")
    if len(chunks) == 1:
        return chunks[0]
    keys = [k for k in chunks[0] if all(k in c and torch.is_tensor(c[k]) for c in chunks)]
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def load_extra_soft_shards(paths: list[Path], log=None) -> list[tuple[str, dict]]:
    loaded = []
    for p in paths:
        p = Path(p)
        cache = p / "soft_cache.pt" if p.is_dir() else p
        if not cache.exists():
            if log:
                log(f"skip extra soft {p}: missing {cache}")
            continue
        data = torch.load(cache, map_location="cpu", weights_only=False)
        attach_static_targets(data)
        loaded.append((str(p), data))
        if log:
            log(f"loaded extra soft {p} n={int(data['board_array'].shape[0]):,}")
    return loaded


def filter_disjoint(data: dict, seen: np.ndarray | None) -> tuple[dict, np.ndarray, dict[str, int]]:
    """Keep first copy of each position hash; drop any already in ``seen``."""
    hs = position_hashes(data)
    n = int(hs.size)
    _, first = np.unique(hs, return_index=True)
    keep = np.zeros(n, dtype=np.bool_)
    keep[first] = True
    internal = n - int(first.size)
    vs_seen = 0
    if seen is not None and seen.size:
        collide = np.isin(hs, seen)
        vs_seen = int((collide & keep).sum())
        keep &= ~collide
    n_keep = int(keep.sum())
    if n_keep != n:
        data = {
            k: (v[keep] if torch.is_tensor(v) or isinstance(v, np.ndarray) else v)
            for k, v in data.items()
        }
        hs = hs[keep]
    return data, hs.astype(np.uint64, copy=False), {
        "n_in": n,
        "n_out": n_keep,
        "internal_dups": internal,
        "vs_seen": vs_seen,
    }


def load_position_hashes(path: Path) -> np.ndarray:
    """Unique position hashes from a shard dir or soft_cache.pt. Drops the table."""
    cache = Path(path)
    if cache.is_dir():
        cache = cache / "soft_cache.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    hs = np.unique(position_hashes(data).astype(np.uint64, copy=False))
    del data
    return hs


def list_attached_shards(queue_dir: Path) -> list[Path]:
    found: list[Path] = []
    for p in sorted(Path(queue_dir).glob("shard_*/ATTACHED")):
        if not p.is_file():
            continue
        sh = p.parent
        if (sh / "READY").exists() or not (sh / "soft_cache.pt").exists():
            continue
        found.append(sh)
    return found


def attach_static_targets(data: dict) -> dict:
    """Precompute WDL and ep_file so the train loop does not redo them."""
    from data_loader import compute_wdl, ep_square_to_file

    if "wdl" not in data:
        data["wdl"] = compute_wdl(data["cp"], data["mate"])
    if "ep_file" not in data:
        data["ep_file"] = ep_square_to_file(data["ep_square"]).to(torch.int8)
    return data


def _hflip_soft_indices(soft_i: torch.Tensor, flip_mask: torch.Tensor) -> torch.Tensor:
    from data_loader import hflip_move_idx

    if not flip_mask.any():
        return soft_i
    si = soft_i[flip_mask]
    valid = si >= 0
    if valid.any():
        si2 = si.clone()
        si2[valid] = hflip_move_idx(si[valid]).to(si.dtype)
        soft_i = soft_i.clone()
        soft_i[flip_mask] = si2
    return soft_i


def prepare_soft_batch(data, indices, device, hflip_p=0.0, rng=None):
    """Build a training batch.

    Horizontal reflection is applied only to rows whose *original* castling
    rights are already zero. Ineligible rows keep board, EP, rights, and
    teacher targets unchanged.
    """
    from data_loader import (
        board_array_to_fused,
        compute_wdl,
        ep_square_to_file,
        hflip_board_array,
        hflip_ep_square,
        hflip_move_idx,
    )

    ba = data["board_array"][indices].clone()
    turn = data["turn"][indices].clone()
    castling = data["castling"][indices].clone()
    ep = data["ep_square"][indices].clone()
    move_idx = data["move_idx"][indices].clone()
    soft_i = data["soft_indices"][indices].clone()
    soft_p = data["soft_probs"][indices].clone()

    if hflip_p > 0:
        eligible = castling == 0
        flip_draw = torch.rand(ba.shape[0], generator=rng) < float(hflip_p)
        flip_mask = eligible & flip_draw
        if flip_mask.any():
            ba[flip_mask] = hflip_board_array(ba[flip_mask])
            move_idx[flip_mask] = hflip_move_idx(move_idx[flip_mask]).to(move_idx.dtype)
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
            soft_i = _hflip_soft_indices(soft_i, flip_mask)

    if "wdl" in data:
        wdl = data["wdl"][indices]
    else:
        wdl = compute_wdl(data["cp"][indices], data["mate"][indices])
    if "ep_file" in data and hflip_p <= 0:
        ep_file = data["ep_file"][indices]
    else:
        ep_file = ep_square_to_file(ep)

    nb = device.type == "cuda"
    board_input = {
        "fused_ids": board_array_to_fused(ba).to(device, non_blocking=nb),
        "turn": turn.long().to(device, non_blocking=nb),
        "castling": castling.long().to(device, non_blocking=nb),
        "ep_file": ep_file.long().to(device, non_blocking=nb),
    }
    return (
        board_input,
        move_idx.long().to(device, non_blocking=nb),
        wdl.to(device, non_blocking=nb),
        soft_i.to(device, non_blocking=nb),
        soft_p.to(device, non_blocking=nb),
    )


def position_hashes(data: dict) -> np.ndarray:
    from build_hf_elo_mix import position_hashes as _ph

    return _ph(data)


def hflip_cache_slice(data: dict, idx: torch.Tensor) -> dict:
    from data_loader import hflip_board_array, hflip_ep_square

    ba = hflip_board_array(data["board_array"][idx].clone())
    ep = hflip_ep_square(data["ep_square"][idx].clone())
    return {
        "board_array": ba,
        "turn": data["turn"][idx].clone(),
        "castling": data["castling"][idx].clone(),
        "ep_square": ep.to(data["ep_square"].dtype),
    }


def make_val_membership(
    data: dict,
    *,
    n_hold: int,
    seed: int = 201,
    source: str = "unknown",
) -> dict[str, Any]:
    """Deterministic hash holdout that survives shard merge / reshuffle.

    Game-level IDs are not in the current caches. Membership is by canonical
    position hash (board+turn+castling+ep). Horizontal-flip equivalents of
    held-out positions are also excluded from training.

    Leakage limit: other positions from the same game may still appear in
    train if they were sampled independently.
    """
    hs = position_hashes(data)
    n = int(hs.shape[0])
    n_hold = int(max(0, min(n_hold, n)))
    rng = np.random.RandomState(seed)
    pick = rng.choice(n, size=n_hold, replace=False) if n_hold else np.zeros(0, dtype=np.int64)
    val_h = np.unique(hs[pick].astype(np.uint64))

    cast = data["castling"].cpu().numpy()
    val_rows = np.isin(hs, val_h)
    extra = np.zeros(0, dtype=np.uint64)
    flip_src = np.flatnonzero(val_rows & (cast == 0))
    if flip_src.size:
        extra = position_hashes(
            hflip_cache_slice(data, torch.from_numpy(flip_src.astype(np.int64)))
        ).astype(np.uint64)
    blocked = np.unique(np.concatenate([val_h, extra])) if extra.size else val_h
    return {
        "method": "position_hash_v1",
        "source": source,
        "seed": seed,
        "n_rows": n,
        "n_hold": int(val_h.size),
        "n_blocked": int(blocked.size),
        "hashes": [int(x) for x in val_h.tolist()],
        "blocked_hashes": [int(x) for x in blocked.tolist()],
        "leakage": (
            "Position-hash holdout. Same-game positions may leak across the split "
            "because caches have no game id. Flip-equivalents of val positions "
            "are blocked from training."
        ),
    }


def apply_membership(data: dict, manifest: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_index, val_index) into data."""
    hs = position_hashes(data)
    val_h = np.asarray(manifest["hashes"], dtype=np.uint64)
    blocked = np.asarray(manifest.get("blocked_hashes") or manifest["hashes"], dtype=np.uint64)
    is_val = np.isin(hs, val_h)
    is_blocked = np.isin(hs, blocked)
    train_idx = torch.from_numpy(np.flatnonzero(~is_blocked).astype(np.int64))
    val_idx = torch.from_numpy(np.flatnonzero(is_val).astype(np.int64))
    if train_idx.numel() == 0:
        train_idx = torch.arange(hs.shape[0], dtype=torch.int64)
    return train_idx, val_idx


def audit_soft_targets(data: dict, max_rows: int = 4096) -> dict[str, Any]:
    n = int(data["soft_indices"].shape[0])
    take = min(n, max_rows)
    if take <= 0:
        return {"rows_checked": 0, "empty_rows": 0, "negative_probs": 0, "unnormalized_rows": 0, "mean_valid_mass": 0.0, "ok": True}
    if n <= take:
        idx = torch.arange(n)
    else:
        idx = (torch.arange(take, dtype=torch.int64) * (n // take)).clamp(max=n - 1)
    si = data["soft_indices"][idx]
    sp = data["soft_probs"][idx].float()
    valid = soft_target_valid_mask(si, sp)
    row_sum = (sp * valid.float()).sum(dim=-1)
    empty = int((~valid.any(dim=-1)).sum())
    bad_neg = int((sp[valid] < 0).sum()) if valid.any() else 0
    unnorm = int(((row_sum > 0) & ((row_sum - 1.0).abs() > 1e-3)).sum())
    return {
        "rows_checked": take,
        "empty_rows": empty,
        "negative_probs": bad_neg,
        "unnormalized_rows": unnorm,
        "mean_valid_mass": float(row_sum.mean()) if take else 0.0,
        "ok": empty == 0 and bad_neg == 0,
    }


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def collect_rng_state(
    sampler_rng: torch.Generator | None = None,
    *,
    include_cuda: bool = False,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if include_cuda and torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = None
    if sampler_rng is not None:
        state["sampler"] = sampler_rng.get_state()
    return state


def restore_rng_state(state: dict[str, Any], sampler_rng: torch.Generator | None = None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass
    if sampler_rng is not None and state.get("sampler") is not None:
        sampler_rng.set_state(state["sampler"])


def classify_checkpoint(ckpt: dict) -> str:
    """'full' if optimizer + step are present; else 'weights_only'."""
    if not isinstance(ckpt, dict):
        return "weights_only"
    has_opt = ckpt.get("optimizer_state_dict") is not None
    has_step = ckpt.get("steps") is not None or ckpt.get("global_step") is not None
    fmt = int(ckpt.get("format", 0) or 0)
    if has_opt and has_step and fmt >= CKPT_FORMAT:
        return "full"
    if has_opt and has_step:
        return "full"
    return "weights_only"


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def save_training_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    positions: int,
    model_cfg: dict,
    train_cfg: dict,
    trial_id: str,
    arch: str,
    n_params: int,
    sampler_rng: torch.Generator | None,
    manifest: dict | None,
    swa_state: dict[str, torch.Tensor] | None,
    swa_n: int,
    resume_kind: str,
    status: str,
    extra: dict | None = None,
    eval_path: Path | None = None,
    known_good_path: Path | None = None,
    include_cuda_rng: bool = False,
) -> dict[str, Any]:
    """Save live weights (+ optimizer/rng) atomically. SWA goes to eval_path only."""
    live = unwrap_model(model)
    payload: dict[str, Any] = {
        "format": CKPT_FORMAT,
        "resume_kind": resume_kind,
        "model_state_dict": {k: v.detach().cpu() for k, v in live.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "schedule": {
            "warmup": int(train_cfg.get("warmup", 0)),
            "min_lr_frac": float(train_cfg.get("min_lr_frac", 1.0)),
            "muon_lr": float(train_cfg.get("muon_lr", 0.0)),
            "adam_lr": float(train_cfg.get("adam_lr", 0.0)),
            "max_steps": int(train_cfg.get("max_steps", 0) or 0),
        },
        "config": model_cfg,
        "train": dict(train_cfg),
        "arch": arch,
        "vocab": "compact",
        "vocab_version": "compact",
        "trial_id": trial_id,
        "steps": int(step),
        "positions": int(positions),
        "n_params": int(n_params),
        "rng": collect_rng_state(sampler_rng, include_cuda=include_cuda_rng),
        "manifest": manifest,
        "swa_n": int(swa_n),
        "status": status,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)
    if eval_path is not None and swa_state is not None and swa_n > 0:
        atomic_torch_save(
            {
                "format": CKPT_FORMAT,
                "eval_only": True,
                "model_state_dict": {k: v.detach().cpu() for k, v in swa_state.items()},
                "config": model_cfg,
                "arch": arch,
                "vocab": "compact",
                "trial_id": trial_id,
                "steps": int(step),
                "swa_n": int(swa_n),
                "n_params": int(n_params),
                "status": "swa_eval",
            },
            eval_path,
        )
    if known_good_path is not None:
        shutil.copy2(path, known_good_path)
    return payload


def load_model_state(ckpt: dict) -> dict[str, torch.Tensor]:
    state = ckpt.get("model_state_dict", ckpt)
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


@torch.no_grad()
def cheap_eval_losses(
    model: nn.Module,
    data: dict,
    indices: torch.Tensor,
    device: torch.device,
    *,
    soft_temp: float = 4.0,
) -> dict[str, float]:
    if indices.numel() == 0:
        return {}
    bi, hard, wdl, si, sp = prepare_soft_batch(data, indices, device, hflip_p=0.0)
    model.eval()
    out = model(bi)
    hard_ce = F.cross_entropy(out["policy_logits"].float(), hard)
    soft_ce = soft_policy_loss(out["policy_logits"], si, sp)
    temp_ce = soft_temp_policy_loss(out["policy_logits"], si, sp, temperature=soft_temp)
    v_loss = F.cross_entropy(out["value_logits"].float(), wdl)
    ent = teacher_entropy(si, sp)
    kl = teacher_kl(out["policy_logits"], si, sp)
    return {
        "hard_ce": float(hard_ce.item()),
        "soft_ce": float(soft_ce.item()),
        "soft_temp_ce": float(temp_ce.item()),
        "wdl_ce": float(v_loss.item()),
        "teacher_entropy": float(ent.item()),
        "teacher_kl": float(kl.item()),
    }


@torch.no_grad()
def legal_policy_diagnostics(
    model: nn.Module,
    data: dict,
    indices: torch.Tensor,
    device: torch.device,
    *,
    topk: int = 3,
    max_rows: int = 64,
) -> dict[str, float]:
    """Legal-masked accuracy / illegal mass / teacher regret. Slow (python-chess)."""
    import chess
    from data_loader import PIECE_MAP
    from move_vocab import IDX_TO_UCI, legal_move_mask

    id_to_piece = {v: k for k, v in PIECE_MAP.items()}
    take = min(int(indices.numel()), max_rows)
    if take == 0:
        return {}
    sel = indices[:take]
    bi, hard, _wdl, si, sp = prepare_soft_batch(data, sel, device, hflip_p=0.0)
    model.eval()
    logits = model(bi)["policy_logits"].float()
    probs = F.softmax(logits, dim=-1)

    top1 = topk_hit = illegal = 0.0
    regret = severe = 0.0
    n = take
    for i in range(take):
        board = _row_to_board(data, int(sel[i].item()), id_to_piece)
        mask = legal_move_mask(board).to(device)
        illegal += float(probs[i][~mask].sum().item())
        masked = logits[i].clone()
        masked[~mask] = float("-inf")
        pred = int(masked.argmax().item())
        hard_i = int(hard[i].item())
        top1 += float(pred == hard_i)
        tk = masked.topk(min(topk, int(mask.sum().item()))).indices.tolist()
        topk_hit += float(hard_i in tk)
        valid = (si[i] >= 0) & (sp[i] > 0)
        if valid.any():
            teacher_best = int(si[i][valid][sp[i][valid].argmax()].item())
            # teacher prob of model pick vs teacher best (on original support)
            p_map = {int(si[i][j]): float(sp[i][j]) for j in range(si.shape[1]) if valid[j]}
            p_pred = p_map.get(pred, 0.0)
            p_best = p_map.get(teacher_best, 0.0)
            regret += max(0.0, p_best - p_pred)
            if p_best >= 0.4 and p_pred < 0.05:
                severe += 1.0
    return {
        "legal_top1": top1 / n,
        f"legal_top{topk}": topk_hit / n,
        "illegal_mass": illegal / n,
        "teacher_regret": regret / n,
        "severe_blunder_rate": severe / n,
        "n": float(n),
    }


def _row_to_board(data: dict, i: int, id_to_piece: dict) -> "chess.Board":
    import chess

    board = chess.Board.empty()
    ba = data["board_array"][i]
    for sq in range(64):
        pid = int(ba[sq].item())
        if pid == 0:
            continue
        sym = id_to_piece.get(pid)
        if sym is None:
            continue
        board.set_piece_at(sq, chess.Piece.from_symbol(sym))
    board.turn = chess.WHITE if int(data["turn"][i].item()) == 0 else chess.BLACK
    # data_loader FEN bits: K=8 Q=4 k=2 q=1
    c = int(data["castling"][i].item())
    rights = 0
    if c & 8:
        rights |= chess.BB_H1
    if c & 4:
        rights |= chess.BB_A1
    if c & 2:
        rights |= chess.BB_H8
    if c & 1:
        rights |= chess.BB_A8
    board.castling_rights = rights
    ep = int(data["ep_square"][i].item())
    board.ep_square = ep if ep >= 0 else None
    return board


def board_to_cache_row(board: "chess.Board", move: "chess.Move | None" = None) -> dict:
    """Encode a python-chess board using data_loader FEN castling bits (K=8)."""
    from data_loader import CASTLING_MAP, PIECE_MAP
    from move_vocab import move_to_index

    ba = torch.zeros(64, dtype=torch.int8)
    for sq, piece in board.piece_map().items():
        ba[sq] = PIECE_MAP[piece.symbol()]
    fen_c = board.fen().split()[2]
    castling = 0
    if fen_c != "-":
        for ch in fen_c:
            castling |= CASTLING_MAP.get(ch, 0)
    ep = -1 if board.ep_square is None else int(board.ep_square)
    mid = 0
    if move is not None:
        mid = int(move_to_index(move))
    return {
        "board_array": ba,
        "turn": torch.tensor(0 if board.turn else 1, dtype=torch.int8),
        "castling": torch.tensor(castling, dtype=torch.int8),
        "ep_square": torch.tensor(ep, dtype=torch.int16),
        "move_idx": torch.tensor(mid, dtype=torch.int64),
        "cp": torch.tensor(0, dtype=torch.int32),
        "mate": torch.tensor(0, dtype=torch.int32),
        "soft_indices": torch.tensor([mid if move is not None else -1, -1, -1, -1], dtype=torch.int64),
        "soft_probs": torch.tensor([1.0 if move is not None else 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    }


def stack_rows(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}


def exposure_report(
    *,
    shallow_n: int,
    deep_n: int,
    deep_mix_frac: float,
    shallow_seen: int,
    deep_seen: int,
) -> dict[str, float]:
    """Per-source sampling probability vs observed counts.

    With replacement sampling, P(draw a given deep row | deep batch) = 1/deep_n,
    and P(deep batch) = deep_mix. Relative to a shallow row:
      (deep_mix / deep_n) / ((1-deep_mix) / shallow_n)
    """
    dm = float(deep_mix_frac)
    p_deep_row = (dm / max(deep_n, 1)) if deep_n else 0.0
    p_shallow_row = ((1.0 - dm) / max(shallow_n, 1)) if shallow_n else 0.0
    rel = p_deep_row / max(p_shallow_row, 1e-18)
    return {
        "shallow_n": float(shallow_n),
        "deep_n": float(deep_n),
        "deep_mix_frac": dm,
        "p_sample_per_deep_row": p_deep_row,
        "p_sample_per_shallow_row": p_shallow_row,
        "deep_vs_shallow_odds": rel,
        "shallow_seen": float(shallow_seen),
        "deep_seen": float(deep_seen),
    }


def list_ready_shards(queue_dir: Path) -> list[Path]:
    """Shards not yet trained. ATTACHED shards stay out of the next attach."""
    found: list[Path] = []
    for p in sorted(Path(queue_dir).glob("shard_*/READY")):
        if not p.is_file():
            continue
        sh = p.parent
        if not (sh / "soft_cache.pt").exists():
            continue
        found.append(sh)
    return found


def ingest_ready_bonus_shards(
    inbox_dir: Path,
    seen: np.ndarray | None,
) -> tuple[list[dict], np.ndarray, list[dict[str, Any]]]:
    """Consume READY inbox shards; drop hashes in ``seen`` (bonus + holdout).

    Marks each shard ATTACHED and removes READY so a live trainer can pick
    new hunts without rewriting the 10M+ soft cache.
    """
    inbox = Path(inbox_dir)
    chunks: list[dict] = []
    reports: list[dict[str, Any]] = []
    seen_h = None if seen is None or not getattr(seen, "size", 0) else seen.astype(np.uint64, copy=False)
    if not inbox.is_dir():
        empty = np.array([], dtype=np.uint64) if seen_h is None else seen_h
        return chunks, empty, reports
    for sh in list_ready_shards(inbox):
        data = torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)
        data, hs, stats = filter_disjoint(data, seen_h)
        n_out = int(stats["n_out"])
        if n_out:
            chunks.append(data)
            if seen_h is None:
                seen_h = np.unique(hs.astype(np.uint64, copy=False))
            else:
                seen_h = np.unique(np.concatenate([seen_h, hs.astype(np.uint64, copy=False)]))
        ready = sh / "READY"
        if ready.exists():
            ready.unlink()
        (sh / "ATTACHED").write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **stats}), encoding="utf-8")
        reports.append({"shard": sh.name, **stats})
    empty = np.array([], dtype=np.uint64) if seen_h is None else seen_h
    return chunks, empty, reports


def write_shard_manifest(
    path: Path,
    *,
    live: Path,
    shards: list[Path],
    val_manifests: dict | None = None,
    notes: str = "",
) -> dict[str, Any]:
    payload = {
        "live": str(live),
        "shards": [str(s) for s in shards],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "val": val_manifests,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class ShardedSoftSource:
    """Sample from live cache + READY shards without concatenating them."""

    def __init__(self, tables: list[tuple[str, dict, torch.Tensor]]):
        # each: (name, data, train_index)
        self.tables = tables
        self.sizes = [int(idx.numel()) for _, _, idx in tables]
        self.total = sum(self.sizes)
        self.seen = {name: 0 for name, _, _ in tables}

    @classmethod
    def from_paths(
        cls,
        live: Path,
        shards: list[Path],
        *,
        val_manifest: dict,
        hold_name: str,
    ) -> "ShardedSoftSource":
        tables = []
        for name, path in [("live", live), *[(p.name, p / "soft_cache.pt") for p in shards]]:
            if not Path(path).exists():
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            attach_static_targets(data)
            tr, _va = apply_membership(data, val_manifest)
            tables.append((name, data, tr))
        return cls(tables)

    def sample(self, bs: int, rng: torch.Generator) -> tuple[dict, torch.Tensor, str]:
        pick = int(torch.randint(0, self.total, (1,), generator=rng).item())
        acc = 0
        for name, data, idx in self.tables:
            if pick < acc + int(idx.numel()):
                local = torch.randint(0, int(idx.numel()), (bs,), generator=rng)
                rows = idx[local]
                self.seen[name] = self.seen.get(name, 0) + bs
                return data, rows, name
            acc += int(idx.numel())
        name, data, idx = self.tables[0]
        local = torch.randint(0, int(idx.numel()), (bs,), generator=rng)
        self.seen[name] = self.seen.get(name, 0) + bs
        return data, idx[local], name


def muon_update_scale_note() -> str:
    return (
        "average_recurrent_grads() divides bank .grad by unroll count. "
        "That does NOT divide optimizer updates by the same amount: Polar-NorMuon "
        "is scale-invariant after orthogonalization, and RMS-Adam (m/sqrt(v)) is "
        "also approximately scale-invariant for a stationary grad. What does "
        "scale is SGD-style updates and the pre-clip global grad norm (so "
        "clipping fires less often after averaging). Use explicit param-group "
        "LRs if different update sizes are intended."
    )


def lr_scale(step: int, *, warmup: int, max_steps: int, min_lr_frac: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_frac + (1.0 - min_lr_frac) * cosine
