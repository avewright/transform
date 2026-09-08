#!/usr/bin/env python3
"""Scan overnight SWA disagreements, then SF19-analyze unknown moves.

Fixes vs the run2 scanner:
  - checkpoint is overnight eval_swa, not run2 latest
  - SF19 cp/mate are white-absolute; soft_cps/soft_mates are STM (do not flip both)
  - off-PV / unlabeled rows are not given an invented 80cp penalty
  - every source is filtered by the overnight union of holdout + flip hashes

Usage:
  MOVE_VOCAB_VERSION=compact python3 -u scripts/harvest_swa_mistakes.py --go
  MOVE_VOCAB_VERSION=compact python3 -u scripts/harvest_swa_mistakes.py --analyze
  MOVE_VOCAB_VERSION=compact python3 -u scripts/harvest_swa_mistakes.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chess  # noqa: E402
from chess_inference import load_checkpoint  # noqa: E402
from data_loader import board_array_to_fused, ep_square_to_file  # noqa: E402
from harvest_hf100m_bulk import _write_shard  # noqa: E402
from move_vocab import UCI_TO_IDX, VOCAB_SIZE, index_to_move  # noqa: E402

OVERNIGHT = ROOT / "outputs/sf19_ft/overnight_20260908"
DEFAULT_CKPT = OVERNIGHT / "eval_swa.pt"
DEFAULT_OUT = ROOT / "outputs/swa_mistakes"

_ID_TO_SYMBOL = {
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}
_CASTLE_BITS = ((8, "K"), (4, "Q"), (2, "k"), (1, "q"))

TAG_TO_I = {
    "ok": 0,
    "off_pv": 1,
    "inaccuracy": 2,
    "blunder": 3,
    "conversion": 4,
    "major": 5,
    "disagree": 6,
}
I_TO_TAG = {v: k for k, v in TAG_TO_I.items()}
ORIGIN = {"sf19_overnight": 0, "lichess_replay": 1, "syzygy": 2, "sf19_generated": 0}


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _np(t, dtype):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(t, dtype=dtype)


def board_array_to_fen(ba, turn, castling, ep_square) -> str:
    ranks = []
    for rank in range(7, -1, -1):
        empty = 0
        cells = []
        for file_idx in range(8):
            pid = int(ba[rank * 8 + file_idx])
            if pid <= 0:
                empty += 1
                continue
            if empty:
                cells.append(str(empty))
                empty = 0
            cells.append(_ID_TO_SYMBOL.get(pid, "1"))
        if empty:
            cells.append(str(empty))
        ranks.append("".join(cells))
    castle = "".join(ch for bit, ch in _CASTLE_BITS if int(castling) & bit) or "-"
    ep = "-"
    ep_i = int(ep_square)
    if 0 <= ep_i <= 63:
        ep = chess.square_name(ep_i)
    stm = "b" if int(turn) else "w"
    return f"{'/'.join(ranks)} {stm} {castle} {ep} 0 1"


def to_stm(cp: int, mate: int, turn: int, *, white_abs: bool) -> tuple[int, int]:
    """Convert packed eval to side-to-move. soft_cps are already STM."""
    if white_abs and int(turn) == 1:
        return -int(cp), -int(mate)
    return int(cp), int(mate)


def classify_lapse(*, best_cp: int, best_mate: int, model_cp: int, model_mate: int, model_in_pv: bool) -> dict:
    best_mate = int(best_mate or 0)
    model_mate = int(model_mate or 0)
    if best_mate > 0 and model_mate <= 0:
        return {"tag": "major", "kind": "missed_mate", "drop_cp": None}
    if model_mate < 0 and best_mate >= 0:
        return {"tag": "major", "kind": "allowed_mate", "drop_cp": None}
    if best_mate > 0 and model_mate > best_mate:
        return {"tag": "inaccuracy", "kind": "slower_mate", "drop_cp": None}
    drop = int(best_cp) - int(model_cp)
    if drop >= 300:
        tag = "major"
    elif int(best_cp) >= 200 and int(model_cp) < 50 and drop >= 150:
        tag = "conversion"
    elif drop >= 150:
        tag = "blunder"
    elif drop >= 75:
        tag = "inaccuracy"
    elif not model_in_pv and drop >= 50:
        tag = "off_pv"
    else:
        tag = "ok"
    return {"tag": tag, "kind": "cp", "drop_cp": drop}


def load_blocked_hashes() -> np.ndarray:
    """Overnight union of holdout hashes including flip-equivalents (20,961)."""
    paths = [
        OVERNIGHT / "val_manifest_soft.json",
        OVERNIGHT / "val_manifest_deep.json",
        OVERNIGHT / "val_manifest_replay.json",
        ROOT / "outputs/sf19_ft/run2/val_manifest_soft.json",
        ROOT / "outputs/sf19_ft/run2/val_manifest_deep.json",
    ]
    chunks = []
    for p in paths:
        if not p.exists():
            continue
        man = json.loads(p.read_text())
        raw = man.get("blocked_hashes") or man.get("hashes") or []
        if raw:
            chunks.append(np.asarray(raw, dtype=np.uint64))
    if chunks:
        return np.unique(np.concatenate(chunks))
    from scripts.autoresearch_8gb.pipeline import make_val_membership

    parts = []
    specs = [
        (ROOT / "outputs/sf19_ft/soft_cache.pt", 201, "soft"),
        (ROOT / "outputs/hf_elo_mix/soft_cache.pt", 203, "replay"),
        (ROOT / "outputs/hf_elo_mix/deep_cache.pt", 202, "deep"),
    ]
    for path, seed, name in specs:
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        man = make_val_membership(data, n_hold=2000, seed=seed, source=name)
        parts.append(np.asarray(man["blocked_hashes"], dtype=np.uint64))
        del data
    if not parts:
        raise SystemExit("no holdout manifests or source caches")
    return np.unique(np.concatenate(parts))


def list_scan_caches() -> list[tuple[Path, bool, str]]:
    out: list[tuple[Path, bool, str]] = []
    soft = OVERNIGHT / "soft_cache.pt"
    replay = OVERNIGHT / "replay_cache.pt"
    deep = OVERNIGHT / "deep_cache.pt"
    if soft.exists():
        out.append((soft, True, "sf19_overnight"))
    gen = OVERNIGHT / "generated_verified"
    if gen.is_dir():
        for p in sorted(gen.glob("*.pt")):
            out.append((p, True, f"sf19_generated_{p.stem}"))
    if replay.exists():
        out.append((replay, False, "lichess_replay"))
    if deep.exists():
        out.append((deep, False, "syzygy"))
    return out


def tag_batch(pred, teacher, turn, cp, mate, soft_i, soft_c, soft_m, *, white_abs: bool):
    n = int(pred.shape[0])
    disagree = pred != teacher
    tags = np.full(n, TAG_TO_I["ok"], dtype=np.int8)
    tags[disagree] = TAG_TO_I["disagree"]
    drop = np.zeros(n, dtype=np.int32)
    in_pv = np.zeros(n, dtype=np.int8)
    needs_sf = np.zeros(n, dtype=np.int8)
    needs_sf[disagree] = 1
    if soft_i is None or soft_c is None:
        return disagree, tags, drop, in_pv, needs_sf
    for i in range(n):
        if not disagree[i]:
            continue
        row = soft_i[i]
        hits = np.flatnonzero(row == pred[i])
        best_cp, best_mate = to_stm(int(cp[i]), int(mate[i]), int(turn[i]), white_abs=white_abs)
        if hits.size == 0:
            tags[i] = TAG_TO_I["off_pv"]
            drop[i] = 0
            needs_sf[i] = 1
            continue
        j = int(hits[0])
        in_pv[i] = 1
        # soft_cps / soft_mates are STM on SF19 rows. Never apply white-abs flip.
        model_cp = int(soft_c[i, j])
        model_mate = int(soft_m[i, j]) if soft_m is not None else 0
        info = classify_lapse(
            best_cp=best_cp,
            best_mate=best_mate,
            model_cp=model_cp,
            model_mate=model_mate,
            model_in_pv=True,
        )
        tags[i] = TAG_TO_I.get(info["tag"], TAG_TO_I["disagree"])
        drop[i] = int(info["drop_cp"] or 0)
        needs_sf[i] = 0
    return disagree, tags, drop, in_pv, needs_sf


def _take(t, idx, dtype):
    return torch.from_numpy(np.ascontiguousarray(_np(t, dtype)[idx]))


def pack_keep(data, mask, pred, tags, drop, in_pv, needs_sf, *, source: int, origin: int) -> dict:
    idx = np.flatnonzero(mask)
    n = int(idx.size)
    out = {
        "board_array": _take(data["board_array"], idx, np.int8),
        "turn": _take(data["turn"], idx, np.int8),
        "castling": _take(data["castling"], idx, np.int8),
        "ep_square": _take(data["ep_square"], idx, np.int8),
        "move_idx": _take(data["move_idx"], idx, np.int64),
        "cp": _take(data["cp"], idx, np.int32),
        "mate": _take(data["mate"], idx, np.int32),
        "soft_indices": _take(data["soft_indices"], idx, np.int64),
        "soft_probs": _take(data["soft_probs"], idx, np.float32),
        "label_depth": (
            _take(data["label_depth"], idx, np.int16)
            if "label_depth" in data
            else torch.zeros(n, dtype=torch.int16)
        ),
        "phase": (
            _take(data["phase"], idx, np.int8)
            if "phase" in data
            else torch.zeros(n, dtype=torch.int8)
        ),
        "source": torch.full((n,), source, dtype=torch.int8),
        "origin": torch.full((n,), origin, dtype=torch.int8),
        "model_move_idx": torch.from_numpy(pred[idx].astype(np.int64)),
        "tag": torch.from_numpy(tags[idx]),
        "drop_cp": torch.from_numpy(drop[idx]),
        "model_in_pv": torch.from_numpy(in_pv[idx]),
        "needs_sf": torch.from_numpy(needs_sf[idx]),
    }
    out["soft_cps"] = (
        _take(data["soft_cps"], idx, np.int32)
        if "soft_cps" in data
        else torch.zeros((n, 8), dtype=torch.int32)
    )
    out["soft_mates"] = (
        _take(data["soft_mates"], idx, np.int32)
        if "soft_mates" in data
        else torch.zeros((n, 8), dtype=torch.int32)
    )
    return out


def _pick_legal(topk_idx: np.ndarray, ba, turn, castling, ep) -> np.ndarray:
    n = int(topk_idx.shape[0])
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        board = chess.Board(board_array_to_fen(ba[i], turn[i], castling[i], ep[i]))
        picked = None
        for raw in topk_idx[i]:
            mv = index_to_move(int(raw))
            if mv in board.legal_moves:
                picked = int(raw)
                break
        if picked is None:
            legal = next(iter(board.legal_moves), None)
            picked = UCI_TO_IDX.get(legal.uci(), int(topk_idx[i, 0])) if legal is not None else int(topk_idx[i, 0])
        out[i] = picked
    return out


@torch.inference_mode()
def predict_legal(model, fused, turn, castling, ep_file, ba, turn_np, cast_np, ep_np, device, micro: int) -> np.ndarray:
    preds = []
    n = fused.shape[0]
    use_amp = device.type == "cuda"
    for i in range(0, n, micro):
        sl = slice(i, i + micro)
        inp = {
            "fused_ids": fused[sl].to(device, non_blocking=True),
            "turn": turn[sl].to(device, non_blocking=True),
            "castling": castling[sl].to(device, non_blocking=True),
            "ep_file": ep_file[sl].to(device, non_blocking=True),
        }
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            logits = model(inp)["policy_logits"].float()
        topk = logits.topk(16, dim=-1).indices.cpu().numpy()
        preds.append(_pick_legal(topk, ba[sl], turn_np[sl], cast_np[sl], ep_np[sl]))
    return np.concatenate(preds, axis=0)


def holdout_keep(data: dict, blocked: np.ndarray) -> np.ndarray:
    from scripts.autoresearch_8gb.pipeline import position_hashes

    n = int(data["move_idx"].shape[0])
    keep = np.ones(n, dtype=bool)
    if "split" in data:
        keep &= _np(data["split"], np.int8) == 0
    hs = position_hashes(data).astype(np.uint64, copy=False)
    if blocked.size:
        keep &= ~np.isin(hs, blocked)
    return keep


def _slice_cache(data: dict, sl: slice) -> dict:
    n = int(data["move_idx"].shape[0])
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v) and v.ndim and len(v) == n:
            out[k] = v[sl]
        elif isinstance(v, np.ndarray) and v.ndim and len(v) == n:
            out[k] = v[sl]
    return out


def scan_cache(
    model, device, cache: Path, *, white_abs: bool, micro: int, origin: int,
    blocked: np.ndarray, log_path: Path, chunk: int = 32768, start_row: int = 0,
    publish=None,
):
    """Scan a cache. If ``publish`` is set, emit keep-rows after every chunk."""
    data = torch.load(cache, map_location="cpu", weights_only=False)
    n = int(data["move_idx"].shape[0])
    keep = holdout_keep(data, blocked)
    n_hold = int((~keep).sum())
    turn64 = _np(data["turn"], np.int64)
    cast64 = _np(data["castling"], np.int64)
    ep64 = _np(data["ep_square"], np.int64)
    ba = _np(data["board_array"], np.int8)
    teacher = _np(data["move_idx"], np.int64)
    si = _np(data["soft_indices"], np.int64) if "soft_indices" in data else None
    sc = _np(data["soft_cps"], np.int32) if "soft_cps" in data else None
    sm = _np(data["soft_mates"], np.int32) if "soft_mates" in data else None
    turn8 = _np(data["turn"], np.int8)
    cp32 = _np(data["cp"], np.int32)
    mate32 = _np(data["mate"], np.int32)
    t0 = time.time()
    totals = {
        "n": n, "holdout": n_hold, "disagree": 0, "keep": 0, "needs_sf": 0,
        "tags": {k: 0 for k in TAG_TO_I},
    }
    start_row = max(0, min(int(start_row), n))
    for i in range(start_row, n, chunk):
        sl = slice(i, min(n, i + chunk))
        fused = board_array_to_fused(torch.from_numpy(np.ascontiguousarray(ba[sl])))
        pred = predict_legal(
            model, fused,
            torch.from_numpy(np.ascontiguousarray(turn64[sl])),
            torch.from_numpy(np.ascontiguousarray(cast64[sl])),
            ep_square_to_file(torch.from_numpy(np.ascontiguousarray(ep64[sl]))),
            ba[sl], turn64[sl], cast64[sl], ep64[sl],
            device, micro,
        )
        disagree, tags, drop, in_pv, needs_sf = tag_batch(
            pred, teacher[sl], turn8[sl], cp32[sl], mate32[sl],
            None if si is None else si[sl],
            None if sc is None else sc[sl],
            None if sm is None else sm[sl],
            white_abs=white_abs,
        )
        mask = keep[sl] & disagree
        n_keep = int(mask.sum())
        totals["disagree"] += int((disagree & keep[sl]).sum())
        totals["keep"] += n_keep
        totals["needs_sf"] += int((needs_sf[mask] == 1).sum()) if n_keep else 0
        for key, val in TAG_TO_I.items():
            totals["tags"][key] += int(((tags == val) & mask).sum())
        if n_keep and publish is not None:
            packed = pack_keep(
                _slice_cache(data, sl), mask, pred, tags, drop, in_pv, needs_sf,
                source=3, origin=origin,
            )
            publish(packed, min(n, i + chunk))
        done = min(n, i + chunk)
        dt = max(time.time() - t0, 1e-6)
        log(
            f"    {cache.name} {done:,}/{n:,} {done / dt:.0f} pos/s "
            f"keep={totals['keep']:,} needs_sf={totals['needs_sf']:,} holdout_drop={n_hold:,}",
            log_path,
        )
    log(
        f"  {cache.name} n={n:,} holdout={n_hold:,} disagree={totals['disagree']:,} keep={totals['keep']:,} "
        f"in_pv={totals['tags'].get('ok', 0) and totals['keep'] - totals['tags'].get('off_pv', 0)} "
        f"off_pv={totals['tags']['off_pv']} bl={totals['tags']['blunder']} "
        f"inacc={totals['tags']['inaccuracy']} maj={totals['tags']['major']} "
        f"needs_sf={totals['needs_sf']:,}",
        log_path,
    )
    del data
    return totals


def concat_pack(chunks: list[dict]) -> dict:
    keys = chunks[0].keys()
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def self_test() -> None:
    # Black to move. Teacher STM +200 (white-abs cp=-200). Model in-PV STM +100.
    # Real drop is 100cp. The old bug flipped soft_cps again → drop 300 "major".
    pred = np.array([5])
    teacher = np.array([1])
    turn = np.array([1], dtype=np.int8)
    cp = np.array([-200], dtype=np.int32)
    mate = np.array([0], dtype=np.int32)
    soft_i = np.array([[5, 1, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    soft_c = np.array([[100, 200, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    soft_m = np.zeros((1, 8), dtype=np.int32)
    disagree, tags, drop, in_pv, needs = tag_batch(
        pred, teacher, turn, cp, mate, soft_i, soft_c, soft_m, white_abs=True,
    )
    assert disagree[0] and in_pv[0] == 1
    assert int(drop[0]) == 100, f"expected drop 100, got {drop[0]}"
    assert int(tags[0]) == TAG_TO_I["inaccuracy"], f"expected inaccuracy, got {I_TO_TAG[int(tags[0])]}"
    assert int(needs[0]) == 0

    # Off-PV must not invent an 80cp (or any) penalty.
    pred = np.array([9])
    soft_i = np.array([[1, 2, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    disagree, tags, drop, in_pv, needs = tag_batch(
        pred, teacher, turn, cp, mate, soft_i, soft_c, soft_m, white_abs=True,
    )
    assert in_pv[0] == 0
    assert int(tags[0]) == TAG_TO_I["off_pv"]
    assert int(drop[0]) == 0
    assert int(needs[0]) == 1

    # White to move, same STM numbers: white-abs == STM.
    pred = np.array([5])
    turn = np.array([0], dtype=np.int8)
    cp = np.array([200], dtype=np.int32)
    soft_i = np.array([[5, 1, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    soft_c = np.array([[100, 200, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    disagree, tags, drop, in_pv, needs = tag_batch(
        pred, teacher, turn, cp, mate, soft_i, soft_c, soft_m, white_abs=True,
    )
    assert int(drop[0]) == 100 and int(tags[0]) == TAG_TO_I["inaccuracy"]
    print("self-test ok")


def run_scan(args) -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(f"need compact vocab, got {VOCAB_SIZE}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    blocked = load_blocked_hashes()
    log(f"blocked_hashes={blocked.size:,} ckpt={args.ckpt}", log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = load_checkpoint(args.ckpt, device=device)
    if device.type == "cuda" and not args.no_compile:
        log("torch.compile ...", log_path)
        model = torch.compile(model)

    caches = list_scan_caches()
    cursor_path = out / "cursor.json"
    done = set()
    if cursor_path.exists():
        prev = json.loads(cursor_path.read_text())
        done = set(prev.get("done") or [])
    log(f"caches={len(caches)} skip_done={len(done)}", log_path)
    buf: list[dict] = []
    n_buf = 0
    shard_i = 0
    inbox = out / "inbox"
    if inbox.exists():
        existing = sorted(inbox.glob("shard_*"))
        if existing:
            shard_i = int(existing[-1].name.split("_", 1)[1]) + 1
    totals = {"seen": 0, "keep": 0, "disagree": 0, "holdout": 0, "needs_sf": 0, "tags": {}}
    if cursor_path.exists():
        prev = json.loads(cursor_path.read_text())
        for k in ("seen", "keep", "disagree", "holdout", "needs_sf"):
            totals[k] = int(prev.get(k) or 0)
        totals["tags"] = dict(prev.get("tags") or {})
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True
        log("stop requested", log_path)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def flush(force: bool = False) -> None:
        nonlocal buf, n_buf, shard_i
        if n_buf == 0 or (n_buf < args.shard_size and not force):
            return
        data = concat_pack(buf)
        n = int(data["move_idx"].shape[0])
        start = 0
        while start < n:
            remain = n - start
            if remain < args.shard_size and not force:
                buf = [{k: v[start:] for k, v in data.items()}]
                n_buf = remain
                return
            end = n if force and remain < args.shard_size * 2 else min(n, start + args.shard_size)
            _write_shard({k: v[start:end] for k, v in data.items()}, out, shard_i)
            log(f"wrote shard_{shard_i:06d} n={end - start:,} total_keep={totals['keep']:,}", log_path)
            shard_i += 1
            start = end
        buf = []
        n_buf = 0

    t0 = time.time()
    partial = {}
    if cursor_path.exists():
        partial = dict((json.loads(cursor_path.read_text()) or {}).get("partial") or {})

    def write_cursor(last: str, extra: dict | None = None) -> None:
        payload = {
            "phase": "local", "last": last, "done": sorted(done),
            **totals, "secs": time.time() - t0, "next_shard": shard_i,
        }
        if extra:
            payload.update(extra)
        cursor_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for path, white_abs, name in caches:
        if stop:
            break
        if name in done:
            log(f"skip done {name}", log_path)
            continue
        origin = 0 if name.startswith("sf19") else ORIGIN.get(name, 3)
        start_row = int(partial.get("row") or 0) if partial.get("name") == name else 0
        log(f"scan {name} white_abs={white_abs} origin={origin} start_row={start_row}", log_path)

        def publish(packed: dict, row_done: int) -> None:
            nonlocal n_buf
            buf.append(packed)
            n_buf += int(packed["move_idx"].shape[0])
            flush(False)
            write_cursor(name, {"partial": {"name": name, "row": row_done}})

        stats = scan_cache(
            model, device, path,
            white_abs=white_abs, micro=args.micro_batch,
            origin=origin, blocked=blocked, log_path=log_path,
            start_row=start_row, publish=publish,
        )
        totals["seen"] += stats["n"]
        totals["disagree"] += stats["disagree"]
        totals["keep"] += stats["keep"]
        totals["holdout"] += stats.get("holdout", 0)
        totals["needs_sf"] += stats.get("needs_sf", 0)
        for k, v in (stats.get("tags") or {}).items():
            totals["tags"][k] = totals["tags"].get(k, 0) + v
        done.add(name)
        partial = {}
        write_cursor(name, {"partial": {}})
    flush(True)
    (out / "summary.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "blocked": int(blocked.size), **totals,
        "secs": time.time() - t0, "shards": shard_i,
    }, indent=2), encoding="utf-8")
    log(json.dumps({"done_local": True, **totals, "secs": round(time.time() - t0, 1)}), log_path)


def _sf_model_eval(engine, board: chess.Board, move: chess.Move, *, nodes: int, watchdog_s: float) -> tuple[int, int]:
    limit = chess.engine.Limit(nodes=max(1, nodes), time=max(0.5, watchdog_s))
    info = engine.analyse(board, limit, root_moves=[move])
    if isinstance(info, list):
        info = info[0]
    sc = info.get("score")
    if sc is None:
        return 0, 0
    pov = sc.pov(board.turn)
    if pov.is_mate():
        return 0, int(pov.mate() or 0)
    raw = pov.score(mate_score=None)
    return int(raw or 0), 0


def analyze_one(engine, fen: str, model_idx: int, *, nodes: int, multipv: int, tau: float, watchdog_s: float) -> dict | None:
    from sf19_soft_dataset import analyze_board

    board = chess.Board(fen)
    row = analyze_board(engine, board, nodes=nodes, multipv=multipv, tau=tau, watchdog_s=watchdog_s)
    if row is None or int(row.get("move_idx", -1)) < 0:
        return None
    try:
        model_mv = index_to_move(int(model_idx))
    except Exception:
        return None
    if model_mv not in board.legal_moves:
        return None
    soft_i = np.asarray(row["soft_indices"]).reshape(-1)
    soft_c = np.asarray(row["soft_cps"]).reshape(-1)
    soft_m = np.asarray(row["soft_mates"]).reshape(-1)
    hits = np.flatnonzero(soft_i == int(model_idx))
    best_cp, best_mate = to_stm(int(row["cp"]), int(row["mate"]), int(row["turn"]), white_abs=True)
    if hits.size:
        in_pv = 1
        model_cp = int(soft_c[int(hits[0])])
        model_mate = int(soft_m[int(hits[0])])
    else:
        in_pv = 0
        model_cp, model_mate = _sf_model_eval(engine, board, model_mv, nodes=nodes, watchdog_s=watchdog_s)
    info = classify_lapse(
        best_cp=best_cp, best_mate=best_mate,
        model_cp=model_cp, model_mate=model_mate, model_in_pv=bool(in_pv),
    )
    row["teacher_move_idx"] = np.int64(-1)
    row["model_move_idx"] = np.int64(model_idx)
    row["tag"] = np.int8(TAG_TO_I.get(info["tag"], TAG_TO_I["disagree"]))
    row["drop_cp"] = np.int32(int(info["drop_cp"] or 0))
    row["model_in_pv"] = np.int8(in_pv)
    row["needs_sf"] = np.int8(0)
    row["model_cp_stm"] = np.int32(model_cp)
    row["model_mate_stm"] = np.int32(model_mate)
    return row


def _priority(tag: int, cp: int, turn: int, white_abs: bool) -> int:
    stm, _ = to_stm(int(cp), 0, int(turn), white_abs=white_abs)
    score = 0
    if tag in (TAG_TO_I["off_pv"], TAG_TO_I["disagree"]):
        score += 1000
    if abs(stm) >= 150:
        score += 200
    elif abs(stm) < 50:
        score += 80
    return score


def run_analyze(args) -> None:
    from multiprocessing import Process, Queue

    from sf19_soft_dataset import resolve_sf

    out = Path(args.out_dir)
    inbox = out / "inbox"
    dest_root = out / "analyzed"
    dest_root.mkdir(parents=True, exist_ok=True)
    log_path = out / "analyze.log"
    state_path = out / "analyze_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "n": 0, "tags": {}}
    sf = resolve_sf()
    log(f"analyze sf={sf} nodes={args.nodes} workers={args.workers} limit={args.analyze_limit:,}", log_path)

    def collect(limit: int) -> tuple[list[dict], list[str]]:
        rows: list[dict] = []
        claimed: list[str] = []
        if not inbox.exists() or limit <= 0:
            return rows, claimed
        for sh in sorted(inbox.glob("shard_*")):
            if sh.name in state["done"]:
                continue
            pt = sh / "soft_cache.pt"
            if not pt.exists():
                continue
            d = torch.load(pt, map_location="cpu", weights_only=False)
            n = int(d["move_idx"].shape[0])
            needs = _np(d["needs_sf"], np.int8) if "needs_sf" in d else np.ones(n, dtype=np.int8)
            tags = _np(d["tag"], np.int8)
            ba = _np(d["board_array"], np.int8)
            turn = _np(d["turn"], np.int8)
            cast = _np(d["castling"], np.int8)
            ep = _np(d["ep_square"], np.int8)
            cp = _np(d["cp"], np.int32)
            model = _np(d["model_move_idx"], np.int64)
            teacher = _np(d["move_idx"], np.int64)
            origin = _np(d["origin"], np.int8) if "origin" in d else np.zeros(n, dtype=np.int8)
            white_abs = origin == 0
            shard_rows = []
            for i in range(n):
                if needs[i] != 1:
                    continue
                shard_rows.append({
                    "fen": board_array_to_fen(ba[i], turn[i], cast[i], ep[i]),
                    "model_idx": int(model[i]),
                    "teacher_idx": int(teacher[i]),
                    "origin": int(origin[i]),
                    "shard": sh.name,
                    "pri": _priority(int(tags[i]), int(cp[i]), int(turn[i]), bool(white_abs[i])),
                })
            shard_rows.sort(key=lambda r: -r["pri"])
            room = limit - len(rows)
            rows.extend(shard_rows[:room])
            if len(shard_rows) <= room:
                claimed.append(sh.name)
            if len(rows) >= limit:
                break
        return rows, claimed

    job_q: Queue = Queue()
    res_q: Queue = Queue()

    def worker(wid: int) -> None:
        import chess.engine
        from sf19_soft_dataset import analyze_board  # noqa: F401

        engine = chess.engine.SimpleEngine.popen_uci(sf)
        engine.configure({"Threads": 1, "Hash": 32, "UCI_ShowWDL": True})
        while True:
            item = job_q.get()
            if item is None:
                break
            try:
                row = analyze_one(
                    engine, item["fen"], item["model_idx"],
                    nodes=args.nodes, multipv=8, tau=120.0, watchdog_s=30.0,
                )
                if row is not None:
                    row["teacher_move_idx"] = np.int64(item["teacher_idx"])
                    row["origin"] = np.int8(item["origin"])
                    row["source"] = np.int8(3)
            except Exception as exc:
                row = None
                log(f"worker{wid} err {type(exc).__name__}", log_path)
            res_q.put(row)
        engine.quit()

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    procs = [Process(target=worker, args=(i,), daemon=True) for i in range(args.workers)]
    for p in procs:
        p.start()
    pending: list[dict] = []
    shard_i = 0
    existing = sorted(dest_root.glob("shard_*"))
    if existing:
        shard_i = int(existing[-1].name.split("_", 1)[1]) + 1
    n_done = int(state.get("n") or 0)
    t0 = time.time()

    def flush_pending(force: bool = False) -> None:
        nonlocal pending, shard_i
        if not pending or (len(pending) < args.shard_size and not force):
            return
        take = pending[: args.shard_size] if not force else pending
        pending = pending[len(take):]
        keys = [
            "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
            "soft_indices", "soft_probs", "soft_cps", "soft_mates", "label_depth",
            "phase", "source", "origin", "model_move_idx", "tag", "drop_cp",
            "model_in_pv", "needs_sf", "teacher_move_idx",
        ]
        packed = {}
        for k in keys:
            packed[k] = torch.from_numpy(np.stack([np.asarray(r[k]) for r in take]))
        _write_shard(packed, dest_root, shard_i)
        log(f"analyzed shard_{shard_i:06d} n={len(take):,} total={n_done:,}", log_path)
        shard_i += 1

    while not stop and n_done < args.analyze_limit:
        remain = args.analyze_limit - n_done
        batch, claimed = collect(remain)
        if not batch:
            if args.watch:
                time.sleep(max(15.0, args.poll_s))
                continue
            break
        inflight = 0
        for item in batch:
            job_q.put(item)
            inflight += 1
        got = 0
        while got < inflight and not stop:
            row = res_q.get()
            got += 1
            if row is None:
                continue
            pending.append(row)
            n_done += 1
            tag = I_TO_TAG.get(int(row["tag"]), str(int(row["tag"])))
            state.setdefault("tags", {})
            state["tags"][tag] = int(state["tags"].get(tag, 0)) + 1
            if n_done % 50 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                log(f"analyzed {n_done:,}/{args.analyze_limit:,} {rate:.2f}/s last={tag} drop={int(row['drop_cp'])}", log_path)
            flush_pending(False)
        state["n"] = n_done
        for name in claimed:
            if name not in state["done"]:
                state["done"].append(name)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if not args.watch:
            break
    flush_pending(True)
    for _ in procs:
        job_q.put(None)
    for p in procs:
        p.join(timeout=30)
    state["n"] = n_done
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    log(json.dumps({"done_analyze": True, "n": n_done, "tags": state.get("tags")}), log_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true", help="GPU disagreement scan")
    ap.add_argument("--analyze", action="store_true", help="CPU SF19 analysis of needs_sf rows")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--micro-batch", type=int, default=96)
    ap.add_argument("--shard-size", type=int, default=4_096, help="Publish READY inbox shards this large so CPU analysis can overlap")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--nodes", type=int, default=250_000, help="SF19 nodes for candidate analysis")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyze-limit", type=int, default=100_000)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-s", type=float, default=60.0)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.analyze:
        run_analyze(args)
        return
    if not args.go:
        raise SystemExit("pass --go, --analyze, or --self-test")
    run_scan(args)


if __name__ == "__main__":
    main()
