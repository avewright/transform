#!/usr/bin/env python3
"""Scan labeled soft caches for run2 policy mistakes and write harvest shards.

Keeps rows where greedy argmax != teacher best (`move_idx`). If MultiPV cps
exist, tags inaccuracy / blunder / major / conversion from the teacher's PV.

Usage:
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_run2_mistakes.py --go \\
    --ckpt outputs/sf19_ft/run2/latest.pt
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

_ID_TO_SYMBOL = {
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}
_CASTLE_BITS = ((8, "K"), (4, "Q"), (2, "k"), (1, "q"))


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


def list_local_caches() -> list[tuple[Path, bool, str]]:
    """(path, white_abs, name). Skip frozen SF19 eval via split later."""
    out: list[tuple[Path, bool, str]] = []
    base = ROOT / "outputs/sf19_ft/soft_cache.pt"
    if base.exists():
        out.append((base, True, "sf19_base"))
    expand = ROOT / "outputs/sf19_soft/expand1/inbox"
    if expand.is_dir():
        for sh in sorted(expand.glob("shard_*/soft_cache.pt")):
            out.append((sh, True, f"sf19_{sh.parent.name}"))
    mix = ROOT / "outputs/hf_elo_mix/soft_cache.pt"
    if mix.exists():
        out.append((mix, False, "lichess_mix"))
    deep = ROOT / "outputs/hf_elo_mix/deep_cache.pt"
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
    if soft_i is None or soft_c is None:
        return disagree, tags, drop, in_pv
    for i in range(n):
        if not disagree[i]:
            continue
        row = soft_i[i]
        hits = np.flatnonzero(row == pred[i])
        stm_sign = -1 if (white_abs and int(turn[i]) == 1) else 1
        best_cp = int(cp[i]) * stm_sign
        best_mate = int(mate[i]) * stm_sign
        if hits.size:
            j = int(hits[0])
            in_pv[i] = 1
            model_cp = int(soft_c[i, j]) * stm_sign
            model_mate = int(soft_m[i, j]) * stm_sign if soft_m is not None else 0
        else:
            model_cp = best_cp - 80
            model_mate = 0
        info = classify_lapse(
            best_cp=best_cp,
            best_mate=best_mate,
            model_cp=model_cp,
            model_mate=model_mate,
            model_in_pv=bool(in_pv[i]),
        )
        tags[i] = TAG_TO_I.get(info["tag"], TAG_TO_I["disagree"])
        drop[i] = int(info["drop_cp"] or 0)
    return disagree, tags, drop, in_pv


def _take(t, idx, dtype):
    return torch.from_numpy(np.ascontiguousarray(_np(t, dtype)[idx]))


ORIGIN = {"sf19_base": 0, "lichess_mix": 1, "syzygy": 2}


def pack_keep(data: dict, mask, pred, tags, drop, in_pv, *, source: int, origin: int) -> dict:
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
    """First legal move in model top-k; matches play-time legal masking."""
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


def scan_cache(model, device, cache: Path, *, white_abs: bool, micro: int, skip_eval: bool, origin: int, log_path: Path, chunk: int = 32768):
    data = torch.load(cache, map_location="cpu", weights_only=False)
    n = int(data["move_idx"].shape[0])
    keep = np.ones(n, dtype=bool)
    if skip_eval and "split" in data:
        keep &= _np(data["split"], np.int8) == 0
    pred = np.empty(n, dtype=np.int64)
    turn64 = _np(data["turn"], np.int64)
    cast64 = _np(data["castling"], np.int64)
    ep64 = _np(data["ep_square"], np.int64)
    ba = _np(data["board_array"], np.int8)
    for i in range(0, n, chunk):
        sl = slice(i, min(n, i + chunk))
        fused = board_array_to_fused(torch.from_numpy(np.ascontiguousarray(ba[sl])))
        pred[sl] = predict_legal(
            model, fused,
            torch.from_numpy(np.ascontiguousarray(turn64[sl])),
            torch.from_numpy(np.ascontiguousarray(cast64[sl])),
            ep_square_to_file(torch.from_numpy(np.ascontiguousarray(ep64[sl]))),
            ba[sl], turn64[sl], cast64[sl], ep64[sl],
            device, micro,
        )
    teacher = _np(data["move_idx"], np.int64)
    si = _np(data["soft_indices"], np.int64) if "soft_indices" in data else None
    sc = _np(data["soft_cps"], np.int32) if "soft_cps" in data else None
    sm = _np(data["soft_mates"], np.int32) if "soft_mates" in data else None
    disagree, tags, drop, in_pv = tag_batch(
        pred, teacher, _np(data["turn"], np.int8),
        _np(data["cp"], np.int32), _np(data["mate"], np.int32),
        si, sc, sm, white_abs=white_abs,
    )
    mask = keep & disagree
    n_keep = int(mask.sum())
    log(
        f"  {cache.name} n={n:,} disagree={int(disagree.sum()):,} keep={n_keep:,} "
        f"bl={int(((tags == TAG_TO_I['blunder']) & mask).sum())} "
        f"inacc={int(((tags == TAG_TO_I['inaccuracy']) & mask).sum())} "
        f"maj={int(((tags == TAG_TO_I['major']) & mask).sum())}",
        log_path,
    )
    if n_keep == 0:
        del data
        return None, {"n": n, "disagree": int(disagree.sum()), "keep": 0}
    packed = pack_keep(data, mask, pred, tags, drop, in_pv, source=3, origin=origin)
    stats = {
        "n": n,
        "disagree": int(disagree.sum()),
        "keep": n_keep,
        "tags": {I_TO_TAG[i]: int(((tags == i) & mask).sum()) for i in TAG_TO_I.values()},
    }
    del data
    return packed, stats


def concat_pack(chunks: list[dict]) -> dict:
    keys = chunks[0].keys()
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--ckpt", default="outputs/sf19_ft/run2/latest.pt")
    ap.add_argument("--out-dir", default="outputs/run2_mistakes")
    ap.add_argument("--micro-batch", type=int, default=96)
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--also-stream", action="store_true", help="Then scan HF lichess parquets")
    ap.add_argument("--stream-repo", default="avewright/chess-soft-multipv-lichess")
    ap.add_argument("--stream-target", type=int, default=3_000_000)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")
    if VOCAB_SIZE != 1968:
        raise SystemExit(f"need compact vocab, got {VOCAB_SIZE}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log(f"ckpt={args.ckpt} device={device}", log_path)
    model = load_checkpoint(args.ckpt, device=device)
    if device.type == "cuda" and not args.no_compile:
        log("torch.compile ...", log_path)
        model = torch.compile(model)
    caches = list_local_caches()
    cursor_path = out / "cursor.json"
    done = set()
    if cursor_path.exists():
        prev = json.loads(cursor_path.read_text())
        done = set(prev.get("done") or [])
        log(f"resume skip={len(done)} {sorted(done)[:8]}", log_path)
    log(f"local caches={len(caches)} skip_done={len(done)}", log_path)
    buf: list[dict] = []
    n_buf = 0
    shard_i = 0
    inbox = out / "inbox"
    if inbox.exists():
        existing = sorted(inbox.glob("shard_*"))
        if existing:
            shard_i = int(existing[-1].name.split("_", 1)[1]) + 1
    totals = {"seen": 0, "keep": 0, "disagree": 0, "tags": {}}
    if cursor_path.exists():
        prev = json.loads(cursor_path.read_text())
        for k in ("seen", "keep", "disagree"):
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
            piece = {k: v[start:end] for k, v in data.items()}
            _write_shard(piece, out, shard_i)
            log(f"wrote shard_{shard_i:06d} n={end - start:,} total_keep={totals['keep']:,}", log_path)
            shard_i += 1
            start = end
        buf = []
        n_buf = 0

    t0 = time.time()
    for path, white_abs, name in caches:
        if stop:
            break
        if name in done:
            log(f"skip done {name}", log_path)
            continue
        origin = 0 if name.startswith("sf19") else ORIGIN.get(name, 3)
        log(f"scan {name} white_abs={white_abs} origin={origin}", log_path)
        packed, stats = scan_cache(
            model, device, path,
            white_abs=white_abs, micro=args.micro_batch,
            skip_eval=name == "sf19_base", origin=origin, log_path=log_path,
        )
        totals["seen"] += stats["n"]
        totals["disagree"] += stats["disagree"]
        totals["keep"] += stats["keep"]
        for k, v in (stats.get("tags") or {}).items():
            totals["tags"][k] = totals["tags"].get(k, 0) + v
        if packed is not None:
            buf.append(packed)
            n_buf += int(packed["move_idx"].shape[0])
            flush(False)
        done.add(name)
        cursor_path.write_text(json.dumps({
            "phase": "local", "last": name, "done": sorted(done),
            **totals, "secs": time.time() - t0, "next_shard": shard_i,
        }, indent=2), encoding="utf-8")

    flush(True)
    (out / "summary.json").write_text(json.dumps({
        "ckpt": args.ckpt, **totals, "secs": time.time() - t0, "shards": shard_i,
    }, indent=2), encoding="utf-8")
    log(json.dumps({"done_local": True, **totals, "secs": round(time.time() - t0, 1)}), log_path)

    if args.also_stream and not stop:
        from harvest_hf100m_bulk import main as stream_main

        log("starting HF lichess stream filter", log_path)
        sys.argv = [
            "harvest_hf100m_bulk.py", "--go",
            "--ckpt", args.ckpt,
            "--out-dir", str(out),
            "--repo", args.stream_repo,
            "--target", str(args.stream_target),
            "--micro-batch", str(args.micro_batch),
        ]
        stream_main()


if __name__ == "__main__":
    main()
