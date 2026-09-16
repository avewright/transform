#!/usr/bin/env python3
"""Positions where the model is inaccurate or blunders, with SF19 soft targets.

Scan labeled caches with greedy-legal policy. Keep inaccuracy / blunder /
conversion / major. Off-PV rows get a fresh Stockfish 19 MultiPV label, then
the same filter. Dedup by compact board hash. Every row stores n_pieces.

Usage:
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_model_blunders.py --go
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_model_blunders.py --scan
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_model_blunders.py --label
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from chess_inference import load_checkpoint
from harvest_swa_mistakes import (
    I_TO_TAG,
    TAG_TO_I,
    analyze_one,
    board_array_to_fen,
    holdout_keep,
    load_blocked_hashes,
    pack_keep,
    predict_legal,
    tag_batch,
    _np,
    _slice_cache,
)
from sf19_soft_dataset import (
    SOFT_K,
    SeenDB,
    compact_key_bytes,
    count_soft_targets,
    n_pieces_from_row,
    normalize_harvest_row,
    resolve_sf,
    stack_rows,
    write_shard,
)

DEFAULT_CKPT = ROOT / "outputs/sf19_ft/overnight_20260908/eval_swa.pt"
FALLBACK_CKPT = ROOT / "outputs/hf100m_lapse_ft_30m/eval_swa.pt"
DEFAULT_OUT = ROOT / "outputs/sf19_soft/blunders"
SUBSTANTIAL = frozenset({
    TAG_TO_I["inaccuracy"], TAG_TO_I["blunder"],
    TAG_TO_I["conversion"], TAG_TO_I["major"],
})
NEED_SF = frozenset({TAG_TO_I["off_pv"], TAG_TO_I["disagree"]})


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def default_ckpt() -> Path:
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT
    return FALLBACK_CKPT


def list_sources() -> list[tuple[Path, bool, str]]:
    out: list[tuple[Path, bool, str]] = []
    harvest = ROOT / "outputs/sf19_soft/harvest/inbox"
    if harvest.is_dir():
        for sh in sorted(harvest.glob("shard_*/soft_cache.pt")):
            out.append((sh, True, f"harvest_{sh.parent.name}"))
    for name, path, white_abs in (
        ("sf19_train", ROOT / "outputs/organized_chess_v1/sf19_train.pt", True),
        ("lichess_train", ROOT / "outputs/organized_chess_v1/lichess_train.pt", False),
        ("puzzles_train", ROOT / "outputs/organized_chess_v1/puzzles_train.pt", False),
        ("hf_elo_mix", ROOT / "outputs/hf_elo_mix/soft_cache.pt", False),
        ("hf_elo_deep", ROOT / "outputs/hf_elo_mix/deep_cache.pt", False),
    ):
        if path.exists():
            out.append((path, white_abs, name))
    return out


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def row_hash(ba, turn, castle, ep) -> bytes:
    return compact_key_bytes(ba, turn, castle, ep)


def packed_to_label_row(packed: dict, i: int) -> dict | None:
    raw = {
        "board_array": packed["board_array"][i],
        "turn": packed["turn"][i],
        "castling": packed["castling"][i],
        "ep_square": packed["ep_square"][i],
        "move_idx": packed["move_idx"][i],
        "cp": packed["cp"][i],
        "mate": packed["mate"][i],
        "soft_indices": packed["soft_indices"][i],
        "soft_probs": packed["soft_probs"][i],
        "soft_cps": packed["soft_cps"][i],
        "soft_mates": packed["soft_mates"][i],
        "label_depth": packed["label_depth"][i] if "label_depth" in packed else 12,
        "phase": packed["phase"][i] if "phase" in packed else 1,
        "wdl": packed["wdl"][i] if "wdl" in packed else None,
    }
    if raw["wdl"] is None:
        raw.pop("wdl")
    row = normalize_harvest_row(raw)
    if row is None:
        return None
    row["model_move_idx"] = np.int64(int(packed["model_move_idx"][i]))
    row["tag"] = np.int8(int(packed["tag"][i]))
    row["drop_cp"] = np.int32(int(packed["drop_cp"][i]))
    row["model_in_pv"] = np.int8(int(packed["model_in_pv"][i]))
    return row


def attach_meta(row: dict) -> dict:
    row["n_pieces"] = np.int8(int(row["n_pieces"]) if "n_pieces" in row else n_pieces_from_row(row))
    row["n_soft"] = np.int8(int(row["n_soft"]) if "n_soft" in row else count_soft_targets(
        row["soft_indices"], row.get("soft_probs"),
    ))
    row["pos_hash"] = np.frombuffer(row_hash(
        row["board_array"], row["turn"], row["castling"], row["ep_square"],
    ), dtype=np.uint64)[0]
    return row


def stack_blunder_rows(rows: list[dict]) -> dict:
    data = stack_rows(rows)
    extra = ("model_move_idx", "tag", "drop_cp", "model_in_pv", "pos_hash")
    for k in extra:
        data[k] = torch.from_numpy(np.stack([r[k] for r in rows]))
    return data


def write_keep_shard(rows: list[dict], dest: Path, meta: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    data = stack_blunder_rows(rows)
    write_shard(data, dest, meta)


# Queued SF jobs are written as jsonl during scan to avoid mixing with keep shards.
_QUEUE_F = None


def _queue_open(out: Path):
    global _QUEUE_F
    path = out / "sf_queue.jsonl"
    _QUEUE_F = path.open("a", encoding="utf-8")
    return path


def _queue_write(item: dict) -> None:
    if _QUEUE_F is None:
        return
    _QUEUE_F.write(json.dumps({"fen": item["fen"], "model_move_idx": int(item["model_move_idx"]), "tag": int(item["tag"])}) + "\n")


def run_scan_v2(args) -> None:
    """Scan caches; write keep shards and an SF queue file."""
    out = Path(args.out_dir)
    inbox = out / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    log_path = out / "scan.log"
    qpath = _queue_open(out)
    blocked = load_blocked_hashes(None)
    ckpt = Path(args.ckpt)
    device = pick_device()
    log(f"scan ckpt={ckpt} device={device} blocked={blocked.size:,} queue={qpath}", log_path)
    model = load_checkpoint(ckpt, device=device)
    model.eval()
    sources = list_sources()
    cursor = out / "scan_cursor.json"
    done = set()
    if cursor.exists():
        done = set(json.loads(cursor.read_text()).get("done") or [])
    shard_i = 0
    existing = sorted(inbox.glob("shard_*"))
    if existing:
        shard_i = int(existing[-1].name.split("_", 1)[1]) + 1
    seen = SeenDB(out / "seen.sqlite")
    totals = {"seen": 0, "keep": 0, "queue_sf": 0, "dup": 0, "tags": {k: 0 for k in TAG_TO_I}}
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    buf: list[dict] = []

    def flush(force: bool = False) -> None:
        nonlocal buf, shard_i
        if not buf or (len(buf) < args.shard_size and not force):
            return
        take = buf[: args.shard_size] if not force else buf
        buf = buf[len(take):] if not force else []
        keys = [row_hash(r["board_array"], r["turn"], r["castling"], r["ep_square"]) for r in take]
        sh = inbox / f"shard_{shard_i:06d}"
        write_keep_shard(take, sh, {
            "phase": "labeled", "n": len(take), "ckpt": str(ckpt), "teacher": "Stockfish 19",
        })
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        log(f"wrote {sh.name} n={len(take)} keep={totals['keep']:,} queue_sf={totals['queue_sf']:,}", log_path)
        shard_i += 1

    try:
        for path, white_abs, name in sources:
            if stop:
                break
            if name in done:
                log(f"skip {name}", log_path)
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            n = int(data["move_idx"].shape[0])
            keep_mask = holdout_keep(data, blocked)
            ba = _np(data["board_array"], np.int8)
            turn64 = _np(data["turn"], np.int64)
            cast64 = _np(data["castling"], np.int64)
            ep64 = _np(data["ep_square"], np.int64)
            teacher = _np(data["move_idx"], np.int64)
            si = _np(data["soft_indices"], np.int64) if "soft_indices" in data else None
            sc = _np(data["soft_cps"], np.int32) if "soft_cps" in data else None
            sm = _np(data["soft_mates"], np.int32) if "soft_mates" in data else None
            turn8 = _np(data["turn"], np.int8)
            cp32 = _np(data["cp"], np.int32)
            mate32 = _np(data["mate"], np.int32)
            t0 = time.time()
            log(f"source {name} n={n:,} white_abs={white_abs}", log_path)
            for i in range(0, n, args.chunk):
                if stop:
                    break
                sl = slice(i, min(n, i + args.chunk))
                from data_loader import board_array_to_fused, ep_square_to_file
                fused = board_array_to_fused(torch.from_numpy(np.ascontiguousarray(ba[sl])))
                pred = predict_legal(
                    model, fused,
                    torch.from_numpy(np.ascontiguousarray(turn64[sl])),
                    torch.from_numpy(np.ascontiguousarray(cast64[sl])),
                    ep_square_to_file(torch.from_numpy(np.ascontiguousarray(ep64[sl]))),
                    ba[sl], turn64[sl], cast64[sl], ep64[sl],
                    device, args.micro,
                )
                disagree, tags, drop, in_pv, needs_sf = tag_batch(
                    pred, teacher[sl], turn8[sl], cp32[sl], mate32[sl],
                    None if si is None else si[sl],
                    None if sc is None else sc[sl],
                    None if sm is None else sm[sl],
                    white_abs=white_abs,
                )
                mask = keep_mask[sl] & disagree
                packed = pack_keep(
                    _slice_cache(data, sl), mask, pred, tags, drop, in_pv, needs_sf,
                    source=3, origin=0 if white_abs else 1,
                )
                m = int(packed["move_idx"].shape[0])
                totals["seen"] += int(sl.stop - sl.start)
                for j in range(m):
                    tag = int(packed["tag"][j])
                    totals["tags"][I_TO_TAG[tag]] = totals["tags"].get(I_TO_TAG[tag], 0) + 1
                    key = row_hash(
                        packed["board_array"][j], packed["turn"][j],
                        packed["castling"][j], packed["ep_square"][j],
                    )
                    if seen.has(key):
                        totals["dup"] += 1
                        continue
                    if tag in SUBSTANTIAL and int(packed["needs_sf"][j]) == 0:
                        row = packed_to_label_row(packed, j)
                        if row is None:
                            totals["queue_sf"] += 1
                            _queue_write({
                                "fen": board_array_to_fen(
                                    packed["board_array"][j].numpy(),
                                    int(packed["turn"][j]), int(packed["castling"][j]),
                                    int(packed["ep_square"][j]),
                                ),
                                "model_move_idx": int(packed["model_move_idx"][j]),
                                "tag": tag,
                            })
                            continue
                        row = attach_meta(row)
                        seen.remember_hot([key])
                        buf.append(row)
                        totals["keep"] += 1
                    elif tag in NEED_SF or int(packed["needs_sf"][j]) == 1:
                        totals["queue_sf"] += 1
                        _queue_write({
                            "fen": board_array_to_fen(
                                packed["board_array"][j].numpy(),
                                int(packed["turn"][j]), int(packed["castling"][j]),
                                int(packed["ep_square"][j]),
                            ),
                            "model_move_idx": int(packed["model_move_idx"][j]),
                            "tag": tag,
                        })
                flush(False)
                dt = max(time.time() - t0, 1e-6)
                if (i // args.chunk) % 2 == 0:
                    log(
                        f"  {name} {min(n, sl.stop):,}/{n:,} {min(n, sl.stop)/dt:.0f}/s "
                        f"keep={totals['keep']:,} queue={totals['queue_sf']:,} dup={totals['dup']:,}",
                        log_path,
                    )
            done.add(name)
            cursor.write_text(json.dumps({"done": sorted(done), **totals}, indent=2), encoding="utf-8")
            del data
            log(f"done {name} keep={totals['keep']:,} queue={totals['queue_sf']:,} dup={totals['dup']:,}", log_path)
    finally:
        flush(True)
        if _QUEUE_F is not None:
            _QUEUE_F.close()
    (out / "scan_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    log(json.dumps({"scan_done": True, **totals}), log_path)


def run_label(args) -> None:
    from multiprocessing import Process, Queue

    out = Path(args.out_dir)
    inbox = out / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    log_path = out / "label.log"
    qpath = out / "sf_queue.jsonl"
    if not qpath.exists():
        log("no sf_queue.jsonl — scan first", log_path)
        return
    seen = SeenDB(out / "seen.sqlite")
    jobs = []
    for line in qpath.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        jobs.append(json.loads(line))
    log(f"label queue={len(jobs):,} sf={resolve_sf()} workers={args.workers} nodes={args.nodes}", log_path)
    if not jobs:
        return
    shard_i = 0
    existing = sorted(inbox.glob("shard_*"))
    if existing:
        shard_i = int(existing[-1].name.split("_", 1)[1]) + 1
    job_q: Queue = Queue()
    res_q: Queue = Queue()
    sf = resolve_sf()

    def worker() -> None:
        import chess.engine
        from sf19_soft_dataset import _init_worker  # noqa: F401

        engine = chess.engine.SimpleEngine.popen_uci(sf)
        engine.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
        try:
            while True:
                item = job_q.get()
                if item is None:
                    break
                try:
                    rec = analyze_one(
                        engine, item["fen"], int(item["model_move_idx"]),
                        nodes=args.nodes, multipv=SOFT_K, tau=120.0, watchdog_s=20.0,
                    )
                except Exception:
                    rec = None
                res_q.put(rec)
        finally:
            try:
                engine.quit()
            except Exception:
                pass

    procs = [Process(target=worker, daemon=True) for _ in range(args.workers)]
    for p in procs:
        p.start()
    inflight = 0
    sent = 0
    buf: list[dict] = []
    kept = 0
    skipped = {"ok": 0, "dup": 0, "fail": 0}

    def flush(force: bool = False) -> None:
        nonlocal buf, shard_i
        if not buf or (len(buf) < args.shard_size and not force):
            return
        take = buf[: args.shard_size] if not force else buf
        buf = buf[len(take):] if not force else []
        keys = [row_hash(r["board_array"], r["turn"], r["castling"], r["ep_square"]) for r in take]
        sh = inbox / f"shard_{shard_i:06d}"
        write_keep_shard(take, sh, {"phase": "sf19", "n": len(take), "nodes": args.nodes})
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        log(f"wrote {sh.name} n={len(take)} labeled_keep={kept:,}", log_path)
        shard_i += 1

    try:
        while sent < len(jobs) or inflight:
            while inflight < args.workers * 4 and sent < len(jobs):
                job_q.put(jobs[sent])
                sent += 1
                inflight += 1
            rec = res_q.get()
            inflight -= 1
            if rec is None:
                skipped["fail"] += 1
                continue
            if int(rec.get("tag", 0)) not in SUBSTANTIAL:
                skipped["ok"] += 1
                continue
            rec = attach_meta(rec)
            key = row_hash(rec["board_array"], rec["turn"], rec["castling"], rec["ep_square"])
            if seen.has(key):
                skipped["dup"] += 1
                continue
            seen.remember_hot([key])
            buf.append(rec)
            kept += 1
            flush(False)
            if kept % 32 == 0:
                log(f"labeled keep={kept:,} skip={skipped} sent={sent:,}/{len(jobs):,}", log_path)
    finally:
        for _ in procs:
            job_q.put(None)
        flush(True)
        for p in procs:
            p.join(timeout=5)
    (out / "label_summary.json").write_text(json.dumps({"keep": kept, "skipped": skipped}, indent=2), encoding="utf-8")
    log(json.dumps({"label_done": True, "keep": kept, "skipped": skipped}), log_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="scan then SF19-label the queue")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--ckpt", default=str(default_ckpt()))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=100_000)
    args = ap.parse_args()
    if not (args.go or args.scan or args.label):
        raise SystemExit("pass --go, --scan, or --label")
    if args.go or args.scan:
        run_scan_v2(args)
    if args.go or args.label:
        run_label(args)


if __name__ == "__main__":
    main()
