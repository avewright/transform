"""exp096: Selective deep relabel for contested positions.

Hypothesis: Positions where the top-2 move cp gap is small (< 50cp at d8)
have the noisiest targets. Re-labeling these at depth 12+ sharpens exactly
the targets that matter most, for a fraction of the cost of deep-labeling
everything.

Pipeline:
  1. Load d8-relabeled dataset
  2. Filter to positions with cp_gap_top1_top2 < threshold
  3. Re-label those positions at depth 12 (or 16)
  4. Merge back: deep-relabeled positions replace their d8 versions
  5. Write merged dataset

Usage:
    python experiments/exp096_selective_deep_relabel.py \
        --input-dir outputs/exp087_relabeled_d8/dataset \
        --output-dir outputs/exp096_selective_d12/dataset \
        --depth 12 --gap-threshold 50 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import chess
import chess.engine
import torch
import torch.nn.functional as F

STOP_REQUESTED = False
LOG_FILE = None
LABEL_TAU = 120.0


def log(message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(stamped + "\n")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_stockfish_path() -> Path:
    configured = os.environ.get("STOCKFISH_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    binary = shutil.which("stockfish")
    if binary:
        candidates.append(Path(binary))
    candidates.extend([
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
        Path("stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"),
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Stockfish not found")


def score_to_cp(score_obj: chess.engine.PovScore, pov_color: chess.Color) -> tuple[int, str]:
    pov = score_obj.pov(pov_color)
    if pov.is_mate():
        mate = pov.mate()
        if mate is None:
            return 0, "cp"
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate"
    cp = pov.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp"


def cp_to_value_class(cp: int) -> int:
    if cp > 100:
        return 2
    if cp < -100:
        return 0
    return 1


def softmax_probs(cps: list[int], tau: float) -> list[float]:
    scores = torch.tensor(cps, dtype=torch.float32)
    return F.softmax(scores / tau, dim=0).tolist()


def parse_pv(info: dict) -> list[str]:
    pv = info.get("pv") or []
    return [move.uci() for move in pv[:8]]


def relabel_position(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int, tau: float) -> dict:
    legal_moves = list(board.legal_moves)
    num_legal = len(legal_moves)
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=max(1, num_legal))
    if isinstance(infos, dict):
        infos = [infos]

    move_values: dict[str, dict] = {}
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        uci = pv[0].uci()
        if uci in move_values:
            continue
        cp, eval_type = score_to_cp(info["score"], board.turn)
        move_values[uci] = {"uci": uci, "cp": cp, "eval_type": eval_type, "pv": parse_pv(info)}

    for move in legal_moves:
        uci = move.uci()
        if uci in move_values:
            continue
        info = engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
        cp, eval_type = score_to_cp(info["score"], board.turn)
        move_values[uci] = {"uci": uci, "cp": cp, "eval_type": eval_type, "pv": parse_pv(info) or [uci]}

    sorted_moves = sorted(move_values.values(), key=lambda x: x["cp"], reverse=True)
    for rank, item in enumerate(sorted_moves, start=1):
        item["rank"] = rank

    probs = softmax_probs([m["cp"] for m in sorted_moves], tau)
    soft_targets = []
    for item, prob in zip(sorted_moves, probs):
        soft_targets.append({
            "uci": item["uci"], "prob": float(prob), "cp": item["cp"],
            "eval_type": item["eval_type"], "rank": item["rank"], "pv": item["pv"],
        })

    teacher_entropy = -sum(t["prob"] * math.log(max(t["prob"], 1e-12)) for t in soft_targets)
    best_cp = sorted_moves[0]["cp"]
    second_cp = sorted_moves[1]["cp"] if len(sorted_moves) > 1 else best_cp

    return {
        "label_mode": "all_legal_moves",
        "best_move": sorted_moves[0]["uci"],
        "best_cp": best_cp,
        "value_target": cp_to_value_class(best_cp),
        "soft_targets": soft_targets,
        "num_legal": num_legal,
        "num_labeled": len(sorted_moves),
        "unlabeled_legal": 0,
        "teacher_entropy": float(teacher_entropy),
        "cp_gap_top1_top2": int(best_cp - second_cp),
        "full_legal_coverage": True,
    }


def worker_fn(worker_id, task_queue, result_queue, stop_event, depth, tau, sf_threads, sf_hash):
    sf_path = resolve_stockfish_path()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    engine.configure({"Threads": sf_threads, "Hash": sf_hash})
    try:
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                continue
            if task is None:
                break
            board = chess.Board(task["fen"])
            if board.is_game_over():
                result_queue.put(None)
                continue
            label = relabel_position(board, engine, depth, tau)
            record = dict(task)
            record.update(label)
            record["label_depth"] = depth
            record["deep_relabel_source"] = "selective_deep_relabel"
            record["deep_relabel_timestamp"] = utcnow_iso()
            result_queue.put(record)
    finally:
        engine.quit()
        result_queue.put({"_done": worker_id})


def signal_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args():
    p = argparse.ArgumentParser(description="exp096: Selective deep relabel for contested positions")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--gap-threshold", type=int, default=50, help="Relabel positions with cp_gap < this")
    p.add_argument("--tau", type=float, default=LABEL_TAU)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sf-threads", type=int, default=1)
    p.add_argument("--sf-hash-mb", type=int, default=128)
    p.add_argument("--shard-records", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    global LOG_FILE, STOP_REQUESTED
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = args.output_dir.parent / "exp096_selective_relabel.log"

    signal.signal(signal.SIGINT, signal_handler)

    log("=" * 72)
    log(f"exp096: Selective deep relabel at depth {args.depth}")
    log(f"input_dir={args.input_dir}")
    log(f"output_dir={args.output_dir}")
    log(f"gap_threshold={args.gap_threshold}cp, workers={args.workers}")
    log("=" * 72)

    # Load all records
    all_records = []
    for path in sorted(args.input_dir.glob("positions_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    log(f"loaded {len(all_records)} total records")

    # Split into contested (needs relabel) and confident (keep as-is)
    contested = []
    confident = []
    for r in all_records:
        gap = float(r.get("cp_gap_top1_top2", 999))
        if gap < args.gap_threshold:
            contested.append(r)
        else:
            confident.append(r)

    log(f"contested (gap < {args.gap_threshold}cp): {len(contested)} positions → relabeling at d{args.depth}")
    log(f"confident (gap >= {args.gap_threshold}cp): {len(confident)} positions → keeping as-is")

    if not contested:
        log("no contested positions to relabel, copying all records")
        # Just copy everything to output
        shard_idx = 0
        shard_count = 0
        handle = None
        for r in all_records:
            if handle is None or shard_count >= args.shard_records:
                if handle:
                    handle.close()
                shard_idx += 1
                shard_count = 0
                handle = open(args.output_dir / f"positions_{shard_idx:06d}.jsonl", "w", encoding="utf-8")
            handle.write(json.dumps(r) + "\n")
            shard_count += 1
        if handle:
            handle.close()
        log("done")
        return

    # Resume support
    done_fens: set[str] = set()
    relabeled_records: list[dict] = []
    if args.resume:
        for path in sorted(args.output_dir.glob("relabeled_*.jsonl")):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            done_fens.add(rec["fen"])
                            relabeled_records.append(rec)
                        except (json.JSONDecodeError, KeyError):
                            pass
        log(f"resume: {len(done_fens)} already relabeled")
        contested = [r for r in contested if r["fen"] not in done_fens]
        log(f"remaining: {len(contested)}")

    if not contested:
        log("all contested positions already relabeled")
    else:
        # Relabel contested positions
        stop_event = threading.Event()
        task_queue: Queue = Queue(maxsize=256)
        result_queue: Queue = Queue(maxsize=256)

        workers = []
        for wid in range(args.workers):
            t = threading.Thread(
                target=worker_fn,
                args=(wid, task_queue, result_queue, stop_event, args.depth, args.tau,
                      args.sf_threads, args.sf_hash_mb),
                daemon=True,
            )
            t.start()
            workers.append(t)

        def feeder():
            for r in contested:
                if stop_event.is_set():
                    break
                while not stop_event.is_set():
                    try:
                        task_queue.put(r, timeout=1.0)
                        break
                    except Full:
                        continue
            for _ in range(args.workers):
                while not stop_event.is_set():
                    try:
                        task_queue.put(None, timeout=1.0)
                        break
                    except Full:
                        continue

        feeder_thread = threading.Thread(target=feeder, daemon=True)
        feeder_thread.start()

        # Collect relabeled results
        finished = 0
        written = 0
        start = time.time()
        last_log = start

        # Write relabeled to temp file for resume
        relabel_tmp = args.output_dir / f"relabeled_{int(time.time())}.jsonl"
        tmp_handle = open(relabel_tmp, "w", encoding="utf-8")

        try:
            while finished < args.workers and not STOP_REQUESTED:
                try:
                    item = result_queue.get(timeout=1.0)
                except Empty:
                    continue
                if item is None:
                    continue
                if "_done" in item:
                    finished += 1
                    continue

                relabeled_records.append(item)
                tmp_handle.write(json.dumps(item) + "\n")
                written += 1

                now = time.time()
                if now - last_log >= 15.0:
                    elapsed = now - start
                    rate = written / max(elapsed, 1e-6) * 60
                    log(f"relabeled {written}/{len(contested)} ({100*written/len(contested):.1f}%) rate={rate:.0f}/min")
                    tmp_handle.flush()
                    last_log = now
        finally:
            stop_event.set()
            tmp_handle.close()
            feeder_thread.join(timeout=5)
            for w in workers:
                w.join(timeout=10)

        log(f"relabeled {written} contested positions at depth {args.depth}")

    # Merge: replace contested records with relabeled versions
    relabeled_by_fen = {r["fen"]: r for r in relabeled_records}
    merged = []
    for r in all_records:
        if r["fen"] in relabeled_by_fen:
            merged.append(relabeled_by_fen[r["fen"]])
        else:
            merged.append(r)

    # Write merged dataset
    shard_idx = 0
    shard_count = 0
    handle = None
    for r in merged:
        if handle is None or shard_count >= args.shard_records:
            if handle:
                handle.close()
            shard_idx += 1
            shard_count = 0
            handle = open(args.output_dir / f"positions_{shard_idx:06d}.jsonl", "w", encoding="utf-8")
        handle.write(json.dumps(r) + "\n")
        shard_count += 1
    if handle:
        handle.close()

    elapsed = time.time() - time.time()  # placeholder
    log(f"wrote {len(merged)} merged records ({len(relabeled_records)} deep-relabeled) to {args.output_dir}")

    atomic_write_json(args.output_dir.parent / "exp096_status.json", {
        "completed_at": utcnow_iso(),
        "total_records": len(merged),
        "contested_relabeled": len(relabeled_records),
        "confident_kept": len(confident),
        "original_depth": 8,
        "relabel_depth": args.depth,
        "gap_threshold": args.gap_threshold,
    })
    log("done")


if __name__ == "__main__":
    main()
