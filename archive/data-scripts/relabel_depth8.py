"""relabel_depth8: Re-analyze existing harvested positions at deeper Stockfish depth.

Takes JSONL shards from exp087 (labeled at depth 4) and re-labels each position
at depth 8+ with full legal move coverage. Writes new shards to a separate
output directory.  Designed to run on CPU alongside other processes.

Usage:
    python relabel_depth8.py --input-dir outputs/exp087_full_legal_harvest/dataset \
                             --output-dir outputs/exp087_relabeled_d8/dataset \
                             --depth 8 --workers 4
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
        Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Stockfish not found. Checked: {candidates}")


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


def relabel_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    tau: float,
) -> dict:
    """Full legal move analysis at the given depth."""
    legal_moves = list(board.legal_moves)
    num_legal = len(legal_moves)
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=max(1, num_legal))
    if isinstance(infos, dict):
        infos = [infos]

    move_values_by_uci: dict[str, dict] = {}
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        uci = pv[0].uci()
        if uci in move_values_by_uci:
            continue
        cp, eval_type = score_to_cp(info["score"], board.turn)
        move_values_by_uci[uci] = {
            "uci": uci, "cp": cp, "eval_type": eval_type, "pv": parse_pv(info),
        }

    # Fill in any moves missed by multipv
    fallback_count = 0
    for move in legal_moves:
        uci = move.uci()
        if uci in move_values_by_uci:
            continue
        info = engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
        cp, eval_type = score_to_cp(info["score"], board.turn)
        pv = parse_pv(info) or [uci]
        move_values_by_uci[uci] = {
            "uci": uci, "cp": cp, "eval_type": eval_type, "pv": pv,
        }
        fallback_count += 1

    move_values = sorted(move_values_by_uci.values(), key=lambda x: x["cp"], reverse=True)
    for rank, item in enumerate(move_values, start=1):
        item["rank"] = rank

    probs = softmax_probs([item["cp"] for item in move_values], tau)
    soft_targets = []
    for item, prob in zip(move_values, probs):
        soft_targets.append({
            "uci": item["uci"], "prob": float(prob), "cp": item["cp"],
            "eval_type": item["eval_type"], "rank": item["rank"], "pv": item["pv"],
        })

    teacher_entropy = -sum(
        t["prob"] * math.log(max(t["prob"], 1e-12)) for t in soft_targets
    )
    best_cp = move_values[0]["cp"]
    second_cp = move_values[1]["cp"] if len(move_values) > 1 else best_cp
    return {
        "label_mode": "all_legal_moves",
        "best_move": move_values[0]["uci"],
        "best_cp": best_cp,
        "value_target": cp_to_value_class(best_cp),
        "soft_targets": soft_targets,
        "num_legal": num_legal,
        "num_labeled": len(move_values),
        "unlabeled_legal": 0,
        "teacher_entropy": float(teacher_entropy),
        "cp_gap_top1_top2": int(best_cp - second_cp),
        "full_legal_coverage": True,
        "fallback_single_move_calls": fallback_count,
    }


def worker_fn(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
    depth: int,
    tau: float,
    sf_threads: int,
    sf_hash_mb: int,
) -> None:
    sf_path = resolve_stockfish_path()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    engine.configure({"Threads": sf_threads, "Hash": sf_hash_mb})
    try:
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                continue
            if task is None:
                break

            fen = task["fen"]
            board = chess.Board(fen)
            if board.is_game_over():
                result_queue.put(None)
                continue

            label = relabel_position(board, engine, depth, tau)

            # Merge new labels into existing record, preserving metadata
            record = dict(task)
            record.update(label)
            record["label_depth"] = depth
            record["relabel_source"] = "relabel_depth8"
            record["relabel_timestamp"] = utcnow_iso()

            result_queue.put(record)
    finally:
        engine.quit()
        result_queue.put({"_done": worker_id})


def load_all_records(input_dir: Path) -> list[dict]:
    """Load all JSONL shards from the input directory."""
    records = []
    paths = sorted(input_dir.glob("positions_*.jsonl"))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def signal_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> None:
    global LOG_FILE, STOP_REQUESTED

    parser = argparse.ArgumentParser(description="Re-label harvested positions at deeper SF depth")
    parser.add_argument("--input-dir", type=Path, required=True, help="Dir with positions_*.jsonl from exp087")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dir to write relabeled shards")
    parser.add_argument("--depth", type=int, default=8, help="Stockfish analysis depth (default: 8)")
    parser.add_argument("--tau", type=float, default=LABEL_TAU, help="Softmax temperature for soft targets")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel SF workers")
    parser.add_argument("--sf-threads", type=int, default=1)
    parser.add_argument("--sf-hash-mb", type=int, default=128)
    parser.add_argument("--shard-records", type=int, default=1000, help="Records per output shard")
    parser.add_argument("--resume", action="store_true", help="Skip already-relabeled FENs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir.parent
    LOG_FILE = log_dir / "relabel_depth8.log"

    signal.signal(signal.SIGINT, signal_handler)

    log("=" * 72)
    log(f"relabel_depth8: re-analyzing at depth {args.depth}")
    log(f"input_dir={args.input_dir}")
    log(f"output_dir={args.output_dir}")
    log(f"workers={args.workers} sf_threads={args.sf_threads} sf_hash={args.sf_hash_mb}MB")
    log("=" * 72)

    # Load existing records
    records = load_all_records(args.input_dir)
    log(f"loaded {len(records)} records from {args.input_dir}")

    # Resume support: skip FENs already in output
    done_fens: set[str] = set()
    if args.resume:
        existing_shards = sorted(args.output_dir.glob("positions_*.jsonl"))
        for path in existing_shards:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            done_fens.add(json.loads(line)["fen"])
                        except (json.JSONDecodeError, KeyError):
                            pass
        log(f"resume: skipping {len(done_fens)} already-relabeled FENs")
        records = [r for r in records if r["fen"] not in done_fens]
        log(f"remaining: {len(records)} records to relabel")

    if not records:
        log("nothing to relabel")
        return

    stop_event = threading.Event()
    task_queue: Queue = Queue(maxsize=256)
    result_queue: Queue = Queue(maxsize=256)

    # Start workers
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

    # Feed tasks
    def feeder():
        for rec in records:
            if stop_event.is_set():
                break
            while not stop_event.is_set():
                try:
                    task_queue.put(rec, timeout=1.0)
                    break
                except Full:
                    continue
        # Poison pills
        for _ in range(args.workers):
            while not stop_event.is_set():
                try:
                    task_queue.put(None, timeout=1.0)
                    break
                except Full:
                    continue

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    # Collect results and write shards
    shard_idx = 0
    shard_records = 0
    shard_handle = None
    finished_workers = 0
    written = 0
    start_time = time.time()
    last_log_time = start_time

    def open_shard():
        nonlocal shard_idx, shard_records, shard_handle
        shard_idx += 1
        shard_records = 0
        path = args.output_dir / f"positions_{shard_idx:06d}.jsonl"
        shard_handle = open(path, "a", encoding="utf-8")

    # Resume shard numbering
    existing = sorted(args.output_dir.glob("positions_*.jsonl"))
    if existing and args.resume:
        last = existing[-1]
        try:
            shard_idx = int(last.stem.split("_")[1])
            with open(last) as f:
                shard_records = sum(1 for _ in f)
            if shard_records >= args.shard_records:
                open_shard()
            else:
                shard_handle = open(last, "a", encoding="utf-8")
        except Exception:
            open_shard()
    else:
        open_shard()

    try:
        while finished_workers < args.workers:
            if STOP_REQUESTED:
                stop_event.set()
                break

            try:
                item = result_queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                continue
            if "_done" in item:
                finished_workers += 1
                continue

            assert shard_handle is not None
            shard_handle.write(json.dumps(item) + "\n")
            shard_records += 1
            written += 1

            if shard_records >= args.shard_records:
                shard_handle.flush()
                shard_handle.close()
                open_shard()

            now = time.time()
            if now - last_log_time >= 15.0:
                elapsed = now - start_time
                rate = written / max(elapsed, 1e-6) * 60
                log(f"relabeled {written}/{len(records)} ({100*written/len(records):.1f}%) rate={rate:.0f}/min")
                last_log_time = now
    finally:
        if shard_handle is not None:
            shard_handle.flush()
            shard_handle.close()
        stop_event.set()
        feeder_thread.join(timeout=5)
        for w in workers:
            w.join(timeout=10)

    elapsed = time.time() - start_time
    rate = written / max(elapsed, 1e-6) * 60
    log(f"done: relabeled {written} positions in {elapsed:.0f}s ({rate:.0f}/min)")

    # Write summary
    atomic_write_json(args.output_dir.parent / "relabel_status.json", {
        "completed_at": utcnow_iso(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "original_depth": 4,
        "new_depth": args.depth,
        "total_input": len(records) + len(done_fens),
        "relabeled": written,
        "skipped_resume": len(done_fens),
        "elapsed_sec": round(elapsed, 1),
        "records_per_min": round(rate, 1),
    })


if __name__ == "__main__":
    main()
