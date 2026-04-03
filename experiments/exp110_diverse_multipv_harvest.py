"""exp110: Diverse multi-PV harvesting from HF lichess-sf positions.

Hypothesis: The current model (~1850 ELO) was trained mostly on opening
positions. Generating multi-PV soft targets from middlegame and endgame
positions will dramatically improve gameplay in those phases, pushing ELO
past 1900+.

Pipeline:
  1. Stream positions from avewright/chess-positions-lichess-sf (832M positions)
  2. Filter for diverse game phases (favor middlegame + endgame)
  3. Score top-8 moves per position at depth 12 using parallel Stockfish workers
  4. Write sharded JSONL compatible with training pipeline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import chess
import chess.engine
import chess.polyglot
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ──
OUTPUT_DIR = Path("outputs/exp110_diverse_harvest")
DATASET_DIR = OUTPUT_DIR / "dataset"
STATUS_PATH = OUTPUT_DIR / "status.json"
LOG_PATH = OUTPUT_DIR / "exp110.log"
DB_PATH = OUTPUT_DIR / "seen_positions.sqlite"

HF_DATASET = "avewright/chess-positions-lichess-sf"
LABEL_TAU = 120.0
SHARD_SIZE = 5000
DB_COMMIT_INTERVAL = 200

# Phase targets: we want MORE middlegame/endgame (underrepresented in prior training)
PHASE_TARGETS = {"opening": 0.20, "middlegame": 0.45, "endgame": 0.35}

STOP_REQUESTED = False
LOG_FILE = None


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
    raise FileNotFoundError(f"Stockfish not found. Checked: {', '.join(str(c) for c in candidates)}")


SF_PATH = resolve_stockfish_path()


def log(msg: str):
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(stamped + "\n")


def phase_name(fen: str) -> str:
    """Determine game phase from FEN without creating chess.Board."""
    board_part = fen.split()[0]
    pieces = sum(1 for c in board_part if c.isalpha() and c.lower() != 'k')
    if pieces >= 20:
        return "opening"
    if pieces >= 10:
        return "middlegame"
    return "endgame"


def fen_to_board(fen: str) -> chess.Board:
    return chess.Board(fen)


def score_to_cp(score_obj: chess.engine.PovScore, pov: chess.Color) -> tuple[int, str]:
    s = score_obj.pov(pov)
    if s.is_mate():
        mate = s.mate()
        if mate is None:
            return 0, "cp"
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate"
    cp = s.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp"


def cp_to_wdl_class(cp: int) -> int:
    if cp > 100:
        return 2  # win
    if cp < -100:
        return 0  # loss
    return 1  # draw


def softmax_probs(cps: list[int], tau: float) -> list[float]:
    t = torch.tensor(cps, dtype=torch.float32)
    return F.softmax(t / tau, dim=0).tolist()


def label_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    tau: float,
) -> dict | None:
    """Analyze position with Stockfish multipv and return soft target record."""
    try:
        n_legal = board.legal_moves.count()
        if n_legal == 0:
            return None

        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=min(multipv, n_legal),
        )
        if isinstance(infos, dict):
            infos = [infos]

        moves = []
        seen = set()
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            uci = pv[0].uci()
            if uci in seen:
                continue
            seen.add(uci)
            cp, et = score_to_cp(info["score"], board.turn)
            moves.append({
                "uci": uci,
                "cp": cp,
                "eval_type": et,
                "rank": len(moves) + 1,
                "pv": [m.uci() for m in pv[:8]],
            })

        if not moves:
            return None

        moves.sort(key=lambda m: m["cp"], reverse=True)
        probs = softmax_probs([m["cp"] for m in moves], tau)

        soft_targets = []
        for m, p in zip(moves, probs):
            soft_targets.append({
                "uci": m["uci"],
                "prob": float(p),
                "cp": m["cp"],
                "eval_type": m["eval_type"],
                "rank": m["rank"],
                "pv": m["pv"],
            })

        best = moves[0]
        cp_gap = moves[0]["cp"] - moves[1]["cp"] if len(moves) > 1 else 0

        fen = board.fen()
        pos_key = f"{chess.polyglot.zobrist_hash(board):016x}"

        return {
            "source": "exp110_diverse_harvest",
            "fen": fen,
            "position_key": pos_key,
            "phase": phase_name(fen),
            "ply": board.ply(),
            "best_move": best["uci"],
            "best_cp": best["cp"],
            "value_target": cp_to_wdl_class(best["cp"]),
            "label_depth": depth,
            "label_multipv": multipv,
            "label_tau": tau,
            "soft_targets": soft_targets,
            "num_legal": n_legal,
            "num_labeled": len(soft_targets),
            "unlabeled_legal": n_legal - len(soft_targets),
            "cp_gap_top1_top2": cp_gap,
        }
    except Exception as e:
        return None


class PositionDB:
    """SQLite dedup for positions."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY)")
        self.conn.commit()
        self.lock = threading.Lock()
        self.pending = 0

    def is_new(self, key: str) -> bool:
        with self.lock:
            cur = self.conn.execute("SELECT 1 FROM seen WHERE key=?", (key,))
            if cur.fetchone():
                return False
            self.conn.execute("INSERT INTO seen (key) VALUES (?)", (key,))
            self.pending += 1
            if self.pending >= DB_COMMIT_INTERVAL:
                self.conn.commit()
                self.pending = 0
            return True

    def flush(self):
        with self.lock:
            self.conn.commit()
            self.pending = 0

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM seen")
        return cur.fetchone()[0]


class ShardWriter:
    """Write records to sharded JSONL files."""

    def __init__(self, out_dir: Path, shard_size: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.shard_idx = 1
        self.records_in_shard = 0
        self.total_records = 0
        self.current_file = None
        self._open_next_shard()

    def _open_next_shard(self):
        if self.current_file:
            self.current_file.close()
        path = self.out_dir / f"positions_{self.shard_idx:06d}.jsonl"
        self.current_file = open(path, "w")
        self.records_in_shard = 0

    def write(self, record: dict):
        self.current_file.write(json.dumps(record) + "\n")
        self.records_in_shard += 1
        self.total_records += 1
        if self.records_in_shard >= self.shard_size:
            self.shard_idx += 1
            self._open_next_shard()

    def close(self):
        if self.current_file:
            self.current_file.flush()
            self.current_file.close()
            self.current_file = None


def worker_fn(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    depth: int,
    multipv: int,
    tau: float,
):
    """Worker: pull FENs from task_queue, analyze with Stockfish, push results."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
        engine.configure({"Threads": 1, "Hash": 32})
    except Exception as e:
        log(f"Worker {worker_id}: failed to start Stockfish: {e}")
        return

    analyzed = 0
    failures = 0
    try:
        while not STOP_REQUESTED:
            try:
                fen = task_queue.get(timeout=2.0)
            except Empty:
                if STOP_REQUESTED:
                    break
                continue

            if fen is None:  # poison pill
                break

            try:
                board = chess.Board(fen)
                record = label_position(board, engine, depth, multipv, tau)
                if record:
                    result_queue.put(record)
                    analyzed += 1
                    failures = 0  # reset on success
                else:
                    failures += 1
            except Exception:
                failures += 1
                # Restart engine if it crashed
                if failures >= 3:
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    try:
                        engine = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
                        engine.configure({"Threads": 1, "Hash": 32})
                        failures = 0
                    except Exception:
                        break  # can't restart, exit worker
    finally:
        try:
            engine.quit()
        except Exception:
            pass
        log(f"Worker {worker_id}: analyzed {analyzed} positions, exiting.")


def stream_diverse_fens(
    max_positions: int,
    seed: int,
    files_to_sample: int = 50,
) -> list[str]:
    """Stream FENs from HF lichess-sf dataset, balanced across game phases."""
    from data_loader import _hf_token

    log(f"Streaming diverse FENs from {HF_DATASET}...")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=_hf_token())
        all_files = [
            f for f in api.list_repo_files(HF_DATASET, repo_type="dataset")
            if f.startswith("data/train-src") and f.endswith(".parquet")
        ]
    except Exception:
        all_files = [f"data/train-src{i:05d}-of-03275.parquet" for i in range(3275)]

    rng = random.Random(seed)
    selected_files = rng.sample(all_files, min(files_to_sample, len(all_files)))
    log(f"Selected {len(selected_files)} parquet files to sample from")

    phase_counts = Counter()
    phase_limits = {
        phase: int(max_positions * frac)
        for phase, frac in PHASE_TARGETS.items()
    }

    collected = []
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for fi, fname in enumerate(selected_files):
        if len(collected) >= max_positions or STOP_REQUESTED:
            break

        try:
            local = hf_hub_download(
                HF_DATASET, fname, repo_type="dataset", token=_hf_token()
            )
            table = pq.read_table(local, columns=["fen", "ply"])
            fens = table.column("fen").to_pylist()
            plies = table.column("ply").to_pylist() if "ply" in table.column_names else [None] * len(fens)

            rng.shuffle(fens_with_ply := list(zip(fens, plies)))
            for fen, ply in fens_with_ply:
                if len(collected) >= max_positions:
                    break

                phase = phase_name(fen)
                if phase_counts[phase] >= phase_limits[phase]:
                    continue

                # Basic validation
                try:
                    b = chess.Board(fen)
                    if b.legal_moves.count() < 2:
                        continue
                except Exception:
                    continue

                collected.append(fen)
                phase_counts[phase] += 1

            log(f"  File {fi+1}/{len(selected_files)}: collected {len(collected)}/{max_positions} "
                f"(open={phase_counts['opening']}, mid={phase_counts['middlegame']}, end={phase_counts['endgame']})")
        except Exception as e:
            log(f"  File {fi+1} error: {e}")
            continue

    rng.shuffle(collected)
    log(f"Collected {len(collected)} FENs: {dict(phase_counts)}")
    return collected


def main():
    global STOP_REQUESTED, LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-positions", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--tau", type=float, default=LABEL_TAU)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--files-to-sample", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        DATASET_DIR = OUTPUT_DIR / "dataset"
        LOG_PATH = OUTPUT_DIR / "exp110.log"
        DB_PATH = OUTPUT_DIR / "seen_positions.sqlite"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_PATH

    def signal_handler(sig, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log("Shutdown requested...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log(f"exp110: Diverse multi-PV harvest")
    log(f"  Workers: {args.workers}, Depth: {args.depth}, MultiPV: {args.multipv}")
    log(f"  Target positions: {args.max_positions}")
    log(f"  Phase targets: {PHASE_TARGETS}")

    # Step 1: Collect diverse FENs
    fens = stream_diverse_fens(args.max_positions, args.seed, args.files_to_sample)
    if not fens:
        log("No FENs collected, exiting.")
        return

    # Step 2: Set up parallel analysis pipeline
    db = PositionDB(DB_PATH)
    writer = ShardWriter(DATASET_DIR, SHARD_SIZE)

    task_queue = Queue(maxsize=4096)
    result_queue = Queue(maxsize=4096)

    # Start workers
    workers = []
    for i in range(args.workers):
        t = threading.Thread(
            target=worker_fn,
            args=(i, task_queue, result_queue, args.depth, args.multipv, args.tau),
            daemon=True,
        )
        t.start()
        workers.append(t)
    log(f"Started {args.workers} Stockfish workers")

    # Feed FENs into task queue (in a thread)
    def feeder():
        for fen in fens:
            if STOP_REQUESTED:
                break
            task_queue.put(fen)
        # Send poison pills
        for _ in range(args.workers):
            task_queue.put(None)

    feed_thread = threading.Thread(target=feeder, daemon=True)
    feed_thread.start()

    # Collect results
    t0 = time.time()
    written = 0
    phase_written = Counter()
    last_log = time.time()

    while not STOP_REQUESTED:
        try:
            record = result_queue.get(timeout=2.0)
        except Empty:
            # Check if all workers are done
            if not any(t.is_alive() for t in workers):
                # Drain remaining
                while not result_queue.empty():
                    try:
                        record = result_queue.get_nowait()
                        if db.is_new(record["position_key"]):
                            writer.write(record)
                            written += 1
                            phase_written[record["phase"]] += 1
                    except Empty:
                        break
                break
            continue

        if db.is_new(record["position_key"]):
            writer.write(record)
            written += 1
            phase_written[record["phase"]] += 1

        now = time.time()
        if now - last_log > 30:
            elapsed = now - t0
            rate = written / elapsed if elapsed > 0 else 0
            remaining = (len(fens) - written) / rate if rate > 0 else 0
            log(f"Written: {written}/{len(fens)} ({rate:.1f}/s, ~{remaining/60:.0f}m left) "
                f"phases: {dict(phase_written)}")
            last_log = now

    # Cleanup
    db.flush()
    writer.close()
    elapsed = time.time() - t0
    rate = written / elapsed if elapsed > 0 else 0

    log(f"\n=== HARVEST COMPLETE ===")
    log(f"  Written: {written} positions in {elapsed/60:.1f} minutes ({rate:.1f}/s)")
    log(f"  Phases: {dict(phase_written)}")
    log(f"  Shards: {writer.shard_idx}")
    log(f"  Output: {DATASET_DIR}")

    # Save status
    status = {
        "completed": True,
        "written": written,
        "phases": dict(phase_written),
        "elapsed_sec": round(elapsed),
        "rate_per_sec": round(rate, 1),
        "config": {
            "max_positions": args.max_positions,
            "workers": args.workers,
            "depth": args.depth,
            "multipv": args.multipv,
            "tau": args.tau,
            "seed": args.seed,
        },
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
