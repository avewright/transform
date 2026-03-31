"""exp085: Parallel Stockfish multipv soft-target harvesting with durable dedupe.

Pipeline:
  1. Sample diverse trajectories from a curated opening set using Stockfish multipv move sampling.
  2. Select a few positions per trajectory instead of labeling every ply.
  3. Label each selected position with a single multipv analysis call on parallel workers.
  4. Deduplicate positions with a SQLite-backed position index before writing.
  5. Write unique records into sharded JSONL files for downstream training.

This script is designed to maximize useful soft-target throughput, not to train.
It keeps exact-position repeats out of the dataset while preserving metadata that
lets a later trainer sample by freshness, phase, and difficulty.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sqlite3
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import chess
import chess.engine
import chess.polyglot
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs/exp085_parallel_multipv_harvest")
DATASET_DIR = OUTPUT_DIR / "dataset"
STATUS_PATH = OUTPUT_DIR / "status.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LOG_PATH = OUTPUT_DIR / "exp085.log"
DB_PATH = OUTPUT_DIR / "seen_positions.sqlite"

LABEL_TAU = 120.0
STATUS_INTERVAL_SEC = 15.0
TASK_QUEUE_MAXSIZE = 8192
RESULT_QUEUE_MAXSIZE = 8192
DEFAULT_SHARD_RECORDS = 5000
DB_COMMIT_INTERVAL = 100
WRITER_FLUSH_INTERVAL = 100

DEFAULT_SF_THREADS = 1
DEFAULT_SF_HASH_MB = 128

OPENINGS = [
    {"name": "startpos", "moves": []},
    {"name": "king_pawn_game", "moves": ["e2e4", "e7e5"]},
    {"name": "sicilian_defense", "moves": ["e2e4", "c7c5"]},
    {"name": "french_defense", "moves": ["e2e4", "e7e6"]},
    {"name": "caro_kann", "moves": ["e2e4", "c7c6"]},
    {"name": "scandinavian_defense", "moves": ["e2e4", "d7d5"]},
    {"name": "queen_pawn_game", "moves": ["d2d4", "d7d5"]},
    {"name": "slav_defense", "moves": ["d2d4", "d7d5", "c2c4", "c7c6"]},
    {"name": "queens_gambit_declined", "moves": ["d2d4", "d7d5", "c2c4", "e7e6"]},
    {"name": "kings_indian_defense", "moves": ["d2d4", "g8f6", "c2c4", "g7g6"]},
    {"name": "grunfeld_defense", "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"]},
    {"name": "nimzo_indian_defense", "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]},
    {"name": "english_reversed_sicilian", "moves": ["c2c4", "e7e5"]},
    {"name": "reti_opening", "moves": ["g1f3", "d7d5", "c2c4"]},
    {"name": "italian_game", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]},
    {"name": "ruy_lopez", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]},
    {"name": "petroff_defense", "moves": ["e2e4", "e7e5", "g1f3", "g8f6"]},
    {"name": "four_knights_game", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"]},
    {"name": "sicilian_najdorf_setup", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]},
    {"name": "sicilian_dragon_setup", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"]},
]

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
    candidates.extend(
        [
            Path("/usr/games/stockfish"),
            Path("/usr/bin/stockfish"),
            Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate Stockfish binary. Checked: {checked}")


STOCKFISH_PATH = resolve_stockfish_path()


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


def phase_name(board: chess.Board) -> str:
    pieces = sum(1 for piece in board.piece_map().values() if piece.piece_type != chess.KING)
    if pieces >= 20:
        return "opening"
    if pieces >= 10:
        return "middlegame"
    return "endgame"


def create_engine(*, threads: int, hash_mb: int) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    engine.configure({"Threads": threads, "Hash": hash_mb})
    return engine


def board_position_fen(board: chess.Board) -> str:
    ep = "-"
    if board.ep_square is not None and board.has_legal_en_passant():
        ep = chess.square_name(board.ep_square)
    return f"{board.board_fen()} {'w' if board.turn == chess.WHITE else 'b'} {board.castling_xfen()} {ep}"


def board_position_key(board: chess.Board) -> str:
    return f"{chess.polyglot.zobrist_hash(board):016x}"


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


def sample_engine_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    multipv: int,
    tau: float,
) -> tuple[chess.Move, dict]:
    num_legal = board.legal_moves.count()
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=min(multipv, num_legal))
    if isinstance(infos, dict):
        infos = [infos]

    candidates = []
    seen = set()
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        if move.uci() in seen:
            continue
        seen.add(move.uci())
        cp, eval_type = score_to_cp(info["score"], board.turn)
        candidates.append(
            {
                "move": move,
                "uci": move.uci(),
                "cp": cp,
                "eval_type": eval_type,
                "pv": parse_pv(info),
            }
        )

    if not candidates:
        fallback = next(iter(board.legal_moves))
        return fallback, {
            "selected_uci": fallback.uci(),
            "selected_cp": 0,
            "candidate_count": 1,
            "selection_mode": "fallback",
            "candidates": [{"uci": fallback.uci(), "cp": 0, "eval_type": "cp", "pv": [fallback.uci()]}],
        }

    candidates.sort(key=lambda item: item["cp"], reverse=True)
    if tau <= 0 or len(candidates) == 1:
        selected_idx = 0
    else:
        probs = softmax_probs([item["cp"] for item in candidates], tau)
        selected_idx = torch.multinomial(torch.tensor(probs), 1).item()

    selected = candidates[selected_idx]
    return selected["move"], {
        "selected_uci": selected["uci"],
        "selected_cp": selected["cp"],
        "candidate_count": len(candidates),
        "selection_mode": "sampled" if tau > 0 and len(candidates) > 1 else "greedy",
        "candidates": [{k: v for k, v in item.items() if k != "move"} for item in candidates],
    }


def label_position_multipv(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    tau: float,
) -> dict:
    num_legal = board.legal_moves.count()
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=min(multipv, num_legal))
    if isinstance(infos, dict):
        infos = [infos]

    move_values = []
    seen = set()
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        uci = move.uci()
        if uci in seen:
            continue
        seen.add(uci)
        cp, eval_type = score_to_cp(info["score"], board.turn)
        move_values.append(
            {
                "uci": uci,
                "cp": cp,
                "eval_type": eval_type,
                "rank": len(move_values) + 1,
                "pv": parse_pv(info),
            }
        )

    if not move_values:
        fallback = next(iter(board.legal_moves)).uci()
        move_values = [{"uci": fallback, "cp": 0, "eval_type": "cp", "rank": 1, "pv": [fallback]}]

    move_values.sort(key=lambda item: item["cp"], reverse=True)
    probs = softmax_probs([item["cp"] for item in move_values], tau)
    soft_targets = []
    for item, prob in zip(move_values, probs):
        soft_targets.append(
            {
                "uci": item["uci"],
                "prob": float(prob),
                "cp": item["cp"],
                "eval_type": item["eval_type"],
                "rank": item["rank"],
                "pv": item["pv"],
            }
        )

    teacher_entropy = -sum(
        target["prob"] * math.log(max(target["prob"], 1e-12))
        for target in soft_targets
    )
    best_cp = move_values[0]["cp"]
    second_cp = move_values[1]["cp"] if len(move_values) > 1 else best_cp
    return {
        "label_mode": "multipv_topk",
        "best_move": move_values[0]["uci"],
        "best_cp": best_cp,
        "value_target": cp_to_value_class(best_cp),
        "soft_targets": soft_targets,
        "num_legal": num_legal,
        "num_labeled": len(move_values),
        "unlabeled_legal": max(num_legal - len(move_values), 0),
        "teacher_entropy": float(teacher_entropy),
        "cp_gap_top1_top2": int(best_cp - second_cp),
    }


def create_board_from_opening(opening_moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal opening move {uci} in {opening_moves}")
        board.push(move)
    return board


@dataclass
class HarvestStats:
    lineages_started: int = 0
    candidate_positions: int = 0
    queued_positions: int = 0
    labeled_positions: int = 0
    written_positions: int = 0
    duplicate_skips_prequeue: int = 0
    duplicate_skips_postlabel: int = 0
    write_failures: int = 0
    shard_index: int = 0
    current_shard_records: int = 0
    exact_positions_seen: int = 0
    worker_done_count: int = 0
    start_time: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def incr(self, field_name: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    def set_counts(self, *, shard_index: int, current_shard_records: int, exact_positions_seen: int) -> None:
        with self.lock:
            self.shard_index = shard_index
            self.current_shard_records = current_shard_records
            self.exact_positions_seen = exact_positions_seen

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = max(time.time() - self.start_time, 1e-6)
            payload = {
                "lineages_started": self.lineages_started,
                "candidate_positions": self.candidate_positions,
                "queued_positions": self.queued_positions,
                "labeled_positions": self.labeled_positions,
                "written_positions": self.written_positions,
                "duplicate_skips_prequeue": self.duplicate_skips_prequeue,
                "duplicate_skips_postlabel": self.duplicate_skips_postlabel,
                "write_failures": self.write_failures,
                "shard_index": self.shard_index,
                "current_shard_records": self.current_shard_records,
                "exact_positions_seen": self.exact_positions_seen,
                "worker_done_count": self.worker_done_count,
                "elapsed_sec": round(elapsed, 1),
                "records_per_min": round(self.written_positions * 60.0 / elapsed, 2),
            }
        return payload


class PositionIndex:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_key TEXT PRIMARY KEY,
                position_fen TEXT NOT NULL,
                created_at TEXT NOT NULL,
                opening_name TEXT,
                phase TEXT,
                ply INTEGER,
                shard_name TEXT
            )
            """
        )
        self.conn.commit()
        self.lock = threading.Lock()
        row = self.conn.execute("SELECT COUNT(*) FROM positions").fetchone()
        self.cached_count = int(row[0] if row else 0)
        self.pending_writes = 0

    def contains(self, position_key: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM positions WHERE position_key = ? LIMIT 1",
                (position_key,),
            ).fetchone()
        return row is not None

    def insert(self, record: dict, shard_name: str) -> bool:
        with self.lock:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO positions (
                    position_key, position_fen, created_at, opening_name, phase, ply, shard_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["position_key"],
                    record["position_fen"],
                    record["created_at"],
                    record.get("opening_name"),
                    record.get("phase"),
                    int(record.get("ply", 0)),
                    shard_name,
                ),
            )
            inserted = cursor.rowcount > 0
            if inserted:
                self.cached_count += 1
                self.pending_writes += 1
                if self.pending_writes >= DB_COMMIT_INTERVAL:
                    self.conn.commit()
                    self.pending_writes = 0
        return inserted

    def count(self) -> int:
        with self.lock:
            return self.cached_count

    def close(self) -> None:
        with self.lock:
            if self.pending_writes:
                self.conn.commit()
                self.pending_writes = 0
            self.conn.close()


class ShardedJsonlWriter:
    def __init__(self, root: Path, shard_records: int):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_records = shard_records
        self.shard_index = 0
        self.current_records = 0
        self.current_path: Path | None = None
        self.current_handle = None
        self.lock = threading.Lock()
        self.pending_flush = 0
        self._rotate()

    def _rotate(self) -> None:
        if self.current_handle is not None:
            self.current_handle.flush()
            self.current_handle.close()
        self.shard_index += 1
        self.current_records = 0
        self.pending_flush = 0
        self.current_path = self.root / f"positions_{self.shard_index:06d}.jsonl"
        self.current_handle = open(self.current_path, "a", encoding="utf-8")

    def write(self, record: dict) -> str:
        with self.lock:
            if self.current_records >= self.shard_records:
                self._rotate()
            assert self.current_handle is not None
            assert self.current_path is not None
            self.current_handle.write(json.dumps(record) + "\n")
            self.current_records += 1
            self.pending_flush += 1
            if self.pending_flush >= WRITER_FLUSH_INTERVAL:
                self.current_handle.flush()
                self.pending_flush = 0
            return self.current_path.name

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "shard_index": self.shard_index,
                "current_shard_records": self.current_records,
                "current_shard_path": str(self.current_path) if self.current_path else None,
            }

    def close(self) -> None:
        with self.lock:
            if self.current_handle is not None:
                self.current_handle.flush()
                self.current_handle.close()
                self.current_handle = None
                self.pending_flush = 0


def build_trajectory(
    engine: chess.engine.SimpleEngine,
    opening: dict,
    target_plies: int,
    play_depth: int,
    play_multipv: int,
    play_tau: float,
) -> tuple[list[dict], list[str]]:
    board = create_board_from_opening(opening["moves"])
    history = [move.uci() for move in board.move_stack]
    positions = []
    local_seen = set()

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < target_plies:
        key = board_position_key(board)
        if key not in local_seen:
            local_seen.add(key)
            positions.append(
                {
                    "fen": board.fen(),
                    "position_fen": board_position_fen(board),
                    "position_key": key,
                    "ply": len(board.move_stack),
                    "phase": phase_name(board),
                }
            )

        move, _meta = sample_engine_move(
            engine=engine,
            board=board,
            depth=play_depth,
            multipv=play_multipv,
            tau=play_tau,
        )
        if move not in board.legal_moves:
            break
        board.push(move)
        history.append(move.uci())

    return positions, history


def choose_positions_from_trajectory(
    positions: list[dict],
    positions_per_lineage: int,
    rng: random.Random,
) -> list[dict]:
    eligible = [item for item in positions if item["ply"] >= 8]
    if len(eligible) <= positions_per_lineage:
        return eligible
    return rng.sample(eligible, positions_per_lineage)


def scheduler_worker(
    scheduler_id: int,
    task_queue: Queue,
    stop_event: threading.Event,
    store: PositionIndex,
    inflight: set[str],
    inflight_lock: threading.Lock,
    stats: HarvestStats,
    args: argparse.Namespace,
) -> None:
    rng = random.Random(args.seed + 17 + scheduler_id)
    engine = create_engine(threads=args.sf_threads, hash_mb=args.sf_hash_mb)
    try:
        while not stop_event.is_set():
            if args.max_records > 0:
                snap = stats.snapshot()
                with inflight_lock:
                    inflight_count = len(inflight)
                if snap["written_positions"] + inflight_count >= args.max_records:
                    break

            opening = rng.choice(OPENINGS)
            target_plies = rng.randint(args.min_target_plies, args.max_target_plies)
            positions, history = build_trajectory(
                engine=engine,
                opening=opening,
                target_plies=target_plies,
                play_depth=args.play_depth,
                play_multipv=args.play_multipv,
                play_tau=args.play_tau,
            )
            stats.incr("lineages_started", 1)
            chosen = choose_positions_from_trajectory(positions, args.positions_per_lineage, rng)
            lineage_id = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"s{scheduler_id:02d}_{stats.snapshot()['lineages_started']:08d}"
            )

            for candidate in chosen:
                if stop_event.is_set():
                    break
                if args.max_records > 0:
                    snap = stats.snapshot()
                    with inflight_lock:
                        inflight_count = len(inflight)
                    if snap["written_positions"] + inflight_count >= args.max_records:
                        stop_event.set()
                        break
                stats.incr("candidate_positions", 1)
                key = candidate["position_key"]
                with inflight_lock:
                    already_inflight = key in inflight
                if already_inflight or store.contains(key):
                    stats.incr("duplicate_skips_prequeue", 1)
                    continue

                task = {
                    "trajectory_id": lineage_id,
                    "created_at": utcnow_iso(),
                    "opening_name": opening["name"],
                    "opening_moves": opening["moves"],
                    "target_plies": target_plies,
                    "move_history": history,
                    **candidate,
                }
                with inflight_lock:
                    inflight.add(key)
                while not stop_event.is_set():
                    try:
                        task_queue.put(task, timeout=1.0)
                        stats.incr("queued_positions", 1)
                        break
                    except Full:
                        continue
                else:
                    with inflight_lock:
                        inflight.discard(key)
    finally:
        engine.quit()


def label_worker(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
    args: argparse.Namespace,
) -> None:
    engine = create_engine(threads=args.sf_threads, hash_mb=args.sf_hash_mb)
    try:
        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                if stop_event.is_set():
                    break
                continue

            if task is None:
                break

            board = chess.Board(task["fen"])
            label = label_position_multipv(
                board=board,
                engine=engine,
                depth=args.label_depth,
                multipv=args.label_multipv,
                tau=args.label_tau,
            )
            record = {
                "source": "sf_multipv_parallel_v1",
                "created_at": task["created_at"],
                "trajectory_id": task["trajectory_id"],
                "worker_id": worker_id,
                "position_key": task["position_key"],
                "position_fen": task["position_fen"],
                "fen": task["fen"],
                "phase": task["phase"],
                "ply": task["ply"],
                "opening_name": task["opening_name"],
                "opening_moves": task["opening_moves"],
                "target_plies": task["target_plies"],
                "move_history": task["move_history"],
                "label_depth": args.label_depth,
                "label_multipv": min(args.label_multipv, label["num_legal"]),
                "label_tau": args.label_tau,
                "play_depth": args.play_depth,
                "play_multipv": args.play_multipv,
                "play_tau": args.play_tau,
                **label,
            }

            while True:
                try:
                    result_queue.put(record, timeout=1.0)
                    break
                except Full:
                    if stop_event.is_set():
                        break
                    continue
    finally:
        try:
            engine.quit()
        finally:
            result_queue.put({"_worker_done": worker_id})


def writer_worker(
    result_queue: Queue,
    stop_event: threading.Event,
    store: PositionIndex,
    writer: ShardedJsonlWriter,
    inflight: set[str],
    inflight_lock: threading.Lock,
    stats: HarvestStats,
    worker_count: int,
) -> None:
    while True:
        try:
            item = result_queue.get(timeout=1.0)
        except Empty:
            if stop_event.is_set():
                snap = stats.snapshot()
                if snap["worker_done_count"] >= worker_count:
                    break
            continue

        if "_worker_done" in item:
            stats.incr("worker_done_count", 1)
            if stop_event.is_set() and stats.snapshot()["worker_done_count"] >= worker_count:
                break
            continue

        stats.incr("labeled_positions", 1)
        shard_name = writer.snapshot()["current_shard_path"] or "pending"
        inserted = store.insert(item, Path(shard_name).name if shard_name else "pending")
        with inflight_lock:
            inflight.discard(item["position_key"])

        if not inserted:
            stats.incr("duplicate_skips_postlabel", 1)
            continue

        try:
            shard_name = writer.write(item)
        except Exception as exc:
            stats.incr("write_failures", 1)
            log(f"writer failure for {item['position_key']}: {exc}")
            continue

        stats.incr("written_positions", 1)
        snap = writer.snapshot()
        stats.set_counts(
            shard_index=snap["shard_index"],
            current_shard_records=snap["current_shard_records"],
            exact_positions_seen=store.count(),
        )


def write_manifest(args: argparse.Namespace) -> None:
    atomic_write_json(
        MANIFEST_PATH,
        {
            "created_at": utcnow_iso(),
            "source": "exp085_parallel_multipv_harvest",
            "stockfish_path": str(STOCKFISH_PATH),
            "dataset_dir": str(DATASET_DIR),
            "db_path": str(DB_PATH),
            "config": {
                "scheduler_count": args.scheduler_count,
                "worker_count": args.worker_count,
                "sf_threads": args.sf_threads,
                "sf_hash_mb": args.sf_hash_mb,
                "label_depth": args.label_depth,
                "label_multipv": args.label_multipv,
                "label_tau": args.label_tau,
                "play_depth": args.play_depth,
                "play_multipv": args.play_multipv,
                "play_tau": args.play_tau,
                "positions_per_lineage": args.positions_per_lineage,
                "min_target_plies": args.min_target_plies,
                "max_target_plies": args.max_target_plies,
                "max_records": args.max_records if args.max_records > 0 else None,
                "shard_records": args.shard_records,
                "seed": args.seed,
            },
            "dedupe": {
                "key": "polyglot zobrist hash over board/side/castling/ep",
                "exact_position_fen": "board turn castling ep (no move clocks)",
            },
            "openings": [opening["name"] for opening in OPENINGS],
        },
    )


def write_status(
    stats: HarvestStats,
    task_queue: Queue,
    result_queue: Queue,
    inflight: set[str],
    inflight_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    with inflight_lock:
        inflight_count = len(inflight)
    payload = {
        "updated_at": utcnow_iso(),
        "stop_requested": stop_event.is_set(),
        "task_queue_size": task_queue.qsize(),
        "result_queue_size": result_queue.qsize(),
        "inflight_positions": inflight_count,
        "stats": stats.snapshot(),
    }
    atomic_write_json(STATUS_PATH, payload)


def signal_handler(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Stockfish multipv soft-target harvester")
    parser.add_argument("--scheduler-count", type=int, default=8)
    parser.add_argument("--worker-count", type=int, default=96)
    parser.add_argument("--sf-threads", type=int, default=DEFAULT_SF_THREADS)
    parser.add_argument("--sf-hash-mb", type=int, default=DEFAULT_SF_HASH_MB)
    parser.add_argument("--label-depth", type=int, default=10)
    parser.add_argument("--label-multipv", type=int, default=8)
    parser.add_argument("--label-tau", type=float, default=LABEL_TAU)
    parser.add_argument("--play-depth", type=int, default=8)
    parser.add_argument("--play-multipv", type=int, default=4)
    parser.add_argument("--play-tau", type=float, default=75.0)
    parser.add_argument("--positions-per-lineage", type=int, default=3)
    parser.add_argument("--min-target-plies", type=int, default=14)
    parser.add_argument("--max-target-plies", type=int, default=80)
    parser.add_argument("--max-records", type=int, default=0, help="0 means run indefinitely.")
    parser.add_argument("--shard-records", type=int, default=DEFAULT_SHARD_RECORDS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    global LOG_FILE
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_PATH
    write_manifest(args)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    log("=" * 72)
    log("exp085: parallel Stockfish multipv harvesting")
    log("=" * 72)
    log(
        f"scheduler_count={args.scheduler_count} worker_count={args.worker_count} "
        f"sf_threads={args.sf_threads} sf_hash_mb={args.sf_hash_mb}"
    )
    log(f"label_depth={args.label_depth} label_multipv={args.label_multipv}")
    log(f"play_depth={args.play_depth} play_multipv={args.play_multipv} positions_per_lineage={args.positions_per_lineage}")
    if args.max_records > 0:
        log(f"max_records={args.max_records}")

    store = PositionIndex(DB_PATH)
    writer = ShardedJsonlWriter(DATASET_DIR, shard_records=args.shard_records)
    stats = HarvestStats()
    stats.set_counts(
        shard_index=writer.snapshot()["shard_index"],
        current_shard_records=writer.snapshot()["current_shard_records"],
        exact_positions_seen=store.count(),
    )

    task_queue: Queue = Queue(maxsize=TASK_QUEUE_MAXSIZE)
    result_queue: Queue = Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    stop_event = threading.Event()
    inflight: set[str] = set()
    inflight_lock = threading.Lock()

    schedulers = []
    for scheduler_id in range(args.scheduler_count):
        thread = threading.Thread(
            target=scheduler_worker,
            args=(scheduler_id, task_queue, stop_event, store, inflight, inflight_lock, stats, args),
            daemon=True,
            name=f"scheduler-{scheduler_id}",
        )
        thread.start()
        schedulers.append(thread)

    workers = []
    for worker_id in range(args.worker_count):
        thread = threading.Thread(
            target=label_worker,
            args=(worker_id, task_queue, result_queue, stop_event, args),
            daemon=True,
            name=f"label-{worker_id}",
        )
        thread.start()
        workers.append(thread)

    writer_thread = threading.Thread(
        target=writer_worker,
        args=(result_queue, stop_event, store, writer, inflight, inflight_lock, stats, args.worker_count),
        daemon=True,
        name="writer",
    )
    writer_thread.start()

    last_status = 0.0
    try:
        while True:
            time.sleep(1.0)
            now = time.time()
            if STOP_REQUESTED:
                stop_event.set()

            if args.max_records > 0 and stats.snapshot()["written_positions"] >= args.max_records:
                stop_event.set()

            if now - last_status >= STATUS_INTERVAL_SEC:
                last_status = now
                write_status(stats, task_queue, result_queue, inflight, inflight_lock, stop_event)
                snap = stats.snapshot()
                log(
                    f"written={snap['written_positions']} queued={snap['queued_positions']} "
                    f"dupe_pre={snap['duplicate_skips_prequeue']} dupe_post={snap['duplicate_skips_postlabel']} "
                    f"task_q={task_queue.qsize()} result_q={result_queue.qsize()}"
                )

            if stop_event.is_set():
                break
    finally:
        stop_event.set()
        for scheduler in schedulers:
            scheduler.join(timeout=5.0)
        for _ in workers:
            while True:
                try:
                    task_queue.put(None, timeout=1.0)
                    break
                except Full:
                    continue
        for worker in workers:
            worker.join(timeout=30.0)
        writer_thread.join(timeout=30.0)
        write_status(stats, task_queue, result_queue, inflight, inflight_lock, stop_event)
        writer.close()
        store.close()
        log("done")


if __name__ == "__main__":
    main()
