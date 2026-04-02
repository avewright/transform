"""exp095: Endgame-focused position harvest.

Hypothesis: The model is weakest in endgames (24-26% accuracy vs 30%+ in
middlegame). Current harvest only generates positions at ply 14-24 (middlegame).
Targeted endgame data should close this gap and improve overall ELO.

Generates endgame positions via three strategies:
  1. Synthetic construction: random K+1-4 pieces on random squares
  2. Trade-down: play aggressive games that force trades, capture positions
     once material drops below endgame threshold
  3. Tablebases-adjacent: simple endgame structures (KP vs K, KR vs K, etc.)

All positions are labeled with full legal move SF analysis at configurable depth.

Usage:
    python experiments/exp095_endgame_harvest.py \
        --output-dir outputs/exp095_endgame_harvest \
        --depth 8 --workers 4 --max-records 25000
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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ---------------------------------------------------------------------------
# Endgame position generators
# ---------------------------------------------------------------------------

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECE_NAMES = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q"}

# Common endgame templates (piece lists for white, black)
ENDGAME_TEMPLATES = [
    # KP vs K
    ([chess.PAWN], []),
    # KR vs K
    ([chess.ROOK], []),
    # KQ vs K
    ([chess.QUEEN], []),
    # KBN vs K
    ([chess.BISHOP, chess.KNIGHT], []),
    # KR vs KP
    ([chess.ROOK], [chess.PAWN]),
    # KR vs KB
    ([chess.ROOK], [chess.BISHOP]),
    # KR vs KN
    ([chess.ROOK], [chess.KNIGHT]),
    # KRR vs KR
    ([chess.ROOK, chess.ROOK], [chess.ROOK]),
    # KPP vs KP
    ([chess.PAWN, chess.PAWN], [chess.PAWN]),
    # KQ vs KR
    ([chess.QUEEN], [chess.ROOK]),
    # KBB vs KN
    ([chess.BISHOP, chess.BISHOP], [chess.KNIGHT]),
    # KRP vs KR (Lucena/Philidor type)
    ([chess.ROOK, chess.PAWN], [chess.ROOK]),
    # KP vs KP
    ([chess.PAWN], [chess.PAWN]),
    # KNN vs KP
    ([chess.KNIGHT, chess.KNIGHT], [chess.PAWN]),
    # KBP vs K
    ([chess.BISHOP, chess.PAWN], []),
    # KNP vs K
    ([chess.KNIGHT, chess.PAWN], []),
]


def random_square_excluding(rng: random.Random, exclude: set[int]) -> int:
    while True:
        sq = rng.randint(0, 63)
        if sq not in exclude:
            return sq


def generate_synthetic_endgame(rng: random.Random) -> chess.Board | None:
    """Generate a random endgame position from a template."""
    template = rng.choice(ENDGAME_TEMPLATES)
    white_pieces, black_pieces = template

    # Randomly decide who is stronger (swap colors 50% of time)
    if rng.random() < 0.5:
        white_pieces, black_pieces = black_pieces, white_pieces

    board = chess.Board(fen=None)  # empty board
    occupied: set[int] = set()

    # Place white king
    wk_sq = random_square_excluding(rng, occupied)
    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
    occupied.add(wk_sq)

    # Place black king (must not be adjacent to white king)
    for _ in range(100):
        bk_sq = random_square_excluding(rng, occupied)
        if chess.square_distance(wk_sq, bk_sq) >= 2:
            break
    else:
        return None
    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
    occupied.add(bk_sq)

    # Place white pieces
    for pt in white_pieces:
        sq = random_square_excluding(rng, occupied)
        # Pawns can't be on rank 1 or 8
        if pt == chess.PAWN:
            for _ in range(50):
                sq = random_square_excluding(rng, occupied)
                rank = chess.square_rank(sq)
                if 1 <= rank <= 6:
                    break
            else:
                return None
        board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
        occupied.add(sq)

    # Place black pieces
    for pt in black_pieces:
        sq = random_square_excluding(rng, occupied)
        if pt == chess.PAWN:
            for _ in range(50):
                sq = random_square_excluding(rng, occupied)
                rank = chess.square_rank(sq)
                if 1 <= rank <= 6:
                    break
            else:
                return None
        board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
        occupied.add(sq)

    # Random side to move
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    # No castling in endgames
    board.castling_rights = 0

    # Validate
    if not board.is_valid():
        return None
    if board.is_game_over():
        return None

    return board


def generate_tradedown_endgame(
    rng: random.Random,
    engine: chess.engine.SimpleEngine,
    depth: int = 4,
) -> chess.Board | None:
    """Play a game with aggressive capture bias until material drops below endgame threshold."""
    board = chess.Board()
    max_ply = 200

    for _ in range(max_ply):
        if board.is_game_over():
            return None

        # Count non-king pieces
        pieces = sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING)
        if pieces <= 6:  # Endgame threshold
            if pieces >= 2:  # At least something interesting
                return board.copy()
            return None

        # Prefer captures heavily
        legal = list(board.legal_moves)
        captures = [m for m in legal if board.is_capture(m)]
        if captures and rng.random() < 0.85:
            move = rng.choice(captures)
        else:
            # Use SF for non-capture moves to keep game reasonable
            try:
                result = engine.play(board, chess.engine.Limit(depth=depth))
                move = result.move
            except Exception:
                move = rng.choice(legal)

        if move not in board.legal_moves:
            move = rng.choice(legal)
        board.push(move)

    return None


def generate_random_endgame(rng: random.Random) -> chess.Board | None:
    """Generate a random endgame with 2-6 non-king pieces."""
    num_pieces = rng.randint(2, 6)
    board = chess.Board(fen=None)
    occupied: set[int] = set()

    # Kings
    wk = random_square_excluding(rng, occupied)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    occupied.add(wk)

    for _ in range(100):
        bk = random_square_excluding(rng, occupied)
        if chess.square_distance(wk, bk) >= 2:
            break
    else:
        return None
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    occupied.add(bk)

    for _ in range(num_pieces):
        pt = rng.choice(PIECE_TYPES)
        color = rng.choice([chess.WHITE, chess.BLACK])
        sq = random_square_excluding(rng, occupied)
        if pt == chess.PAWN:
            for _ in range(50):
                sq = random_square_excluding(rng, occupied)
                rank = chess.square_rank(sq)
                if 1 <= rank <= 6:
                    break
            else:
                continue
        board.set_piece_at(sq, chess.Piece(pt, color))
        occupied.add(sq)

    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = 0

    if not board.is_valid() or board.is_game_over():
        return None
    return board


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_position(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int, tau: float) -> dict:
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

    # Fill gaps
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


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def generator_worker(
    worker_id: int,
    task_queue: Queue,
    stop_event: threading.Event,
    engine: chess.engine.SimpleEngine | None,
    seed: int,
    method_weights: dict[str, float],
):
    """Generates endgame positions and puts them on the task queue."""
    rng = random.Random(seed + worker_id * 1000)
    methods = list(method_weights.keys())
    weights = list(method_weights.values())

    while not stop_event.is_set():
        method = rng.choices(methods, weights=weights, k=1)[0]

        if method == "synthetic":
            board = generate_synthetic_endgame(rng)
        elif method == "tradedown" and engine is not None:
            board = generate_tradedown_endgame(rng, engine, depth=4)
        else:
            board = generate_random_endgame(rng)

        if board is None:
            continue

        task = {
            "fen": board.fen(),
            "source": f"endgame_{method}",
            "phase": "endgame",
            "num_pieces": sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING),
        }
        try:
            task_queue.put(task, timeout=1.0)
        except Full:
            continue


def label_worker(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
    depth: int,
    tau: float,
    sf_threads: int,
    sf_hash: int,
):
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
                continue

            label = label_position(board, engine, depth, tau)
            record = {
                "created_at": utcnow_iso(),
                "source": task["source"],
                "fen": task["fen"],
                "phase": task["phase"],
                "num_pieces": task["num_pieces"],
                "label_depth": depth,
                "label_tau": tau,
                **label,
            }
            try:
                result_queue.put(record, timeout=2.0)
            except Full:
                pass
    finally:
        engine.quit()
        result_queue.put({"_done": worker_id})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def signal_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args():
    p = argparse.ArgumentParser(description="exp095: Endgame-focused position harvest")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/exp095_endgame_harvest"))
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--tau", type=float, default=LABEL_TAU)
    p.add_argument("--workers", type=int, default=4, help="Labeling workers")
    p.add_argument("--generators", type=int, default=2, help="Position generator threads")
    p.add_argument("--sf-threads", type=int, default=1)
    p.add_argument("--sf-hash-mb", type=int, default=64)
    p.add_argument("--max-records", type=int, default=25000)
    p.add_argument("--shard-records", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    # Strategy weights
    p.add_argument("--synthetic-weight", type=float, default=0.4)
    p.add_argument("--tradedown-weight", type=float, default=0.3)
    p.add_argument("--random-weight", type=float, default=0.3)
    return p.parse_args()


def main():
    global LOG_FILE, STOP_REQUESTED
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = args.output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = args.output_dir / "exp095.log"

    signal.signal(signal.SIGINT, signal_handler)

    log("=" * 72)
    log("exp095: Endgame-focused position harvest")
    log(f"depth={args.depth} workers={args.workers} generators={args.generators}")
    log(f"max_records={args.max_records}")
    log(f"strategy weights: synthetic={args.synthetic_weight} tradedown={args.tradedown_weight} random={args.random_weight}")
    log("=" * 72)

    method_weights = {
        "synthetic": args.synthetic_weight,
        "tradedown": args.tradedown_weight,
        "random": args.random_weight,
    }

    stop_event = threading.Event()
    task_queue: Queue = Queue(maxsize=256)
    result_queue: Queue = Queue(maxsize=256)

    # Generator threads (lightweight, no SF needed for synthetic/random)
    gen_threads = []
    for gid in range(args.generators):
        t = threading.Thread(
            target=generator_worker,
            args=(gid, task_queue, stop_event, None, args.seed, method_weights),
            daemon=True,
        )
        t.start()
        gen_threads.append(t)

    # Label workers
    label_threads = []
    for wid in range(args.workers):
        t = threading.Thread(
            target=label_worker,
            args=(wid, task_queue, result_queue, stop_event, args.depth, args.tau,
                  args.sf_threads, args.sf_hash_mb),
            daemon=True,
        )
        t.start()
        label_threads.append(t)

    # Writer
    shard_idx = 0
    shard_records = 0
    shard_handle = None
    written = 0
    finished_workers = 0
    seen_fens: set[str] = set()
    start_time = time.time()
    last_log_time = start_time
    source_counts: dict[str, int] = {}

    def open_shard():
        nonlocal shard_idx, shard_records, shard_handle
        if shard_handle:
            shard_handle.flush()
            shard_handle.close()
        shard_idx += 1
        shard_records = 0
        path = dataset_dir / f"positions_{shard_idx:06d}.jsonl"
        shard_handle = open(path, "a", encoding="utf-8")

    open_shard()

    try:
        while written < args.max_records and not STOP_REQUESTED:
            try:
                item = result_queue.get(timeout=1.0)
            except Empty:
                continue

            if "_done" in item:
                finished_workers += 1
                continue

            # Dedup
            fen = item["fen"]
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            assert shard_handle is not None
            shard_handle.write(json.dumps(item) + "\n")
            shard_records += 1
            written += 1
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1

            if shard_records >= args.shard_records:
                open_shard()

            now = time.time()
            if now - last_log_time >= 15.0:
                elapsed = now - start_time
                rate = written / max(elapsed, 1e-6) * 60
                log(f"written={written}/{args.max_records} ({100*written/args.max_records:.1f}%) "
                    f"rate={rate:.0f}/min sources={source_counts}")
                last_log_time = now

    finally:
        stop_event.set()
        if shard_handle:
            shard_handle.flush()
            shard_handle.close()
        for t in gen_threads:
            t.join(timeout=3)
        for _ in label_threads:
            try:
                task_queue.put(None, timeout=1)
            except Full:
                pass
        for t in label_threads:
            t.join(timeout=10)

    elapsed = time.time() - start_time
    rate = written / max(elapsed, 1e-6) * 60
    log(f"done: {written} endgame positions in {elapsed:.0f}s ({rate:.0f}/min)")
    log(f"sources: {source_counts}")

    atomic_write_json(args.output_dir / "status.json", {
        "completed_at": utcnow_iso(),
        "total_written": written,
        "source_counts": source_counts,
        "unique_fens": len(seen_fens),
        "elapsed_sec": round(elapsed, 1),
        "depth": args.depth,
    })


if __name__ == "__main__":
    main()
