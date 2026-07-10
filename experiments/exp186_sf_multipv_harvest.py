"""exp186: Extensive Stockfish MultiPV soft-target labeling (CPU).

Full-strength Stockfish 18 (no UCI_LimitStrength), MultiPV=8, depths in [2,8]
for throughput. Streams FENs from HF (lichess-sf + optional extras), dedupes,
writes JSONL shards + a training-ready soft_cache.pt compatible with exp185.

Designed to run on CPU while the A40 GPU trains.

Usage:
  python experiments/exp186_sf_multipv_harvest.py --go --smoke
  python experiments/exp186_sf_multipv_harvest.py --go --workers 48 --target 2_000_000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import chess
import chess.engine
import chess.polyglot
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "exp186_sf_multipv"
DATASET_DIR = OUTPUT_DIR / "dataset"
STATUS_PATH = OUTPUT_DIR / "status.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LOG_PATH = OUTPUT_DIR / "harvest.log"
DB_PATH = OUTPUT_DIR / "seen_positions.sqlite"
SOFT_CACHE_PATH = OUTPUT_DIR / "soft_cache.pt"

LABEL_TAU = 120.0
DEFAULT_MULTIPV = 8
DEFAULT_DEPTH_MIN = 2
DEFAULT_DEPTH_MAX = 8
# Prefer mid depths: more signal than d=2, much faster than always d=8
DEPTH_WEIGHTS = {2: 1, 3: 2, 4: 4, 5: 4, 6: 3, 7: 2, 8: 1}

HF_SOURCES = [
    "avewright/chess-positions-lichess-sf",
    "avewright/chess-positions-sf-labeled",
]

STOP = False
LOG_FILE = None


def resolve_stockfish() -> Path:
    configured = os.environ.get("STOCKFISH_PATH")
    cands = []
    if configured:
        cands.append(Path(configured).expanduser())
    which = shutil.which("stockfish")
    if which:
        cands.append(Path(which))
    cands.extend([
        ROOT / "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
    ])
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"Stockfish not found. Checked: {cands}")


SF_PATH = resolve_stockfish()


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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
    pieces = sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING)
    if pieces >= 20:
        return "opening"
    if pieces >= 10:
        return "middlegame"
    return "endgame"


def position_key(board: chess.Board) -> str:
    return f"{chess.polyglot.zobrist_hash(board):016x}"


def score_to_cp(score_obj: chess.engine.PovScore, pov: chess.Color) -> tuple[int, str]:
    s = score_obj.pov(pov)
    if s.is_mate():
        mate = s.mate() or 0
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate"
    cp = s.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp"


def sample_depth(rng: random.Random, dmin: int, dmax: int) -> int:
    weights = {d: DEPTH_WEIGHTS.get(d, 1) for d in range(dmin, dmax + 1)}
    ds = list(weights.keys())
    ws = [weights[d] for d in ds]
    return rng.choices(ds, weights=ws, k=1)[0]


def analyze_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    multipv: int,
    tau: float,
) -> dict | None:
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return None
    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=min(multipv, n_legal),
        )
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return None
    if not isinstance(infos, list):
        infos = [infos]

    moves, cps, eval_types, pvs = [], [], [], []
    for info in infos:
        pv = info.get("pv") or []
        score = info.get("score")
        if not pv or score is None:
            continue
        cp, et = score_to_cp(score, board.turn)
        moves.append(pv[0].uci())
        cps.append(cp)
        eval_types.append(et)
        pvs.append([m.uci() for m in pv[:8]])

    if not moves:
        return None

    # Dedup identical first moves (keep best cp)
    best: dict[str, tuple[int, str, list[str]]] = {}
    for uci, cp, et, pv in zip(moves, cps, eval_types, pvs):
        if uci not in best or cp > best[uci][0]:
            best[uci] = (cp, et, pv)
    items = sorted(best.items(), key=lambda x: -x[1][0])
    ucis = [u for u, _ in items]
    cps2 = [v[0] for _, v in items]
    ets = [v[1] for _, v in items]
    pvs2 = [v[2] for _, v in items]
    probs = F.softmax(torch.tensor(cps2, dtype=torch.float32) / tau, dim=0).tolist()

    soft = []
    for i, (uci, cp, et, pv, pr) in enumerate(zip(ucis, cps2, ets, pvs2, probs)):
        soft.append({
            "uci": uci,
            "prob": float(pr),
            "cp": int(cp),
            "eval_type": et,
            "rank": i + 1,
            "pv": pv,
        })

    return {
        "fen": board.fen(),
        "position_fen": board.board_fen() + f" {'w' if board.turn else 'b'}",
        "position_key": position_key(board),
        "best_move": soft[0]["uci"],
        "best_cp": soft[0]["cp"],
        "soft_targets": soft,
        "label_depth": depth,
        "label_multipv": len(soft),
        "label_tau": tau,
        "label_mode": "multipv_topk",
        "num_legal": n_legal,
        "phase": phase_name(board),
        "ply": board.fullmove_number * 2 - (0 if board.turn else 1),
        "source": "exp186_sf_multipv",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "value_target": 2 if soft[0]["cp"] > 100 else (0 if soft[0]["cp"] < -100 else 1),
        "cp_gap_top1_top2": int(soft[0]["cp"] - soft[1]["cp"]) if len(soft) > 1 else 0,
        "teacher_entropy": float(-(sum(p * math.log(max(p, 1e-12)) for p in probs))),
    }


class SeenDB:
    """Fast in-memory dedupe with periodic SQLite persistence."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.mem: set[str] = set()
        self._pending: list[str] = []
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, ts REAL)"
            )
        # Load existing keys into RAM for O(1) checks
        rows = self.conn.execute("SELECT key FROM seen").fetchall()
        self.mem = {r[0] for r in rows}
        log(f"SeenDB loaded {len(self.mem):,} keys from {path}")

    def try_insert(self, key: str) -> bool:
        with self.lock:
            if key in self.mem:
                return False
            self.mem.add(key)
            self._pending.append(key)
            if len(self._pending) >= 2000:
                self._flush_unlocked()
            return True

    def _flush_unlocked(self) -> None:
        if not self._pending:
            return
        ts = time.time()
        self.conn.executemany(
            "INSERT OR IGNORE INTO seen(key, ts) VALUES (?, ?)",
            [(k, ts) for k in self._pending],
        )
        self.conn.commit()
        self._pending.clear()

    def flush(self) -> None:
        with self.lock:
            self._flush_unlocked()

    def count(self) -> int:
        with self.lock:
            return len(self.mem)

    def close(self) -> None:
        self.flush()
        self.conn.close()


def fen_key(fen: str) -> str:
    """Cheap stable key: board+turn+castling+ep (drop move clocks)."""
    parts = fen.split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return fen


def fen_stream(sources: list[str], seed: int = 42):
    """Yield (fen, meta) from HF datasets forever (reshuffles).

    Prefers non-streaming materialization for smaller repos (fast random access).
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    epoch = 0
    while True:
        srcs = list(sources)
        rng.shuffle(srcs)
        for repo in srcs:
            if STOP:
                return
            log(f"Streaming FENs from {repo} (epoch={epoch})")
            # Small/medium repos: load fully for speed
            try:
                if "lichess-sf" in repo:
                    ds = load_dataset(repo, split="train", streaming=True)
                    ds = ds.shuffle(seed=seed + epoch, buffer_size=50_000)
                    for row in ds:
                        if STOP:
                            return
                        fen = row.get("fen") or row.get("position_fen")
                        if fen:
                            yield fen, {"hf_repo": repo, "phase_hint": row.get("phase")}
                else:
                    ds = load_dataset(repo, split="train")
                    n = len(ds)
                    log(f"  materialized {n:,} rows from {repo}")
                    order = list(range(n))
                    rng.shuffle(order)
                    for i in order:
                        if STOP:
                            return
                        row = ds[i]
                        fen = row.get("fen") or row.get("position_fen")
                        if fen:
                            yield fen, {"hf_repo": repo, "phase_hint": row.get("phase")}
                    del ds
            except Exception as e:
                log(f"  skip {repo}: {e}")
                continue
        epoch += 1


def worker_loop(
    wid: int,
    task_q: Queue,
    result_q: Queue,
    depth_min: int,
    depth_max: int,
    multipv: int,
    tau: float,
    hash_mb: int,
):
    rng = random.Random(10_000 + wid)
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
        # Full strength: do NOT enable UCI_LimitStrength
        engine.configure({"Threads": 1, "Hash": hash_mb})
        while not STOP:
            try:
                item = task_q.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            fen, meta = item
            try:
                board = chess.Board(fen)
            except ValueError:
                result_q.put(("bad", None))
                continue
            if board.is_game_over(claim_draw=True):
                result_q.put(("skip", None))
                continue
            depth = sample_depth(rng, depth_min, depth_max)
            rec = analyze_multipv(engine, board, depth, multipv, tau)
            if rec is None:
                result_q.put(("fail", None))
                continue
            rec["hf_repo"] = meta.get("hf_repo")
            result_q.put(("ok", rec))
    except Exception as e:
        log(f"worker {wid} crashed: {e}")
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def build_soft_cache_from_jsonl(dataset_dir: Path, out_path: Path, max_rows: int | None = None) -> int:
    """Convert harvested JSONL → soft_cache.pt for exp185 training."""
    from move_vocab import UCI_TO_IDX
    from data_loader import board_array_to_fused  # noqa: F401 — ensure path ok

    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

    shards = sorted(dataset_dir.glob("positions_*.jsonl"))
    boards, turns, castles, eps, moves, cps, mates = [], [], [], [], [], [], []
    soft_idx, soft_pr = [], []
    skipped = 0

    def fen_to_tensors(fen: str):
        board = chess.Board(fen)
        arr = [0] * 64
        for sq, piece in board.piece_map().items():
            arr[sq] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
        ba = torch.tensor([arr], dtype=torch.int8)
        turn = torch.tensor([0 if board.turn == chess.WHITE else 1], dtype=torch.int8)
        castling = torch.tensor([0], dtype=torch.int8)
        if board.has_kingside_castling_rights(chess.WHITE):
            castling[0] |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling[0] |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling[0] |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling[0] |= 8
        ep_sq = torch.tensor(
            [board.ep_square if board.ep_square is not None else 0], dtype=torch.int8
        )
        return ba, turn, castling, ep_sq, board

    for shard in shards:
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if max_rows is not None and len(boards) >= max_rows:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                fen = row.get("fen")
                best = row.get("best_move")
                soft = row.get("soft_targets") or []
                if not fen or not best or best not in UCI_TO_IDX or not soft:
                    skipped += 1
                    continue
                try:
                    ba, turn, castling, ep_sq, board = fen_to_tensors(fen)
                except Exception:
                    skipped += 1
                    continue
                move = chess.Move.from_uci(best)
                if move not in board.legal_moves:
                    skipped += 1
                    continue
                idx, pr = [], []
                for item in soft[:8]:
                    uci = item.get("uci")
                    if uci and uci in UCI_TO_IDX:
                        idx.append(UCI_TO_IDX[uci])
                        pr.append(float(item.get("prob", 0.0)))
                if not idx:
                    skipped += 1
                    continue
                s = sum(pr) or 1.0
                pr = [p / s for p in pr]
                while len(idx) < 8:
                    idx.append(-1)
                    pr.append(0.0)
                boards.append(ba)
                turns.append(turn)
                castles.append(castling)
                eps.append(ep_sq)
                moves.append(torch.tensor([UCI_TO_IDX[best]], dtype=torch.long))
                cps.append(torch.tensor([int(row.get("best_cp", 0) or 0)], dtype=torch.int32))
                mates.append(torch.tensor([0], dtype=torch.int32))
                soft_idx.append(torch.tensor(idx, dtype=torch.long))
                soft_pr.append(torch.tensor(pr, dtype=torch.float32))
        if max_rows is not None and len(boards) >= max_rows:
            break

    if not boards:
        log("soft_cache: no rows")
        return 0

    data = {
        "board_array": torch.cat(boards, dim=0),
        "turn": torch.cat(turns, dim=0),
        "castling": torch.cat(castles, dim=0),
        "ep_square": torch.cat(eps, dim=0),
        "move_idx": torch.cat(moves, dim=0),
        "cp": torch.cat(cps, dim=0),
        "mate": torch.cat(mates, dim=0),
        "soft_indices": torch.stack(soft_idx, dim=0),
        "soft_probs": torch.stack(soft_pr, dim=0),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_path)
    log(f"soft_cache saved {data['board_array'].shape[0]:,} → {out_path} (skipped {skipped})")
    return int(data["board_array"].shape[0])


def main():
    global STOP, LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--target", type=int, default=2_000_000,
                        help="Stop after this many unique labeled positions")
    parser.add_argument("--depth-min", type=int, default=DEFAULT_DEPTH_MIN)
    parser.add_argument("--depth-max", type=int, default=DEFAULT_DEPTH_MAX)
    parser.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV)
    parser.add_argument("--tau", type=float, default=LABEL_TAU)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--cache-every", type=int, default=50_000,
                        help="Rebuild soft_cache.pt every N new labels")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--build-cache-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    dataset_dir = out / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.build_cache_only:
        LOG_FILE = out / "harvest.log"
        n = build_soft_cache_from_jsonl(dataset_dir, out / "soft_cache.pt")
        print(f"built cache n={n}")
        return

    if not args.go:
        print("DRY RUN. Pass --go to harvest.")
        print("  Smoke: python experiments/exp186_sf_multipv_harvest.py --go --smoke")
        print("  Full:  python experiments/exp186_sf_multipv_harvest.py --go --workers 48 --target 2000000")
        return

    if args.smoke:
        args.target = 200
        args.workers = min(args.workers, 8)
        args.cache_every = 100
        args.shard_size = 100

    LOG_FILE = out / "harvest.log"
    log("=" * 60)
    log(f"exp186 SF MultiPV harvest | SF={SF_PATH}")
    log(f"  workers={args.workers} depth=[{args.depth_min},{args.depth_max}] "
        f"multipv={args.multipv} tau={args.tau} target={args.target:,}")
    log("  strength=FULL (no UCI_LimitStrength) Threads=1/worker")

    def _stop(signum, frame):
        global STOP
        STOP = True
        log("STOP requested")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    db = SeenDB(out / "seen_positions.sqlite")
    task_q: Queue = Queue(maxsize=args.workers * 32)
    result_q: Queue = Queue(maxsize=args.workers * 32)

    workers = []
    for wid in range(args.workers):
        t = threading.Thread(
            target=worker_loop,
            args=(wid, task_q, result_q, args.depth_min, args.depth_max,
                  args.multipv, args.tau, args.hash_mb),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Producer thread — cheap FEN-key dedupe (no Board parse on feed path)
    def producer():
        for fen, meta in fen_stream(HF_SOURCES):
            if STOP:
                break
            key = fen_key(fen)
            if not db.try_insert(key):
                continue
            while not STOP:
                try:
                    task_q.put((fen, meta), timeout=0.5)
                    break
                except Full:
                    continue
        for _ in workers:
            try:
                task_q.put(None, timeout=1.0)
            except Full:
                pass

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    written = 0
    # Resume count from existing shards
    for sp in sorted(dataset_dir.glob("positions_*.jsonl")):
        with open(sp, "r", encoding="utf-8") as f:
            for _ in f:
                written += 1
    if written:
        log(f"Resuming with {written:,} already-labeled positions on disk")

    ok = fail = skip = bad = 0
    shard_idx = 1
    existing = sorted(dataset_dir.glob("positions_*.jsonl"))
    if existing:
        # continue last shard if not full
        last = existing[-1]
        with open(last, "r", encoding="utf-8") as f:
            shard_count = sum(1 for _ in f)
        shard_idx = int(last.stem.split("_")[-1])
        if shard_count >= args.shard_size:
            shard_idx += 1
            shard_count = 0
            shard_path = dataset_dir / f"positions_{shard_idx:06d}.jsonl"
            shard_f = open(shard_path, "a", encoding="utf-8")
        else:
            shard_path = last
            shard_f = open(shard_path, "a", encoding="utf-8")
    else:
        shard_count = 0
        shard_path = dataset_dir / f"positions_{shard_idx:06d}.jsonl"
        shard_f = open(shard_path, "a", encoding="utf-8")
    t0 = time.time()
    last_status = t0
    last_cache_at = written
    depth_hist: dict[int, int] = {}
    start_written = written

    try:
        while written < args.target and not STOP:
            try:
                kind, rec = result_q.get(timeout=1.0)
            except Empty:
                if not prod.is_alive() and task_q.empty() and result_q.empty():
                    break
                continue

            if kind != "ok":
                if kind == "fail":
                    fail += 1
                elif kind == "skip":
                    skip += 1
                else:
                    bad += 1
                continue

            shard_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            written += 1
            shard_count += 1
            ok += 1
            depth_hist[rec["label_depth"]] = depth_hist.get(rec["label_depth"], 0) + 1

            if shard_count >= args.shard_size:
                shard_f.flush()
                os.fsync(shard_f.fileno())
                shard_f.close()
                shard_idx += 1
                shard_count = 0
                shard_path = dataset_dir / f"positions_{shard_idx:06d}.jsonl"
                shard_f = open(shard_path, "a", encoding="utf-8")

            now = time.time()
            if now - last_status >= 15:
                newly = written - start_written
                rate = newly / max(now - t0, 1e-6)
                eta = (args.target - written) / max(rate, 1e-6)
                log(
                    f"labeled={written:,}/{args.target:,} ({100*written/args.target:.1f}%) "
                    f"| {rate:.1f} pos/s | eta={eta/3600:.1f}h | "
                    f"ok={ok} fail={fail} skip={skip} bad={bad} | "
                    f"depths={dict(sorted(depth_hist.items()))}"
                )
                atomic_write_json(out / "status.json", {
                    "written": written,
                    "target": args.target,
                    "rate_pos_s": rate,
                    "ok": ok, "fail": fail, "skip": skip, "bad": bad,
                    "depth_hist": depth_hist,
                    "seen_db": db.count(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                last_status = now

            if written - last_cache_at >= args.cache_every:
                shard_f.flush()
                build_soft_cache_from_jsonl(dataset_dir, out / "soft_cache.pt")
                last_cache_at = written

    finally:
        STOP = True
        try:
            shard_f.flush()
            os.fsync(shard_f.fileno())
            shard_f.close()
        except Exception:
            pass
        for _ in workers:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        for t in workers:
            t.join(timeout=5)
        db.close()

    build_soft_cache_from_jsonl(dataset_dir, out / "soft_cache.pt")
    atomic_write_json(out / "manifest.json", {
        "written": written,
        "target": args.target,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "multipv": args.multipv,
        "tau": args.tau,
        "workers": args.workers,
        "stockfish": str(SF_PATH),
        "depth_hist": depth_hist,
        "elapsed_s": time.time() - t0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    log(f"Done. labeled={written:,} in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
