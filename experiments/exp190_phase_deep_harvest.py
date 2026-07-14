#!/usr/bin/env python3
"""exp190: Phase-balanced deep MultiPV harvest (Stockfish 18 full strength).

Runs on CPU alongside GPU training. Fixes the ~56% opening skew in exp186
by enforcing phase quotas and labeling deeper where it matters.

Architecture (data → training efficiency):
  1. Phase quotas (opening/middlegame/endgame) so soft cache is saturated,
     not opening-heavy.
  2. Depth-by-phase: openings shallower, middlegame/endgame deeper.
  3. Multi-source FENs: HF stream + book playouts + endgame templates +
     random walks — round-robin so one source can't dominate.
  4. soft_cache includes phase + label_depth tensors for stratified /
     depth-weighted training (no bogging on one section).
  5. Syzygy path wired when tables exist (perfect endgame teacher).

Usage:
  STOCKFISH_PATH=stockfish/stockfish-latest \\
  MOVE_VOCAB_VERSION=compact python -u experiments/exp190_phase_deep_harvest.py --go
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
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Event, Process, Queue, cpu_count
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

LABEL_TAU = 120.0
PHASE_TARGETS = {"opening": 0.22, "middlegame": 0.48, "endgame": 0.30}
# Soft slack: accept a phase until it exceeds target + slack
PHASE_SLACK = 0.04
DEPTH_BY_PHASE = {
    "opening": (10, 14),
    "middlegame": (12, 16),
    "endgame": (14, 18),
}
HF_SOURCES = [
    "avewright/chess-positions-lichess-sf",
    "avewright/chess-positions-sf-labeled",
]
ENDGAME_TEMPLATES = [
    ([chess.QUEEN], []),
    ([chess.ROOK], []),
    ([chess.ROOK, chess.ROOK], []),
    ([chess.BISHOP, chess.BISHOP], []),
    ([chess.BISHOP, chess.KNIGHT], []),
    ([chess.PAWN], []),
    ([chess.PAWN, chess.PAWN], []),
    ([chess.QUEEN], [chess.ROOK]),
    ([chess.QUEEN], [chess.PAWN]),
    ([chess.ROOK], [chess.BISHOP]),
    ([chess.ROOK], [chess.KNIGHT]),
    ([chess.ROOK], [chess.PAWN]),
    ([chess.ROOK, chess.PAWN], [chess.ROOK]),
    ([chess.QUEEN, chess.PAWN], [chess.QUEEN]),
    ([chess.ROOK, chess.PAWN, chess.PAWN], []),
    ([chess.BISHOP, chess.PAWN], [chess.BISHOP]),
    ([chess.ROOK, chess.BISHOP], [chess.ROOK]),
    ([chess.ROOK, chess.KNIGHT], [chess.ROOK]),
    ([chess.ROOK, chess.PAWN], [chess.ROOK, chess.PAWN]),
    ([chess.QUEEN, chess.PAWN, chess.PAWN], [chess.QUEEN]),
]

# Short book seeds for opening→middlegame generation
BOOK_SEEDS = [
    [],
    ["e2e4"], ["d2d4"], ["c2c4"], ["g1f3"],
    ["e2e4", "e7e5"], ["e2e4", "c7c5"], ["e2e4", "e7e6"], ["e2e4", "c7c6"],
    ["d2d4", "d7d5"], ["d2d4", "g8f6"], ["d2d4", "g8f6", "c2c4"],
    ["e2e4", "e7e5", "g1f3", "b8c6"], ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"], ["e2e4", "e7e5", "f2f4"],
]


def resolve_sf() -> Path:
    c = os.environ.get("STOCKFISH_PATH")
    cands = [Path(c)] if c else []
    w = shutil.which("stockfish")
    if w:
        cands.append(Path(w))
    cands += [
        ROOT / "stockfish/stockfish-latest",
        ROOT / "stockfish/latest/stockfish/stockfish-ubuntu-x86-64-vnni512",
        ROOT / "stockfish/latest/stockfish/stockfish-ubuntu-x86-64-avx2",
        ROOT / "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
    ]
    for p in cands:
        if p and p.exists():
            return p.resolve()
    raise FileNotFoundError(cands)


SF = resolve_sf()
SYZYGY = ROOT / "syzygy"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fen_key(fen: str) -> str:
    parts = fen.split()
    return " ".join(parts[:4]) if len(parts) >= 4 else fen


def phase_name(board: chess.Board) -> str:
    pieces = sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING)
    if pieces >= 20:
        return "opening"
    if pieces >= 10:
        return "middlegame"
    return "endgame"


def phase_id(name: str) -> int:
    return {"opening": 0, "middlegame": 1, "endgame": 2}[name]


def score_to_cp(score_obj, pov):
    s = score_obj.pov(pov)
    if s.is_mate():
        mate = s.mate() or 0
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate", mate
    cp = s.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp", 0


def sample_depth(rng: random.Random, phase: str) -> int:
    dmin, dmax = DEPTH_BY_PHASE[phase]
    # Slight bias toward mid depths
    ds = list(range(dmin, dmax + 1))
    mid = (dmin + dmax) / 2
    ws = [1.0 / (1.0 + abs(d - mid)) for d in ds]
    return rng.choices(ds, weights=ws, k=1)[0]


def analyze(engine, board, depth, multipv, tau):
    n = board.legal_moves.count()
    if n == 0:
        return None
    try:
        infos = engine.analyse(
            board, chess.engine.Limit(depth=depth), multipv=min(multipv, n),
        )
    except Exception:
        return None
    if not isinstance(infos, list):
        infos = [infos]
    best = {}
    mate_best = 0
    for info in infos:
        pv = info.get("pv") or []
        score = info.get("score")
        if not pv or score is None:
            continue
        cp, et, mate = score_to_cp(score, board.turn)
        uci = pv[0].uci()
        if uci not in best or cp > best[uci][0]:
            best[uci] = (cp, et, [m.uci() for m in pv[:8]], mate)
            if et == "mate":
                mate_best = mate
    if not best:
        return None
    items = sorted(best.items(), key=lambda x: -x[1][0])
    cps = [v[0] for _, v in items]
    probs = F.softmax(torch.tensor(cps, dtype=torch.float32) / tau, dim=0).tolist()
    soft = []
    for i, ((uci, (cp, et, pv, _)), pr) in enumerate(zip(items, probs)):
        soft.append({
            "uci": uci, "prob": float(pr), "cp": int(cp),
            "eval_type": et, "rank": i + 1, "pv": pv,
        })
    ph = phase_name(board)
    return {
        "fen": board.fen(),
        "best_move": soft[0]["uci"],
        "best_cp": soft[0]["cp"],
        "soft_targets": soft,
        "label_depth": depth,
        "label_multipv": len(soft),
        "label_tau": tau,
        "label_mode": "multipv_topk",
        "num_legal": n,
        "phase": ph,
        "phase_id": phase_id(ph),
        "ply": board.fullmove_number * 2 - (0 if board.turn else 1),
        "source": "exp190_phase_deep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "value_target": 2 if soft[0]["cp"] > 100 else (0 if soft[0]["cp"] < -100 else 1),
        "cp_gap_top1_top2": int(soft[0]["cp"] - soft[1]["cp"]) if len(soft) > 1 else 0,
        "teacher_entropy": float(-(sum(p * math.log(max(p, 1e-12)) for p in probs))),
        "mate": int(mate_best),
    }


def worker(wid, task_q, result_q, stop_ev, multipv, tau, hash_mb):
    rng = random.Random(20000 + wid)
    # Stagger + retry: mass-parallel popen_uci often hits asyncio init timeouts.
    time.sleep(0.15 * (wid % 16))
    eng = None
    for attempt in range(8):
        try:
            eng = chess.engine.SimpleEngine.popen_uci(str(SF))
            cfg = {"Threads": 1, "Hash": hash_mb}
            if SYZYGY.exists() and any(SYZYGY.glob("*.rtbw")):
                cfg["SyzygyPath"] = str(SYZYGY)
            eng.configure(cfg)
            try:
                eng.configure({"UCI_LimitStrength": False})
            except Exception:
                pass
            break
        except Exception:
            try:
                if eng is not None:
                    eng.quit()
            except Exception:
                pass
            eng = None
            time.sleep(0.5 * (attempt + 1) + rng.random())
    if eng is None:
        return
    try:
        while not stop_ev.is_set():
            try:
                item = task_q.get(timeout=1.0)
            except Exception:
                continue
            if item is None:
                break
            fen, meta = item
            try:
                board = chess.Board(fen)
            except Exception:
                result_q.put(("bad", None))
                continue
            if board.is_game_over(claim_draw=True):
                result_q.put(("skip", None))
                continue
            ph = phase_name(board)
            depth = sample_depth(rng, ph)
            rec = analyze(eng, board, depth, multipv, tau)
            if rec is None:
                result_q.put(("fail", None))
                continue
            rec["hf_repo"] = meta.get("hf_repo")
            rec["gen_source"] = meta.get("gen_source", "hf")
            result_q.put(("ok", rec))
    finally:
        try:
            eng.quit()
        except Exception:
            pass


def _rand_sq(rng, occupied):
    for _ in range(64):
        sq = rng.randint(0, 63)
        if sq not in occupied:
            return sq
    return None


def gen_endgame_fen(rng: random.Random) -> str | None:
    white_pcs, black_pcs = rng.choice(ENDGAME_TEMPLATES)
    for _ in range(40):
        board = chess.Board.empty()
        occupied = set()
        wk = _rand_sq(rng, occupied)
        if wk is None:
            continue
        occupied.add(wk)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        bk = _rand_sq(rng, occupied)
        if bk is None:
            continue
        occupied.add(bk)
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        ok = True
        for pt in white_pcs:
            sq = _rand_sq(rng, occupied)
            if sq is None:
                ok = False
                break
            # pawns not on back ranks
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
        if not ok:
            continue
        for pt in black_pcs:
            sq = _rand_sq(rng, occupied)
            if sq is None:
                ok = False
                break
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
        if not ok:
            continue
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        if board.is_valid() and not board.is_game_over(claim_draw=True):
            return board.fen()
    return None


def gen_book_playout_fen(rng: random.Random) -> str | None:
    """Play book seed then random/legal noise into midgame."""
    board = chess.Board()
    for uci in rng.choice(BOOK_SEEDS):
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
        except Exception:
            return None
    # Continue 4–20 plies with random legal (biased to captures/checks)
    extra = rng.randint(4, 20)
    for _ in range(extra):
        legal = list(board.legal_moves)
        if not legal:
            break
        checks = [m for m in legal if board.gives_check(m)]
        caps = [m for m in legal if board.is_capture(m)]
        pool = checks or caps or legal
        if checks and caps and rng.random() < 0.5:
            pool = checks + caps
        board.push(rng.choice(pool))
        if board.is_game_over(claim_draw=True):
            board.pop()
            break
    if board.is_game_over(claim_draw=True):
        return None
    return board.fen()


def gen_random_walk_fen(rng: random.Random) -> str | None:
    board = chess.Board()
    n = rng.randint(8, 60)
    for _ in range(n):
        legal = list(board.legal_moves)
        if not legal:
            return None
        board.push(rng.choice(legal))
        if board.is_game_over(claim_draw=True):
            return None
    return board.fen()


def want_phase(phase: str, counts: Counter, total: int, hard: bool = False) -> bool:
    """Accept if this phase is at or below target+slack (deficit-aware)."""
    # Tiny free-pass only at cold start so workers aren't starved
    if total < 64:
        return True
    frac = counts[phase] / max(total, 1)
    slack = 0.01 if hard else PHASE_SLACK
    return frac <= PHASE_TARGETS[phase] + slack


def most_needed_phase(counts: Counter, total: int) -> str:
    if total < 32:
        return "endgame"  # hardest/slowest — seed early
    deficits = {
        ph: PHASE_TARGETS[ph] - counts[ph] / max(total, 1)
        for ph in PHASE_TARGETS
    }
    return max(deficits, key=deficits.get)


def build_cache(dataset_dir: Path, out_path: Path, max_rows: int | None = None) -> int:
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    from move_vocab import UCI_TO_IDX

    boards, turns, castles, eps = [], [], [], []
    moves, cps, mates = [], [], []
    soft_idx, soft_pr = [], []
    phases, depths = [], []
    skipped = 0
    for shard in sorted(Path(dataset_dir).glob("positions_*.jsonl")):
        with open(shard) as f:
            for line in f:
                if max_rows is not None and len(boards) >= max_rows:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                fen, best, soft = row.get("fen"), row.get("best_move"), row.get("soft_targets") or []
                if not fen or not best or best not in UCI_TO_IDX or not soft:
                    skipped += 1
                    continue
                try:
                    board = chess.Board(fen)
                except Exception:
                    skipped += 1
                    continue
                if chess.Move.from_uci(best) not in board.legal_moves:
                    skipped += 1
                    continue
                arr = [0] * 64
                for sq, piece in board.piece_map().items():
                    arr[sq] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
                boards.append(torch.tensor([arr], dtype=torch.int8))
                turns.append(torch.tensor([0 if board.turn else 1], dtype=torch.int8))
                castling = torch.tensor([0], dtype=torch.int8)
                if board.has_kingside_castling_rights(chess.WHITE):
                    castling[0] |= 1
                if board.has_queenside_castling_rights(chess.WHITE):
                    castling[0] |= 2
                if board.has_kingside_castling_rights(chess.BLACK):
                    castling[0] |= 4
                if board.has_queenside_castling_rights(chess.BLACK):
                    castling[0] |= 8
                castles.append(castling)
                eps.append(torch.tensor(
                    [board.ep_square if board.ep_square is not None else 0], dtype=torch.int8,
                ))
                idx, pr = [], []
                for it in soft[:8]:
                    u = it.get("uci")
                    if u and u in UCI_TO_IDX:
                        idx.append(UCI_TO_IDX[u])
                        pr.append(float(it.get("prob", 0)))
                if not idx:
                    skipped += 1
                    continue
                s = sum(pr) or 1.0
                pr = [p / s for p in pr]
                while len(idx) < 8:
                    idx.append(-1)
                    pr.append(0.0)
                moves.append(torch.tensor([UCI_TO_IDX[best]], dtype=torch.long))
                cps.append(torch.tensor([int(row.get("best_cp", 0) or 0)], dtype=torch.int32))
                mates.append(torch.tensor([int(row.get("mate", 0) or 0)], dtype=torch.int32))
                soft_idx.append(torch.tensor(idx, dtype=torch.long))
                soft_pr.append(torch.tensor(pr, dtype=torch.float32))
                ph = row.get("phase") or phase_name(board)
                phases.append(torch.tensor([phase_id(ph) if isinstance(ph, str) else int(ph)], dtype=torch.int8))
                depths.append(torch.tensor([int(row.get("label_depth", 0) or 0)], dtype=torch.int16))
        if max_rows is not None and len(boards) >= max_rows:
            break
    if not boards:
        return 0
    data = {
        "board_array": torch.cat(boards),
        "turn": torch.cat(turns),
        "castling": torch.cat(castles),
        "ep_square": torch.cat(eps),
        "move_idx": torch.cat(moves),
        "cp": torch.cat(cps),
        "mate": torch.cat(mates),
        "soft_indices": torch.stack(soft_idx),
        "soft_probs": torch.stack(soft_pr),
        "phase": torch.cat(phases),
        "label_depth": torch.cat(depths),
    }
    tmp = Path(str(out_path) + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_path)
    # Phase summary
    ph = data["phase"]
    n = ph.numel()
    summary = {
        "opening": int((ph == 0).sum()),
        "middlegame": int((ph == 1).sum()),
        "endgame": int((ph == 2).sum()),
    }
    log(f"soft_cache {n:,} → {out_path} phases={summary} skip={skipped}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--target", type=int, default=1_000_000)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--tau", type=float, default=LABEL_TAU)
    ap.add_argument("--hash-mb", type=int, default=96)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--cache-every", type=int, default=25000)
    ap.add_argument("--output-dir", type=str, default=str(ROOT / "outputs/exp190_phase_deep"))
    ap.add_argument("--build-cache-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    dsdir = out / "dataset"
    dsdir.mkdir(parents=True, exist_ok=True)

    if args.build_cache_only:
        build_cache(dsdir, out / "soft_cache.pt")
        return
    if not args.go:
        print("Pass --go")
        return
    if args.smoke:
        args.target = 200
        args.workers = min(8, args.workers)
        args.cache_every = 100
        args.shard_size = 50

    log(f"exp190 phase-deep SF={SF} workers={args.workers} target={args.target:,}")
    log(f"  phase_targets={PHASE_TARGETS} depth_by_phase={DEPTH_BY_PHASE}")
    log(f"  syzygy={'yes' if SYZYGY.exists() else 'no'} hash_mb={args.hash_mb}")
    # RAM estimate
    ram_est_gb = args.workers * args.hash_mb / 1024 + 8
    log(f"  est SF hash RAM ≈ {ram_est_gb:.1f} GB (+ python overhead)")

    stop_ev = Event()

    def _stop(*_):
        stop_ev.set()
        log("STOP")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    dbp = out / "seen_positions.sqlite"
    boot = sqlite3.connect(str(dbp))
    boot.execute("CREATE TABLE IF NOT EXISTS seen(key TEXT PRIMARY KEY, ts REAL)")
    seen = {r[0] for r in boot.execute("SELECT key FROM seen")}
    boot.close()
    log(f"seen loaded {len(seen):,}")

    written = 0
    phase_counts: Counter = Counter()
    for sp in sorted(dsdir.glob("positions_*.jsonl")):
        with open(sp) as f:
            for line in f:
                written += 1
                try:
                    phase_counts[json.loads(line).get("phase", "middlegame")] += 1
                except Exception:
                    pass
    log(f"resume written={written:,} phases={dict(phase_counts)}")

    task_q: Queue = Queue(maxsize=args.workers * 48)
    result_q: Queue = Queue(maxsize=args.workers * 48)
    procs = []
    for wid in range(args.workers):
        p = Process(
            target=worker,
            args=(wid, task_q, result_q, stop_ev, args.multipv, args.tau, args.hash_mb),
            daemon=True,
        )
        p.start()
        procs.append(p)

    def enqueue(fen: str, meta: dict) -> bool:
        while not stop_ev.is_set():
            try:
                task_q.put((fen, meta), timeout=0.5)
                return True
            except Exception:
                continue
        return False

    def producer():
        from datasets import load_dataset

        rng = random.Random(190)
        epoch = 0
        pending = []
        conn = sqlite3.connect(str(dbp), check_same_thread=False)
        local_counts = Counter(phase_counts)
        local_total = written

        def try_offer(fen: str, meta: dict) -> bool:
            nonlocal local_total
            k = fen_key(fen)
            if k in seen:
                return False
            try:
                board = chess.Board(fen)
            except Exception:
                return False
            if board.is_game_over(claim_draw=True):
                return False
            ph = phase_name(board)
            if not want_phase(ph, local_counts, local_total):
                return False
            seen.add(k)
            pending.append(k)
            local_counts[ph] += 1
            local_total += 1
            enqueue(fen, {**meta, "phase_hint": ph})
            if len(pending) >= 1500:
                conn.executemany(
                    "INSERT OR IGNORE INTO seen(key,ts) VALUES (?,?)",
                    [(k, time.time()) for k in pending],
                )
                conn.commit()
                pending.clear()
            return True

        while not stop_ev.is_set():
            need = most_needed_phase(local_counts, local_total)
            # Deficit-first: oversample the lagging phase before HF flood
            if need == "endgame":
                for _ in range(256):
                    if stop_ev.is_set():
                        break
                    fen = gen_endgame_fen(rng)
                    if fen:
                        try_offer(fen, {"gen_source": "endgame_template"})
            elif need == "middlegame":
                for _ in range(128):
                    fen = gen_book_playout_fen(rng)
                    if fen:
                        try_offer(fen, {"gen_source": "book_playout"})
                for _ in range(64):
                    fen = gen_random_walk_fen(rng)
                    if fen:
                        try_offer(fen, {"gen_source": "random_walk"})
            else:  # opening
                for _ in range(64):
                    fen = gen_book_playout_fen(rng)
                    if fen:
                        try_offer(fen, {"gen_source": "book_playout"})

            # Always sprinkle some of every phase for diversity
            for _ in range(48):
                fen = gen_endgame_fen(rng)
                if fen:
                    try_offer(fen, {"gen_source": "endgame_template"})
            for _ in range(32):
                fen = gen_book_playout_fen(rng)
                if fen:
                    try_offer(fen, {"gen_source": "book_playout"})
            for _ in range(24):
                fen = gen_random_walk_fen(rng)
                if fen:
                    try_offer(fen, {"gen_source": "random_walk"})

            # Keep workers fed: skip heavy HF pull when the queue is already healthy
            try:
                qsz = task_q.qsize()
            except Exception:
                qsz = 0
            if qsz >= max(32, args.workers):
                time.sleep(0.2)
                continue

            repo = HF_SOURCES[epoch % len(HF_SOURCES)]
            log(f"feed {repo} epoch={epoch} need={need} counts={dict(local_counts)} q={qsz}")
            try:
                if "lichess-sf" in repo:
                    ds = load_dataset(repo, split="train", streaming=True).shuffle(
                        seed=190 + epoch, buffer_size=10000,
                    )
                    it = iter(ds)
                    offered = 0
                    scanned = 0
                    # Smaller HF batches so synthetics keep interleaving
                    while not stop_ev.is_set() and offered < 1500 and scanned < 12000:
                        try:
                            row = next(it)
                        except StopIteration:
                            break
                        scanned += 1
                        fen = row.get("fen")
                        if not fen:
                            continue
                        if try_offer(fen, {"hf_repo": repo, "gen_source": "hf"}):
                            offered += 1
                        if scanned % 400 == 0:
                            # Mid-batch synthetic top-up so workers never idle
                            for _ in range(16):
                                fen_s = gen_endgame_fen(rng) if need == "endgame" else gen_book_playout_fen(rng)
                                if fen_s:
                                    try_offer(fen_s, {"gen_source": "endgame_template" if need == "endgame" else "book_playout"})
                else:
                    ds = load_dataset(repo, split="train")
                    order = list(range(len(ds)))
                    rng.shuffle(order)
                    offered = 0
                    for i in order:
                        if stop_ev.is_set() or offered >= 2000:
                            break
                        fen = ds[i].get("fen")
                        if fen and try_offer(fen, {"hf_repo": repo, "gen_source": "hf"}):
                            offered += 1
                    del ds
            except Exception as e:
                log(f"feed err {repo}: {e}")
            epoch += 1
            if local_total > 0:
                fracs = {k: local_counts[k] / local_total for k in PHASE_TARGETS}
                frac_s = ", ".join(f"{k[0]}={v:.0%}" for k, v in fracs.items())
                log(f"producer quotas [{frac_s}] total={local_total:,}")

        if pending:
            conn.executemany(
                "INSERT OR IGNORE INTO seen(key,ts) VALUES (?,?)",
                [(k, time.time()) for k in pending],
            )
            conn.commit()
        try:
            conn.close()
        except Exception:
            pass
        for _ in procs:
            try:
                task_q.put(None, timeout=1)
            except Exception:
                pass

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    # Writer
    existing = sorted(dsdir.glob("positions_*.jsonl"))
    if existing:
        last = existing[-1]
        shard_idx = int(last.stem.split("_")[-1])
        with open(last) as f:
            shard_count = sum(1 for _ in f)
        if shard_count >= args.shard_size:
            shard_idx += 1
            shard_count = 0
            shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
        else:
            shard_path = last
    else:
        shard_idx = 1
        shard_count = 0
        shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
    shard_f = open(shard_path, "a", encoding="utf-8")

    t0 = time.time()
    start_w = written
    last_st = t0
    last_cache = written
    ok = fail = skip = bad = 0
    depth_hist: Counter = Counter()
    gen_hist: Counter = Counter()

    try:
        while written < args.target and not stop_ev.is_set():
            try:
                kind, rec = result_q.get(timeout=1.0)
            except Exception:
                if not prod.is_alive() and task_q.empty():
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
            # Hard quota gate — openings label faster; don't let them flood
            ph = rec["phase"]
            if not want_phase(ph, phase_counts, written, hard=False):
                skip += 1
                continue
            shard_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            written += 1
            shard_count += 1
            ok += 1
            phase_counts[ph] += 1
            depth_hist[rec["label_depth"]] += 1
            gen_hist[rec.get("gen_source", "?")] += 1
            if shard_count >= args.shard_size:
                shard_f.flush()
                os.fsync(shard_f.fileno())
                shard_f.close()
                shard_idx += 1
                shard_count = 0
                shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
                shard_f = open(shard_path, "a", encoding="utf-8")
            now = time.time()
            if now - last_st >= 20:
                rate = (written - start_w) / max(now - t0, 1e-6)
                eta = (args.target - written) / max(rate, 1e-6)
                fracs = {k: phase_counts[k] / max(written, 1) for k in PHASE_TARGETS}
                log(
                    f"labeled={written:,}/{args.target:,} | {rate:.2f}/s | eta={eta/3600:.1f}h | "
                    f"phases={{{', '.join(f'{k[0]}={v:.0%}' for k,v in fracs.items())}}} | "
                    f"depths={dict(sorted(depth_hist.items()))}"
                )
                Path(out / "status.json").write_text(json.dumps({
                    "written": written,
                    "target": args.target,
                    "rate_pos_s": rate,
                    "phase_counts": dict(phase_counts),
                    "phase_fracs": fracs,
                    "depth_hist": dict(depth_hist),
                    "gen_hist": dict(gen_hist),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, indent=2))
                last_st = now
            if written - last_cache >= args.cache_every:
                shard_f.flush()
                build_cache(dsdir, out / "soft_cache.pt")
                last_cache = written
    finally:
        stop_ev.set()
        try:
            shard_f.flush()
            shard_f.close()
        except Exception:
            pass
        for p in procs:
            p.join(timeout=3)
    build_cache(dsdir, out / "soft_cache.pt")
    log(f"Done written={written:,} phases={dict(phase_counts)}")


if __name__ == "__main__":
    main()
