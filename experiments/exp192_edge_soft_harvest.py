#!/usr/bin/env python3
"""exp192: Edge-case / max-Elo soft MultiPV harvest (CPU, parallel to GPU train).

Complements exp190 (phase-balanced deep soft) with positions that matter most
for pure-policy Elo:

  - Black STM oversample (known side weakness)
  - In-check / check-evasion tactics
  - Forced tactics (large top1–top2 cp gap) AND sharp near-equal MultiPV
  - Promotions / underpromotions
  - EP captures, castling rights live
  - Mate-in-N / mating-net adjacent
  - High-rated Lichess puzzles (clean one-best supervision + soft SF)
  - Syzygy-aware endgames (tables already on disk)

Teacher: Stockfish full strength, MultiPV=8, deeper than shallow exp186.
Writes exp190-compatible jsonl + soft_cache.pt (phase + label_depth tags).

Usage:
  STOCKFISH_PATH=stockfish/stockfish-latest MOVE_VOCAB_VERSION=compact \\
    python -u experiments/exp192_edge_soft_harvest.py --go --workers 32
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

LABEL_TAU = 100.0  # slightly sharper than exp186/190 (120) for max-Elo teacher
MULTIPV = 8
# Prefer middlegame/endgame + tactics; openings only for book edges
PHASE_TARGETS = {"opening": 0.12, "middlegame": 0.48, "endgame": 0.40}
PHASE_SLACK = 0.05
DEPTH_BY_PHASE = {
    "opening": (12, 16),
    "middlegame": (14, 18),
    "endgame": (16, 20),
}
# Extra depth bump for checks / puzzles / mate-ish
DEPTH_BUMP = {"check": 2, "puzzle": 2, "mateish": 2, "promo": 1}

HF_SOURCES = [
    "avewright/chess-positions-lichess-sf",
    "avewright/chess-positions-sf-labeled",
]

ENDGAME_TEMPLATES = [
    ([chess.QUEEN], []),
    ([chess.ROOK], []),
    ([chess.ROOK, chess.ROOK], []),
    ([chess.BISHOP, chess.KNIGHT], []),
    ([chess.PAWN], []),
    ([chess.PAWN, chess.PAWN], []),
    ([chess.QUEEN], [chess.ROOK]),
    ([chess.ROOK], [chess.PAWN]),
    ([chess.ROOK, chess.PAWN], [chess.ROOK]),
    ([chess.QUEEN, chess.PAWN], [chess.QUEEN]),
    ([chess.ROOK, chess.PAWN, chess.PAWN], []),
    ([chess.BISHOP, chess.PAWN], [chess.KNIGHT]),
    ([chess.ROOK, chess.BISHOP], [chess.ROOK]),
    ([chess.QUEEN, chess.PAWN, chess.PAWN], [chess.ROOK, chess.PAWN]),
    ([chess.KNIGHT, chess.PAWN], [chess.BISHOP]),
    ([chess.ROOK, chess.KNIGHT], [chess.ROOK, chess.PAWN]),
]

BOOK_SEEDS = [
    [],
    ["e2e4"], ["d2d4"], ["c2c4"], ["g1f3"],
    ["e2e4", "e7e5"], ["e2e4", "c7c5"], ["e2e4", "e7e6"], ["e2e4", "c7c6"],
    ["d2d4", "d7d5"], ["d2d4", "g8f6"], ["d2d4", "g8f6", "c2c4"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],
    ["d2d4", "d7d5", "c2c4", "e7e6"], ["e2e4", "e7e5", "f2f4"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
]


def resolve_sf() -> Path:
    c = os.environ.get("STOCKFISH_PATH")
    cands = [Path(c)] if c else []
    w = shutil.which("stockfish")
    if w:
        cands.append(Path(w))
    cands += [
        ROOT / "stockfish/stockfish-latest",
        ROOT / "stockfish/stockfish/stockfish-ubuntu-x86-64-vnni512",
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


def sample_depth(rng: random.Random, phase: str, tags: list[str]) -> int:
    dmin, dmax = DEPTH_BY_PHASE[phase]
    ds = list(range(dmin, dmax + 1))
    mid = (dmin + dmax) / 2
    ws = [1.0 / (1.0 + abs(d - mid)) for d in ds]
    d = rng.choices(ds, weights=ws, k=1)[0]
    bump = 0
    for t in tags:
        bump = max(bump, DEPTH_BUMP.get(t, 0))
    return min(d + bump, 22)


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
        "source": "exp192_edge_soft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "value_target": 2 if soft[0]["cp"] > 100 else (0 if soft[0]["cp"] < -100 else 1),
        "cp_gap_top1_top2": int(soft[0]["cp"] - soft[1]["cp"]) if len(soft) > 1 else 0,
        "teacher_entropy": float(-(sum(p * math.log(max(p, 1e-12)) for p in probs))),
        "mate": int(mate_best),
        "stm_black": int(board.turn == chess.BLACK),
        "in_check": int(board.is_check()),
    }


def worker(wid, task_q, result_q, stop_ev, multipv, tau, hash_mb):
    rng = random.Random(30000 + wid)
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
            tags = list(meta.get("tags") or [])
            if board.is_check():
                tags.append("check")
            if board.turn == chess.BLACK:
                tags.append("black_stm")
            ph = phase_name(board)
            depth = sample_depth(rng, ph, tags)
            puzzle_uci = meta.get("puzzle_best")
            rec = analyze(eng, board, depth, multipv, tau)
            if rec is None:
                result_q.put(("fail", None))
                continue
            if puzzle_uci:
                rec["puzzle_best"] = puzzle_uci
                rec["puzzle_sf_agree"] = int(rec["best_move"] == puzzle_uci)
                if meta.get("force_puzzle_hard"):
                    from move_vocab import UCI_TO_IDX
                    if puzzle_uci in UCI_TO_IDX:
                        rec["best_move"] = puzzle_uci
            rec["tags"] = sorted(set(tags + list(meta.get("tags") or [])))
            rec["gen_source"] = meta.get("gen_source", "edge")
            rec["hf_repo"] = meta.get("hf_repo")
            rec["puzzle_rating"] = meta.get("puzzle_rating")
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


def gen_endgame_fen(rng: random.Random, prefer_black: bool = False) -> tuple[str, list[str]] | None:
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
            if sq is None or (pt == chess.PAWN and chess.square_rank(sq) in (0, 7)):
                ok = False
                break
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
        if not ok:
            continue
        for pt in black_pcs:
            sq = _rand_sq(rng, occupied)
            if sq is None or (pt == chess.PAWN and chess.square_rank(sq) in (0, 7)):
                ok = False
                break
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
        if not ok:
            continue
        if prefer_black or rng.random() < 0.55:
            board.turn = chess.BLACK
        else:
            board.turn = chess.WHITE
        if board.is_valid() and not board.is_game_over(claim_draw=True):
            tags = ["endgame_template"]
            if board.turn == chess.BLACK:
                tags.append("black_stm")
            return board.fen(), tags
    return None


def gen_tactical_fen(rng: random.Random) -> tuple[str, list[str]] | None:
    """Book seed + capture/check-biased playout → sharp middlegame."""
    board = chess.Board()
    for uci in rng.choice(BOOK_SEEDS):
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
        except Exception:
            return None
    extra = rng.randint(6, 28)
    for _ in range(extra):
        legal = list(board.legal_moves)
        if not legal:
            break
        checks = [m for m in legal if board.gives_check(m)]
        caps = [m for m in legal if board.is_capture(m)]
        promos = [m for m in legal if m.promotion]
        # Bias heavily toward forcing moves
        pool = promos or checks or caps or legal
        if checks and caps and rng.random() < 0.6:
            pool = checks + caps
        board.push(rng.choice(pool))
        if board.is_game_over(claim_draw=True):
            board.pop()
            break
    if board.is_game_over(claim_draw=True):
        return None
    # Prefer leave position with side-to-move that faces a check or has promo
    tags = ["tactical_playout"]
    if board.is_check():
        tags.append("check")
    if any(m.promotion for m in board.legal_moves):
        tags.append("promo")
    if board.ep_square is not None:
        tags.append("ep")
    if board.castling_rights:
        tags.append("castle_rights")
    # Black STM oversample
    if board.turn == chess.WHITE and rng.random() < 0.45:
        # One more forcing ply if possible so Black to move
        legal = list(board.legal_moves)
        if legal:
            checks = [m for m in legal if board.gives_check(m)]
            board.push(rng.choice(checks or legal))
            if board.is_game_over(claim_draw=True):
                board.pop()
            else:
                tags.append("black_stm")
    if board.turn == chess.BLACK:
        tags.append("black_stm")
    return board.fen(), tags


def gen_promo_fen(rng: random.Random) -> tuple[str, list[str]] | None:
    """Construct near-promotion positions (7th/2nd rank pawns)."""
    for _ in range(50):
        board = chess.Board.empty()
        occupied = set()
        wk = chess.square(rng.randint(0, 7), rng.randint(0, 2))
        bk = chess.square(rng.randint(0, 7), rng.randint(5, 7))
        if wk == bk:
            continue
        occupied |= {wk, bk}
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        # White pawn on 7th
        file = rng.randint(0, 7)
        wp = chess.square(file, 6)
        if wp in occupied:
            continue
        occupied.add(wp)
        board.set_piece_at(wp, chess.Piece(chess.PAWN, chess.WHITE))
        # Optional blockers / defenders
        for _k in range(rng.randint(0, 3)):
            pt = rng.choice([chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.PAWN])
            color = chess.WHITE if rng.random() < 0.4 else chess.BLACK
            sq = _rand_sq(rng, occupied)
            if sq is None:
                break
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, color))
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            continue
        if not any(m.promotion for m in board.legal_moves):
            # Still useful if pawn on 7th and White to move next-ish
            if board.turn != chess.WHITE:
                continue
        tags = ["promo"]
        if board.turn == chess.BLACK:
            tags.append("black_stm")
        return board.fen(), tags
    return None


def gen_check_net_fen(rng: random.Random) -> tuple[str, list[str]] | None:
    """Play until in check, then capture that position."""
    board = chess.Board()
    for uci in rng.choice(BOOK_SEEDS):
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
        except Exception:
            return None
    for _ in range(rng.randint(10, 40)):
        legal = list(board.legal_moves)
        if not legal:
            return None
        checks = [m for m in legal if board.gives_check(m)]
        board.push(rng.choice(checks if checks and rng.random() < 0.7 else legal))
        if board.is_game_over(claim_draw=True):
            return None
        if board.is_check():
            tags = ["check", "check_net"]
            if board.turn == chess.BLACK:
                tags.append("black_stm")
            return board.fen(), tags
    return None


def want_phase(phase: str, counts: Counter, total: int, hard: bool = False) -> bool:
    if total < 64:
        return True
    frac = counts[phase] / max(total, 1)
    slack = 0.01 if hard else PHASE_SLACK
    return frac <= PHASE_TARGETS[phase] + slack


def most_needed_phase(counts: Counter, total: int) -> str:
    if total < 32:
        return "endgame"
    deficits = {ph: PHASE_TARGETS[ph] - counts[ph] / max(total, 1) for ph in PHASE_TARGETS}
    return max(deficits, key=deficits.get)


def build_cache(dataset_dir: Path, out_path: Path, max_rows: int | None = None) -> int:
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    from move_vocab import UCI_TO_IDX

    boards, turns, castles, eps = [], [], [], []
    moves, cps, mates = [], [], []
    soft_idx, soft_pr = [], []
    phases, depths, stm_black, in_check = [], [], [], []
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
                stm_black.append(torch.tensor([int(row.get("stm_black", board.turn == chess.BLACK))], dtype=torch.int8))
                in_check.append(torch.tensor([int(row.get("in_check", board.is_check()))], dtype=torch.int8))
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
        "stm_black": torch.cat(stm_black),
        "in_check": torch.cat(in_check),
    }
    tmp = Path(str(out_path) + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_path)
    ph = data["phase"]
    summary = {
        "opening": int((ph == 0).sum()),
        "middlegame": int((ph == 1).sum()),
        "endgame": int((ph == 2).sum()),
        "black_stm": int(data["stm_black"].sum()),
        "in_check": int(data["in_check"].sum()),
    }
    log(f"soft_cache {ph.numel():,} → {out_path} meta={summary} skip={skipped}")
    return int(ph.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--target", type=int, default=500_000)
    ap.add_argument("--multipv", type=int, default=MULTIPV)
    ap.add_argument("--tau", type=float, default=LABEL_TAU)
    ap.add_argument("--hash-mb", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--cache-every", type=int, default=20000)
    ap.add_argument("--output-dir", type=str, default=str(ROOT / "outputs/exp192_edge_soft"))
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument("--min-puzzle-rating", type=int, default=1400)
    ap.add_argument("--max-puzzle-rating", type=int, default=2800)
    args = ap.parse_args()

    if not args.go and not args.build_cache_only:
        print("DRY RUN. Pass --go")
        return
    if args.smoke:
        args.target = 200
        args.workers = min(8, args.workers)

    out = Path(args.output_dir)
    dsdir = out / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    dsdir.mkdir(parents=True, exist_ok=True)

    if args.build_cache_only:
        build_cache(dsdir, out / "soft_cache.pt")
        return

    log(f"exp192 edge-soft SF={SF} workers={args.workers} target={args.target:,}")
    log(f"  phase_targets={PHASE_TARGETS} depth_by_phase={DEPTH_BY_PHASE} tau={args.tau}")
    log(f"  syzygy={'yes' if SYZYGY.exists() else 'no'} hash_mb={args.hash_mb}")
    ram_est = args.workers * args.hash_mb / 1024 + 6
    log(f"  est SF hash RAM ≈ {ram_est:.1f} GB")

    db_path = out / "seen_positions.sqlite"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY)")
    conn.commit()
    seen_n = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
    log(f"seen loaded {seen_n:,}")

    written = 0
    phase_counts: Counter = Counter()
    tag_hist: Counter = Counter()
    gen_hist: Counter = Counter()
    for shard in sorted(dsdir.glob("positions_*.jsonl")):
        with open(shard) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    written += 1
                    phase_counts[row.get("phase", "middlegame")] += 1
                    for t in row.get("tags") or []:
                        tag_hist[t] += 1
                    gen_hist[row.get("gen_source", "?")] += 1
                except Exception:
                    pass
    log(f"resume written={written:,} phases={dict(phase_counts)}")

    stop_ev = Event()

    def _sig(*_):
        stop_ev.set()
        log("shutdown requested")

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    task_q: Queue = Queue(maxsize=args.workers * 64)
    result_q: Queue = Queue(maxsize=args.workers * 64)
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
        try:
            task_q.put((fen, meta), timeout=0.5)
            return True
        except Exception:
            return False

    db_lock = threading.Lock()

    def mark_seen(k: str) -> bool:
        with db_lock:
            try:
                conn.execute("INSERT INTO seen(k) VALUES (?)", (k,))
                return True
            except sqlite3.IntegrityError:
                return False

    def producer():
        rng = random.Random(192)
        local_counts = Counter(phase_counts)
        local_total = written
        puzzle_iter = None
        puzzle_epoch = 0

        def try_offer(fen: str, meta: dict) -> bool:
            nonlocal local_total
            k = fen_key(fen)
            if not mark_seen(k):
                return False
            try:
                board = chess.Board(fen)
            except Exception:
                return False
            if board.is_game_over(claim_draw=True):
                return False
            ph = phase_name(board)
            if not want_phase(ph, local_counts, local_total):
                # Still allow high-value tags through slack
                tags = meta.get("tags") or []
                if not (set(tags) & {"check", "promo", "puzzle", "mateish"}):
                    return False
            local_counts[ph] += 1
            local_total += 1
            return enqueue(fen, {**meta, "phase_hint": ph})

        while not stop_ev.is_set() and local_total < args.target + args.workers * 100:
            need = most_needed_phase(local_counts, local_total)
            # 1) Deficit-first synthetic
            for _ in range(8):
                if need == "endgame":
                    got = gen_endgame_fen(rng, prefer_black=True)
                    if got:
                        fen, tags = got
                        try_offer(fen, {"gen_source": "endgame_template", "tags": tags})
                elif need == "opening":
                    got = gen_tactical_fen(rng)
                    if got:
                        fen, tags = got
                        try_offer(fen, {"gen_source": "book_playout", "tags": tags})
                else:
                    got = gen_tactical_fen(rng)
                    if got:
                        fen, tags = got
                        try_offer(fen, {"gen_source": "tactical_playout", "tags": tags})

            # 2) Always sprinkle edge generators
            for gen, gname in (
                (gen_check_net_fen, "check_net"),
                (gen_promo_fen, "promo_construct"),
                (lambda r: gen_endgame_fen(r, prefer_black=True), "endgame_template"),
                (gen_tactical_fen, "tactical_playout"),
            ):
                got = gen(rng)
                if got:
                    fen, tags = got
                    try_offer(fen, {"gen_source": gname, "tags": tags})

            # 3) High-rated puzzles (clean tactics)
            try:
                if puzzle_iter is None:
                    from datasets import load_dataset
                    puzzle_iter = iter(load_dataset("Lichess/chess-puzzles", split="train", streaming=True))
                    puzzle_epoch += 1
                    log(f"puzzle stream epoch={puzzle_epoch}")
                for _ in range(40):
                    try:
                        puzzle = next(puzzle_iter)
                    except StopIteration:
                        puzzle_iter = None
                        break
                    rating = int(puzzle.get("Rating") or 0)
                    if rating < args.min_puzzle_rating or rating > args.max_puzzle_rating:
                        continue
                    fen0 = puzzle.get("FEN")
                    moves_str = puzzle.get("Moves") or ""
                    moves = moves_str.split()
                    if not fen0 or len(moves) < 2:
                        continue
                    try:
                        board = chess.Board(fen0)
                        board.push(chess.Move.from_uci(moves[0]))
                    except Exception:
                        continue
                    if board.is_game_over(claim_draw=True):
                        continue
                    tags = ["puzzle"]
                    if board.is_check():
                        tags.append("check")
                    if board.turn == chess.BLACK:
                        tags.append("black_stm")
                    themes = puzzle.get("Themes") or ""
                    if isinstance(themes, str) and ("mate" in themes.lower() or "mateIn" in themes):
                        tags.append("mateish")
                    try_offer(board.fen(), {
                        "gen_source": "puzzle",
                        "tags": tags,
                        "puzzle_best": moves[1],
                        "puzzle_rating": rating,
                        "force_puzzle_hard": True,
                    })
            except Exception as e:
                log(f"puzzle stream err: {e}")
                puzzle_iter = None
                time.sleep(2)

            # 4) HF diversity ballast (filter: black / check / endgame preferred)
            try:
                from datasets import load_dataset
                repo = rng.choice(HF_SOURCES)
                ds = load_dataset(repo, split="train", streaming=True)
                n_take = 0
                for row in ds:
                    if stop_ev.is_set() or n_take >= 80:
                        break
                    fen = row.get("fen")
                    if not fen:
                        continue
                    try:
                        board = chess.Board(fen)
                    except Exception:
                        continue
                    tags = ["hf"]
                    score = 0
                    if board.turn == chess.BLACK:
                        tags.append("black_stm")
                        score += 2
                    if board.is_check():
                        tags.append("check")
                        score += 3
                    if phase_name(board) == "endgame":
                        score += 2
                    if phase_name(board) == "middlegame":
                        score += 1
                    if score < 2 and rng.random() > 0.15:
                        continue
                    if try_offer(fen, {"gen_source": "hf", "hf_repo": repo, "tags": tags}):
                        n_take += 1
            except Exception as e:
                log(f"hf stream err: {e}")
                time.sleep(2)

            time.sleep(0.05)

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    shard_idx = 1
    existing = sorted(dsdir.glob("positions_*.jsonl"))
    if existing:
        shard_idx = int(existing[-1].stem.split("_")[-1]) + 1
    shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
    shard_f = open(shard_path, "a", encoding="utf-8")
    in_shard = 0
    t0 = time.time()
    last_log = t0
    ok = fail = skip = 0

    try:
        while written < args.target and not stop_ev.is_set():
            try:
                status, rec = result_q.get(timeout=2.0)
            except Exception:
                continue
            if status != "ok" or rec is None:
                if status == "fail":
                    fail += 1
                elif status == "skip":
                    skip += 1
                continue
            ph = rec["phase"]
            if not want_phase(ph, phase_counts, written, hard=True):
                tags = set(rec.get("tags") or [])
                if not (tags & {"check", "promo", "puzzle", "mateish"}):
                    skip += 1
                    continue
            shard_f.write(json.dumps(rec) + "\n")
            in_shard += 1
            written += 1
            ok += 1
            phase_counts[ph] += 1
            gen_hist[rec.get("gen_source", "?")] += 1
            for t in rec.get("tags") or []:
                tag_hist[t] += 1
            if in_shard >= args.shard_size:
                shard_f.flush()
                shard_f.close()
                conn.commit()
                shard_idx += 1
                shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
                shard_f = open(shard_path, "a", encoding="utf-8")
                in_shard = 0
            now = time.time()
            if now - last_log >= 20:
                rate = written / max(now - t0, 1e-6)
                eta = (args.target - written) / max(rate, 1e-6)
                fracs = {k: phase_counts[k] / max(written, 1) for k in PHASE_TARGETS}
                top_tags = dict(tag_hist.most_common(6))
                log(
                    f"labeled={written:,}/{args.target:,} | {rate:.2f}/s | eta={eta/3600:.1f}h | "
                    f"phases={{o={fracs['opening']:.0%}, m={fracs['middlegame']:.0%}, e={fracs['endgame']:.0%}}} | "
                    f"tags={top_tags} | gen={dict(gen_hist)}"
                )
                with open(out / "status.json", "w") as sf:
                    json.dump({
                        "written": written,
                        "target": args.target,
                        "rate_pos_s": rate,
                        "phase_counts": dict(phase_counts),
                        "phase_fracs": fracs,
                        "tag_hist": dict(tag_hist),
                        "gen_hist": dict(gen_hist),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }, sf, indent=2)
                last_log = now
            if written > 0 and written % args.cache_every == 0:
                shard_f.flush()
                build_cache(dsdir, out / "soft_cache.pt")
    finally:
        stop_ev.set()
        for _ in procs:
            try:
                task_q.put(None, timeout=0.1)
            except Exception:
                pass
        shard_f.flush()
        shard_f.close()
        conn.commit()
        conn.close()
        for p in procs:
            p.join(timeout=5)
        build_cache(dsdir, out / "soft_cache.pt")
        log(f"Done written={written:,} phases={dict(phase_counts)} tags={dict(tag_hist.most_common(12))}")


if __name__ == "__main__":
    main()
