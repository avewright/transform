#!/usr/bin/env python3
"""Harvest Stockfish MultiPV soft labels from local JSONL FENs → soft_cache.pt.

Bypasses HuggingFace (private/offline). Feeds autoresearch_8gb.

Usage:
  python scripts/harvest_local_multipv.py --go --target 40000 --workers 12
  python scripts/harvest_local_multipv.py --go --smoke
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
import time
from multiprocessing import Process, Queue, Event
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABEL_TAU = 120.0
STOP = None


def resolve_sf() -> Path:
    env = os.environ.get("STOCKFISH_PATH")
    cands = []
    if env:
        cands.append(Path(env))
    which = shutil.which("stockfish")
    if which:
        cands.append(Path(which))
    cands.extend([
        ROOT / "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
        Path("/opt/homebrew/bin/stockfish"),
        Path("/usr/local/bin/stockfish"),
    ])
    for p in cands:
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"Stockfish not found: {cands}")


def score_to_cp(score_obj: chess.engine.PovScore, pov: chess.Color) -> tuple[int, int | None]:
    s = score_obj.pov(pov)
    if s.is_mate():
        mate = s.mate()
        return (10000 if mate and mate > 0 else -10000), mate
    return int(s.score(mate_score=10000) or 0), None


def analyze(engine, board, depth, multipv, tau):
    n = board.legal_moves.count()
    if n == 0:
        return None
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=min(multipv, n))
    if not isinstance(infos, list):
        infos = [infos]
    cps, ucis, mate0 = [], [], None
    for info in infos:
        mv = info.get("pv", [None])[0]
        if mv is None:
            continue
        cp, mate = score_to_cp(info["score"], board.turn)
        cps.append(cp)
        ucis.append(mv.uci())
        if mate0 is None and mate is not None:
            mate0 = mate
    if not ucis:
        return None
    t = torch.tensor(cps, dtype=torch.float32)
    probs = F.softmax(t / tau, dim=0).tolist()
    soft = [{"uci": u, "cp": int(c), "prob": float(p)} for u, c, p in zip(ucis, cps, probs)]
    return {
        "fen": board.fen(),
        "best_move": ucis[0],
        "best_cp": int(cps[0]),
        "mate": int(mate0) if mate0 is not None else 0,
        "soft_targets": soft,
        "label_depth": depth,
        "label_multipv": len(soft),
    }


def worker(wid, task_q, result_q, stop_ev, dmin, dmax, multipv, tau, hash_mb, sf_path):
    eng = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    eng.configure({"Threads": 1, "Hash": hash_mb})
    rng = random.Random(1000 + wid)
    try:
        while not stop_ev.is_set():
            try:
                fen = task_q.get(timeout=0.5)
            except Exception:
                continue
            if fen is None:
                break
            try:
                board = chess.Board(fen)
                if board.is_game_over():
                    continue
                depth = rng.randint(dmin, dmax)
                rec = analyze(eng, board, depth, multipv, tau)
                if rec is not None:
                    result_q.put(rec)
            except Exception:
                continue
    finally:
        eng.quit()


def build_cache(shard_path: Path, out_path: Path, max_rows: int | None = None) -> int:
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    from move_vocab import UCI_TO_IDX

    boards, turns, castles, eps, moves, cps, mates = [], [], [], [], [], [], []
    soft_idx, soft_pr, depths = [], [], []
    skipped = 0
    with open(shard_path, encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and len(boards) >= max_rows:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
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
            ep = torch.tensor([board.ep_square if board.ep_square is not None else 0], dtype=torch.int8)
            move = chess.Move.from_uci(best)
            if move not in board.legal_moves:
                skipped += 1
                continue
            idx, pr = [], []
            for item in soft[:8]:
                u = item.get("uci")
                if u and u in UCI_TO_IDX:
                    idx.append(UCI_TO_IDX[u])
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
            eps.append(ep)
            moves.append(torch.tensor([UCI_TO_IDX[best]], dtype=torch.long))
            cps.append(torch.tensor([int(row.get("best_cp", 0) or 0)], dtype=torch.int32))
            mates.append(torch.tensor([int(row.get("mate", 0) or 0)], dtype=torch.int32))
            soft_idx.append(torch.tensor(idx, dtype=torch.long))
            soft_pr.append(torch.tensor(pr, dtype=torch.float32))
            depths.append(torch.tensor([int(row.get("label_depth", 0) or 0)], dtype=torch.int16))

    if not boards:
        print("no rows")
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
        "label_depth": torch.cat(depths),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_path)
    print(f"soft_cache n={data['board_array'].shape[0]:,} → {out_path} (skipped {skipped})")
    return int(data["board_array"].shape[0])


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def load_fens(paths: list[Path], max_fens: int) -> list[str]:
    seen, out = set(), []
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    fen = json.loads(line).get("fen")
                except json.JSONDecodeError:
                    continue
                if not fen:
                    continue
                key = fen_key(fen)
                if key in seen:
                    continue
                seen.add(key)
                out.append(fen)
                if len(out) >= max_fens:
                    return out
    return out


def expand_fens_by_walks(
    seeds: list[str],
    target: int,
    *,
    walks_per_seed: int = 4,
    walk_plies: int = 12,
    seed: int = 42,
) -> list[str]:
    """Grow unique FENs via random legal walks from seed positions."""
    rng = random.Random(seed)
    seen = {fen_key(f) for f in seeds}
    out = list(seeds)
    i = 0
    while len(out) < target and i < len(seeds) * max(1, walks_per_seed) * 4:
        base = seeds[i % len(seeds)]
        i += 1
        try:
            board = chess.Board(base)
        except Exception:
            continue
        for _ in range(walks_per_seed):
            b = board.copy(stack=False)
            for _ply in range(rng.randint(1, walk_plies)):
                moves = list(b.legal_moves)
                if not moves or b.is_game_over():
                    break
                b.push(rng.choice(moves))
                fen = b.fen()
                key = fen_key(fen)
                if key in seen:
                    continue
                seen.add(key)
                out.append(fen)
                if len(out) >= target:
                    return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--target", type=int, default=40000)
    ap.add_argument("--depth-min", type=int, default=3)
    ap.add_argument("--depth-max", type=int, default=6)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--tau", type=float, default=LABEL_TAU)
    ap.add_argument("--hash-mb", type=int, default=32)
    ap.add_argument("--out", type=str, default="outputs/autoresearch_8gb/soft_cache_200k.pt")
    ap.add_argument(
        "--jsonl",
        nargs="+",
        default=[
            "data/lichess_sf_cached_200k.jsonl",
            "data/sf_labels_10k_d8.jsonl",
            "data/sf_labels_2000_d6.jsonl",
        ],
    )
    ap.add_argument("--expand-walks", action="store_true",
                    help="Grow FENs via random legal walks from seeds")
    ap.add_argument("--walk-plies", type=int, default=12)
    ap.add_argument("--shard-name", type=str, default="local_multipv_positions.jsonl")
    args = ap.parse_args()
    if not args.go:
        print("Pass --go")
        return
    if args.smoke:
        args.target = 200
        args.workers = min(4, args.workers)

    sf = resolve_sf()
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / args.shard_name

    fens = load_fens([Path(p) for p in args.jsonl], max_fens=max(args.target * 2, 200_000))
    if args.expand_walks and len(fens) < args.target:
        before = len(fens)
        fens = expand_fens_by_walks(
            fens, args.target * 2, walk_plies=args.walk_plies, seed=42,
        )
        print(f"expanded FENs {before:,} → {len(fens):,} via walks")
    random.Random(42).shuffle(fens)
    print(f"SF={sf} workers={args.workers} fens={len(fens):,} target={args.target:,}")

    stop_ev = Event()

    def _stop(*_):
        stop_ev.set()
        print("STOP")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    task_q: Queue = Queue(maxsize=args.workers * 64)
    result_q: Queue = Queue(maxsize=args.workers * 64)
    procs = []
    for wid in range(args.workers):
        p = Process(
            target=worker,
            args=(
                wid, task_q, result_q, stop_ev,
                args.depth_min, args.depth_max, args.multipv, args.tau, args.hash_mb, sf,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    written = 0
    if shard.exists():
        with open(shard, encoding="utf-8") as f:
            written = sum(1 for _ in f)
        print(f"resume written={written:,}")

    t0 = time.time()
    fi = 0
    with open(shard, "a", encoding="utf-8") as out_f:
        while written < args.target and not stop_ev.is_set():
            # fill tasks
            while fi < len(fens) and not task_q.full() and not stop_ev.is_set():
                task_q.put(fens[fi])
                fi += 1
            try:
                rec = result_q.get(timeout=0.5)
            except Exception:
                if fi >= len(fens) and task_q.empty():
                    # drain remaining
                    try:
                        rec = result_q.get(timeout=2.0)
                    except Exception:
                        break
                else:
                    continue
            out_f.write(json.dumps(rec) + "\n")
            written += 1
            if written % 200 == 0:
                out_f.flush()
                rate = written / max(time.time() - t0, 1e-6)
                print(f"labeled {written:,}/{args.target:,} ({rate:.1f}/s)", flush=True)

    stop_ev.set()
    for _ in procs:
        try:
            task_q.put_nowait(None)
        except Exception:
            pass
    for p in procs:
        p.join(timeout=5)

    n = build_cache(shard, out_path, max_rows=args.target)
    print(f"done n={n} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
