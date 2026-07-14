#!/usr/bin/env python3
"""exp193: Fast Lichess puzzle → soft_cache.pt (CPU, no MultiPV required).

Uses the puzzle solution as a hard one-hot soft target. Clean tactical
supervision for mixing into exp191-style training. Optional light Stockfish
MultiPV can refine soft labels without blocking the hard puzzle answer.

Source: HuggingFace Lichess/chess-puzzles (streaming).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from move_vocab import UCI_TO_IDX

ROOT = Path(__file__).resolve().parent.parent
STOP = False
LOG_PATH: Path | None = None


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _handle(*_):
    global STOP
    STOP = True
    log("shutdown requested")


def phase_name(board: chess.Board) -> str:
    pieces = sum(
        1 for sq in chess.SQUARES
        if board.piece_at(sq) and board.piece_type_at(sq) != chess.KING
    )
    if pieces >= 20:
        return "opening"
    if pieces >= 10:
        return "middlegame"
    return "endgame"


def phase_id(name: str) -> int:
    return {"opening": 0, "middlegame": 1, "endgame": 2}.get(name, 1)


def board_to_arr(board: chess.Board) -> list[int]:
    arr = [0] * 64
    for sq, piece in board.piece_map().items():
        arr[sq] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
    return arr


def castling_byte(board: chess.Board) -> int:
    c = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        c |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        c |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        c |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        c |= 8
    return c


def puzzle_to_record(puzzle: dict, min_rating: int, max_rating: int) -> dict | None:
    rating = int(puzzle.get("Rating") or 0)
    if rating < min_rating or rating > max_rating:
        return None
    fen0 = puzzle.get("FEN")
    moves_str = puzzle.get("Moves") or ""
    moves = moves_str.split()
    if not fen0 or len(moves) < 2:
        return None
    try:
        board = chess.Board(fen0)
        opp = chess.Move.from_uci(moves[0])
        if opp not in board.legal_moves:
            return None
        board.push(opp)
    except Exception:
        return None
    if board.is_game_over(claim_draw=True):
        return None
    best = moves[1]
    if best not in UCI_TO_IDX:
        return None
    try:
        mv = chess.Move.from_uci(best)
        if mv not in board.legal_moves:
            return None
    except Exception:
        return None

    themes = puzzle.get("Themes") or ""
    if isinstance(themes, list):
        themes = " ".join(themes)
    soft = [{
        "uci": best,
        "prob": 1.0,
        "cp": 500,
        "eval_type": "puzzle",
        "rank": 1,
        "pv": [best],
    }]
    # Include remaining puzzle PV moves as zero-prob placeholders? No — keep one-hot.
    return {
        "source": "exp193_puzzle",
        "fen": board.fen(),
        "best_move": best,
        "best_cp": 500,
        "mate": 0,
        "phase": phase_name(board),
        "label_depth": 0,
        "soft_targets": soft,
        "puzzle_id": puzzle.get("PuzzleId"),
        "puzzle_rating": rating,
        "puzzle_themes": themes,
        "stm_black": int(board.turn == chess.BLACK),
        "in_check": int(board.is_check()),
        "tags": ["puzzle"]
        + (["check"] if board.is_check() else [])
        + (["black_stm"] if board.turn == chess.BLACK else [])
        + (["mateish"] if "mate" in themes.lower() else []),
    }


def build_cache(dataset_dir: Path, out_path: Path, max_rows: int | None = None) -> int:
    boards, turns, castles, eps = [], [], [], []
    moves, cps, mates = [], [], []
    soft_idx, soft_pr = [], []
    phases, depths, ratings = [], [], []
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
                boards.append(torch.tensor([board_to_arr(board)], dtype=torch.int8))
                turns.append(torch.tensor([0 if board.turn else 1], dtype=torch.int8))
                castles.append(torch.tensor([castling_byte(board)], dtype=torch.int8))
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
                cps.append(torch.tensor([int(row.get("best_cp", 500) or 500)], dtype=torch.int32))
                mates.append(torch.tensor([int(row.get("mate", 0) or 0)], dtype=torch.int32))
                soft_idx.append(torch.tensor(idx, dtype=torch.long))
                soft_pr.append(torch.tensor(pr, dtype=torch.float32))
                phases.append(torch.tensor([phase_id(row.get("phase") or phase_name(board))], dtype=torch.int8))
                depths.append(torch.tensor([int(row.get("label_depth", 0) or 0)], dtype=torch.int16))
                ratings.append(torch.tensor([int(row.get("puzzle_rating", 0) or 0)], dtype=torch.int16))
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
        "puzzle_rating": torch.cat(ratings),
    }
    tmp = Path(str(out_path) + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_path)
    ph = data["phase"]
    log(
        f"soft_cache {ph.numel():,} → {out_path} "
        f"phases={{o={int((ph==0).sum())} m={int((ph==1).sum())} e={int((ph==2).sum())}}} "
        f"skip={skipped}"
    )
    return int(ph.numel())


def main() -> None:
    global LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--target", type=int, default=200_000)
    ap.add_argument("--min-rating", type=int, default=1500)
    ap.add_argument("--max-rating", type=int, default=2800)
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--cache-every", type=int, default=25_000)
    ap.add_argument("--output-dir", default="outputs/exp193_puzzle_soft")
    ap.add_argument("--build-cache-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    dsdir = out / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    dsdir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out / "run.log"
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    if args.build_cache_only:
        build_cache(dsdir, out / "soft_cache.pt")
        return
    if not args.go:
        print("DRY RUN. Pass --go to harvest.")
        return

    log("=" * 64)
    log("exp193: Lichess puzzle soft harvest (hard one-hot, no SF)")
    log(f"  target={args.target:,} rating={args.min_rating}-{args.max_rating}")
    log(f"  out={out}")
    log("=" * 64)

    from datasets import load_dataset

    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    written = 0
    skipped = 0
    phase_counts: Counter = Counter()
    theme_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    shard_idx = 1
    # Resume: count existing shards, append a new shard
    existing = sorted(dsdir.glob("positions_*.jsonl"))
    if existing:
        for p in existing:
            with open(p) as f:
                written += sum(1 for _ in f)
            shard_idx = max(shard_idx, int(p.stem.split("_")[-1]) + 1)
        log(f"  resume written≈{written:,} next_shard={shard_idx}")
    if written >= args.target:
        log(f"  already at target ({written:,}); rebuilding cache only")
        build_cache(dsdir, out / "soft_cache.pt")
        return

    shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
    shard_f = open(shard_path, "w", encoding="utf-8")
    in_shard = 0
    t0 = time.time()
    last_cache = written
    status_path = out / "status.json"

    for puzzle in ds:
        if STOP or written >= args.target:
            break
        rec = puzzle_to_record(puzzle, args.min_rating, args.max_rating)
        if rec is None:
            skipped += 1
            continue
        shard_f.write(json.dumps(rec) + "\n")
        written += 1
        in_shard += 1
        phase_counts[rec["phase"]] += 1
        for t in rec.get("tags") or []:
            tag_counts[t] += 1
        themes = rec.get("puzzle_themes") or ""
        for th in themes.replace(",", " ").split():
            if th:
                theme_counts[th] += 1

        if in_shard >= args.shard_size:
            shard_f.close()
            shard_idx += 1
            shard_path = dsdir / f"positions_{shard_idx:06d}.jsonl"
            shard_f = open(shard_path, "w", encoding="utf-8")
            in_shard = 0

        if written % 2000 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            rate = written / elapsed
            eta_h = (args.target - written) / rate / 3600 if rate > 0 else 0
            log(
                f"labeled={written:,}/{args.target:,} | {rate:.0f}/s | eta={eta_h:.1f}h | "
                f"phases={dict(phase_counts)} | tags={dict(tag_counts)} | skipped={skipped}"
            )
            status_path.write_text(json.dumps({
                "written": written,
                "target": args.target,
                "rate_pos_s": rate,
                "phase_counts": dict(phase_counts),
                "tag_hist": dict(tag_counts),
                "top_themes": dict(theme_counts.most_common(12)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))

        if written - last_cache >= args.cache_every:
            shard_f.flush()
            build_cache(dsdir, out / "soft_cache.pt")
            last_cache = written

    shard_f.close()
    build_cache(dsdir, out / "soft_cache.pt")
    elapsed = (time.time() - t0) / 60
    log(f"DONE written={written:,} skipped={skipped:,} time={elapsed:.1f}m")
    log(f"  phases={dict(phase_counts)}")
    log(f"  tags={dict(tag_counts)}")
    log(f"  top themes={theme_counts.most_common(15)}")


if __name__ == "__main__":
    main()
