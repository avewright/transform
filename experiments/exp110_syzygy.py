"""exp110_syzygy: Generate perfectly-solved endgame positions via Lichess tablebase API.

Uses the Lichess Syzygy HTTP API to get provably correct labels for
endgame positions with ≤7 pieces. Every label is 100% accurate — no
Stockfish approximation. The API returns exact WDL, DTZ, DTM and
scores for ALL legal moves.

This is the highest possible quality training data for endgames.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
import urllib.request
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess
import chess.polyglot
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs/exp110_syzygy")
DATASET_DIR = OUTPUT_DIR / "dataset"
LOG_PATH = OUTPUT_DIR / "exp110_syzygy.log"

API_URL = "http://tablebase.lichess.ovh/standard"
# Rate limit: be respectful (~50 req/s max, we'll do ~20)
API_DELAY = 0.05

STOP_REQUESTED = False
SHARD_SIZE = 5000

ENDGAME_TEMPLATES = [
    ("KQ_v_K",    [chess.QUEEN],  []),
    ("KR_v_K",    [chess.ROOK],   []),
    ("KRR_v_K",   [chess.ROOK, chess.ROOK], []),
    ("KBB_v_K",   [chess.BISHOP, chess.BISHOP], []),
    ("KBN_v_K",   [chess.BISHOP, chess.KNIGHT], []),
    ("KNN_v_K",   [chess.KNIGHT, chess.KNIGHT], []),
    ("KP_v_K",    [chess.PAWN],   []),
    ("KPP_v_K",   [chess.PAWN, chess.PAWN], []),
    ("KQ_v_KR",   [chess.QUEEN],  [chess.ROOK]),
    ("KQ_v_KB",   [chess.QUEEN],  [chess.BISHOP]),
    ("KQ_v_KN",   [chess.QUEEN],  [chess.KNIGHT]),
    ("KQ_v_KP",   [chess.QUEEN],  [chess.PAWN]),
    ("KR_v_KB",   [chess.ROOK],   [chess.BISHOP]),
    ("KR_v_KN",   [chess.ROOK],   [chess.KNIGHT]),
    ("KR_v_KP",   [chess.ROOK],   [chess.PAWN]),
    ("KBP_v_K",   [chess.BISHOP, chess.PAWN], []),
    ("KNP_v_K",   [chess.KNIGHT, chess.PAWN], []),
    ("KRP_v_KR",  [chess.ROOK, chess.PAWN], [chess.ROOK]),
    ("KBP_v_KB",  [chess.BISHOP, chess.PAWN], [chess.BISHOP]),
    ("KRP_v_KB",  [chess.ROOK, chess.PAWN], [chess.BISHOP]),
    ("KRP_v_KN",  [chess.ROOK, chess.PAWN], [chess.KNIGHT]),
    ("KQP_v_KQ",  [chess.QUEEN, chess.PAWN], [chess.QUEEN]),
    ("KBB_v_KN",  [chess.BISHOP, chess.BISHOP], [chess.KNIGHT]),
    ("KRB_v_KR",  [chess.ROOK, chess.BISHOP], [chess.ROOK]),
    ("KRN_v_KR",  [chess.ROOK, chess.KNIGHT], [chess.ROOK]),
    # 6-piece endgames
    ("KQPP_v_K",  [chess.QUEEN, chess.PAWN, chess.PAWN], []),
    ("KRPP_v_K",  [chess.ROOK, chess.PAWN, chess.PAWN], []),
    ("KBPP_v_K",  [chess.BISHOP, chess.PAWN, chess.PAWN], []),
    ("KNPP_v_K",  [chess.KNIGHT, chess.PAWN, chess.PAWN], []),
    ("KRP_v_KRP", [chess.ROOK, chess.PAWN], [chess.ROOK, chess.PAWN]),
    ("KBP_v_KBP", [chess.BISHOP, chess.PAWN], [chess.BISHOP, chess.PAWN]),
    ("KRP_v_KBP", [chess.ROOK, chess.PAWN], [chess.BISHOP, chess.PAWN]),
    ("KQR_v_KQ",  [chess.QUEEN, chess.ROOK], [chess.QUEEN]),
    # 7-piece (API supports these too)
    ("KRPP_v_KRP", [chess.ROOK, chess.PAWN, chess.PAWN], [chess.ROOK, chess.PAWN]),
    ("KBPP_v_KBP", [chess.BISHOP, chess.PAWN, chess.PAWN], [chess.BISHOP, chess.PAWN]),
]


def log(msg: str):
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(stamped + "\n")
    except Exception:
        pass


def random_square(rng, exclude):
    while True:
        sq = rng.randint(0, 63)
        if sq not in exclude:
            return sq


def generate_position(template_name, white_pieces, black_pieces, rng):
    """Generate a random legal position from a template."""
    for _ in range(50):
        board = chess.Board.empty()
        occupied = set()

        wk = random_square(rng, occupied)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        occupied.add(wk)

        for _ in range(20):
            bk = random_square(rng, occupied)
            if abs(wk // 8 - bk // 8) > 1 or abs(wk % 8 - bk % 8) > 1:
                break
        else:
            continue
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        occupied.add(bk)

        valid = True
        for pt in white_pieces:
            sq = random_square(rng, occupied)
            if pt == chess.PAWN:
                rank = sq // 8
                if rank == 0 or rank == 7:
                    sq = rng.randint(1, 6) * 8 + (sq % 8)
                    if sq in occupied:
                        valid = False
                        break
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
            occupied.add(sq)
        if not valid:
            continue

        for pt in black_pieces:
            sq = random_square(rng, occupied)
            if pt == chess.PAWN:
                rank = sq // 8
                if rank == 0 or rank == 7:
                    sq = rng.randint(1, 6) * 8 + (sq % 8)
                    if sq in occupied:
                        valid = False
                        break
            board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
            occupied.add(sq)
        if not valid:
            continue

        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        try:
            if board.is_valid() and not board.is_game_over() and board.legal_moves.count() >= 2:
                return board
        except Exception:
            continue
    return None


def query_tablebase(fen: str) -> dict | None:
    """Query Lichess tablebase API. Returns parsed JSON or None."""
    try:
        encoded = urllib.parse.quote(fen, safe='')
        url = f"{API_URL}?fen={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "chess-transformer-research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def dtz_to_cp(dtz: int | None, category: str) -> int:
    """Convert DTZ + category to a centipawn-like score for soft target compatibility."""
    if category == "win":
        if dtz is not None and dtz > 0:
            return max(500, 100000 - abs(dtz) * 100)
        return 10000
    elif category == "loss":
        if dtz is not None and dtz < 0:
            return min(-500, -(100000 - abs(dtz) * 100))
        return -10000
    elif category == "cursed-win":
        return 200
    elif category == "blessed-loss":
        return -200
    else:  # draw
        return 0


def process_position(board: chess.Board) -> dict | None:
    """Query the tablebase API and build a training record."""
    fen = board.fen()
    data = query_tablebase(fen)
    if data is None:
        return None

    if data.get("checkmate") or data.get("stalemate"):
        return None

    category = data.get("category")
    if category is None:
        return None

    moves = data.get("moves", [])
    if len(moves) < 2:
        return None

    # Build soft targets from all legal moves with exact scores
    move_scores = []
    for m in moves:
        uci = m.get("uci")
        if not uci:
            continue
        m_cat = m.get("category", "draw")
        m_dtz = m.get("dtz")
        # Opponent's result after our move — flip perspective
        # If opponent is in "loss" after our move, that's a win for us
        if m_cat == "loss":
            cp = max(500, 100000 - abs(m_dtz or 50) * 100)
        elif m_cat == "win":
            cp = min(-500, -(100000 - abs(m_dtz or 50) * 100))
        elif m_cat == "cursed-win":
            cp = -200
        elif m_cat == "blessed-loss":
            cp = 200
        elif m_cat == "checkmate":
            cp = 100000  # we just checkmated opponent
        elif m_cat == "stalemate":
            cp = 0
        else:
            cp = 0

        # Moves that lead to our win are positive (opponent in loss)
        # DTZ closer to 0 = faster mate = better
        move_scores.append({
            "uci": uci,
            "cp": cp,
            "eval_type": "tablebase",
            "dtz": m_dtz,
            "category": m_cat,
        })

    if not move_scores:
        return None

    move_scores.sort(key=lambda x: x["cp"], reverse=True)

    # Compute softmax probabilities (tau=120 for compatibility)
    cps = [m["cp"] for m in move_scores]
    probs = F.softmax(torch.tensor(cps, dtype=torch.float32) / 120.0, dim=0).tolist()

    soft_targets = []
    for m, p in zip(move_scores, probs):
        soft_targets.append({
            "uci": m["uci"],
            "prob": float(p),
            "cp": m["cp"],
            "eval_type": m["eval_type"],
            "rank": len(soft_targets) + 1,
            "pv": [m["uci"]],  # tablebase doesn't give PV, just best move
        })

    best = move_scores[0]
    cp_gap = move_scores[0]["cp"] - move_scores[1]["cp"] if len(move_scores) > 1 else 0

    # Value target
    if best["cp"] > 100:
        value_target = 2
    elif best["cp"] < -100:
        value_target = 0
    else:
        value_target = 1

    pos_key = f"{chess.polyglot.zobrist_hash(board):016x}"
    piece_count = sum(1 for _ in board.piece_map())

    # Classify: forced_mate if winning with exact DTZ
    is_forced_mate = category in ("win",) and data.get("dtm") is not None

    return {
        "source": "exp110_syzygy_api",
        "fen": fen,
        "position_key": pos_key,
        "phase": "endgame",
        "ply": 80,
        "best_move": best["uci"],
        "best_cp": best["cp"],
        "value_target": value_target,
        "label_depth": 999,  # perfect
        "label_multipv": len(soft_targets),
        "label_tau": 120.0,
        "soft_targets": soft_targets,
        "num_legal": len(moves),
        "num_labeled": len(soft_targets),
        "unlabeled_legal": 0,
        "cp_gap_top1_top2": cp_gap,
        "endgame_type": "forced_mate" if is_forced_mate else "tablebase_" + category,
        "piece_count": piece_count,
        "dtm": data.get("dtm"),
        "dtz": data.get("dtz"),
        "category": category,
    }


def main():
    global STOP_REQUESTED

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=16, help="Concurrent API requests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    def sig_handler(sig, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log("Shutdown requested...")

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    rng = random.Random(args.seed)

    log(f"exp110_syzygy: Generating solved endgame positions via Lichess API")
    log(f"  Target: {args.target}, Workers: {args.workers}")

    # Generate candidate positions
    log("Generating candidate positions...")
    candidates = []
    seen_fens = set()

    # Generate many more than target (API may reject some)
    n_generate = args.target * 3
    for i in range(n_generate):
        if len(candidates) >= args.target * 2:
            break
        tmpl = rng.choice(ENDGAME_TEMPLATES)
        board = generate_position(tmpl[0], tmpl[1], tmpl[2], rng)
        if board:
            fen = board.fen()
            board_fen = board.board_fen()
            if board_fen not in seen_fens:
                seen_fens.add(board_fen)
                candidates.append(board)

    log(f"Generated {len(candidates)} unique candidate positions")

    # Process with rate-limited thread pool (API returns 429 above ~10 req/s)
    # Use 8 workers with inter-request delay = ~8 concurrent, ~40 req/s peak
    # Back off on 429 errors
    import threading
    rate_lock = threading.Lock()
    MIN_INTERVAL = 0.15  # seconds between API calls globally (~6-7 req/s total)
    last_call_time = [time.time()]  # mutable for closure
    retry_delay = [0.0]  # increase on 429

    original_query = query_tablebase

    def rate_limited_query(fen):
        """Query with global rate limiting and 429 backoff."""
        with rate_lock:
            now = time.time()
            wait = (last_call_time[0] + MIN_INTERVAL + retry_delay[0]) - now
            if wait > 0:
                time.sleep(wait)
            last_call_time[0] = time.time()

        result = original_query(fen)
        if result is None:
            # Possible 429 — back off gently
            with rate_lock:
                retry_delay[0] = min(retry_delay[0] + 0.02, 0.5)
        else:
            with rate_lock:
                retry_delay[0] = max(retry_delay[0] - 0.005, 0.0)
        return result

    # Monkey-patch for rate limiting
    import experiments.exp110_syzygy as self_mod
    original_process = process_position

    def rate_limited_process(board):
        fen = board.fen()
        data = rate_limited_query(fen)
        if data is None:
            return None
        if data.get("checkmate") or data.get("stalemate"):
            return None
        category = data.get("category")
        if category is None:
            return None
        moves = data.get("moves", [])
        if len(moves) < 2:
            return None
        move_scores = []
        for m in moves:
            uci = m.get("uci")
            if not uci:
                continue
            m_cat = m.get("category", "draw")
            m_dtz = m.get("dtz")
            if m_cat == "loss":
                cp = max(500, 100000 - abs(m_dtz or 50) * 100)
            elif m_cat == "win":
                cp = min(-500, -(100000 - abs(m_dtz or 50) * 100))
            elif m_cat == "cursed-win":
                cp = -200
            elif m_cat == "blessed-loss":
                cp = 200
            elif m_cat == "checkmate":
                cp = 100000
            elif m_cat == "stalemate":
                cp = 0
            else:
                cp = 0
            move_scores.append({"uci": uci, "cp": cp, "eval_type": "tablebase", "dtz": m_dtz, "category": m_cat})
        if not move_scores:
            return None
        move_scores.sort(key=lambda x: x["cp"], reverse=True)
        cps = [m["cp"] for m in move_scores]
        probs = F.softmax(torch.tensor(cps, dtype=torch.float32) / 120.0, dim=0).tolist()
        soft_targets = []
        for m, p in zip(move_scores, probs):
            soft_targets.append({"uci": m["uci"], "prob": float(p), "cp": m["cp"], "eval_type": m["eval_type"], "rank": len(soft_targets) + 1, "pv": [m["uci"]]})
        best = move_scores[0]
        cp_gap = move_scores[0]["cp"] - move_scores[1]["cp"] if len(move_scores) > 1 else 0
        if best["cp"] > 100:
            value_target = 2
        elif best["cp"] < -100:
            value_target = 0
        else:
            value_target = 1
        pos_key = f"{chess.polyglot.zobrist_hash(board):016x}"
        piece_count = sum(1 for _ in board.piece_map())
        is_forced_mate = category in ("win",) and data.get("dtm") is not None
        return {
            "source": "exp110_syzygy_api", "fen": fen, "position_key": pos_key,
            "phase": "endgame", "ply": 80, "best_move": best["uci"],
            "best_cp": best["cp"], "value_target": value_target,
            "label_depth": 999, "label_multipv": len(soft_targets),
            "label_tau": 120.0, "soft_targets": soft_targets,
            "num_legal": len(moves), "num_labeled": len(soft_targets),
            "unlabeled_legal": 0, "cp_gap_top1_top2": cp_gap,
            "endgame_type": "forced_mate" if is_forced_mate else "tablebase_" + category,
            "piece_count": piece_count, "dtm": data.get("dtm"),
            "dtz": data.get("dtz"), "category": category,
        }

    t0 = time.time()
    written = 0
    type_counts = Counter()
    category_counts = Counter()
    api_fails = 0

    shard_idx = 1
    shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
    records_in_shard = 0
    last_log = time.time()

    actual_workers = min(args.workers, 8)  # cap at 8 to avoid 429s
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(rate_limited_process, b): b for b in candidates}
        for future in as_completed(futures):
            if STOP_REQUESTED or written >= args.target:
                for f in futures:
                    f.cancel()
                break
            try:
                record = future.result()
                if record:
                    shard_file.write(json.dumps(record) + "\n")
                    written += 1
                    records_in_shard += 1
                    type_counts[record.get("endgame_type", "?")] += 1
                    category_counts[record.get("category", "?")] += 1

                    if records_in_shard >= SHARD_SIZE:
                        shard_file.close()
                        shard_idx += 1
                        shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
                        records_in_shard = 0
                else:
                    api_fails += 1
            except Exception:
                api_fails += 1

            now = time.time()
            if now - last_log > 30:
                elapsed = now - t0
                rate = written / elapsed if elapsed > 0 else 0
                log(f"Written: {written}/{args.target} ({rate:.1f}/s) fails={api_fails} delay={retry_delay[0]:.2f}s categories: {dict(category_counts)}")
                last_log = now

    shard_file.close()
    elapsed = time.time() - t0
    rate = written / elapsed if elapsed > 0 else 0

    log(f"\n=== SYZYGY HARVEST COMPLETE ===")
    log(f"  Written: {written} positions in {elapsed/60:.1f} minutes ({rate:.1f}/s)")
    log(f"  Categories: {dict(category_counts)}")
    log(f"  Types: {dict(type_counts)}")
    log(f"  Output: {DATASET_DIR}")

    status = {
        "completed": True,
        "written": written,
        "categories": dict(category_counts),
        "types": dict(type_counts),
        "elapsed_sec": round(elapsed),
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
