"""exp110_syzygy_local: Generate solved endgame positions via local Syzygy tables.

Uses python-chess Syzygy tablebase probing for instant, exact WDL+DTZ lookups.
Generates random 3-5 piece positions, probes locally, and builds soft targets
by evaluating ALL legal moves via tablebase.

This is infinitely faster than the HTTP API approach.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chess
import chess.syzygy
import chess.polyglot
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs/exp110_syzygy")
DATASET_DIR = OUTPUT_DIR / "dataset"
LOG_PATH = OUTPUT_DIR / "exp110_syzygy.log"
SYZYGY_DIR = Path("syzygy")

STOP_REQUESTED = False
SHARD_SIZE = 5000

# Templates for generating endgame positions (3-5 piece, within our tablebase range)
ENDGAME_TEMPLATES = [
    # 3-piece (KXvK)
    ("KQvK",    [chess.QUEEN],  []),
    ("KRvK",    [chess.ROOK],   []),
    ("KBvK",    [chess.BISHOP], []),
    ("KNvK",    [chess.KNIGHT], []),
    ("KPvK",    [chess.PAWN],   []),
    # 4-piece (KXvKY, KXYvK)
    ("KQvKR",   [chess.QUEEN],  [chess.ROOK]),
    ("KQvKB",   [chess.QUEEN],  [chess.BISHOP]),
    ("KQvKN",   [chess.QUEEN],  [chess.KNIGHT]),
    ("KQvKP",   [chess.QUEEN],  [chess.PAWN]),
    ("KQvKQ",   [chess.QUEEN],  [chess.QUEEN]),
    ("KRvKR",   [chess.ROOK],   [chess.ROOK]),
    ("KRvKB",   [chess.ROOK],   [chess.BISHOP]),
    ("KRvKN",   [chess.ROOK],   [chess.KNIGHT]),
    ("KRvKP",   [chess.ROOK],   [chess.PAWN]),
    ("KBvKB",   [chess.BISHOP], [chess.BISHOP]),
    ("KBvKN",   [chess.BISHOP], [chess.KNIGHT]),
    ("KBvKP",   [chess.BISHOP], [chess.PAWN]),
    ("KNvKN",   [chess.KNIGHT], [chess.KNIGHT]),
    ("KNvKP",   [chess.KNIGHT], [chess.PAWN]),
    ("KPvKP",   [chess.PAWN],   [chess.PAWN]),
    ("KQQvK",   [chess.QUEEN, chess.QUEEN], []),
    ("KQRvK",   [chess.QUEEN, chess.ROOK], []),
    ("KQBvK",   [chess.QUEEN, chess.BISHOP], []),
    ("KQNvK",   [chess.QUEEN, chess.KNIGHT], []),
    ("KQPvK",   [chess.QUEEN, chess.PAWN], []),
    ("KRRvK",   [chess.ROOK, chess.ROOK], []),
    ("KRBvK",   [chess.ROOK, chess.BISHOP], []),
    ("KRNvK",   [chess.ROOK, chess.KNIGHT], []),
    ("KRPvK",   [chess.ROOK, chess.PAWN], []),
    ("KBBvK",   [chess.BISHOP, chess.BISHOP], []),
    ("KBNvK",   [chess.BISHOP, chess.KNIGHT], []),
    ("KBPvK",   [chess.BISHOP, chess.PAWN], []),
    ("KNNvK",   [chess.KNIGHT, chess.KNIGHT], []),
    ("KNPvK",   [chess.KNIGHT, chess.PAWN], []),
    ("KPPvK",   [chess.PAWN, chess.PAWN], []),
    # 5-piece (KXYvKZ, KXvKYZ, KXYZvK)
    ("KQRvKQ",  [chess.QUEEN, chess.ROOK], [chess.QUEEN]),
    ("KQRvKR",  [chess.QUEEN, chess.ROOK], [chess.ROOK]),
    ("KQBvKQ",  [chess.QUEEN, chess.BISHOP], [chess.QUEEN]),
    ("KQNvKQ",  [chess.QUEEN, chess.KNIGHT], [chess.QUEEN]),
    ("KQPvKQ",  [chess.QUEEN, chess.PAWN], [chess.QUEEN]),
    ("KQPvKR",  [chess.QUEEN, chess.PAWN], [chess.ROOK]),
    ("KRRvKR",  [chess.ROOK, chess.ROOK], [chess.ROOK]),
    ("KRRvKQ",  [chess.ROOK, chess.ROOK], [chess.QUEEN]),
    ("KRBvKR",  [chess.ROOK, chess.BISHOP], [chess.ROOK]),
    ("KRNvKR",  [chess.ROOK, chess.KNIGHT], [chess.ROOK]),
    ("KRPvKR",  [chess.ROOK, chess.PAWN], [chess.ROOK]),
    ("KRPvKB",  [chess.ROOK, chess.PAWN], [chess.BISHOP]),
    ("KRPvKN",  [chess.ROOK, chess.KNIGHT], [chess.KNIGHT]),
    ("KRPvKP",  [chess.ROOK, chess.PAWN], [chess.PAWN]),
    ("KRPvKQ",  [chess.ROOK, chess.PAWN], [chess.QUEEN]),
    ("KBBvKN",  [chess.BISHOP, chess.BISHOP], [chess.KNIGHT]),
    ("KBBvKR",  [chess.BISHOP, chess.BISHOP], [chess.ROOK]),
    ("KBBvKP",  [chess.BISHOP, chess.BISHOP], [chess.PAWN]),
    ("KBNvKR",  [chess.BISHOP, chess.KNIGHT], [chess.ROOK]),
    ("KBNvKP",  [chess.BISHOP, chess.KNIGHT], [chess.PAWN]),
    ("KBNvKB",  [chess.BISHOP, chess.KNIGHT], [chess.BISHOP]),
    ("KBNvKN",  [chess.BISHOP, chess.KNIGHT], [chess.KNIGHT]),
    ("KBPvKB",  [chess.BISHOP, chess.PAWN], [chess.BISHOP]),
    ("KBPvKN",  [chess.BISHOP, chess.PAWN], [chess.KNIGHT]),
    ("KBPvKP",  [chess.BISHOP, chess.PAWN], [chess.PAWN]),
    ("KNPvKN",  [chess.KNIGHT, chess.PAWN], [chess.KNIGHT]),
    ("KNPvKP",  [chess.KNIGHT, chess.PAWN], [chess.PAWN]),
    ("KPPvKP",  [chess.PAWN, chess.PAWN], [chess.PAWN]),
    ("KPPvKB",  [chess.PAWN, chess.PAWN], [chess.BISHOP]),
    ("KPPvKN",  [chess.PAWN, chess.PAWN], [chess.KNIGHT]),
    ("KPPvKR",  [chess.PAWN, chess.PAWN], [chess.ROOK]),
    ("KPPvKQ",  [chess.PAWN, chess.PAWN], [chess.QUEEN]),
    # Triple pieces vs K
    ("KQQQvK",  [chess.QUEEN, chess.QUEEN, chess.QUEEN], []),
    ("KQQRvK",  [chess.QUEEN, chess.QUEEN, chess.ROOK], []),
    ("KRRRvK",  [chess.ROOK, chess.ROOK, chess.ROOK], []),
    ("KRRBvK",  [chess.ROOK, chess.ROOK, chess.BISHOP], []),
    ("KRRNvK",  [chess.ROOK, chess.ROOK, chess.KNIGHT], []),
    ("KRRPvK",  [chess.ROOK, chess.ROOK, chess.PAWN], []),
    ("KRBPvK",  [chess.ROOK, chess.BISHOP, chess.PAWN], []),
    ("KRNPvK",  [chess.ROOK, chess.KNIGHT, chess.PAWN], []),
    ("KRPPvK",  [chess.ROOK, chess.PAWN, chess.PAWN], []),
    ("KBBPvK",  [chess.BISHOP, chess.BISHOP, chess.PAWN], []),
    ("KBNPvK",  [chess.BISHOP, chess.KNIGHT, chess.PAWN], []),
    ("KBPPvK",  [chess.BISHOP, chess.PAWN, chess.PAWN], []),
    ("KNPPvK",  [chess.KNIGHT, chess.PAWN, chess.PAWN], []),
    ("KPPPvK",  [chess.PAWN, chess.PAWN, chess.PAWN], []),
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

        # Place black king non-adjacent to white king
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


def wdl_to_cp(wdl: int, dtz: int) -> int:
    """Convert WDL + DTZ to a centipawn-like score.
    
    WDL: 2 = win, 1 = cursed-win, 0 = draw, -1 = blessed-loss, -2 = loss
    """
    if wdl == 2:  # win
        return max(500, 100000 - abs(dtz) * 100)
    elif wdl == -2:  # loss
        return min(-500, -(100000 - abs(dtz) * 100))
    elif wdl == 1:  # cursed-win (50-move draw)
        return 200
    elif wdl == -1:  # blessed-loss
        return -200
    else:  # draw
        return 0


def process_position(board: chess.Board, tb: chess.syzygy.Tablebase) -> dict | None:
    """Probe tablebase for a position and build a training record."""
    try:
        wdl = tb.probe_wdl(board)
        dtz = tb.probe_dtz(board)
    except KeyError:
        return None  # position not in our tables

    if board.is_game_over():
        return None

    legal_moves = list(board.legal_moves)
    if len(legal_moves) < 2:
        return None

    # Probe WDL for every legal move (from opponent's perspective after move)
    move_scores = []
    for mv in legal_moves:
        board.push(mv)
        try:
            m_wdl = tb.probe_wdl(board)
            m_dtz = tb.probe_dtz(board)
        except (KeyError, chess.IllegalMoveError):
            board.pop()
            continue
        board.pop()

        # Flip perspective: opponent's loss = our win
        our_wdl = -m_wdl
        # DTZ after our move, from opponent perspective, flip sign
        cp = wdl_to_cp(our_wdl, m_dtz if m_dtz is not None else 50)
        
        move_scores.append({
            "uci": mv.uci(),
            "cp": cp,
            "wdl": our_wdl,
            "dtz": m_dtz,
        })

    if len(move_scores) < 2:
        return None

    move_scores.sort(key=lambda x: x["cp"], reverse=True)

    # Compute softmax probabilities
    cps = [m["cp"] for m in move_scores]
    probs = F.softmax(torch.tensor(cps, dtype=torch.float32) / 120.0, dim=0).tolist()

    soft_targets = []
    for m, p in zip(move_scores, probs):
        soft_targets.append({
            "uci": m["uci"],
            "prob": float(p),
            "cp": m["cp"],
            "eval_type": "tablebase",
            "rank": len(soft_targets) + 1,
            "pv": [m["uci"]],
        })

    best = move_scores[0]
    cp_gap = move_scores[0]["cp"] - move_scores[1]["cp"] if len(move_scores) > 1 else 0

    # Value target
    if wdl > 0:
        value_target = 2  # white winning
    elif wdl < 0:
        value_target = 0  # white losing
    else:
        value_target = 1  # draw

    # Adjust for Black's perspective
    if board.turn == chess.BLACK:
        value_target = 2 - value_target  # flip

    fen = board.fen()
    pos_key = f"{chess.polyglot.zobrist_hash(board):016x}"
    piece_count = sum(1 for _ in board.piece_map())

    # Category
    if wdl == 2:
        endgame_type = "forced_mate"
    elif wdl == -2:
        endgame_type = "losing"
    elif wdl in (1, -1):
        endgame_type = "cursed_blessed"
    else:
        endgame_type = "tablebase_draw"

    return {
        "source": "exp110_syzygy_local",
        "fen": fen,
        "position_key": pos_key,
        "phase": "endgame",
        "ply": 80,
        "best_move": best["uci"],
        "best_cp": best["cp"],
        "value_target": value_target,
        "label_depth": 999,
        "label_multipv": len(soft_targets),
        "label_tau": 120.0,
        "soft_targets": soft_targets,
        "num_legal": len(legal_moves),
        "num_labeled": len(soft_targets),
        "unlabeled_legal": len(legal_moves) - len(soft_targets),
        "cp_gap_top1_top2": cp_gap,
        "endgame_type": endgame_type,
        "piece_count": piece_count,
        "dtz": dtz,
        "wdl": wdl,
    }


def main():
    global STOP_REQUESTED

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50000)
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

    log(f"exp110_syzygy_local: Generating solved endgame positions via local Syzygy tables")
    log(f"  Target: {args.target}")
    log(f"  Syzygy dir: {SYZYGY_DIR}")

    tb = chess.syzygy.open_tablebase(str(SYZYGY_DIR))
    log(f"  Tablebase opened successfully")

    t0 = time.time()
    written = 0
    attempted = 0
    type_counts = Counter()
    wdl_counts = Counter()
    template_counts = Counter()

    shard_idx = 1
    shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
    records_in_shard = 0
    seen_fens = set()

    last_log = time.time()

    while written < args.target and not STOP_REQUESTED:
        # Pick a random template
        tmpl = rng.choice(ENDGAME_TEMPLATES)
        board = generate_position(tmpl[0], tmpl[1], tmpl[2], rng)
        if board is None:
            continue

        # Dedup
        board_fen = board.board_fen() + (" w" if board.turn else " b")
        if board_fen in seen_fens:
            continue
        seen_fens.add(board_fen)

        attempted += 1
        record = process_position(board, tb)

        if record:
            shard_file.write(json.dumps(record) + "\n")
            written += 1
            records_in_shard += 1
            type_counts[record["endgame_type"]] += 1
            wdl_counts[record["wdl"]] += 1
            template_counts[tmpl[0]] += 1

            if records_in_shard >= SHARD_SIZE:
                shard_file.close()
                shard_idx += 1
                shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
                records_in_shard = 0

        now = time.time()
        if now - last_log > 15:
            elapsed = now - t0
            rate = written / elapsed if elapsed > 0 else 0
            yield_pct = written / attempted * 100 if attempted > 0 else 0
            log(f"Written: {written}/{args.target} ({rate:.0f}/s, yield={yield_pct:.0f}%) "
                f"types: {dict(type_counts)} wdl: {dict(wdl_counts)}")
            last_log = now

    shard_file.close()
    tb.close()

    elapsed = time.time() - t0
    rate = written / elapsed if elapsed > 0 else 0

    log(f"\n=== SYZYGY LOCAL HARVEST COMPLETE ===")
    log(f"  Written: {written} positions in {elapsed:.1f}s ({rate:.0f}/s)")
    log(f"  Attempted: {attempted}, Yield: {written/attempted*100:.0f}%")
    log(f"  Types: {dict(type_counts)}")
    log(f"  WDL: {dict(wdl_counts)}")
    log(f"  Templates: {dict(template_counts)}")
    log(f"  Output: {DATASET_DIR}")

    status = {
        "completed": True,
        "written": written,
        "types": dict(type_counts),
        "wdl": dict(wdl_counts),
        "elapsed_sec": round(elapsed),
        "rate_per_sec": round(rate),
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
