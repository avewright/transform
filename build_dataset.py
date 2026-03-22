"""build_dataset.py — Generate and upload chess training data to HuggingFace.

Creates a well-structured, streamable HF dataset of chess positions labeled
with Stockfish evaluations. Designed for incremental growth: run it repeatedly
to add more positions to the same dataset.

Dataset schema (avewright/chess-positions):
  - fen: str               — board position
  - best_move: str          — SF best move UCI
  - eval_type: str          — "cp" or "mate"
  - eval_value: int         — centipawn or mate-in-N from side to move
  - wdl_win: float          — win probability [0,1]
  - wdl_draw: float         — draw probability [0,1]
  - wdl_loss: float         — loss probability [0,1]
  - phase: str              — "opening" / "middlegame" / "endgame"
  - num_legal: int          — number of legal moves
  - source: str             — generation method
  - game_id: str            — for game-level splits (empty if N/A)
  - top_moves: str          — JSON list of top-5 moves with evals (enrichment)
  - ply: int                — half-move count in the position

Usage:
  python build_dataset.py --num 50000 --depth 8
  python build_dataset.py --num 50000 --depth 8 --seed 123   # append more
  python build_dataset.py --num 1000 --depth 6 --dry-run     # test locally
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import chess
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from huggingface_hub import HfApi

# Load HF token from .env
def _load_hf_token():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")

HF_TOKEN = _load_hf_token()
REPO_ID = "avewright/chess-positions"

# Auto-detect Stockfish
def _find_stockfish():
    candidates = [
        "stockfish/stockfish/stockfish-windows-x86-64-avx2.exe",
        "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
    ]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c))
    return "stockfish"

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or _find_stockfish()


# === Position Generation ===

# 50 common opening lines (ECO-style) for realistic opening positions
OPENING_BOOK = [
    # Sicilian variations
    ["e2e4", "c7c5"], ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["e2e4", "c7c5", "g1f3", "b8c6"], ["e2e4", "c7c5", "g1f3", "e7e6"],
    # French
    ["e2e4", "e7e6"], ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"],
    # Caro-Kann
    ["e2e4", "c7c6"], ["e2e4", "c7c6", "d2d4", "d7d5"],
    # Italian
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "f1a4"],
    # Scotch
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],
    # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6"],
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    # Queen's Gambit
    ["d2d4", "d7d5", "c2c4"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],  # QGD
    ["d2d4", "d7d5", "c2c4", "d5c4"],  # QGA
    ["d2d4", "d7d5", "c2c4", "c7c6"],  # Slav
    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    # English
    ["c2c4"], ["c2c4", "e7e5"], ["c2c4", "g8f6"],
    # Pirc
    ["e2e4", "d7d6", "d2d4", "g8f6"],
    # Scandinavian
    ["e2e4", "d7d5"],
    # Dutch
    ["d2d4", "f7f5"],
    # Grunfeld
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
    # London
    ["d2d4", "d7d5", "c1f4"],
    ["d2d4", "g8f6", "c1f4"],
    # Catalan
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],
    # Vienna
    ["e2e4", "e7e5", "b1c3"],
    # Alekhine
    ["e2e4", "g8f6"],
    # Benoni
    ["d2d4", "g8f6", "c2c4", "c7c5"],
    # Philidor
    ["e2e4", "e7e5", "g1f3", "d7d6"],
    # Two Knights
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    # Four Knights
    ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"],
    # Bird
    ["f2f4"],
    # Reti
    ["g1f3", "d7d5", "c2c4"],
    # King's Gambit
    ["e2e4", "e7e5", "f2f4"],
    # Petrov
    ["e2e4", "e7e5", "g1f3", "g8f6"],
]


def classify_phase(board: chess.Board) -> str:
    """Classify game phase based on material count — ply-independent."""
    material = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            material += vals.get(piece.piece_type, 0)
    # Full starting material = 78 per side, 156 total (but we count total non-king)
    # Here we count one-sided style: total material of all non-king pieces
    if material >= 50 and board.fullmove_number <= 12:
        return "opening"
    elif material <= 26:
        return "endgame"
    return "middlegame"


class _PosCollector:
    """Dedup-aware position collector tracking source provenance."""
    def __init__(self):
        self.positions = []   # list of (Board, source_str)
        self.seen = set()

    def add(self, board: chess.Board, source: str) -> bool:
        if board.is_game_over() or not list(board.legal_moves):
            return False
        key = board.board_fen() + (" w " if board.turn else " b ")
        if key in self.seen:
            return False
        self.seen.add(key)
        self.positions.append((board.copy(), source))
        return True

    def __len__(self):
        return len(self.positions)


def _gen_opening_book(collector: _PosCollector, target: int,
                      rng: random.Random):
    """Walk known opening lines then branch randomly for 0-8 more moves."""
    count = 0
    lines = list(OPENING_BOOK)
    max_attempts = target * 30
    attempt = 0
    while count < target and attempt < max_attempts:
        attempt += 1
        line = rng.choice(lines)
        board = chess.Board()
        # Play the book moves
        ok = True
        for uci in line:
            try:
                m = chess.Move.from_uci(uci)
                if m not in board.legal_moves:
                    ok = False
                    break
                board.push(m)
            except Exception:
                ok = False
                break
        if not ok:
            continue
        # Sample the position after book moves
        if collector.add(board, "opening_book"):
            count += 1
        # Branch randomly for 0-8 more moves
        extra = rng.randint(0, 8)
        for _ in range(extra):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            if collector.add(board, "opening_book"):
                count += 1
                if count >= target:
                    break
    return count


def _gen_weighted_play(collector: _PosCollector, target: int,
                       rng: random.Random):
    """Random play with bias toward captures and center moves.
    Avoids expensive gives_check() — uses cheap heuristics only."""
    CENTER = {chess.D4, chess.E4, chess.D5, chess.E5,
              chess.C3, chess.F3, chess.C6, chess.F6}
    DEVELOP_RANKS_W = {0, 1}  # pieces starting on ranks 1-2
    DEVELOP_RANKS_B = {6, 7}
    count = 0
    max_attempts = target * 10
    for _ in range(max_attempts):
        if count >= target:
            break
        board = chess.Board()
        ply = rng.randint(4, 80)
        for _ in range(ply):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            # Weight: captures 4x, center 2x, development 1.5x, other 1x
            # No gives_check — too expensive at scale
            weights = []
            for m in moves:
                w = 1.0
                if board.is_capture(m):
                    w = 4.0
                elif m.to_square in CENTER:
                    w = 2.0
                else:
                    # Bonus for moving pieces out of back ranks (development)
                    fr = chess.square_rank(m.from_square)
                    if board.turn == chess.WHITE and fr in DEVELOP_RANKS_W:
                        w = 1.5
                    elif board.turn == chess.BLACK and fr in DEVELOP_RANKS_B:
                        w = 1.5
                weights.append(w)
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0.0
            chosen = moves[0]
            for m, w in zip(moves, weights):
                cum += w
                if cum >= r:
                    chosen = m
                    break
            board.push(chosen)
        if collector.add(board, "weighted_play"):
            count += 1
    return count


def _gen_aggressive_play(collector: _PosCollector, target: int,
                         rng: random.Random):
    """Play that strongly prefers captures, creating positions
    with material imbalances and tactical complexity."""
    count = 0
    max_attempts = target * 10
    for _ in range(max_attempts):
        if count >= target:
            break
        board = chess.Board()
        ply = rng.randint(6, 60)
        for _ in range(ply):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            # Strongly prefer captures (cheap to check)
            captures = [m for m in moves if board.is_capture(m)]
            if captures and rng.random() < 0.75:
                board.push(rng.choice(captures))
            else:
                board.push(rng.choice(moves))
        if collector.add(board, "aggressive_play"):
            count += 1
    return count


def _gen_endgame(collector: _PosCollector, target: int,
                 rng: random.Random):
    """Generate endgame positions via two methods:
    1. Synthetic: kings + 1-4 pieces (low material, guaranteed endgame)
    2. Trade-down: play aggressive games that trade pieces until endgame
    """
    LIGHT_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK]
    count = 0

    # Method 1: Synthetic construction (~60% of target)
    synth_target = int(target * 0.6)
    attempts = 0
    synth_count = 0
    while synth_count < synth_target and attempts < synth_target * 50:
        attempts += 1
        board = chess.Board.empty()
        all_sq = list(chess.SQUARES)
        rng.shuffle(all_sq)
        wk = all_sq[0]
        bk_cands = [s for s in all_sq[1:] if chess.square_distance(s, wk) > 1]
        if not bk_cands:
            continue
        bk = rng.choice(bk_cands)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        occupied = {wk, bk}
        remaining = [s for s in all_sq if s not in occupied]
        # 1-4 pieces only — keeps material low enough for endgame classification
        n_pieces = rng.randint(1, 4)
        for _ in range(n_pieces):
            if not remaining:
                break
            sq = remaining.pop()
            color = rng.choice([chess.WHITE, chess.BLACK])
            ptype = rng.choice(LIGHT_PIECES)
            rank = chess.square_rank(sq)
            if ptype == chess.PAWN and rank in (0, 7):
                ptype = rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
            board.set_piece_at(sq, chess.Piece(ptype, color))

        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.castling_rights = 0
        board.ep_square = None
        # Set halfmove clock and fullmove number to realistic endgame values
        board.halfmove_clock = rng.randint(0, 30)
        board.fullmove_number = rng.randint(30, 60)

        if not board.is_valid():
            continue
        if collector.add(board, "endgame_synth"):
            synth_count += 1
            count += 1

    # Method 2: Trade-down — play an aggressive game, keep positions after
    # material drops below endgame threshold
    tradedown_target = target - synth_count
    attempts = 0
    td_count = 0
    while td_count < tradedown_target and attempts < tradedown_target * 40:
        attempts += 1
        board = chess.Board()
        for _ in range(rng.randint(20, 80)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            captures = [m for m in moves if board.is_capture(m)]
            if captures and rng.random() < 0.7:
                board.push(rng.choice(captures))
            else:
                board.push(rng.choice(moves))
        # Only accept if we're in endgame
        if classify_phase(board) == "endgame":
            if collector.add(board, "endgame_tradedown"):
                td_count += 1
                count += 1

    return count


def _gen_perturbed(collector: _PosCollector, target: int,
                   rng: random.Random):
    """Take existing positions and mutate them for material imbalance.
    This creates positions the model wouldn't see from normal play:
    - Remove a non-king piece (creates advantage for other side)
    - Swap a piece for a different type (creates unusual material combos)
    """
    count = 0
    # Use positions already in collector as templates
    source_positions = list(collector.positions)
    if not source_positions:
        return 0

    max_attempts = target * 20
    for _ in range(max_attempts):
        if count >= target:
            break
        template_board, _ = rng.choice(source_positions)
        board = template_board.copy()

        # Choose mutation type
        mutation = rng.choice(["remove", "remove", "swap"])  # 2:1 remove:swap

        non_king_squares = [sq for sq in chess.SQUARES
                            if board.piece_at(sq) and
                            board.piece_at(sq).piece_type != chess.KING]
        if not non_king_squares:
            continue

        if mutation == "remove":
            sq = rng.choice(non_king_squares)
            board.remove_piece_at(sq)
        else:  # swap
            sq = rng.choice(non_king_squares)
            piece = board.piece_at(sq)
            new_type = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                   chess.ROOK, chess.QUEEN])
            rank = chess.square_rank(sq)
            if new_type == chess.PAWN and rank in (0, 7):
                new_type = chess.KNIGHT
            board.set_piece_at(sq, chess.Piece(new_type, piece.color))

        # Re-validate
        if not board.is_valid():
            continue
        if collector.add(board, "perturbed"):
            count += 1

    return count


def generate_positions(n: int, seed: int = 42) -> list[tuple[chess.Board, str]]:
    """Generate n diverse chess positions from 5 sources.

    Returns list of (board, source_name) tuples.
    Allocation: 15% opening book, 30% weighted play, 15% aggressive play,
                20% endgame, 20% perturbation.
    """
    rng = random.Random(seed)
    collector = _PosCollector()
    allocations = [
        ("opening_book", 0.15, _gen_opening_book),
        ("weighted_play", 0.30, _gen_weighted_play),
        ("aggressive_play", 0.15, _gen_aggressive_play),
        ("endgame", 0.20, _gen_endgame),
    ]

    # Generate from real sources first
    for name, frac, gen_fn in allocations:
        target = int(n * frac)
        print(f"  [{name}] target={target}...", end=" ", flush=True)
        got = gen_fn(collector, target, rng)
        print(f"got {got}")

    # Perturbation uses existing positions as templates — run last
    perturb_target = n - len(collector)
    if perturb_target > 0:
        print(f"  [perturbed] target={perturb_target}...", end=" ", flush=True)
        got = _gen_perturbed(collector, perturb_target, rng)
        print(f"got {got}")

    # Shuffle and return
    result = list(collector.positions)
    rng.shuffle(result)
    print(f"  Total unique: {len(result)}")
    return result[:n]


def cp_to_wdl(cp: int, eval_type: str = "cp") -> tuple[float, float, float]:
    """Convert eval to (win, draw, loss) probabilities."""
    if eval_type == "mate":
        if cp > 0: return (1.0, 0.0, 0.0)
        elif cp < 0: return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)
    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


# === Labeling ===

def label_positions_fast(positions: list[tuple[chess.Board, str]], depth: int,
                         threads: int, top_k: int = 5) -> list[dict]:
    """Label (board, source) tuples with SF best-move + eval.

    For each position: set_fen → get_top_moves(top_k) → one SF call.
    ~15-30 pos/s at depth 8.
    """
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=depth,
        parameters={"Threads": threads, "Hash": 256},
    )

    labeled = []
    t0 = time.time()

    for i, (board, source) in enumerate(positions):
        try:
            fen = board.fen()
            sf.set_fen_position(fen)
            top_moves = sf.get_top_moves(top_k)

            if not top_moves:
                continue

            best = top_moves[0]
            best_move = best["Move"]
            eval_type = "mate" if best.get("Mate") is not None else "cp"
            if eval_type == "mate":
                eval_value = best["Mate"]
            else:
                eval_value = best.get("Centipawn", 0)

            wdl = cp_to_wdl(eval_value, eval_type)

            # Format top moves as compact JSON
            top_moves_data = []
            for m in top_moves:
                entry = {"uci": m["Move"]}
                if m.get("Mate") is not None:
                    entry["mate"] = m["Mate"]
                else:
                    entry["cp"] = m.get("Centipawn", 0)
                top_moves_data.append(entry)

            labeled.append({
                "fen": fen,
                "best_move": best_move,
                "eval_type": eval_type,
                "eval_value": eval_value,
                "wdl_win": round(wdl[0], 4),
                "wdl_draw": round(wdl[1], 4),
                "wdl_loss": round(wdl[2], 4),
                "phase": classify_phase(board),
                "num_legal": len(list(board.legal_moves)),
                "source": source,
                "game_id": "",
                "top_moves": json.dumps(top_moves_data),
                "ply": board.ply(),
            })

        except Exception as e:
            continue

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(positions) - i - 1) / rate
            print(f"  {i+1}/{len(positions)} | {rate:.1f} pos/s | ETA {eta/60:.1f}m",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Labeled {len(labeled)}/{len(positions)} in {elapsed:.0f}s "
          f"({len(labeled)/elapsed:.1f} pos/s)")
    return labeled


# === HuggingFace Upload ===

def upload_to_hf(records: list[dict], repo_id: str, token: str,
                 append: bool = True) -> Dataset:
    """Upload labeled positions to HuggingFace as a streamable dataset.

    If append=True and the dataset exists, downloads existing data and
    concatenates with new records before re-uploading.
    """
    new_ds = Dataset.from_list(records)

    if append:
        try:
            existing = load_dataset(repo_id, split="train", token=token)
            print(f"  Existing dataset: {len(existing)} rows")
            # Deduplicate by FEN
            existing_fens = set(existing["fen"])
            new_records = [r for r in records if r["fen"] not in existing_fens]
            print(f"  New unique positions: {len(new_records)} "
                  f"(filtered {len(records) - len(new_records)} duplicates)")
            if not new_records:
                print("  Nothing new to upload.")
                return existing
            new_ds = Dataset.from_list(new_records)
            combined = concatenate_datasets([existing, new_ds])
        except Exception as e:
            print(f"  No existing dataset found ({e}), creating new")
            combined = new_ds
    else:
        combined = new_ds

    # Create train/test split (95/5 by index, deterministic)
    split = combined.train_test_split(test_size=0.05, seed=42)
    ds_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    print(f"  Uploading {len(combined)} total positions "
          f"(train={len(split['train'])}, test={len(split['test'])})")

    ds_dict.push_to_hub(
        repo_id,
        token=token,
        commit_message=f"Add {len(new_ds)} positions (total: {len(combined)})",
    )
    print(f"  Uploaded to https://huggingface.co/datasets/{repo_id}")

    return combined


def compute_stats(records: list[dict]) -> dict:
    """Summary stats for a batch of labeled records."""
    phases = {}
    sources = {}
    evals = []
    mate_count = 0
    for r in records:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
        if r["eval_type"] == "cp":
            evals.append(r["eval_value"])
        else:
            mate_count += 1

    cp_mean = sum(evals) / len(evals) if evals else 0
    cp_std = (sum((e - cp_mean)**2 for e in evals) / len(evals)) ** 0.5 if evals else 0
    abs_evals = sorted(abs(e) for e in evals) if evals else []

    return {
        "count": len(records),
        "phases": phases,
        "sources": sources,
        "eval_cp_mean": round(cp_mean, 1),
        "eval_cp_std": round(cp_std, 1),
        "eval_abs_median": abs_evals[len(abs_evals)//2] if abs_evals else 0,
        "eval_abs_p95": abs_evals[int(len(abs_evals)*0.95)] if abs_evals else 0,
        "mate_positions": mate_count,
        "avg_legal_moves": round(sum(r["num_legal"] for r in records) / len(records), 1),
        "avg_ply": round(sum(r["ply"] for r in records) / len(records), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Build and upload chess dataset to HuggingFace")
    parser.add_argument("--num", type=int, default=50000, help="Positions to generate")
    parser.add_argument("--depth", type=int, default=8, help="SF search depth")
    parser.add_argument("--threads", type=int, default=4, help="SF threads")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K moves to record per position")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--repo", type=str, default=REPO_ID, help="HF dataset repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't upload")
    parser.add_argument("--no-append", action="store_true", help="Replace instead of append")
    args = parser.parse_args()

    print(f"=== Chess Dataset Builder ===")
    print(f"  Target: {args.num} positions @ SF depth {args.depth}")
    print(f"  Top-K: {args.top_k} moves per position")
    print(f"  Seed: {args.seed}")
    print(f"  Repo: {args.repo}")
    print(f"  SF: {STOCKFISH_PATH}")
    t0 = time.time()

    # Generate
    print(f"\n[1/3] Generating {args.num} positions...")
    positions = generate_positions(args.num, seed=args.seed)
    print(f"  Generated {len(positions)} unique positions")

    # Label
    print(f"\n[2/3] Labeling with SF depth {args.depth} (top-{args.top_k})...")
    records = label_positions_fast(positions, args.depth, args.threads, args.top_k)

    # Stats
    stats = compute_stats(records)
    print(f"\n  Stats: {json.dumps(stats, indent=2)}")

    # Upload
    if args.dry_run:
        print(f"\n[3/3] Dry run — skipping upload")
        # Save locally instead
        out_path = Path("data") / f"sf_labels_{args.num}_d{args.depth}.jsonl"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved to {out_path}")
    else:
        print(f"\n[3/3] Uploading to HuggingFace...")
        if not HF_TOKEN:
            print("  ERROR: No HF_TOKEN found in .env or environment")
            return
        upload_to_hf(records, args.repo, HF_TOKEN, append=not args.no_append)

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}m)")


if __name__ == "__main__":
    main()
