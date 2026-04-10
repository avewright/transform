"""Generate SF-vs-SF games in PGN format for move-history transformer training.

Usage:
  python generate_sf_games.py --num-games 10000 --output outputs/sf_games_10k.pgn
  python generate_sf_games.py --num-games 50000 --output outputs/sf_games_50k.pgn --depth 8 --workers 4
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import chess.pgn

OPENING_BOOK = [
    "e2e4", "d2d4", "c2c4", "g1f3", "g2g3",
    "e2e4 e7e5", "e2e4 c7c5", "e2e4 e7e6", "e2e4 c7c6", "e2e4 d7d5",
    "d2d4 d7d5", "d2d4 g8f6", "d2d4 e7e6", "d2d4 f7f5",
    "c2c4 e7e5", "c2c4 g8f6", "c2c4 c7c5",
    "g1f3 d7d5", "g1f3 g8f6", "g1f3 c7c5",
    "e2e4 e7e5 g1f3 b8c6", "e2e4 e7e5 g1f3 g8f6",
    "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 g1f3 b8c6", "e2e4 c7c5 g1f3 e7e6",
    "d2d4 d7d5 c2c4", "d2d4 d7d5 g1f3", "d2d4 g8f6 c2c4",
    "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6",
    "e2e4 e7e5 g1f3 b8c6 f1b5",
    "e2e4 e7e5 g1f3 b8c6 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1c4",
    "e2e4 c7c5 g1f3 d7d6 d2d4",
    "d2d4 d7d5 c2c4 e7e6",
    "d2d4 d7d5 c2c4 c7c6",
    "d2d4 g8f6 c2c4 g7g6 b8c6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "e2e4 d7d5 e4d5 d8d5",
    "d2d4 f7f5",
    "e2e4 g7g6",
    "e2e4 d7d6",
    "g1f3 d7d5 g2g3",
    "c2c4 e7e5 b8c3",
    "e2e4 e7e5 f2f4",
    "d2d4 d7d5 c2c4 d5c4",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 f8b4",
]

SF_PATH = str(Path(__file__).parent / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe")


def find_stockfish() -> str:
    if os.path.isfile(SF_PATH):
        return SF_PATH
    env_path = os.environ.get("STOCKFISH_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    return "stockfish"


def play_one_game(seed: int, depth: int, max_moves: int, resign_cp: int) -> chess.pgn.Game | None:
    """Play a single SF-vs-SF game with random opening from book."""
    rng = random.Random(seed)
    sf_path = find_stockfish()

    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"Threads": 1, "Hash": 16})
    except Exception as e:
        print(f"[seed={seed}] Failed to start Stockfish: {e}", file=sys.stderr)
        return None

    try:
        board = chess.Board()

        # Play random opening
        opening = rng.choice(OPENING_BOOK)
        for uci_move in opening.split():
            try:
                board.push_uci(uci_move)
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                break

        move_list: list[chess.Move] = list(board.move_stack)
        draw_counter = 0

        for ply in range(max_moves):
            if board.is_game_over():
                break

            # Vary depth slightly for diversity
            d = max(1, depth + rng.randint(-1, 1))
            result = engine.play(board, chess.engine.Limit(depth=d))
            if result.move is None:
                break

            board.push(result.move)
            move_list.append(result.move)

            # Adjudication: resign if score is extreme
            info = engine.analyse(board, chess.engine.Limit(depth=max(2, depth - 2)))
            score = info.get("score")
            if score:
                cp = score.relative.score(mate_score=10000)
                if cp is not None:
                    if abs(cp) > resign_cp:
                        break
                    # Draw adjudication: 0cp for 10+ consecutive moves
                    if abs(cp) < 10:
                        draw_counter += 1
                    else:
                        draw_counter = 0
                    if draw_counter >= 20:
                        break

        # Build PGN game
        game = chess.pgn.Game()
        game.headers["White"] = f"SF_d{depth}"
        game.headers["Black"] = f"SF_d{depth}"

        outcome = board.outcome()
        if outcome:
            game.headers["Result"] = outcome.result()
        elif draw_counter >= 20:
            game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"

        node = game
        temp_board = chess.Board()
        for move in move_list:
            node = node.add_variation(move)
            temp_board.push(move)

        return game

    finally:
        engine.quit()


def play_batch(seeds: list[int], depth: int, max_moves: int, resign_cp: int) -> list[chess.pgn.Game]:
    """Play a batch of games in one engine process for efficiency."""
    sf_path = find_stockfish()
    games: list[chess.pgn.Game] = []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({"Threads": 1, "Hash": 16})
    except Exception as e:
        print(f"Failed to start Stockfish: {e}", file=sys.stderr)
        return games

    try:
        for seed in seeds:
            rng = random.Random(seed)
            board = chess.Board()

            opening = rng.choice(OPENING_BOOK)
            for uci_move in opening.split():
                try:
                    board.push_uci(uci_move)
                except (chess.InvalidMoveError, chess.IllegalMoveError):
                    break

            move_list: list[chess.Move] = list(board.move_stack)
            draw_counter = 0

            for ply in range(max_moves):
                if board.is_game_over():
                    break

                d = max(1, depth + rng.randint(-1, 1))
                result = engine.play(board, chess.engine.Limit(depth=d))
                if result.move is None:
                    break

                board.push(result.move)
                move_list.append(result.move)

                info = engine.analyse(board, chess.engine.Limit(depth=max(2, depth - 2)))
                score = info.get("score")
                if score:
                    cp = score.relative.score(mate_score=10000)
                    if cp is not None:
                        if abs(cp) > resign_cp:
                            break
                        if abs(cp) < 10:
                            draw_counter += 1
                        else:
                            draw_counter = 0
                        if draw_counter >= 20:
                            break

            game = chess.pgn.Game()
            game.headers["White"] = f"SF_d{depth}"
            game.headers["Black"] = f"SF_d{depth}"

            outcome = board.outcome()
            if outcome:
                game.headers["Result"] = outcome.result()
            elif draw_counter >= 20:
                game.headers["Result"] = "1/2-1/2"
            else:
                game.headers["Result"] = "*"

            node = game
            temp_board = chess.Board()
            for move in move_list:
                node = node.add_variation(move)
                temp_board.push(move)

            games.append(game)

        return games
    finally:
        engine.quit()


def _worker(args: tuple) -> list[str]:
    """Worker that plays a batch of games and returns PGN strings."""
    seeds, depth, max_moves, resign_cp = args
    games = play_batch(seeds, depth, max_moves, resign_cp)
    return [str(g) + "\n\n" for g in games]


def main():
    parser = argparse.ArgumentParser(description="Generate SF-vs-SF games in PGN format")
    parser.add_argument("--num-games", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("outputs/sf_games_10k.pgn"))
    parser.add_argument("--depth", type=int, default=8, help="SF search depth per move")
    parser.add_argument("--workers", type=int, default=2, help="Parallel SF processes")
    parser.add_argument("--max-moves", type=int, default=200, help="Max plies per game")
    parser.add_argument("--resign-cp", type=int, default=1000, help="Resign threshold in cp")
    parser.add_argument("--batch-size", type=int, default=50, help="Games per worker batch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_seeds = list(range(args.seed, args.seed + args.num_games))

    # Split seeds into batches
    batches = []
    for i in range(0, len(all_seeds), args.batch_size):
        batch_seeds = all_seeds[i : i + args.batch_size]
        batches.append((batch_seeds, args.depth, args.max_moves, args.resign_cp))

    total_written = 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as f:
        if args.workers <= 1:
            for batch_args in batches:
                pgn_strings = _worker(batch_args)
                for pgn_str in pgn_strings:
                    f.write(pgn_str)
                total_written += len(pgn_strings)
                elapsed = time.time() - t0
                rate = total_written / elapsed if elapsed > 0 else 0
                print(f"\r  {total_written}/{args.num_games} games ({rate:.1f} games/s)", end="", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, b): b for b in batches}
                for future in as_completed(futures):
                    pgn_strings = future.result()
                    for pgn_str in pgn_strings:
                        f.write(pgn_str)
                    total_written += len(pgn_strings)
                    elapsed = time.time() - t0
                    rate = total_written / elapsed if elapsed > 0 else 0
                    print(f"\r  {total_written}/{args.num_games} games ({rate:.1f} games/s)", end="", flush=True)

    elapsed = time.time() - t0
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nDone: {total_written} games in {elapsed:.0f}s ({size_mb:.1f} MB)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
