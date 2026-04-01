"""Recursive move explorer with alpha-beta style branching and training output.

This script plays model-as-white games from a fixed opening against Stockfish,
but it does more than pick the top policy move. At every model turn it:

1. Searches a recursive tree over top-k model moves.
2. Optionally branches over Stockfish MultiPV replies on black turns.
3. Uses deterministic rollout from leaf nodes to score branches.
4. Logs trainable root records plus richer search traces for future work.

The root training records keep the exp082-style core fields:
    fen, best_move, best_cp, value_target, move_values
but they also mark that only a searched subset of legal moves was labeled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from move_vocab import index_to_move, legal_move_mask
from play import encode_board, load_model

ROOT = Path(__file__).resolve().parent
SF_PATH = ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "exp082_sf_game_softloop" / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ROOT / "outputs" / "deep_search"

OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
]

WIN_SCORE = 100000
DRAW_SCORE = 0
LOSS_SCORE = -100000
LEAF_EVAL_DEPTH = 8


@dataclass
class SearchStats:
    nodes: int = 0
    playouts: int = 0
    prunes: int = 0
    cache_hits: int = 0
    positions_logged: int = 0


@dataclass
class SearchResult:
    score: int
    pv: list[str]
    leaf_reason: str


def log(msg: str, log_file: Path | None = None):
    print(msg, flush=True)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def get_model_candidates(model, board: chess.Board, k: int) -> list[dict]:
    board_input = encode_board(board, DEVICE)
    result = model(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(DEVICE)
    logits[~mask] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(k, int(mask.sum().item())))

    candidates = []
    for rank, (idx, prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
        move = index_to_move(idx)
        if move in board.legal_moves:
            candidates.append(
                {
                    "move": move,
                    "uci": move.uci(),
                    "policy_prob": float(prob),
                    "rank": rank,
                }
            )
    return candidates


def score_terminal(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return DRAW_SCORE
    plies = len(board.move_stack)
    if outcome.winner == chess.WHITE:
        return WIN_SCORE - plies
    return LOSS_SCORE + plies


def cp_to_value_class(cp: int) -> int:
    if cp > 100:
        return 2
    if cp < -100:
        return 0
    return 1


def score_to_result(score: int) -> str:
    if score > 100:
        return "win"
    if score < -100:
        return "loss"
    return "draw"


def engine_eval_white(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
) -> int:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=WIN_SCORE)
    return int(score if score is not None else 0)


def get_sf_candidates(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    k: int,
) -> list[dict]:
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=max(1, k))
    if isinstance(infos, dict):
        infos = [infos]

    candidates = []
    seen = set()
    for rank, info in enumerate(infos, 1):
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        if move.uci() in seen:
            continue
        seen.add(move.uci())
        score = info["score"].white().score(mate_score=WIN_SCORE)
        if score is None:
            score = 0
        candidates.append(
            {
                "move": move,
                "uci": move.uci(),
                "engine_cp_white": int(score),
                "rank": rank,
            }
        )

    candidates.sort(key=lambda x: x["engine_cp_white"])
    return candidates[:k]


def greedy_rollout(
    model,
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    sf_depth: int,
    ply_cap: int,
    leaf_eval_depth: int,
) -> SearchResult:
    rollout = board.copy()
    while not rollout.is_game_over(claim_draw=True) and len(rollout.move_stack) < ply_cap:
        if rollout.turn == chess.WHITE:
            candidates = get_model_candidates(model, rollout, k=1)
            if not candidates:
                break
            rollout.push(candidates[0]["move"])
        else:
            move = engine.play(rollout, chess.engine.Limit(depth=sf_depth)).move
            if move not in rollout.legal_moves:
                move = next(iter(rollout.legal_moves))
            rollout.push(move)

    if rollout.is_game_over(claim_draw=True):
        return SearchResult(score_terminal(rollout), [], "terminal_rollout")
    return SearchResult(
        engine_eval_white(engine, rollout, leaf_eval_depth),
        [],
        "engine_leaf_rollout",
    )


def search(
    model,
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    sf_depth: int,
    ply_cap: int,
    depth: int,
    model_branch_k: int,
    sf_branch_k: int,
    leaf_eval_depth: int,
    alpha: int,
    beta: int,
    stats: SearchStats,
    cache: dict[tuple[str, int, int, int], SearchResult],
) -> SearchResult:
    stats.nodes += 1
    key = (board.fen(), depth, model_branch_k, sf_branch_k)
    cached = cache.get(key)
    if cached is not None:
        stats.cache_hits += 1
        return cached

    if board.is_game_over(claim_draw=True) or len(board.move_stack) >= ply_cap:
        result = SearchResult(score_terminal(board), [], "terminal")
        cache[key] = result
        return result

    if depth <= 0:
        stats.playouts += 1
        result = greedy_rollout(model, engine, board, sf_depth, ply_cap, leaf_eval_depth)
        cache[key] = result
        return result

    if board.turn == chess.WHITE:
        candidates = get_model_candidates(model, board, model_branch_k)
        if not candidates:
            result = SearchResult(DRAW_SCORE, [], "no_model_moves")
            cache[key] = result
            return result

        best_score = -10**9
        best_pv: list[str] = []
        for cand in candidates:
            board.push(cand["move"])
            child = search(
                model,
                engine,
                board,
                sf_depth,
                ply_cap,
                depth - 1,
                model_branch_k,
                sf_branch_k,
                leaf_eval_depth,
                alpha,
                beta,
                stats,
                cache,
            )
            board.pop()

            if child.score > best_score:
                best_score = child.score
                best_pv = [cand["uci"]] + child.pv

            alpha = max(alpha, best_score)
            if alpha >= beta:
                stats.prunes += 1
                break

        result = SearchResult(best_score, best_pv, "search")
        cache[key] = result
        return result

    candidates = get_sf_candidates(engine, board, sf_depth, sf_branch_k)
    if not candidates:
        result = SearchResult(DRAW_SCORE, [], "no_sf_moves")
        cache[key] = result
        return result

    best_score = 10**9
    best_pv = []
    for cand in candidates:
        board.push(cand["move"])
        child = search(
            model,
            engine,
            board,
            sf_depth,
            ply_cap,
            depth,
            model_branch_k,
            sf_branch_k,
            leaf_eval_depth,
            alpha,
            beta,
            stats,
            cache,
        )
        board.pop()

        if child.score < best_score:
            best_score = child.score
            best_pv = [cand["uci"]] + child.pv

        beta = min(beta, best_score)
        if alpha >= beta:
            stats.prunes += 1
            break

    result = SearchResult(best_score, best_pv, "search")
    cache[key] = result
    return result


def explore_root_position(
    model,
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    sf_depth: int,
    ply_cap: int,
    search_depth: int,
    model_branch_k: int,
    sf_branch_k: int,
    leaf_eval_depth: int,
    stats: SearchStats,
    cache: dict[tuple[str, int, int, int], SearchResult],
    game_id: str,
    root_path: list[str],
) -> tuple[dict, list[dict]]:
    candidates = get_model_candidates(model, board, model_branch_k)
    move_values = []
    trace_rows = []

    for cand in candidates:
        board.push(cand["move"])
        child = search(
            model,
            engine,
            board,
            sf_depth,
            ply_cap,
            max(search_depth - 1, 0),
            model_branch_k,
            sf_branch_k,
            leaf_eval_depth,
            -10**9,
            10**9,
            stats,
            cache,
        )
        board.pop()

        move_values.append(
            {
                "uci": cand["uci"],
                "cp": int(child.score),
                "model_prob": round(cand["policy_prob"], 6),
                "eval_type": "recursive_search",
                "pv": child.pv[:12],
                "leaf_reason": child.leaf_reason,
            }
        )
        trace_rows.append(
            {
                "game_id": game_id,
                "fen": board.fen(),
                "ply": len(board.move_stack),
                "path": list(root_path),
                "candidate_move": cand["uci"],
                "candidate_rank": cand["rank"],
                "candidate_policy_prob": round(cand["policy_prob"], 6),
                "search_score": int(child.score),
                "search_result": score_to_result(child.score),
                "pv": child.pv[:16],
                "leaf_reason": child.leaf_reason,
                "search_depth": search_depth,
                "model_branch_k": model_branch_k,
                "sf_branch_k": sf_branch_k,
            }
        )

    move_values.sort(key=lambda mv: mv["cp"], reverse=True)
    best_cp = int(move_values[0]["cp"]) if move_values else 0
    best_move = move_values[0]["uci"] if move_values else ""

    record = {
        "fen": board.fen(),
        "best_move": best_move,
        "best_cp": best_cp,
        "value_target": cp_to_value_class(best_cp),
        "move_values": move_values,
        "source": "deep_recursive_search",
        "game_id": game_id,
        "ply": len(board.move_stack),
        "root_path": list(root_path),
        "search_depth": search_depth,
        "model_branch_k": model_branch_k,
        "sf_branch_k": sf_branch_k,
        "leaf_eval_depth": leaf_eval_depth,
        "num_legal": board.legal_moves.count(),
        "searched_only": True,
        "searched_move_count": len(move_values),
    }
    return record, trace_rows


def play_and_explore_game(
    model,
    engine: chess.engine.SimpleEngine,
    sf_depth: int,
    opening_uci: list[str],
    ply_cap: int,
    search_depth: int,
    model_branch_k: int,
    sf_branch_k: int,
    leaf_eval_depth: int,
    log_file: Path,
    records_path: Path,
    traces_path: Path,
    game_id: str,
) -> dict:
    board = chess.Board()
    for uci in opening_uci:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    stats = SearchStats()
    cache: dict[tuple[str, int, int, int], SearchResult] = {}
    game_moves = [m.uci() for m in board.move_stack]
    root_path = list(game_moves)
    records = 0

    t0 = time.time()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == chess.WHITE:
            move_num = len(board.move_stack) // 2 + 1
            pos_t0 = time.time()
            record, trace_rows = explore_root_position(
                model,
                engine,
                board,
                sf_depth,
                ply_cap,
                search_depth,
                model_branch_k,
                sf_branch_k,
                leaf_eval_depth,
                stats,
                cache,
                game_id,
                root_path,
            )
            elapsed = time.time() - pos_t0
            best = record["move_values"][0] if record["move_values"] else None
            summary = " | ".join(
                f"{mv['uci']}={mv['cp']:+d}({mv['model_prob']*100:.0f}%)"
                for mv in record["move_values"][: min(5, len(record["move_values"]))]
            )
            log(
                f"  {move_num}. {summary} -> {record['best_move']} "
                f"[{elapsed:.1f}s, nodes={stats.nodes}, cache={stats.cache_hits}]",
                log_file,
            )

            append_jsonl(records_path, record)
            for row in trace_rows:
                append_jsonl(traces_path, row)
            stats.positions_logged += 1
            records += 1

            if best is None:
                break
            move = chess.Move.from_uci(best["uci"])
            board.push(move)
            game_moves.append(move.uci())
            root_path.append(move.uci())
        else:
            sf_move = engine.play(board, chess.engine.Limit(depth=sf_depth)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            board.push(sf_move)
            game_moves.append(sf_move.uci())
            root_path.append(sf_move.uci())

    total_time = time.time() - t0
    final_score = score_terminal(board)
    result = score_to_result(final_score)
    outcome = board.outcome(claim_draw=True)
    termination = outcome.termination.name if outcome is not None else "PLY_CAP"

    log(
        f"\nGame {game_id}: {result} in {len(board.move_stack)} plies ({termination})",
        log_file,
    )
    log(
        f"Stats: nodes={stats.nodes}, playouts={stats.playouts}, "
        f"prunes={stats.prunes}, cache_hits={stats.cache_hits}, "
        f"records={stats.positions_logged}, time={total_time:.1f}s",
        log_file,
    )
    log(f"Moves: {' '.join(game_moves)}\n", log_file)

    return {
        "game_id": game_id,
        "result": result,
        "terminal_score": final_score,
        "total_plies": len(board.move_stack),
        "termination": termination,
        "moves": game_moves,
        "records": records,
        "stats": {
            "nodes": stats.nodes,
            "playouts": stats.playouts,
            "prunes": stats.prunes,
            "cache_hits": stats.cache_hits,
            "positions_logged": stats.positions_logged,
            "elapsed_s": round(total_time, 2),
        },
        "final_fen": board.fen(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recursive move explorer with alpha-beta style search and training output"
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--sf-depth", type=int, default=6)
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--branch-k", type=int, default=5)
    parser.add_argument("--sf-branch-k", type=int, default=2)
    parser.add_argument("--leaf-eval-depth", type=int, default=LEAF_EVAL_DEPTH)
    parser.add_argument("--opening", type=int, default=1)
    parser.add_argument("--ply-cap", type=int, default=160)
    parser.add_argument("--num-games", type=int, default=1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "search.log"
    records_path = run_dir / "training_records.jsonl"
    traces_path = run_dir / "search_traces.jsonl"
    summary_path = run_dir / "summary.json"

    opening = OPENINGS[args.opening % len(OPENINGS)]
    opening_str = " ".join(opening) if opening else "startpos"

    log("Deep recursive move explorer", log_path)
    log(f"  checkpoint: {args.checkpoint}", log_path)
    log(f"  sf_depth: {args.sf_depth}", log_path)
    log(f"  search_depth: {args.search_depth}", log_path)
    log(f"  branch_k: {args.branch_k}", log_path)
    log(f"  sf_branch_k: {args.sf_branch_k}", log_path)
    log(f"  leaf_eval_depth: {args.leaf_eval_depth}", log_path)
    log(f"  opening: {opening_str}", log_path)
    log(f"  ply_cap: {args.ply_cap}", log_path)
    log(f"  num_games: {args.num_games}", log_path)
    log(f"  output_dir: {run_dir}", log_path)
    log("", log_path)

    model = load_model(args.checkpoint, DEVICE)
    engine = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
    engine.configure({"Threads": 1, "Hash": 64})

    games = []
    try:
        for game_idx in range(args.num_games):
            game_id = f"{timestamp}_g{game_idx + 1:03d}"
            log("=" * 72, log_path)
            log(f"Game {game_idx + 1}/{args.num_games} ({game_id})", log_path)
            log("=" * 72, log_path)
            game_summary = play_and_explore_game(
                model,
                engine,
                args.sf_depth,
                opening,
                args.ply_cap,
                args.search_depth,
                args.branch_k,
                args.sf_branch_k,
                args.leaf_eval_depth,
                log_path,
                records_path,
                traces_path,
                game_id,
            )
            games.append(game_summary)
    finally:
        engine.quit()

    total_records = sum(g["records"] for g in games)
    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "sf_depth": args.sf_depth,
            "search_depth": args.search_depth,
            "branch_k": args.branch_k,
            "sf_branch_k": args.sf_branch_k,
            "leaf_eval_depth": args.leaf_eval_depth,
            "opening": opening_str,
            "ply_cap": args.ply_cap,
            "num_games": args.num_games,
        },
        "games": games,
        "total_training_records": total_records,
        "training_records_path": str(records_path),
        "search_traces_path": str(traces_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Done. {total_records} training records -> {records_path}", log_path)
    log(f"Search traces -> {traces_path}", log_path)
    log(f"Summary -> {summary_path}", log_path)


if __name__ == "__main__":
    main()
