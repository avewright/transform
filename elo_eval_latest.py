import argparse
import json
import math
import os
import shutil
from pathlib import Path

import chess
import chess.engine
import chess.syzygy
import torch
import torch.nn.functional as F

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT = ROOT / "outputs" / "hf" / "chess-transformer-200m-latest" / "best_model.pt"
DEFAULT_MODEL_CONFIG = None

# Syzygy tablebase for perfect endgame play
SYZYGY_DIR = ROOT / "syzygy"
SYZYGY_TB = None
SYZYGY_MAX_PIECES = 5

def init_syzygy():
    """Initialize Syzygy tablebase if available."""
    global SYZYGY_TB
    if SYZYGY_DIR.exists() and any(SYZYGY_DIR.glob("*.rtbw")):
        try:
            SYZYGY_TB = chess.syzygy.open_tablebase(str(SYZYGY_DIR))
        except Exception:
            SYZYGY_TB = None

def get_syzygy_move(board: chess.Board) -> chess.Move | None:
    """Look up best move from Syzygy tables. Returns None if not available."""
    if SYZYGY_TB is None:
        return None
    if len(board.piece_map()) > SYZYGY_MAX_PIECES:
        return None
    try:
        best_move = None
        best_wdl = -3
        best_dtz = 0
        for move in board.legal_moves:
            board.push(move)
            try:
                wdl = -SYZYGY_TB.probe_wdl(board)
                dtz = -SYZYGY_TB.probe_dtz(board)
                # Prefer winning moves, then draws, then losses
                # Among winning moves, prefer shorter DTZ (faster win)
                # Among losing moves, prefer longer DTZ (slower loss)
                if wdl > best_wdl or (wdl == best_wdl and (
                    (wdl > 0 and dtz < best_dtz) or
                    (wdl < 0 and dtz > best_dtz) or
                    (wdl == 0 and abs(dtz) < abs(best_dtz))
                )):
                    best_move = move
                    best_wdl = wdl
                    best_dtz = dtz
            except Exception:
                pass
            board.pop()
        return best_move
    except Exception:
        return None

DEFAULT_OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
]
DEFAULT_TEST_ELOS = [1320, 1450, 1600, 1750, 1900]

LOG: Path
JSON_OUT: Path


def resolve_stockfish_path() -> Path:
    configured = os.environ.get("STOCKFISH_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    binary = shutil.which("stockfish")
    if binary:
        candidates.append(Path(binary))
    candidates.extend(
        [
            Path("/usr/games/stockfish"),
            Path("/usr/bin/stockfish"),
            ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
            ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate Stockfish binary. Checked: {checked}")


SF = resolve_stockfish_path()


def load_checkpoint_state(checkpoint_path: str | Path) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def load_eval_model(checkpoint_path: str | Path, device: torch.device, model_config: str | None = None):
    if model_config is None:
        from play import load_model
        return load_model(str(checkpoint_path), device)

    model = build_model(model_config)
    state = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_model_move_generic(model, board: chess.Board, device: torch.device, temperature: float = 0.0):
    from move_vocab import IDX_TO_UCI, index_to_move, legal_move_mask

    board_input = batch_boards_to_fused_token_ids([board], device)
    result = model(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    if temperature <= 0:
        move_idx = logits.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()

    move = index_to_move(move_idx)
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, mask.sum().item()))
    top_moves = []
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        top_moves.append((IDX_TO_UCI[idx], f"{p*100:.1f}%"))
    wdl_logits = result["value_logits"][0].float()
    wdl_probs = F.softmax(wdl_logits, dim=-1).tolist()
    # Model WDL is White-absolute: idx0=P(W wins), idx1=P(draw), idx2=P(W loses)
    return move, {"top_moves": top_moves, "wdl": {"win": wdl_probs[0], "draw": wdl_probs[1], "loss": wdl_probs[2]}}


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def opening_name(opening: list[str]) -> str:
    return "startpos" if not opening else " ".join(opening)


def play_one(
    engine: chess.engine.SimpleEngine,
    model,
    move_fn,
    sf_elo: int,
    model_color: chess.Color,
    opening: list[str],
    movetime: float,
    ply_cap: int,
) -> dict:
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            # Try Syzygy tablebase first for perfect endgame play
            tb_move = get_syzygy_move(board)
            if tb_move is not None:
                move = tb_move
            else:
                move, _ = move_fn(model, board, DEVICE, temperature=0.0)
        else:
            move = engine.play(board, chess.engine.Limit(time=movetime)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == model_color:
        score = 1.0
    else:
        score = 0.0

    return {
        "sf_elo": sf_elo,
        "model_color": "white" if model_color == chess.WHITE else "black",
        "opening": opening,
        "opening_name": opening_name(opening),
        "result": board.result(claim_draw=True),
        "score": score,
        "plies": len(board.move_stack),
        "termination": outcome.termination.name if outcome else "PLY_CAP",
        "final_fen": board.fen(),
    }


def summarize_results(sf_elo: int, results: list[dict]) -> dict:
    games = len(results)
    total_score = sum(r["score"] for r in results)
    score = total_score / games if games else 0.0
    ci_low, ci_high = wilson_interval(total_score, games)

    by_color = {}
    for color in ("white", "black"):
        color_results = [r for r in results if r["model_color"] == color]
        color_games = len(color_results)
        color_score = sum(r["score"] for r in color_results) / color_games if color_games else 0.0
        by_color[color] = {
            "games": color_games,
            "score": color_score,
            "w": sum(1 for r in color_results if r["score"] == 1.0),
            "d": sum(1 for r in color_results if r["score"] == 0.5),
            "l": sum(1 for r in color_results if r["score"] == 0.0),
        }

    by_opening = {}
    for r in results:
        name = r["opening_name"]
        bucket = by_opening.setdefault(name, {"games": 0, "score_sum": 0.0, "w": 0, "d": 0, "l": 0})
        bucket["games"] += 1
        bucket["score_sum"] += r["score"]
        if r["score"] == 1.0:
            bucket["w"] += 1
        elif r["score"] == 0.5:
            bucket["d"] += 1
        else:
            bucket["l"] += 1

    openings_summary = []
    for name, bucket in sorted(by_opening.items()):
        openings_summary.append(
            {
                "opening": name,
                "games": bucket["games"],
                "score": bucket["score_sum"] / bucket["games"],
                "w": bucket["w"],
                "d": bucket["d"],
                "l": bucket["l"],
            }
        )

    return {
        "sf_elo": sf_elo,
        "games": games,
        "score": score,
        "score_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "w": sum(1 for r in results if r["score"] == 1.0),
        "d": sum(1 for r in results if r["score"] == 0.5),
        "l": sum(1 for r in results if r["score"] == 0.0),
        "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0.0,
        "terminations": {
            term: sum(1 for r in results if r["termination"] == term)
            for term in sorted({r["termination"] for r in results})
        },
        "by_color": by_color,
        "by_opening": openings_summary,
    }


def eval_level(
    model,
    move_fn,
    sf_elo: int,
    openings: list[list[str]],
    games_per_opening_per_color: int,
    movetime: float,
    ply_cap: int,
) -> tuple[dict, list[dict]]:
    engine = chess.engine.SimpleEngine.popen_uci(str(SF))
    engine.configure(
        {
            "UCI_LimitStrength": True,
            "UCI_Elo": sf_elo,
            "Threads": 1,
            "Hash": 32,
        }
    )
    results = []
    try:
        for opening in openings:
            for color in [chess.WHITE, chess.BLACK]:
                for repeat_idx in range(games_per_opening_per_color):
                    result = play_one(
                        engine=engine,
                        model=model,
                        move_fn=move_fn,
                        sf_elo=sf_elo,
                        model_color=color,
                        opening=opening,
                        movetime=movetime,
                        ply_cap=ply_cap,
                    )
                    result["repeat_idx"] = repeat_idx
                    results.append(result)
                    log(
                        "game "
                        + json.dumps(
                            {
                                "sf_elo": sf_elo,
                                "color": result["model_color"],
                                "opening": result["opening_name"],
                                "repeat_idx": repeat_idx,
                                "result": result["result"],
                                "score": result["score"],
                                "plies": result["plies"],
                                "termination": result["termination"],
                            }
                        )
                    )
    finally:
        engine.quit()

    summary = summarize_results(sf_elo, results)
    log("summary " + json.dumps(summary))
    return summary, results


def estimate_elo(summaries: list[dict]) -> dict:
    if not summaries:
        return {"estimated_elo": None, "lower_bound": None, "upper_bound": None, "note": "no games completed"}

    ordered = sorted(summaries, key=lambda s: s["sf_elo"])
    above = [s for s in ordered if s["score"] >= 0.5]
    below = [s for s in ordered if s["score"] < 0.5]

    lower_bound = max((s["sf_elo"] for s in above), default=None)
    upper_bound = min((s["sf_elo"] for s in below), default=None)

    if lower_bound is None:
        first = ordered[0]
        return {
            "estimated_elo": first["sf_elo"],
            "lower_bound": None,
            "upper_bound": first["sf_elo"],
            "note": f"model stayed below 50% score across tested levels; at {first['sf_elo']} score={first['score']:.3f}",
        }

    if upper_bound is None:
        last = ordered[-1]
        return {
            "estimated_elo": last["sf_elo"],
            "lower_bound": last["sf_elo"],
            "upper_bound": None,
            "note": f"model stayed at or above 50% through the highest tested level; at {last['sf_elo']} score={last['score']:.3f}",
        }

    lower_summary = next(s for s in ordered if s["sf_elo"] == lower_bound)
    upper_summary = next(s for s in ordered if s["sf_elo"] == upper_bound)

    if lower_bound == upper_bound:
        est = lower_bound
    elif upper_summary["score"] == lower_summary["score"]:
        est = lower_bound
    else:
        frac = (0.5 - lower_summary["score"]) / (upper_summary["score"] - lower_summary["score"])
        est = round(lower_bound + frac * (upper_bound - lower_bound))

    return {
        "estimated_elo": est,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "note": (
            f"bracketed by {lower_bound} (score={lower_summary['score']:.3f}) "
            f"and {upper_bound} (score={upper_summary['score']:.3f}); non-monotonic results may make this estimate noisy"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust Elo-style evaluation vs limited-strength Stockfish")
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    parser.add_argument("out_prefix", nargs="?", default=None, help="Output file prefix; defaults to checkpoint parent name")
    parser.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG, help="Optional model config JSON for non-play.py architectures")
    parser.add_argument("--movetime", type=float, default=0.05, help="Stockfish move time in seconds")
    parser.add_argument("--ply-cap", type=int, default=160, help="Maximum plies per game before adjudicating as a draw")
    parser.add_argument(
        "--games-per-opening-per-color",
        type=int,
        default=2,
        help="Repeat count for each opening with each color; total games per level = openings * 2 * repeats",
    )
    parser.add_argument(
        "--elos",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_ELOS,
        help="Stockfish Elo levels to test",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=None,
        help="Optional cap on the number of Elo levels to evaluate",
    )
    parser.add_argument(
        "--stop-after-bracket",
        action="store_true",
        help="Stop after the first bracketed 50%% interval instead of running all requested levels",
    )
    parser.add_argument(
        "--no-syzygy",
        action="store_true",
        help="Disable Syzygy tablebase probing during games",
    )
    return parser.parse_args()


def write_snapshot(
    checkpoint: Path,
    args: argparse.Namespace,
    summaries: list[dict],
    all_games: list[dict],
    estimate: dict,
) -> None:
    JSON_OUT.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "device": str(DEVICE),
                "config": {
                    "movetime": args.movetime,
                    "ply_cap": args.ply_cap,
                "games_per_opening_per_color": args.games_per_opening_per_color,
                "elos": args.elos,
                "openings": [opening_name(o) for o in DEFAULT_OPENINGS],
                "stop_after_bracket": args.stop_after_bracket,
                "model_config": args.model_config,
            },
                "summaries": summaries,
                "games": all_games,
                "estimate": estimate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    out_prefix = args.out_prefix or checkpoint.parent.name
    log_path = ROOT / "outputs" / f"elo_eval_{out_prefix}.log"
    json_path = ROOT / "outputs" / f"elo_eval_{out_prefix}.json"
    elos = args.elos[: args.max_levels] if args.max_levels is not None else args.elos

    global LOG, JSON_OUT
    LOG = log_path
    JSON_OUT = json_path

    if LOG.exists():
        LOG.unlink()

    log(
        "start "
        + json.dumps(
            {
                "device": str(DEVICE),
                "checkpoint": str(checkpoint),
                "movetime": args.movetime,
                "ply_cap": args.ply_cap,
                "games_per_opening_per_color": args.games_per_opening_per_color,
                "openings": [opening_name(o) for o in DEFAULT_OPENINGS],
                "elos": elos,
                "model_config": args.model_config,
            }
        )
    )
    model = load_eval_model(str(checkpoint), DEVICE, model_config=args.model_config)
    move_fn = get_model_move_generic if args.model_config else __import__("play").get_model_move

    # Initialize Syzygy tablebases for perfect endgame play
    if not args.no_syzygy:
        init_syzygy()
    if SYZYGY_TB is not None:
        log(f"Syzygy tablebases loaded from {SYZYGY_DIR} (up to {SYZYGY_MAX_PIECES} pieces)")
    else:
        log("Syzygy tablebases not available" + (" (disabled)" if args.no_syzygy else ""))

    summaries = []
    all_games = []
    for elo in elos:
        log(f"begin sf_elo={elo}")
        summary, results = eval_level(
            model=model,
            move_fn=move_fn,
            sf_elo=elo,
            openings=DEFAULT_OPENINGS,
            games_per_opening_per_color=args.games_per_opening_per_color,
            movetime=args.movetime,
            ply_cap=args.ply_cap,
        )
        summaries.append(summary)
        all_games.extend(results)
        estimate = estimate_elo(summaries)
        write_snapshot(checkpoint, args, summaries, all_games, estimate)
        log("estimate " + json.dumps(estimate))
        if (
            args.stop_after_bracket
            and estimate["lower_bound"] is not None
            and estimate["upper_bound"] is not None
        ):
            log(f"bracketed between {estimate['lower_bound']} and {estimate['upper_bound']}")
            break

    log("done")


if __name__ == "__main__":
    main()
