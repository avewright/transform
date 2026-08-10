#!/usr/bin/env python3
"""Max-Elo evaluation: pure policy (default) or MCTS report.

Promotion default: greedy policy, no book, no Syzygy, protocol.json ladder.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

# Vocab must be set before chess/move_vocab imports when used as __main__
from harness.common import (  # noqa: E402
    ROOT,
    ensure_compact_vocab,
    load_protocol,
    opening_name,
    pick_device,
    resolve_stockfish,
    stockfish_version,
)


def _bootstrap_vocab_from_argv() -> None:
    ckpt = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--ckpt", "--checkpoint", "-c") and i + 1 < len(argv):
            ckpt = argv[i + 1]
            break
        if not a.startswith("-") and (a.endswith(".pt") or a.endswith(".pth")):
            ckpt = a
            break
    ensure_compact_vocab(ckpt)


_bootstrap_vocab_from_argv()

import chess  # noqa: E402
import chess.engine  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
from move_vocab import IDX_TO_UCI, index_to_move, legal_move_mask  # noqa: E402

LOG: Path | None = None
SYZYGY_TB = None
SYZYGY_MAX_PIECES = 5


def init_syzygy(syzygy_dir: Path | None = None) -> None:
    """Optional Syzygy probe for deploy-style eval (off for promotion)."""
    global SYZYGY_TB
    import chess.syzygy

    d = syzygy_dir or (ROOT / "syzygy")
    if d.exists() and any(d.glob("*.rtbw")):
        try:
            SYZYGY_TB = chess.syzygy.open_tablebase(str(d))
        except Exception:
            SYZYGY_TB = None


def get_syzygy_move(board: chess.Board):
    if SYZYGY_TB is None or len(board.piece_map()) > SYZYGY_MAX_PIECES:
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
                if wdl > best_wdl or (
                    wdl == best_wdl
                    and (
                        (wdl > 0 and dtz < best_dtz)
                        or (wdl < 0 and dtz > best_dtz)
                        or (wdl == 0 and abs(dtz) < abs(best_dtz))
                    )
                ):
                    best_move = move
                    best_wdl = wdl
                    best_dtz = dtz
            except Exception:
                pass
            board.pop()
        return best_move
    except Exception:
        return None


def log(msg: str) -> None:
    print(msg, flush=True)
    if LOG is not None:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_checkpoint_state(checkpoint_path: str | Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        return state, ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    return state, {}


def load_eval_model(checkpoint_path: str | Path, device: torch.device):
    from chess_inference import load_checkpoint

    return load_checkpoint(str(checkpoint_path), device)


@torch.no_grad()
def get_model_move(model, board: chess.Board, device: torch.device, temperature: float = 0.0):
    board_input = batch_boards_to_fused_token_ids([board], device)
    mask = legal_move_mask(board).to(device)
    board_input["legal_mask"] = mask.unsqueeze(0)
    result = model(board_input)
    logits = result["policy_logits"][0].float()
    logits[~mask] = float("-inf")
    if temperature <= 0:
        move_idx = logits.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()
    move = index_to_move(move_idx)
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, int(mask.sum().item())))
    top_moves = [(IDX_TO_UCI[i], f"{p * 100:.1f}%") for i, p in zip(topk.indices.tolist(), topk.values.tolist())]
    wdl_logits = result["value_logits"][0].float()
    if wdl_logits.shape[-1] == 3:
        wdl = F.softmax(wdl_logits, dim=-1).tolist()
        info = {"top_moves": top_moves, "wdl": {"win": wdl[0], "draw": wdl[1], "loss": wdl[2]}}
    else:
        n = wdl_logits.shape[-1]
        centers = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=wdl_logits.device)
        win_pct = (F.softmax(wdl_logits, dim=-1) * centers).sum().item()
        info = {"top_moves": top_moves, "wdl": {"win": win_pct, "draw": 0.0, "loss": 1.0 - win_pct}}
    return move, info


def play_one_policy(
    engine: chess.engine.SimpleEngine,
    model,
    move_fn: Callable,
    device: torch.device,
    sf_elo: int,
    model_color: chess.Color,
    opening: list[str],
    movetime: float,
    ply_cap: int,
    *,
    use_book: bool,
    get_book_move,
    get_syzygy_move,
) -> dict:
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            move = None
            source = "policy"
            if get_syzygy_move is not None:
                tb = get_syzygy_move(board)
                if tb is not None:
                    move, source = tb, "syzygy"
            if move is None and use_book and get_book_move is not None:
                bm = get_book_move(board)
                if bm is not None:
                    move, source = bm, "book"
            if move is None:
                move, _ = move_fn(model, board, device, temperature=0.0)
                source = "policy"
        else:
            move = engine.play(board, chess.engine.Limit(time=movetime)).move
            source = "sf"
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
            source = "fallback"
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
        "last_model_source": source if board.turn != model_color else "n/a",
    }


def summarize_results(sf_elo: int, results: list[dict]) -> dict:
    games = len(results)
    total = sum(r["score"] for r in results)
    score = total / games if games else 0.0
    ci_lo, ci_hi = wilson_interval(total, games)
    by_color = {}
    for color in ("white", "black"):
        cr = [r for r in results if r["model_color"] == color]
        n = len(cr)
        by_color[color] = {
            "games": n,
            "score": sum(r["score"] for r in cr) / n if n else 0.0,
            "w": sum(1 for r in cr if r["score"] == 1.0),
            "d": sum(1 for r in cr if r["score"] == 0.5),
            "l": sum(1 for r in cr if r["score"] == 0.0),
        }
    by_opening: dict[str, dict] = {}
    for r in results:
        name = r["opening_name"]
        b = by_opening.setdefault(name, {"games": 0, "score_sum": 0.0, "w": 0, "d": 0, "l": 0})
        b["games"] += 1
        b["score_sum"] += r["score"]
        if r["score"] == 1.0:
            b["w"] += 1
        elif r["score"] == 0.5:
            b["d"] += 1
        else:
            b["l"] += 1
    openings_summary = [
        {
            "opening": name,
            "games": b["games"],
            "score": b["score_sum"] / b["games"],
            "w": b["w"],
            "d": b["d"],
            "l": b["l"],
        }
        for name, b in sorted(by_opening.items())
    ]
    return {
        "sf_elo": sf_elo,
        "games": games,
        "score": score,
        "score_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "w": sum(1 for r in results if r["score"] == 1.0),
        "d": sum(1 for r in results if r["score"] == 0.5),
        "l": sum(1 for r in results if r["score"] == 0.0),
        "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0.0,
        "terminations": {
            t: sum(1 for r in results if r["termination"] == t)
            for t in sorted({r["termination"] for r in results})
        },
        "by_color": by_color,
        "by_opening": openings_summary,
    }


def estimate_elo(summaries: list[dict]) -> dict:
    if not summaries:
        return {"estimated_elo": None, "lower_bound": None, "upper_bound": None, "note": "no games"}
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
            "note": f"below 50% at all levels; at {first['sf_elo']} score={first['score']:.3f}",
        }
    if upper_bound is None:
        last = ordered[-1]
        return {
            "estimated_elo": last["sf_elo"],
            "lower_bound": last["sf_elo"],
            "upper_bound": None,
            "note": f"≥50% through {last['sf_elo']} score={last['score']:.3f}",
        }
    lo_s = next(s for s in ordered if s["sf_elo"] == lower_bound)
    hi_s = next(s for s in ordered if s["sf_elo"] == upper_bound)
    if lower_bound == upper_bound or hi_s["score"] == lo_s["score"]:
        est = lower_bound
    else:
        frac = (0.5 - lo_s["score"]) / (hi_s["score"] - lo_s["score"])
        est = round(lower_bound + frac * (upper_bound - lower_bound))
    return {
        "estimated_elo": est,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "note": (
            f"bracketed by {lower_bound} (score={lo_s['score']:.3f}) "
            f"and {upper_bound} (score={hi_s['score']:.3f})"
        ),
    }


def run_policy_elo(args: argparse.Namespace) -> dict[str, Any]:
    global LOG
    protocol = load_protocol(args.protocol)
    device = pick_device(args.device)
    sf_path = resolve_stockfish()
    sf_ver = stockfish_version(sf_path)
    ckpt = Path(args.ckpt).resolve()
    ensure_compact_vocab(ckpt)

    out_prefix = args.out_prefix or f"{ckpt.parent.name}_policy"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"elo_eval_{out_prefix}.log"
    json_path = out_dir / f"elo_eval_{out_prefix}.json"
    LOG = log_path
    if log_path.exists():
        log_path.unlink()

    use_book = args.book
    use_syzygy = args.syzygy
    movetime = args.movetime if args.movetime is not None else protocol["movetime"]
    ply_cap = args.ply_cap if args.ply_cap is not None else protocol["ply_cap"]
    games = (
        args.games_per_opening_per_color
        if args.games_per_opening_per_color is not None
        else protocol["games_per_opening_per_color"]
    )
    elos = args.elos if args.elos is not None else list(protocol["elos"])
    openings = [list(o) for o in protocol["openings"]]
    stop = args.stop_after_bracket if args.stop_after_bracket is not None else protocol["stop_after_bracket"]

    get_book_move = None
    if use_book:
        from opening_book import get_book_move as _gbm

        get_book_move = _gbm

    syzygy_fn = None
    if use_syzygy:
        init_syzygy()
        syzygy_fn = get_syzygy_move
        if SYZYGY_TB is not None:
            log(f"Syzygy loaded from {ROOT / 'syzygy'}")
        else:
            log("Syzygy requested but not available")

    proto_record = {
        "name": protocol.get("name"),
        "mode": "policy",
        "book": use_book,
        "syzygy": use_syzygy,
        "movetime": movetime,
        "ply_cap": ply_cap,
        "games_per_opening_per_color": games,
        "elos": elos,
        "openings": [opening_name(o) for o in openings],
        "stop_after_bracket": stop,
        "sf_path": str(sf_path),
        "sf_version": sf_ver,
        "vocab": "compact",
        "device": str(device),
        "threads": protocol.get("threads", 1),
        "hash": protocol.get("hash", 32),
    }

    log("start " + json.dumps({"checkpoint": str(ckpt), **proto_record}))
    model = load_eval_model(ckpt, device)
    log(f"Model loaded on {device}; Stockfish={sf_ver} ({sf_path})")

    summaries: list[dict] = []
    all_games: list[dict] = []
    estimate: dict = {}

    def write_snapshot() -> None:
        payload = {
            "checkpoint": str(ckpt),
            "device": str(device),
            "mode": "policy",
            "protocol": proto_record,
            "config": {
                "movetime": movetime,
                "ply_cap": ply_cap,
                "games_per_opening_per_color": games,
                "elos": elos,
                "openings": proto_record["openings"],
                "stop_after_bracket": stop,
                "book": use_book,
                "syzygy": use_syzygy,
            },
            "summaries": summaries,
            "games": all_games,
            "estimate": estimate,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for elo in elos:
        log(f"begin sf_elo={elo}")
        engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
        engine.configure(
            {
                "UCI_LimitStrength": True,
                "UCI_Elo": elo,
                "Threads": protocol.get("threads", 1),
                "Hash": protocol.get("hash", 32),
            }
        )
        results = []
        try:
            for opening in openings:
                for color in (chess.WHITE, chess.BLACK):
                    for repeat_idx in range(games):
                        r = play_one_policy(
                            engine,
                            model,
                            get_model_move,
                            device,
                            elo,
                            color,
                            opening,
                            movetime,
                            ply_cap,
                            use_book=use_book,
                            get_book_move=get_book_move,
                            get_syzygy_move=syzygy_fn,
                        )
                        r["repeat_idx"] = repeat_idx
                        results.append(r)
                        log(
                            "game "
                            + json.dumps(
                                {
                                    "sf_elo": elo,
                                    "color": r["model_color"],
                                    "opening": r["opening_name"],
                                    "repeat_idx": repeat_idx,
                                    "result": r["result"],
                                    "score": r["score"],
                                    "plies": r["plies"],
                                    "termination": r["termination"],
                                }
                            )
                        )
        finally:
            engine.quit()

        summary = summarize_results(elo, results)
        summaries.append(summary)
        all_games.extend(results)
        estimate = estimate_elo(summaries)
        write_snapshot()
        log("summary " + json.dumps(summary))
        log("estimate " + json.dumps(estimate))
        if stop and estimate.get("lower_bound") is not None and estimate.get("upper_bound") is not None:
            log(f"bracketed between {estimate['lower_bound']} and {estimate['upper_bound']}")
            break

    log("done")
    log(f"wrote {json_path}")
    return {
        "json_path": str(json_path),
        "estimate": estimate,
        "elo": estimate.get("estimated_elo"),
        "protocol": proto_record,
    }


def run_mcts_elo(args: argparse.Namespace) -> dict[str, Any]:
    """MCTS report mode — not used for champion promotion."""
    protocol = load_protocol(args.protocol)
    device = pick_device(args.device)
    sf_path = resolve_stockfish()
    sf_ver = stockfish_version(sf_path)
    ckpt = Path(args.ckpt).resolve()
    ensure_compact_vocab(ckpt)

    from chess_inference import load_checkpoint
    from uci_engine import MCTSSearch, SyzygyProbe

    out_prefix = args.out_prefix or f"{ckpt.parent.name}_mcts{args.sims}"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"elo_eval_{out_prefix}.log"
    json_path = out_dir / f"elo_eval_{out_prefix}.json"
    global LOG
    LOG = log_path
    if log_path.exists():
        log_path.unlink()

    use_book = args.book  # default False for harness; deploy can pass --book
    use_syzygy = args.syzygy
    movetime = args.movetime if args.movetime is not None else protocol["movetime"]
    ply_cap = args.ply_cap if args.ply_cap is not None else protocol["ply_cap"]
    games = (
        args.games_per_opening_per_color
        if args.games_per_opening_per_color is not None
        else 1
    )
    elos = args.elos if args.elos is not None else [1750, 1900, 2050]
    openings = [list(o) for o in protocol["openings"]]
    stop = bool(args.stop_after_bracket) if args.stop_after_bracket is not None else True

    proto_record = {
        "name": protocol.get("name"),
        "mode": "mcts",
        "book": use_book,
        "syzygy": use_syzygy,
        "sims": args.sims,
        "search_mode": args.search_mode,
        "reuse_tree": True,
        "movetime": movetime,
        "ply_cap": ply_cap,
        "games_per_opening_per_color": games,
        "elos": elos,
        "openings": [opening_name(o) for o in openings],
        "stop_after_bracket": stop,
        "sf_path": str(sf_path),
        "sf_version": sf_ver,
        "vocab": "compact",
        "device": str(device),
    }
    log("start " + json.dumps({"checkpoint": str(ckpt), **proto_record}))

    model = load_checkpoint(str(ckpt), device)
    model.eval()
    syzygy = SyzygyProbe(str(ROOT / "syzygy") if use_syzygy and (ROOT / "syzygy").exists() else None)

    # Optionally disable book inside MCTS by monkeypatching get_book_move
    if not use_book:
        import opening_book

        opening_book.get_book_move = lambda board: None  # type: ignore

    mcts = MCTSSearch(
        model,
        device,
        syzygy,
        c_puct=2.5,
        batch_size=args.batch_size,
        root_noise_frac=0.0,
        search_mode=args.search_mode,
    )

    summaries: list[dict] = []
    all_games: list[dict] = []
    estimate: dict = {}

    def write_snapshot() -> None:
        payload = {
            "checkpoint": str(ckpt),
            "device": str(device),
            "mode": "mcts",
            "protocol": proto_record,
            "summaries": summaries,
            "games": all_games,
            "estimate": estimate,
            "estimated_elo": estimate.get("estimated_elo"),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    try:
        for elo in elos:
            engine.configure(
                {
                    "UCI_LimitStrength": True,
                    "UCI_Elo": elo,
                    "Threads": 1,
                    "Hash": 32,
                }
            )
            log(f"begin sf_elo={elo}")
            results = []
            for opening in openings:
                for color in (chess.WHITE, chess.BLACK):
                    for repeat_idx in range(games):
                        board = chess.Board()
                        for uci in opening:
                            m = chess.Move.from_uci(uci)
                            if m in board.legal_moves:
                                board.push(m)
                        mcts.new_game()
                        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
                            if board.turn == color:
                                move, info = mcts.search(board, max_sims=args.sims)
                                # Tree reuse ON: do not new_game() per move
                                if move not in board.legal_moves:
                                    move = next(iter(board.legal_moves))
                                board.push(move)
                            else:
                                mv = engine.play(board, chess.engine.Limit(time=movetime)).move
                                if mv not in board.legal_moves:
                                    mv = next(iter(board.legal_moves))
                                board.push(mv)
                        o = board.outcome(claim_draw=True)
                        if o is None or o.winner is None:
                            score = 0.5
                        elif o.winner == color:
                            score = 1.0
                        else:
                            score = 0.0
                        r = {
                            "sf_elo": elo,
                            "model_color": "white" if color == chess.WHITE else "black",
                            "opening_name": opening_name(opening),
                            "score": score,
                            "plies": len(board.move_stack),
                            "termination": o.termination.name if o else "PLY_CAP",
                            "repeat_idx": repeat_idx,
                            "result": board.result(claim_draw=True),
                        }
                        results.append(r)
                        log("game " + json.dumps(r))
            summary = summarize_results(elo, results)
            # normalize model_color key already in summarize
            summaries.append(summary)
            all_games.extend(results)
            estimate = estimate_elo(summaries)
            write_snapshot()
            log("summary " + json.dumps(summary))
            log("estimate " + json.dumps(estimate))
            if stop and estimate.get("lower_bound") is not None and estimate.get("upper_bound") is not None:
                break
    finally:
        engine.quit()

    log("done")
    log(f"wrote {json_path}")
    return {
        "json_path": str(json_path),
        "estimate": estimate,
        "elo": estimate.get("estimated_elo"),
        "protocol": proto_record,
    }


def build_parser() -> argparse.ArgumentParser:
    proto = load_protocol()
    ap = argparse.ArgumentParser(description="Max-Elo gauntlet (pure policy default)")
    ap.add_argument("--ckpt", "--checkpoint", "-c", dest="ckpt", required=False, default=None)
    ap.add_argument("checkpoint_pos", nargs="?", default=None, help="Positional ckpt (legacy)")
    ap.add_argument("out_prefix_pos", nargs="?", default=None, help="Positional out prefix (legacy)")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--mode", choices=("policy", "mcts"), default="policy")
    ap.add_argument("--protocol", default=str(Path(__file__).with_name("protocol.json")))
    ap.add_argument("--device", default=None)
    ap.add_argument("--book", action="store_true", help="Enable opening book (off by default)")
    ap.add_argument("--no-book", action="store_true", help="Explicitly disable book (default)")
    ap.add_argument("--syzygy", action="store_true", help="Enable Syzygy (off by default)")
    ap.add_argument("--no-syzygy", action="store_true", help="Explicitly disable Syzygy (default)")
    ap.add_argument("--movetime", type=float, default=None)
    ap.add_argument("--ply-cap", type=int, default=None)
    ap.add_argument("--games-per-opening-per-color", type=int, default=None)
    ap.add_argument("--elos", type=int, nargs="+", default=None)
    ap.add_argument("--stop-after-bracket", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--sims", type=int, default=200, help="MCTS sims (mcts mode)")
    ap.add_argument("--search-mode", choices=("auto", "puct", "gumbel"), default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    # legacy aliases accepted by shim
    ap.add_argument("--model-config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-levels", type=int, default=None, help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    ckpt = args.ckpt or args.checkpoint_pos
    if not ckpt:
        ap.error("--ckpt / checkpoint required")
    args.ckpt = ckpt
    args.out_prefix = args.out_prefix or args.out_prefix_pos

    # Defaults: book/syzygy OFF unless --book / --syzygy
    if args.no_book:
        args.book = False
    if args.no_syzygy:
        args.syzygy = False

    if args.max_levels is not None and args.elos is not None:
        args.elos = args.elos[: args.max_levels]

    if args.mode == "policy":
        run_policy_elo(args)
    else:
        run_mcts_elo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
