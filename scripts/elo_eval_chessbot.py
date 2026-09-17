#!/usr/bin/env python3
"""Elo gauntlet for Maxlegrec/ChessBot on the maxelo policy protocol.

Card: https://huggingface.co/Maxlegrec/ChessBot
Greedy policy (T=0). Lower T is stronger. No book, no Syzygy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import chess
import chess.engine
import torch

from harness.common import (
    ROOT,
    load_protocol,
    opening_name,
    pick_device,
    resolve_stockfish,
    stockfish_version,
)
from harness import elo as elo_h


REPO = "Maxlegrec/ChessBot"


def load_chessbot(repo, device):
    """Card load, without transformers 5.x from_pretrained finalize.

    AutoModel.from_pretrained leaves this custom class on meta/garbage
    weights. Config + safetensors state_dict matches the published API.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    local = snapshot_download(repo)
    AutoConfig.from_pretrained(local, trust_remote_code=True)
    mod = next(m for n, m in sys.modules.items()
               if n.endswith("modeling_chessbot") and hasattr(m, "ChessBotModel"))
    if not hasattr(mod.ChessBotModel, "all_tied_weights_keys"):
        mod.ChessBotModel.all_tied_weights_keys = {}
    model = mod.ChessBotModel(AutoConfig.from_pretrained(local, trust_remote_code=True))
    missing, unexpected = model.load_state_dict(load_file(str(Path(local) / "model.safetensors")), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ChessBot weight mismatch missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


def chessbot_move(model, board, device, temperature=0.0):
    # Card: get_move_from_fen_no_thinking; T=0 is argmax / strongest.
    t = 0.0 if temperature is None or temperature <= 0 else float(temperature)
    with torch.no_grad():
        uci = model.get_move_from_fen_no_thinking(board.fen(), T=t, device=device)
    if isinstance(uci, dict):
        uci = max(uci.items(), key=lambda kv: kv[1])[0]
    move = chess.Move.from_uci(str(uci))
    if move not in board.legal_moves:
        for suffix in ("q", "n", "r", "b"):
            try:
                alt = chess.Move.from_uci(str(uci) + suffix)
            except ValueError:
                continue
            if alt in board.legal_moves:
                move = alt
                break
    return move, {"source": "chessbot_policy", "temperature": t}


def next_levels(start, protocol_elos):
    ordered = [e for e in protocol_elos if e != start]
    up = [e for e in ordered if e > start]
    down = [e for e in reversed(ordered) if e < start]
    return [start], up, down


def pick_next(score, up, down):
    if score >= 0.5:
        return up.pop(0) if up else None
    return down.pop(0) if down else None


def main() -> int:
    proto = load_protocol()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--out-prefix", default="chessbot_policy")
    p.add_argument("--device", default=None)
    p.add_argument("--start-elo", type=int, default=2050)
    p.add_argument("--elos", type=int, nargs="+", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--movetime", type=float, default=None)
    p.add_argument("--ply-cap", type=int, default=None)
    p.add_argument("--games-per-opening-per-color", type=int, default=None)
    p.add_argument("--stop-after-bracket", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = pick_device(args.device)
    sf_path = resolve_stockfish()
    sf_ver = stockfish_version(sf_path)
    protocol_elos = args.elos or list(proto["elos"])
    if args.start_elo not in protocol_elos:
        protocol_elos = sorted(set(protocol_elos) | {args.start_elo})
    first, up, down = next_levels(args.start_elo, protocol_elos)
    movetime = args.movetime if args.movetime is not None else proto["movetime"]
    ply_cap = args.ply_cap if args.ply_cap is not None else proto["ply_cap"]
    games = (
        args.games_per_opening_per_color
        if args.games_per_opening_per_color is not None
        else proto["games_per_opening_per_color"]
    )
    openings = [list(o) for o in proto["openings"]]
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"elo_eval_{args.out_prefix}.log"
    json_path = out_dir / f"elo_eval_{args.out_prefix}.json"
    if log_path.exists():
        log_path.unlink()
    elo_h.LOG = log_path

    proto_record = {
        "name": proto.get("name"),
        "mode": "policy",
        "model": args.repo,
        "book": False,
        "syzygy": False,
        "temperature": args.temperature,
        "movetime": movetime,
        "ply_cap": ply_cap,
        "games_per_opening_per_color": games,
        "start_elo": args.start_elo,
        "elos": protocol_elos,
        "openings": [opening_name(o) for o in openings],
        "stop_after_bracket": args.stop_after_bracket,
        "sf_path": str(sf_path),
        "sf_version": sf_ver,
        "device": str(device),
        "threads": proto.get("threads", 1),
        "hash": proto.get("hash", 32),
        "note": "External HF ChessBot. Greedy policy per model card (lower T stronger).",
    }
    elo_h.log("start " + json.dumps(proto_record))
    model = load_chessbot(args.repo, device)
    elo_h.log(f"Model loaded on {device}; Stockfish={sf_ver} ({sf_path})")

    summaries, all_games, estimate = [], [], {}

    def write_snapshot():
        json_path.write_text(json.dumps({
            "checkpoint": args.repo,
            "device": str(device),
            "mode": "policy",
            "protocol": proto_record,
            "summaries": summaries,
            "games": all_games,
            "estimate": estimate,
        }, indent=2), encoding="utf-8")

    def play_level(sf_elo):
        elo_h.log(f"begin sf_elo={sf_elo}")
        engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": sf_elo,
            "Threads": proto.get("threads", 1),
            "Hash": proto.get("hash", 32),
        })
        results = []
        try:
            for opening in openings:
                for color in (chess.WHITE, chess.BLACK):
                    for repeat_idx in range(games):
                        r = elo_h.play_one_policy(
                            engine, model, chessbot_move, device, sf_elo, color,
                            opening, movetime, ply_cap, use_book=False,
                            get_book_move=None, get_syzygy_move=None,
                        )
                        r["repeat_idx"] = repeat_idx
                        results.append(r)
                        elo_h.log("game " + json.dumps({
                            "sf_elo": sf_elo, "color": r["model_color"],
                            "opening": r["opening_name"], "repeat_idx": repeat_idx,
                            "result": r["result"], "score": r["score"],
                            "plies": r["plies"], "termination": r["termination"],
                        }))
        finally:
            engine.quit()
        return results

    elo = first[0]
    played = []
    while elo is not None:
        results = play_level(elo)
        played.append(elo)
        summary = elo_h.summarize_results(elo, results)
        summaries.append(summary)
        all_games.extend(results)
        estimate = elo_h.estimate_elo(summaries)
        write_snapshot()
        elo_h.log("summary " + json.dumps(summary))
        elo_h.log("estimate " + json.dumps(estimate))
        if args.stop_after_bracket and estimate.get("lower_bound") is not None and estimate.get("upper_bound") is not None:
            elo_h.log(f"bracketed between {estimate['lower_bound']} and {estimate['upper_bound']}")
            break
        elo = pick_next(summary["score"], up, down)
        if elo in played:
            break

    elo_h.log("done")
    elo_h.log(f"wrote {json_path}")
    print(json.dumps(estimate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
