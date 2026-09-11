#!/usr/bin/env python3
"""270M vs SF19 mistake mining. Searchless model. Reuses audited SF19 soft targets.

Openings come from lichess-org/chess-openings. Book application uses epsilon
exploration (random legal deviation, then leave the book). Model inference
stays greedy / legal-masked, matching the Elo harness.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import signal
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "harness")]

from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
from chess_inference import load_checkpoint  # noqa: E402
from harvest_swa_mistakes import TAG_TO_I, classify_lapse  # noqa: E402
from move_vocab import IDX_TO_UCI, UCI_TO_IDX, VOCAB_SIZE, index_to_move, legal_move_mask  # noqa: E402
from sf19_soft_dataset import (  # noqa: E402
    DEFAULT_TAU,
    encode_board,
    file_sha256,
    label_to_row,
    parse_multipv,
    position_key,
    resolve_sf,
    teacher_fingerprint,
    to_white_abs,
)

# sf19_soft_dataset force-hides the GPU for its CPU labeler. Undo that here.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

LICHESS_OPENINGS_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
LICHESS_OPENING_FILES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")
BENCH_OPENINGS = (
    (),
    ("e2e4", "e7e5"),
    ("d2d4", "d7d5"),
    ("e2e4", "c7c5"),
    ("d2d4", "g8f6"),
    ("e2e4", "e7e6"),
    ("c2c4", "e7e5"),
    ("g1f3", "d7d5"),
    ("e2e4", "c7c6"),
    ("e2e4", "g8f6"),
    ("e2e4", "d7d5"),
    ("e2e4", "g7g6"),
    ("e2e4", "d7d6"),
    ("d2d4", "f7f5"),
    ("d2d4", "e7e6"),
    ("d2d4", "c7c5"),
    ("c2c4", "c7c5"),
    ("c2c4", "e7e6"),
    ("c2c4", "g8f6"),
    ("g1f3", "g8f6"),
    ("g1f3", "c7c5"),
    ("b2b3",),
    ("g2g3",),
    ("f2f4",),
)
THRESHOLDS = {
    "inaccuracy_cp": 75,
    "blunder_cp": 150,
    "major_cp": 300,
    "conversion_best_cp": 200,
    "conversion_model_cp": 50,
    "conversion_drop_cp": 150,
    "off_pv_cp": 50,
    "note": "Tags are labels only. Continuous STM regret is stored separately. Mates are never mapped to cp.",
}
SF_DEPTH = 4
ELO_2500 = {
    "name": "sf19_2500",
    "UCI_LimitStrength": True,
    "UCI_Elo": 2500,
    "Threads": 1,
    "Hash": 16,
    "depth": SF_DEPTH,
}
KEEP_MISTAKE = frozenset({"inaccuracy", "blunder", "conversion", "major"})
NEAR_OPP = ELO_2500
FULL_OPP = ELO_2500
TEACHER = {
    "name": "sf19_teacher",
    "UCI_LimitStrength": False,
    "Threads": 1,
    "Hash": 16,
    "multipv": 4,
    "depth": SF_DEPTH,
    "tau": DEFAULT_TAU,
    "watchdog_s": 2.0,
    "UCI_ShowWDL": True,
    "note": "Depth 4 everywhere. Speed over strength; no 1M/5M escalate.",
}
INFERENCE = {
    "mode": "policy",
    "temperature": 0.0,
    "legal_mask": True,
    "book": False,
    "syzygy": False,
    "search": False,
    "vocab": "compact",
    "vocab_size": 1968,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, path: Path | None = None) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def json_read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def stop_requested(out: Path) -> bool:
    return (out / "STOP").exists()


def install_stop_handlers(out: Path) -> None:
    def _handle(signum, _frame):
        (out / "STOP").write_text(f"signal={signum}\n", encoding="utf-8")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


# --- openings ----------------------------------------------------------------


def pgn_to_uci(pgn: str) -> list[str]:
    text = (pgn or "").strip()
    if not text:
        return []
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        return []
    board = game.board()
    ucis: list[str] = []
    for mv in game.mainline_moves():
        if mv not in board.legal_moves:
            return []
        ucis.append(mv.uci())
        board.push(mv)
    if board.is_checkmate():
        return []
    return ucis


def parse_openings_tsv(text: str, *, source: str) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.lower().startswith("eco"):
            continue
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        eco, name, pgn = parts[0].strip(), parts[1].strip(), parts[2].strip()
        ucis = pgn_to_uci(pgn)
        if len(ucis) < 3:
            continue
        if tuple(ucis) in BENCH_OPENINGS or tuple(ucis[:2]) in BENCH_OPENINGS and len(ucis) == 2:
            continue
        if tuple(ucis[:1]) in BENCH_OPENINGS and len(ucis) == 1:
            continue
        rows.append({
            "eco": eco,
            "name": name,
            "pgn": pgn,
            "uci": ucis,
            "source": source,
        })
    return rows


def download_lichess_openings(cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name in LICHESS_OPENING_FILES:
        dest = cache / name
        if not dest.exists():
            url = f"{LICHESS_OPENINGS_BASE}/{name}"
            urllib.request.urlretrieve(url, dest)
        rows.extend(parse_openings_tsv(dest.read_text(encoding="utf-8"), source=f"lichess-org/chess-openings/{name}"))
    return rows


def explore_opening(ucis: list[str], *, epsilon: float, rng: random.Random) -> dict:
    """Follow the book line; at each ply, with prob epsilon take a random legal alt and stop."""
    board = chess.Board()
    played: list[str] = []
    diverted_at = None
    for i, u in enumerate(ucis):
        try:
            book = chess.Move.from_uci(u)
        except ValueError:
            break
        if book not in board.legal_moves:
            break
        legal = list(board.legal_moves)
        if epsilon > 0 and len(legal) > 1 and rng.random() < float(epsilon):
            alts = [m for m in legal if m != book]
            mv = rng.choice(alts)
            board.push(mv)
            played.append(mv.uci())
            diverted_at = i
            break
        board.push(book)
        played.append(u)
    return {
        "uci": played,
        "book_uci": list(ucis),
        "diverted_at": diverted_at,
        "fen": board.fen(),
        "plies": len(played),
    }


def sample_schedule(
    pool: list[dict],
    *,
    n_openings: int,
    epsilon: float,
    seed: int,
    smoke: bool = False,
) -> dict:
    rng = random.Random(seed)
    by_letter: dict[str, list[dict]] = {}
    for row in pool:
        by_letter.setdefault((row["eco"][:1] or "Z"), []).append(row)
    letters = [k for k in "ABCDE" if by_letter.get(k)]
    chosen: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    while len(chosen) < n_openings and letters:
        letter = letters[len(chosen) % len(letters)]
        cand = rng.choice(by_letter[letter])
        explored = explore_opening(cand["uci"], epsilon=epsilon, rng=rng)
        key = tuple(explored["uci"])
        if len(key) < 2 or key in seen or key in BENCH_OPENINGS:
            continue
        seen.add(key)
        chosen.append({
            "opening_id": f"o{len(chosen):02d}",
            "eco": cand["eco"],
            "name": cand["name"],
            "source": cand["source"],
            "book_uci": cand["uci"],
            "played_uci": explored["uci"],
            "diverted_at": explored["diverted_at"],
            "start_fen": explored["fen"],
            "split": "val" if len(chosen) >= n_openings - max(1, n_openings // 5) else "train",
        })
    opps = [ELO_2500]
    games = []
    for op in chosen:
        for color in ("white", "black"):
            for opp in opps:
                games.append({
                    "game_id": f"{op['opening_id']}_{color}_{opp['name']}",
                    "opening_id": op["opening_id"],
                    "model_color": color,
                    "opponent": opp["name"],
                    "split": op["split"],
                })
    return {
        "seed": seed,
        "epsilon": epsilon,
        "n_openings": len(chosen),
        "n_games": len(games),
        "openings": chosen,
        "games": games,
        "opponents": [ELO_2500],
        "blocked_benchmark": [list(x) for x in BENCH_OPENINGS],
        "dataset": "lichess-org/chess-openings",
        "train_on": "model inaccuracy/blunder + every SF 2500 move",
    }


# --- players -----------------------------------------------------------------


def hash_nnue(sf_path: Path, eval_file: str) -> dict:
    names = [eval_file, Path(eval_file).name] if eval_file else []
    names += ["nn-1a298aa575a0.nnue"]
    search = [sf_path.parent, Path.home() / ".local/share/stockfish", ROOT]
    files = []
    for n in names:
        if not n:
            continue
        p = Path(n)
        if p.is_file():
            files.append(p)
        for d in search:
            cand = d / Path(n).name
            if cand.is_file():
                files.append(cand)
    uniq, seen = [], set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    if not uniq:
        return {"eval_file": eval_file, "sha256": None, "note": "NNUE embedded in SF19 binary"}
    return {"eval_file": str(uniq[0]), "sha256": file_sha256(str(uniq[0]))}


def freeze_players(out: Path, ckpt: Path, *, n_openings: int, epsilon: float, seed: int, smoke: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    install_stop_handlers(out)
    dest = out / "players" / "270m.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != ckpt.stat().st_size:
        shutil.copy2(ckpt, dest)
    ck = torch.load(dest, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()
    sf = Path(resolve_sf())
    sf_fp = teacher_fingerprint(str(sf))
    nnue = hash_nnue(sf, str(sf_fp.get("eval_file") or ""))
    players = {
        "frozen_at": utc_now(),
        "model": {
            "path": str(dest),
            "source": str(ckpt.resolve()),
            "sha256": file_sha256(str(dest)),
            "arch": ck.get("arch", "squares64"),
            "config": cfg,
            "steps": int(ck.get("steps") or 0),
            "n_params": int(ck.get("n_params") or 0),
            "trial_id": ck.get("trial_id"),
            "vocab": "compact",
            "vocab_size": int(VOCAB_SIZE),
            "inference": INFERENCE,
        },
        "stockfish": {
            **sf_fp,
            "nnue": nnue,
            "play": ELO_2500,
            "teacher": TEACHER,
        },
        "thresholds": THRESHOLDS,
        "soft_targets": {
            "module": "scripts/sf19_soft_dataset.py",
            "tau": DEFAULT_TAU,
            "k": 8,
            "note": "Miner does not invent temperature or mate→cp conversion.",
        },
    }
    json_write(out / "players.json", players)
    pool = download_lichess_openings(out / "openings")
    schedule = sample_schedule(pool, n_openings=n_openings, epsilon=epsilon, seed=seed, smoke=smoke)
    json_write(out / "schedule.json", schedule)
    json_write(out / "config.json", {
        "experiment": "exp272_sf19_corrections",
        "ckpt": str(ckpt),
        "out": str(out),
        "n_openings": n_openings,
        "epsilon": epsilon,
        "seed": seed,
        "smoke": smoke,
        "ply_cap": 160,
        "games": schedule["n_games"],
        "launch": (
            "MOVE_VOCAB_VERSION=compact STOCKFISH_PATH=$HOME/.local/bin/stockfish-19 "
            f"python -u experiments/exp272_sf19_corrections.py go --out-dir {out}"
        ),
    })
    log(
        f"froze model sha={players['model']['sha256'][:12]} steps={players['model']['steps']} "
        f"sf={sf_fp.get('uci_name')} openings={schedule['n_openings']} games={schedule['n_games']} "
        f"pool={len(pool)} eps={epsilon}",
        out / "pipeline.log",
    )
    return players


# --- collect -----------------------------------------------------------------


def _probe_from_logits(logits, mask, value_logits, *, top_k: int = 8) -> dict:
    logits = logits.float()
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    idx = int(logits.argmax().item())
    k = min(int(top_k), max(int(mask.sum().item()), 1))
    topv, topi = torch.topk(probs, k)
    wdl = F.softmax(value_logits.float(), dim=-1).tolist() if value_logits.shape[-1] == 3 else None
    return {
        "move_idx": idx,
        "move_uci": IDX_TO_UCI[idx],
        "p": float(probs[idx].item()),
        "top": [
            {"idx": int(i), "uci": IDX_TO_UCI[int(i)], "p": float(p)}
            for i, p in zip(topi.tolist(), topv.tolist())
        ],
        "wdl": wdl,
    }


@torch.no_grad()
def probe_policy(model, board: chess.Board, device: torch.device, *, top_k: int = 8) -> dict:
    return probe_policy_batch(model, [board], device, top_k=top_k)[0]


@torch.no_grad()
def probe_policy_batch(model, boards: list, device: torch.device, *, top_k: int = 8) -> list[dict]:
    if not boards:
        return []
    inp = batch_boards_to_fused_token_ids(boards, device)
    masks = torch.stack([legal_move_mask(b).to(device) for b in boards])
    inp["legal_mask"] = masks
    out = model(inp)
    logits = out["policy_logits"]
    vals = out["value_logits"]
    return [
        _probe_from_logits(logits[i], masks[i], vals[i], top_k=top_k)
        for i in range(len(boards))
    ]


def apply_uci(board: chess.Board, ucis: Iterable[str]) -> None:
    for u in ucis:
        mv = chess.Move.from_uci(u)
        if mv not in board.legal_moves:
            raise ValueError(f"illegal opening move {u} in {board.fen()}")
        board.push(mv)


def configure_play_engine(engine: chess.engine.SimpleEngine, opp: dict) -> None:
    cfg = {"Threads": int(opp["Threads"]), "Hash": int(opp["Hash"])}
    if opp.get("UCI_LimitStrength"):
        cfg["UCI_LimitStrength"] = True
        cfg["UCI_Elo"] = int(opp["UCI_Elo"])
    else:
        cfg["UCI_LimitStrength"] = False
    engine.configure(cfg)


def _pos_meta(board: chess.Board) -> dict:
    return {
        "ply": len(board.move_stack),
        "fen": board.fen(),
        "uci_history": [m.uci() for m in board.move_stack],
        "halfmove": board.halfmove_clock,
        "fullmove": board.fullmove_number,
        "is_repetition": board.is_repetition(2),
        "can_claim_fifty": board.can_claim_fifty_moves(),
        "can_claim_threefold": board.can_claim_threefold_repetition(),
        "history_dependent": bool(
            board.halfmove_clock >= 40
            or board.is_repetition(2)
            or board.can_claim_fifty_moves()
            or board.can_claim_threefold_repetition()
        ),
    }


def hard_teacher_row(board: chess.Board, move_uci: str) -> dict | None:
    enc = encode_board(board)
    if enc is None or move_uci not in UCI_TO_IDX:
        return None
    arr, turn, castling, ep = enc
    idx = int(UCI_TO_IDX[move_uci])
    soft_i = np.full(8, -1, dtype=np.int64)
    soft_p = np.zeros(8, dtype=np.float32)
    soft_i[0] = idx
    soft_p[0] = 1.0
    return {
        "board_array": arr,
        "turn": int(turn),
        "castling": int(castling),
        "ep_square": int(ep),
        "move_idx": idx,
        "soft_indices": soft_i.tolist(),
        "soft_probs": soft_p.tolist(),
        "teacher_uci": move_uci,
    }


def tag_model_decision(engine, decision: dict, *, depth: int, watchdog_s: float) -> dict | None:
    board = chess.Board(decision["fen"])
    try:
        model_mv = chess.Move.from_uci(decision["model_move"])
    except ValueError:
        return None
    if model_mv not in board.legal_moves:
        return None
    row = analyze_board_depth(
        engine, board, depth=depth, multipv=3, tau=DEFAULT_TAU, watchdog_s=watchdog_s,
    )
    if row is None:
        return None
    teacher_mv = index_to_move(int(row["move_idx"]))
    if teacher_mv not in board.legal_moves:
        return None
    teacher = analyse_root(engine, board, teacher_mv, depth=depth, watchdog_s=watchdog_s)
    model = analyse_root(engine, board, model_mv, depth=depth, watchdog_s=watchdog_s)
    in_pv = int(decision["model_idx"]) in {int(x) for x in np.asarray(row["soft_indices"]).reshape(-1) if int(x) >= 0}
    regret = regret_from_roots(teacher, model, in_pv=in_pv)
    return {
        "teacher_uci": teacher_mv.uci(),
        "in_pv": in_pv,
        "regret": regret,
        "soft": _soft_view(row),
    }


def play_game(
    *,
    probe_fn,
    engine: chess.engine.SimpleEngine,
    spec: dict,
    opening: dict,
    opp: dict,
    ply_cap: int,
    ckpt_sha: str,
) -> dict:
    board = chess.Board()
    apply_uci(board, opening["played_uci"])
    decisions: list[dict] = []
    sf_moves: list[dict] = []
    model_color = chess.WHITE if spec["model_color"] == "white" else chess.BLACK
    limit = chess.engine.Limit(depth=int(opp.get("depth") or SF_DEPTH), time=1.0)
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            probe = probe_fn(board)
            mv = chess.Move.from_uci(probe["move_uci"])
            if mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
                probe["fallback"] = True
            rec = _pos_meta(board)
            rec.update({
                "model_move": probe["move_uci"],
                "model_idx": probe["move_idx"],
                "model_p": probe["p"],
                "top": probe["top"],
                "model_wdl": probe["wdl"],
            })
            decisions.append(rec)
            board.push(mv)
        else:
            rec = _pos_meta(board)
            played = engine.play(board, limit).move
            if played not in board.legal_moves:
                played = next(iter(board.legal_moves))
            rec["sf_move"] = played.uci()
            rec["sf_idx"] = int(UCI_TO_IDX.get(played.uci(), -1))
            sf_moves.append(rec)
            board.push(played)

    outcome = board.outcome(claim_draw=True)
    truncated = outcome is None
    if truncated:
        result = "*"
        termination = "truncated"
        winner = None
    else:
        result = board.result(claim_draw=True)
        termination = outcome.termination.name
        winner = (
            "white" if outcome.winner is chess.WHITE
            else "black" if outcome.winner is chess.BLACK
            else None
        )
    game = chess.pgn.Game()
    game.headers["Event"] = "exp272"
    game.headers["White"] = "270m" if spec["model_color"] == "white" else opp["name"]
    game.headers["Black"] = "270m" if spec["model_color"] == "black" else opp["name"]
    game.headers["Result"] = result
    game.headers["Opening"] = opening["name"]
    game.headers["ECO"] = opening["eco"]
    node = game
    tmp = chess.Board()
    for mv in board.move_stack:
        node = node.add_variation(mv)
        tmp.push(mv)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return {
        "game_id": spec["game_id"],
        "opening_id": spec["opening_id"],
        "split": spec["split"],
        "model_color": spec["model_color"],
        "opponent": opp,
        "opening": {
            "eco": opening["eco"],
            "name": opening["name"],
            "book_uci": opening["book_uci"],
            "played_uci": opening["played_uci"],
            "diverted_at": opening["diverted_at"],
        },
        "ckpt_sha256": ckpt_sha,
        "result": result,
        "winner": winner,
        "termination": termination,
        "truncated": truncated,
        "plies": len(board.move_stack),
        "final_fen": board.fen(),
        "pgn": game.accept(exporter),
        "decisions": decisions,
        "sf_moves": sf_moves,
        "n_model_decisions": len(decisions),
        "n_sf_moves": len(sf_moves),
        "collected_at": utc_now(),
    }


def completed_game_ids(out: Path) -> set[str]:
    path = out / "games.jsonl"
    seen: set[str] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("game_id"):
            seen.add(row["game_id"])
    return seen


def ensure_2500_schedule(out: Path) -> dict:
    sch = json_read(out / "schedule.json")
    if not sch:
        raise SystemExit("run freeze first")
    if sch.get("opponents") == [ELO_2500] and all(
        g.get("opponent") == ELO_2500["name"] for g in sch.get("games") or []
    ):
        return sch
    games = []
    for op in sch["openings"]:
        for color in ("white", "black"):
            games.append({
                "game_id": f"{op['opening_id']}_{color}_{ELO_2500['name']}",
                "opening_id": op["opening_id"],
                "model_color": color,
                "opponent": ELO_2500["name"],
                "split": op["split"],
            })
    sch["opponents"] = [ELO_2500]
    sch["games"] = games
    sch["n_games"] = len(games)
    sch["train_on"] = "model inaccuracy/blunder + every SF 2500 move"
    json_write(out / "schedule.json", sch)
    return sch


def example_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def sample_more_jobs(pool: list[dict], *, n: int, seq: int, epsilon: float, rng: random.Random) -> tuple[list[dict], dict]:
    jobs, extra = [], {}
    tries = 0
    while len(jobs) < n and tries < n * 8:
        tries += 1
        cand = rng.choice(pool)
        explored = explore_opening(cand["uci"], epsilon=epsilon, rng=rng)
        if len(explored["uci"]) < 2:
            continue
        oid = f"x{seq + len(jobs):06d}"
        color = "white" if rng.random() < 0.5 else "black"
        extra[oid] = {
            "opening_id": oid,
            "eco": cand["eco"],
            "name": cand["name"],
            "source": cand.get("source"),
            "book_uci": cand["uci"],
            "played_uci": explored["uci"],
            "diverted_at": explored["diverted_at"],
            "start_fen": explored["fen"],
            "split": "val" if (seq + len(jobs)) % 5 == 0 else "train",
        }
        jobs.append({
            "game_id": f"{oid}_{color}_{ELO_2500['name']}",
            "opening_id": oid,
            "model_color": color,
            "opponent": ELO_2500["name"],
            "split": extra[oid]["split"],
        })
    return jobs, extra


def run_collect(
    out: Path,
    *,
    device: str | None = None,
    workers: int = 12,
    target: int = 100_000,
    epsilon: float = 0.20,
    seed: int = 272,
) -> dict:
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    install_stop_handlers(out)
    players = json_read(out / "players.json")
    schedule = ensure_2500_schedule(out)
    if not players:
        raise SystemExit("run freeze first")
    log_path = out / "pipeline.log"
    games_path = out / "games.jsonl"
    pgn_path = out / "games.pgn"
    ex_path = out / "examples.jsonl"
    done = completed_game_ids(out)
    pending = [g for g in schedule["games"] if g["game_id"] not in done]
    have = example_count(ex_path)
    log(
        f"collect have={have} target={target} pending_sched={len(pending)} "
        f"done_games={len(done)} workers={workers} opp=2500",
        log_path,
    )
    if have >= int(target):
        return {"done": len(done), "have": have, "target": target}

    import torch as _torch
    if device:
        dev = _torch.device(device)
    elif _torch.cuda.is_available():
        dev = _torch.device("cuda")
    else:
        dev = _torch.device("cpu")
    if dev.type == "cuda" and not _torch.cuda.is_available():
        log("cuda requested but unavailable; falling back to cpu", log_path)
        dev = _torch.device("cpu")
    log(f"collect device={dev}", log_path)
    model = load_checkpoint(players["model"]["path"], device=dev)
    model.eval()
    openings = {o["opening_id"]: o for o in schedule["openings"]}
    infer_q: queue.Queue = queue.Queue()
    write_lock = threading.Lock()
    n_new = 0
    n_sf = 0
    n_mist = 0
    t0 = time.time()

    def infer_loop() -> None:
        while True:
            item = infer_q.get()
            if item is None:
                break
            batch = [item]
            while len(batch) < 16:
                try:
                    nxt = infer_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    infer_q.put(None)
                    break
                batch.append(nxt)
            boards = [chess.Board(x["fen"]) for x in batch]
            try:
                results = probe_policy_batch(model, boards, dev)
            except Exception as e:
                results = [{"error": str(e)} for _ in batch]
            for x, r in zip(batch, results):
                x["result"] = r
                x["ev"].set()

    infer_t = threading.Thread(target=infer_loop, daemon=True)
    infer_t.start()

    def probe_fn(board: chess.Board) -> dict:
        req = {"fen": board.fen(), "ev": threading.Event(), "result": None}
        infer_q.put(req)
        if not req["ev"].wait(timeout=30):
            raise TimeoutError("gpu infer timeout")
        if not req["result"] or req["result"].get("error"):
            raise RuntimeError(req["result"])
        return req["result"]

    def emit_examples(game: dict, engine) -> tuple[int, int]:
        got_sf = 0
        got_m = 0
        lines = []
        last_mist_ply = -99
        for rec in game.get("sf_moves") or []:
            if rec.get("history_dependent") or rec.get("sf_idx", -1) < 0:
                continue
            board = chess.Board(rec["fen"])
            row = hard_teacher_row(board, rec["sf_move"])
            if row is None:
                continue
            lines.append({
                "role": "sf2500",
                "game_id": game["game_id"],
                "opening_id": game["opening_id"],
                "split": game["split"],
                "ply": rec["ply"],
                "fen": rec["fen"],
                **row,
            })
            got_sf += 1
        for rec in game.get("decisions") or []:
            if rec.get("history_dependent"):
                continue
            tagged = tag_model_decision(engine, rec, depth=SF_DEPTH, watchdog_s=1.0)
            if tagged is None:
                continue
            tag = tagged["regret"].get("tag")
            if tag not in KEEP_MISTAKE:
                continue
            if rec["ply"] - last_mist_ply < 2:
                continue
            last_mist_ply = rec["ply"]
            lines.append({
                "role": "correction",
                "game_id": game["game_id"],
                "opening_id": game["opening_id"],
                "split": game["split"],
                "ply": rec["ply"],
                "fen": rec["fen"],
                "model_move": rec["model_move"],
                "model_idx": rec["model_idx"],
                **tagged,
            })
            got_m += 1
        if lines:
            with write_lock:
                with ex_path.open("a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(json.dumps(line, default=_jsonable) + "\n")
        return got_sf, got_m

    def play_one(spec: dict) -> dict | None:
        if stop_requested(out):
            return None
        sf = resolve_sf()
        engine = chess.engine.SimpleEngine.popen_uci(sf)
        try:
            configure_play_engine(engine, ELO_2500)
            game = play_game(
                probe_fn=probe_fn,
                engine=engine,
                spec=spec,
                opening=openings[spec["opening_id"]],
                opp=ELO_2500,
                ply_cap=160,
                ckpt_sha=players["model"]["sha256"],
            )
            ns, nm = emit_examples(game, engine)
            game["n_sf_examples"] = ns
            game["n_mistake_examples"] = nm
            slim = {k: game[k] for k in game if k not in {"decisions", "sf_moves"}}
            slim["n_model_decisions"] = game["n_model_decisions"]
            slim["n_sf_moves"] = game["n_sf_moves"]
            with write_lock:
                with games_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(slim, default=str) + "\n")
                with pgn_path.open("a", encoding="utf-8") as f:
                    f.write(game["pgn"].rstrip() + "\n\n")
            log(
                f"game {game['game_id']} {game['result']} {game['termination']} "
                f"sf_ex={ns} mist={nm} have={have + ns + nm}/{target}",
                log_path,
            )
            return {"sf": ns, "mist": nm}
        finally:
            engine.quit()

    opening_pool = download_lichess_openings(out / "openings")
    rng = random.Random(int(seed) + 91)
    seq = 0
    have_n = have

    def run_wave(specs: list[dict]) -> bool:
        nonlocal n_new, n_sf, n_mist, have_n
        if not specs:
            return True
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for result in pool.map(play_one, specs):
                if result is None or stop_requested(out):
                    return False
                n_new += 1
                n_sf += result["sf"]
                n_mist += result["mist"]
                have_n += result["sf"] + result["mist"]
                if have_n >= int(target):
                    return False
        return True

    try:
        if pending and have_n < int(target):
            run_wave(pending)
        while have_n < int(target) and not stop_requested(out):
            need = int(target) - have_n
            n_jobs = min(64, max(int(workers), need // 50 + int(workers)))
            jobs, extra = sample_more_jobs(
                opening_pool, n=n_jobs, seq=seq, epsilon=float(epsilon), rng=rng,
            )
            seq += max(n_jobs, 1)
            openings.update(extra)
            log(f"wave seq={seq} jobs={len(jobs)} have={have_n}/{target}", log_path)
            if not run_wave(jobs):
                break
    finally:
        infer_q.put(None)
        infer_t.join(timeout=5)
    report = {
        "done": len(done) + n_new,
        "new": n_new,
        "have": have_n,
        "sf2500_examples": n_sf,
        "mistake_examples": n_mist,
        "seconds": round(time.time() - t0, 1),
        "target": int(target),
        "workers": workers,
        "pos_per_s": round(have_n / max(time.time() - t0, 1e-6), 2),
    }
    json_write(out / "collect_report.json", report)
    log(f"collect done {report}", log_path)
    return report


# --- analyze / regret --------------------------------------------------------


def sf_limit(*, depth: int, watchdog_s: float) -> chess.engine.Limit:
    return chess.engine.Limit(depth=max(1, int(depth)), time=max(0.2, float(watchdog_s)))


def stream_multipv_depth(engine, board: chess.Board, *, depth: int, multipv: int, watchdog_s: float) -> list[dict]:
    from sf19_soft_dataset import SOFT_K

    k = min(int(multipv), max(board.legal_moves.count(), 1), SOFT_K)
    limit = sf_limit(depth=depth, watchdog_s=watchdog_s)
    out: list[dict] = []
    with engine.analysis(board, limit, multipv=k) as analysis:
        for info in analysis:
            out.append({
                "pv": list(info.get("pv") or []),
                "score": info.get("score"),
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
                "multipv": info.get("multipv"),
                "wdl": info.get("wdl"),
                "upperbound": info.get("upperbound"),
                "lowerbound": info.get("lowerbound"),
            })
    return out


def analyze_board_depth(engine, board: chess.Board, *, depth: int, multipv: int, tau: float, watchdog_s: float):
    from sf19_soft_dataset import SOFT_K

    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return label_to_row(board, {"terminal": True}, tau=tau, nodes_budget=0)
    try:
        infos = stream_multipv_depth(engine, board, depth=depth, multipv=multipv, watchdog_s=watchdog_s)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
        return None
    parsed = parse_multipv(infos, board, k=min(int(multipv), n_legal, SOFT_K), tau=tau)
    if parsed is None:
        return None
    return label_to_row(board, parsed, tau=tau, nodes_budget=0)


def analyse_root(engine, board, move, *, depth: int, watchdog_s: float) -> dict:
    limit = sf_limit(depth=depth, watchdog_s=watchdog_s)
    info = engine.analyse(board, limit, root_moves=[move], info=chess.engine.INFO_ALL)
    if isinstance(info, list):
        info = info[0]
    sc = info.get("score")
    wdl = info.get("wdl")
    wdl_stm = None
    if wdl is not None:
        rel = wdl.relative if hasattr(wdl, "relative") else wdl
        trip = [float(rel.wins), float(rel.draws), float(rel.losses)]
        z = sum(trip) or 1.0
        wdl_stm = [x / z for x in trip]
    if sc is None:
        return {"cp_stm": 0, "mate_stm": 0, "depth": 0, "nodes": 0, "wdl_stm": wdl_stm, "pv": [], "bound": False}
    pov = sc.pov(board.turn)
    if pov.is_mate():
        cp, mate = 0, int(pov.mate() or 0)
    else:
        mate = 0
        raw = pov.score(mate_score=None)
        cp = int(raw) if raw is not None else 0
    return {
        "cp_stm": cp,
        "mate_stm": mate,
        "depth": int(info.get("depth") or 0),
        "nodes": int(info.get("nodes") or 0),
        "wdl_stm": wdl_stm,
        "pv": [m.uci() for m in (info.get("pv") or [])],
        "bound": bool(info.get("upperbound") or info.get("lowerbound")),
    }


def outcome_kind(best_cp: int, best_mate: int, model_cp: int, model_mate: int) -> str:
    def bucket(cp: int, mate: int) -> str:
        if mate > 0 or cp >= 150:
            return "win"
        if mate < 0 or cp <= -150:
            return "loss"
        return "draw"

    b, m = bucket(best_cp, best_mate), bucket(model_cp, model_mate)
    if b == m:
        return "same"
    return f"{b}_to_{m}"


def regret_from_roots(teacher: dict, model: dict, *, in_pv: bool) -> dict:
    lapse = classify_lapse(
        best_cp=teacher["cp_stm"],
        best_mate=teacher["mate_stm"],
        model_cp=model["cp_stm"],
        model_mate=model["mate_stm"],
        model_in_pv=in_pv,
    )
    if teacher["mate_stm"] or model["mate_stm"]:
        drop_cp = None
        regret_kind = "mate"
    else:
        drop_cp = int(teacher["cp_stm"]) - int(model["cp_stm"])
        regret_kind = "cp"
    return {
        "teacher_cp_stm": teacher["cp_stm"],
        "teacher_mate_stm": teacher["mate_stm"],
        "teacher_wdl_stm": teacher.get("wdl_stm"),
        "model_cp_stm": model["cp_stm"],
        "model_mate_stm": model["mate_stm"],
        "model_wdl_stm": model.get("wdl_stm"),
        "drop_cp": drop_cp,
        "regret_kind": regret_kind,
        "tag": lapse["tag"],
        "lapse_kind": lapse["kind"],
        "missed_mate": bool(teacher["mate_stm"] > 0 and model["mate_stm"] <= 0),
        "allowed_mate": bool(model["mate_stm"] < 0 and teacher["mate_stm"] >= 0),
        "outcome_transition": outcome_kind(
            teacher["cp_stm"], teacher["mate_stm"], model["cp_stm"], model["mate_stm"]
        ),
        "best_is_draw": bool(teacher["mate_stm"] == 0 and abs(teacher["cp_stm"]) < 30),
    }


def analyze_decision(engine, decision: dict, *, depth: int, multipv: int, tau: float, watchdog_s: float) -> dict | None:
    board = chess.Board(decision["fen"])
    try:
        model_mv = chess.Move.from_uci(decision["model_move"])
    except ValueError:
        return None
    if model_mv not in board.legal_moves:
        return None
    row = analyze_board_depth(engine, board, depth=depth, multipv=multipv, tau=tau, watchdog_s=watchdog_s)
    if row is None:
        return None
    soft_i = [int(x) for x in np.asarray(row["soft_indices"]).reshape(-1) if int(x) >= 0]
    in_pv = int(decision["model_idx"]) in set(soft_i)
    teacher_mv = index_to_move(int(row["move_idx"]))
    if teacher_mv not in board.legal_moves:
        return None
    teacher = analyse_root(engine, board, teacher_mv, depth=depth, watchdog_s=watchdog_s)
    model = analyse_root(engine, board, model_mv, depth=depth, watchdog_s=watchdog_s)
    parsed_ok = bool(soft_i)
    if not parsed_ok:
        return None
    regret = regret_from_roots(teacher, model, in_pv=in_pv)
    return {
        "soft_row": {k: row[k] for k in row},
        "in_pv": in_pv,
        "explicit_model_eval": (not in_pv),
        "teacher_root": teacher,
        "model_root": model,
        "regret": regret,
        "depth": depth,
        "teacher_uci": teacher_mv.uci(),
        "complete": True,
    }


def unstable(a: dict, b: dict) -> bool:
    ra, rb = a["regret"], b["regret"]
    if ra["teacher_mate_stm"] != rb["teacher_mate_stm"] or ra["model_mate_stm"] != rb["model_mate_stm"]:
        return True
    if a["teacher_uci"] != b["teacher_uci"]:
        return True
    if ra["drop_cp"] is None or rb["drop_cp"] is None:
        return ra["drop_cp"] != rb["drop_cp"]
    return abs(int(ra["drop_cp"]) - int(rb["drop_cp"])) >= 50


def costly(screen: dict) -> bool:
    r = screen["regret"]
    if r["missed_mate"] or r["allowed_mate"]:
        return True
    if r["tag"] in {"inaccuracy", "blunder", "conversion", "major"}:
        return True
    if r["outcome_transition"] != "same":
        return True
    return False


def load_games(out: Path) -> list[dict]:
    path = out / "games.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def analyzed_keys(out: Path) -> set[str]:
    path = out / "analysis.jsonl"
    seen: set[str] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen.add(f"{row['game_id']}:{row['ply']}")
    return seen


def run_analyze(out: Path, *, workers: int = 2, smoke: bool = False) -> dict:
    install_stop_handlers(out)
    log_path = out / "pipeline.log"
    games = load_games(out)
    done = analyzed_keys(out)
    jobs = []
    for g in games:
        for d in g.get("decisions") or []:
            key = f"{g['game_id']}:{d['ply']}"
            if key in done:
                continue
            jobs.append((g, d))
    if smoke:
        jobs = jobs[:24]
    log(f"analyze pending={len(jobs)} done={len(done)} workers={workers}", log_path)
    if not jobs:
        return {"pending": 0, "done": len(done)}

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({
        "Threads": 1,
        "Hash": int(TEACHER["Hash"]),
        "UCI_LimitStrength": False,
        "UCI_ShowWDL": True,
    })
    t0 = time.time()
    n = 0
    counts = {"screen": 0, "verify": 0, "escalate": 0, "uncertain": 0, "verified": 0, "rejected": 0}
    try:
        for g, d in jobs:
            if stop_requested(out):
                log("analyze stop requested", log_path)
                break
            screen = analyze_decision(
                engine, d,
                depth=TEACHER["depth"],
                multipv=TEACHER["multipv"],
                tau=TEACHER["tau"],
                watchdog_s=TEACHER["watchdog_s"],
            )
            record = {
                "key": f"{g['game_id']}:{d['ply']}",
                "game_id": g["game_id"],
                "opening_id": g["opening_id"],
                "split": g["split"],
                "ply": d["ply"],
                "fen": d["fen"],
                "model_move": d["model_move"],
                "model_idx": d["model_idx"],
                "model_p": d["model_p"],
                "history_dependent": d.get("history_dependent", False),
                "ckpt_sha256": g["ckpt_sha256"],
                "opponent": g["opponent"]["name"],
                "passes": {},
                "status": "rejected",
            }
            counts["screen"] += 1
            if screen is None:
                record["status"] = "rejected"
                record["reason"] = "screen_failed"
                counts["rejected"] += 1
            else:
                record["passes"]["screen"] = _pass_view(screen)
                record["explicit_model_eval"] = screen["explicit_model_eval"]
                record["status"] = "verified"
                record["in_pv"] = screen["in_pv"]
                record["teacher_uci"] = screen["teacher_uci"]
                record["regret"] = screen["regret"]
                record["soft"] = _soft_view(screen["soft_row"])
                record["depth"] = screen["depth"]
                counts["verified"] += 1
            with (out / "analysis.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=_jsonable) + "\n")
            n += 1
            if n % 25 == 0:
                log(f"analyze {n}/{len(jobs)} {counts}", log_path)
    finally:
        engine.quit()
    seconds = max(time.time() - t0, 1e-6)
    report = {
        **counts,
        "new": n,
        "seconds": round(seconds, 1),
        "pos_per_s": round(n / seconds, 3),
        "done_total": len(done) + n,
    }
    json_write(out / "analyze_report.json", report)
    log(f"analyze done {report}", log_path)
    return report


def _jsonable(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def _pass_view(p: dict) -> dict:
    return {
        "in_pv": p["in_pv"],
        "explicit_model_eval": p["explicit_model_eval"],
        "teacher_uci": p["teacher_uci"],
        "teacher_root": p["teacher_root"],
        "model_root": p["model_root"],
        "regret": p["regret"],
        "depth": p["depth"],
    }


def _soft_view(row: dict) -> dict:
    keep = (
        "move_idx", "cp", "mate", "soft_indices", "soft_probs", "soft_cps", "soft_mates",
        "label_depth", "wdl", "nodes", "tau", "nodes_budget", "phase",
    )
    return {k: _jsonable(row[k]) for k in keep if k in row}


# --- assemble / report -------------------------------------------------------


def load_analysis(out: Path) -> list[dict]:
    path = out / "analysis.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_blocked_hashes() -> set[str]:
    keys = set()
    for p in (
        ROOT / "outputs/sf19_ft/overnight_20260908/val_manifest_soft.json",
        ROOT / "outputs/sf19_ft/overnight_20260908/val_manifest_deep.json",
        ROOT / "outputs/sf19_ft/overnight_20260908/val_manifest_replay.json",
        ROOT / "outputs/exp271_mix/dataset_manifest.json",
    ):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in data if isinstance(data, list) else data.get("holdout") or data.get("hashes") or []:
            if isinstance(item, str):
                keys.add(item)
    return keys


def load_examples(out: Path) -> list[dict]:
    path = out / "examples.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def assemble(out: Path, *, max_corr_per_game: int = 12, max_replay_per_game: int = 4) -> dict:
    rows = load_examples(out) or load_analysis(out)
    blocked = load_blocked_hashes()
    accepted, replay, rejected = [], [], []
    seen_pos: set[str] = set()
    for r in rows:
        board = chess.Board(r["fen"])
        key = position_key(board)
        if key in seen_pos or key in blocked:
            rejected.append({**r, "exclude": "duplicate_or_holdout"})
            continue
        teacher_u = r.get("teacher_uci")
        legal = {m.uci() for m in board.legal_moves}
        if not teacher_u or teacher_u not in legal:
            rejected.append({**r, "exclude": "illegal_teacher"})
            continue
        role = r.get("role")
        if role == "sf2500":
            replay.append(r)
        elif role == "correction" or (r.get("regret") or {}).get("tag") in KEEP_MISTAKE:
            accepted.append(r)
            r["role"] = "correction"
        else:
            rejected.append({**r, "exclude": "not_keep"})
            continue
        seen_pos.add(key)
        r["pos_key"] = key

    pack = accepted + replay
    train = [r for r in pack if r.get("split") != "val"]
    val = [r for r in pack if r.get("split") == "val"]
    dest = out / "pack"
    dest.mkdir(parents=True, exist_ok=True)
    json_write(dest / "accepted.json", {
        "n_correction": len(accepted),
        "n_replay": len(replay),
        "n_train": len(train),
        "n_val": len(val),
        "rows": pack,
    })
    report = {
        "analyzed": len(rows),
        "verified": sum(1 for r in rows if r.get("status") == "verified"),
        "uncertain": sum(1 for r in rows if r.get("status") == "uncertain"),
        "rejected_status": sum(1 for r in rows if r.get("status") == "rejected"),
        "accepted_corrections": len(accepted),
        "replay": len(replay),
        "excluded_after_verify": len(rejected),
        "train": len(train),
        "val": len(val),
        "thresholds": THRESHOLDS,
    }
    json_write(out / "assemble_report.json", report)
    return report


def pick_gallery(rows: list[dict], n: int = 30) -> list[dict]:
    buckets = {
        "harmless": [],
        "tactical_blunder": [],
        "missed_win": [],
        "defensive": [],
        "endgame": [],
        "other": [],
    }
    for r in rows:
        if r.get("status") != "verified" or "regret" not in r:
            continue
        rg = r["regret"]
        fen = r["fen"]
        pieces = sum(ch.isalpha() for ch in fen.split()[0])
        if rg.get("tag") in {"ok", "off_pv", "disagree"}:
            buckets["harmless"].append(r)
        elif rg.get("tag") in {"blunder", "major"} and not rg.get("missed_mate"):
            buckets["tactical_blunder"].append(r)
        elif rg.get("missed_mate") or rg.get("outcome_transition", "").startswith("win_to"):
            buckets["missed_win"].append(r)
        elif rg.get("allowed_mate") or rg.get("outcome_transition") == "draw_to_loss":
            buckets["defensive"].append(r)
        elif pieces <= 14:
            buckets["endgame"].append(r)
        else:
            buckets["other"].append(r)
    out: list[dict] = []
    order = ["tactical_blunder", "missed_win", "defensive", "endgame", "harmless", "other"]
    i = 0
    while len(out) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            r = buckets[k].pop(0)
            out.append({
                "category": k,
                "game_id": r["game_id"],
                "fen": r["fen"],
                "model_move": r["model_move"],
                "teacher_uci": r.get("teacher_uci"),
                "regret": r["regret"],
                "in_pv": r.get("in_pv"),
                "explain": _explain(r, k),
            })
        i += 1
    return out


def _explain(r: dict, cat: str) -> str:
    rg = r["regret"]
    return (
        f"{cat}: model {r['model_move']} vs teacher {r.get('teacher_uci')}; "
        f"tag={rg.get('tag')} drop_cp={rg.get('drop_cp')} "
        f"teacher={rg.get('teacher_cp_stm')}cp/{rg.get('teacher_mate_stm')}m "
        f"model={rg.get('model_cp_stm')}cp/{rg.get('model_mate_stm')}m "
        f"transition={rg.get('outcome_transition')}"
    )


def write_reports(out: Path) -> dict:
    games = load_games(out)
    analysis = load_analysis(out)
    collect = json_read(out / "collect_report.json", {})
    analyze = json_read(out / "analyze_report.json", {})
    assembled = json_read(out / "assemble_report.json") or assemble(out)
    gallery = pick_gallery(analysis, 30)
    json_write(out / "gallery_30.json", gallery)
    verified = assembled.get("accepted_corrections", 0)
    secs = float(analyze.get("seconds") or 0)
    per_1k = None
    if verified > 0 and secs > 0:
        per_1k = {
            "seconds": round(secs / verified * 1000, 1),
            "hours": round(secs / verified * 1000 / 3600, 2),
            "note": "From this run's wall time / accepted corrections. Screening-heavy; 1M/5M escalate adds variance.",
        }
    terms: dict[str, int] = {}
    for g in games:
        terms[g["termination"]] = terms.get(g["termination"], 0) + 1
    completion = {
        "games": len(games),
        "target": json_read(out / "schedule.json", {}).get("n_games"),
        "model_decisions": sum(g.get("n_model_decisions", 0) for g in games),
        "terminations": terms,
        "truncated": sum(1 for g in games if g.get("truncated")),
        "results": {
            "1-0": sum(1 for g in games if g["result"] == "1-0"),
            "0-1": sum(1 for g in games if g["result"] == "0-1"),
            "1/2-1/2": sum(1 for g in games if g["result"] == "1/2-1/2"),
            "*": sum(1 for g in games if g["result"] == "*"),
        },
        "by_opponent": {},
    }
    for name in ("sf19_near_2050", "sf19_full"):
        gs = [g for g in games if g.get("opponent", {}).get("name") == name]
        if not gs:
            continue
        w = sum(1 for g in gs if (g["winner"] == "white" and g["model_color"] == "white") or (g["winner"] == "black" and g["model_color"] == "black"))
        d = sum(1 for g in gs if g["result"] == "1/2-1/2")
        l = sum(1 for g in gs if g["result"] in {"1-0", "0-1"} and g["winner"] and g["winner"] != g["model_color"])
        completion["by_opponent"][name] = {"games": len(gs), "w": w, "d": d, "l": l}
    json_write(out / "completion_report.json", completion)
    (out / "TRAIN_PLAN.md").write_text(_train_plan_text(out, assembled, per_1k), encoding="utf-8")
    summary = {
        "collect": collect,
        "analyze": analyze,
        "assemble": assembled,
        "completion": completion,
        "gallery_n": len(gallery),
        "cost_per_1000_corrections": per_1k,
    }
    json_write(out / "SUMMARY.json", summary)
    log(f"report {json.dumps(summary, default=str)[:800]}", out / "pipeline.log")
    return summary


def _train_plan_text(out: Path, assembled: dict, cost: dict | None) -> str:
    return f"""# exp272 control vs correction plan

Do **not** auto-start this. Review the collection report first.

## Init

- Same 270M weights: `{out / "players" / "270m.pt"}` (see `players.json` SHA-256).
- Optimizer: re-init Polar-NorMuon (public/weights-only ckpt has no optimizer), same LR/batch as exp271 student train (`batch=64`, muon 2e-3 cosine, adam aux 3e-5).
- No 99M distillation.

## Arms

| Arm | Mix | Steps |
|---|---|---|
| Control | 100% existing SF19 baseline (`outputs/exp271_mix/soft_cache.pt`) | 2000–4000 |
| Correction | 90% same baseline + 10% verified corrections from `{out / "pack" / "accepted.json"}` | 2000–4000 |

Correction-set reuse: the 10% slice will repeat if the accepted set is smaller than 10% of `steps × batch`. Document the repeat factor in the run log.

## Held out

- Val openings in `schedule.json` (`split=val`).
- Promotion / held-out protocol openings never used as collection seeds.
- History-dependent rows excluded from board-only training.

## Eval

- Unseen collected val games (replay greedy, no search).
- Established paired-opening benchmark (`harness/protocol.json`), 32 games @ 2050 and 2150.
- Do not publish or replace the incumbent from this FT.

## This run

- accepted corrections: {assembled.get("accepted_corrections")}
- replay: {assembled.get("replay")}
- cost/1000: {cost}
"""


def default_ckpt() -> Path:
    for p in (
        ROOT / "outputs/exp271_distill_99m/hf_upload/latest.pt",
        ROOT / "outputs/exp271_distill_99m/step_045000.pt",
        ROOT / "outputs/hf_models/270m/latest.pt",
    ):
        if p.exists():
            return p
    raise SystemExit("no 270M checkpoint found")


SOURCE_EXP272 = 5


def pack_examples(out: Path) -> dict:
    """Pack examples.jsonl into train/val soft caches. Mixed hard one-hot + MultiPV soft."""
    from sf19_soft_dataset import phase_from_board

    path = out / "examples.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = []
    seen: set[str] = set()
    stats = {"read": 0, "sf2500": 0, "correction": 0, "dup": 0, "bad": 0, "train": 0, "val": 0}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        raw = json.loads(line)
        stats["read"] += 1
        role = raw.get("role")
        fen = raw.get("fen")
        if not fen:
            stats["bad"] += 1
            continue
        board = chess.Board(fen)
        key = position_key(board)
        if key in seen:
            stats["dup"] += 1
            continue
        if role == "correction":
            soft = raw.get("soft") or {}
            si = soft.get("soft_indices") or raw.get("soft_indices")
            sp = soft.get("soft_probs") or raw.get("soft_probs")
            mid = int(soft.get("move_idx", raw.get("move_idx", -1)))
            cp = int(soft.get("cp", 0) or 0)
            mate = int(soft.get("mate", 0) or 0)
            wdl = soft.get("wdl")
            value_valid = 1 if wdl is not None or cp or mate else 0
        elif role == "sf2500":
            si = raw.get("soft_indices")
            sp = raw.get("soft_probs")
            mid = int(raw.get("move_idx", -1))
            cp, mate, wdl, value_valid = 0, 0, None, 0
        else:
            stats["bad"] += 1
            continue
        if si is None or sp is None or mid < 0:
            stats["bad"] += 1
            continue
        si = np.asarray(si, dtype=np.int64).reshape(-1)
        sp = np.asarray(sp, dtype=np.float32).reshape(-1)
        if si[0] < 0 or float(sp[0]) <= 0:
            stats["bad"] += 1
            continue
        enc = encode_board(board)
        if enc is None:
            stats["bad"] += 1
            continue
        arr, turn, castling, ep = enc
        if wdl is None:
            wdl = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        else:
            wdl = np.asarray(wdl, dtype=np.float32).reshape(3)
        rows.append({
            "board_array": np.asarray(arr, dtype=np.int8),
            "turn": np.int8(turn),
            "castling": np.int8(castling),
            "ep_square": np.int8(ep),
            "move_idx": np.int64(mid),
            "cp": np.int32(cp),
            "mate": np.int32(mate),
            "soft_indices": si[:8] if si.size >= 8 else np.pad(si, (0, 8 - si.size), constant_values=-1),
            "soft_probs": sp[:8] if sp.size >= 8 else np.pad(sp, (0, 8 - sp.size)),
            "wdl": wdl,
            "value_valid": np.int8(value_valid),
            "phase": np.int8(phase_from_board(arr)),
            "label_depth": np.int16(4),
            "split": 1 if raw.get("split") == "val" else 0,
            "role": role,
        })
        seen.add(key)
        stats[role] = stats.get(role, 0) + 1
    if not rows:
        raise SystemExit("no packable rows")

    def to_cache(subset: list[dict]) -> dict:
        return {
            "board_array": torch.from_numpy(np.stack([r["board_array"] for r in subset])),
            "turn": torch.tensor([int(r["turn"]) for r in subset], dtype=torch.int8),
            "castling": torch.tensor([int(r["castling"]) for r in subset], dtype=torch.int8),
            "ep_square": torch.tensor([int(r["ep_square"]) for r in subset], dtype=torch.int8),
            "move_idx": torch.tensor([int(r["move_idx"]) for r in subset], dtype=torch.int64),
            "cp": torch.tensor([int(r["cp"]) for r in subset], dtype=torch.int32),
            "mate": torch.tensor([int(r["mate"]) for r in subset], dtype=torch.int32),
            "soft_indices": torch.from_numpy(np.stack([r["soft_indices"] for r in subset])),
            "soft_probs": torch.from_numpy(np.stack([r["soft_probs"] for r in subset])),
            "wdl": torch.from_numpy(np.stack([r["wdl"] for r in subset])),
            "value_valid": torch.tensor([int(r["value_valid"]) for r in subset], dtype=torch.int8),
            "phase": torch.tensor([int(r["phase"]) for r in subset], dtype=torch.int8),
            "label_depth": torch.tensor([int(r["label_depth"]) for r in subset], dtype=torch.int16),
            "source": torch.full((len(subset),), SOURCE_EXP272, dtype=torch.int8),
        }

    train_rows = [r for r in rows if int(r["split"]) == 0]
    val_rows = [r for r in rows if int(r["split"]) == 1]
    dest = out / "pack"
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(to_cache(train_rows), dest / "soft_cache.pt")
    if val_rows:
        torch.save(to_cache(val_rows), dest / "val_cache.pt")
    stats["train"] = len(train_rows)
    stats["val"] = len(val_rows)
    json_write(dest / "pack_report.json", {
        **stats,
        "note": (
            "sf2500 rows are one-hot hard labels (the 2500 move). "
            "correction rows are depth-4 MultiPV soft (tau=120) from the audited converter."
        ),
    })
    log(f"packed {stats}", out / "pipeline.log")
    return stats


def run_train(out: Path, *, max_steps: int = 4000, batch_size: int = 64, minutes: float = 180.0) -> dict:
    from autoresearch_8gb.train_trial import train_trial
    from chess_squares64 import DEFAULT_270M_SQUARES64_CONFIG, EXPECTED_270M_PARAMS

    pack = out / "pack" / "soft_cache.pt"
    if not pack.exists():
        pack_examples(out)
    ckpt = default_ckpt()
    init = out / "pack" / "init_270m.pt"
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch.save({
        "arch": raw.get("arch", "squares64"),
        "config": raw.get("config"),
        "model_state_dict": raw["model_state_dict"],
        "steps": 0,
        "n_params": raw.get("n_params", EXPECTED_270M_PARAMS),
        "trial_id": "exp272_sf19_2500",
        "eval_only": True,
    }, init)
    train_out = ROOT / "outputs" / "exp272_train_2500"
    train_out.mkdir(parents=True, exist_ok=True)
    model = DEFAULT_270M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    trial = {
        "id": "exp272_sf19_2500",
        "arch": "squares64",
        "desc": "270M FT on 2500-move one-hots + model inaccuracy/blunder MultiPV soft. No 99M KD.",
        "model": model,
        "train": {
            "batch_size": int(batch_size),
            "min_batch_size": 8,
            "max_batch_size": int(batch_size),
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.85,
            "soft_temp": 0.0,
            "soft_temp_weight": 0.0,
            "teacher_kd_frac": 0.0,
            "deep_mix_frac": 0.0,
            "deep_in_each_batch": False,
            "bonus_mix_frac": 0.0,
            "quality_mix_frac": 0.0,
            "puzzle_mix_frac": 0.0,
            "use_swa": False,
            "hflip_p": 0.5,
            "value_weight": 0.05,
            "min_depth": 0,
            "optimizer": "polar_normuon",
            "compile_polar": True,
            "force_lr": True,
            "muon_lr": 0.002,
            "adam_lr": 3e-5,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup": 100,
            "min_lr_frac": 0.1,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": False,
            "max_vram_gb": 40.0,
            "save_every_steps": 250,
            "keep_step_every": 1000,
            "keep_last_ckpts": 4,
            "val_every_steps": 250,
            "val_eval_n": 256,
            "elo_every_steps": 0,
        },
    }
    eval_pt = out / "pack" / "val_cache.pt"
    if eval_pt.exists():
        trial["train"]["external_eval"] = {"harvest_val": str(eval_pt.resolve())}
    elif (ROOT / "outputs/exp271_mix/sf19_eval.pt").exists():
        trial["train"]["external_eval"] = {"sf19": str((ROOT / "outputs/exp271_mix/sf19_eval.pt").resolve())}
    log(f"train start steps={max_steps} bs={batch_size} init={init} pack={pack}", out / "pipeline.log")
    result = train_trial(
        trial,
        train_out,
        soft_cache=pack,
        deep_cache=None,
        max_steps=int(max_steps),
        max_minutes=float(minutes),
        smoke=False,
        resume_ckpt=init,
        teacher_ckpt=None,
    )
    json_write(train_out / "train_summary.json", result)
    return result
