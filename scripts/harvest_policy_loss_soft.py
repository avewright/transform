#!/usr/bin/env python3
"""Parallel policy-loss soft harvest + optional FT handoff.

Many workers play FT3h vs limited SF at once (Black / weak openings oversampled).
Each model ply is labeled with full-strength SF MultiPV. Keep:
  - all positions from LOST games
  - blunders/inaccuracies from any game

Usage:
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_policy_loss_soft.py --go \\
    --workers 8 --games 480 --train-after
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.engine
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
from chess_inference import load_checkpoint  # noqa: E402
from data_loader import _fast_parse_fen  # noqa: E402
from move_vocab import UCI_TO_IDX, index_to_move, legal_move_mask  # noqa: E402

SOFT_K = 8
WEAK_OPENINGS = [
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
    ["e2e4", "c7c5"],
    ["e2e4", "c7c6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    ["c2c4", "e7e5", "g2g3"],
]


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def resolve_sf() -> str:
    for p in [
        os.environ.get("STOCKFISH_PATH", ""),
        shutil.which("stockfish") or "",
        "/usr/games/stockfish",
        str(ROOT / "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"),
        str(ROOT / "stockfish/stockfish-latest"),
    ]:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("Stockfish not found")


def phase_id(board: chess.Board) -> int:
    n = len(board.piece_map())
    if n >= 26:
        return 0
    if n >= 14:
        return 1
    return 2


@torch.no_grad()
def model_move(model, board, device) -> chess.Move:
    x = batch_boards_to_fused_token_ids([board], device)
    logits = model(x)["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits = logits.masked_fill(~mask, float("-inf"))
    mv = index_to_move(int(logits.argmax().item()))
    if mv not in board.legal_moves:
        mv = next(iter(board.legal_moves))
    return mv


def score_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    pov = score.pov(turn)
    if pov.is_mate():
        m = pov.mate()
        assert m is not None
        return (100000 - min(abs(m), 1000)) * (1 if m > 0 else -1)
    cp = pov.score(mate_score=100000)
    return int(cp if cp is not None else 0)


def analyze_deep(engine, board, *, depth, multipv, tau, nodes, movetime) -> dict | None:
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return None
    if movetime > 0:
        limit = chess.engine.Limit(time=movetime)
    elif nodes > 0:
        limit = chess.engine.Limit(nodes=nodes)
    else:
        limit = chess.engine.Limit(depth=max(depth, 12))
    try:
        infos = engine.analyse(board, limit, multipv=min(multipv, n_legal))
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return None
    if not isinstance(infos, list):
        infos = [infos]
    best: dict[str, int] = {}
    used_depth = depth
    for info in infos:
        pv = info.get("pv") or []
        sc = info.get("score")
        if not pv or sc is None:
            continue
        uci = pv[0].uci()
        cp = score_cp(sc, board.turn)
        if uci not in best or cp > best[uci]:
            best[uci] = cp
        if "depth" in info:
            used_depth = max(used_depth, int(info["depth"]))
    if not best:
        return None
    items = sorted(best.items(), key=lambda x: -x[1])[:SOFT_K]
    cps = [c for _, c in items]
    mx = max(cps)
    exps = [math.exp((c - mx) / tau) for c in cps]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]
    return {
        "ucis": [u for u, _ in items],
        "cps": cps,
        "probs": probs,
        "best_cp": cps[0],
        "best_uci": items[0][0],
        "depth": used_depth,
    }


def board_row(board: chess.Board, soft: dict) -> dict | None:
    arr = np.zeros(64, dtype=np.int8)
    try:
        _fast_parse_fen(board.fen(), arr)
    except Exception:
        return None
    soft_i = [-1] * SOFT_K
    soft_p = [0.0] * SOFT_K
    for i, (uci, pr) in enumerate(zip(soft["ucis"], soft["probs"])):
        if uci not in UCI_TO_IDX:
            continue
        soft_i[i] = UCI_TO_IDX[uci]
        soft_p[i] = float(pr)
    if soft_i[0] < 0 or soft["best_uci"] not in UCI_TO_IDX:
        return None
    s = sum(soft_p) or 1.0
    soft_p = [p / s for p in soft_p]
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    return {
        "board_array": arr.copy(),
        "turn": 0 if board.turn == chess.WHITE else 1,
        "castling": castling,
        "ep_square": board.ep_square if board.ep_square is not None else -1,
        "move_idx": UCI_TO_IDX[soft["best_uci"]],
        "cp": int(soft["cps"][0]),
        "mate": 0,
        "soft_indices": soft_i,
        "soft_probs": soft_p,
        "label_depth": int(soft["depth"]),
        "phase": phase_id(board),
        "source": 0,
    }


def play_one(model, device, opp, teacher, *, model_color, opening, sf_elo,
             teacher_depth, teacher_nodes, teacher_movetime, tau, ply_cap,
             sf_movetime, blunder_cp, inacc_cp):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    kept = []
    meta = {
        "model_color": "white" if model_color == chess.WHITE else "black",
        "opening": " ".join(opening) if opening else "startpos",
        "sf_elo": sf_elo,
        "n_model_plies": 0,
        "n_blunders": 0,
        "n_inacc": 0,
    }
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            meta["n_model_plies"] += 1
            mv = model_move(model, board, device)
            soft = analyze_deep(
                teacher, board, depth=teacher_depth, multipv=SOFT_K, tau=tau,
                nodes=teacher_nodes, movetime=teacher_movetime,
            )
            if soft is not None:
                if mv.uci() in soft["ucis"]:
                    model_cp = soft["cps"][soft["ucis"].index(mv.uci())]
                else:
                    model_cp = soft["best_cp"] - 400
                drop = soft["best_cp"] - model_cp
                tag = "ok"
                if drop >= blunder_cp:
                    tag = "blunder"
                    meta["n_blunders"] += 1
                elif drop >= inacc_cp:
                    tag = "inaccuracy"
                    meta["n_inacc"] += 1
                row = board_row(board, soft)
                if row is not None:
                    row["_tag"] = tag
                    row["_fen"] = board.fen()
                    kept.append(row)
            if mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            board.push(mv)
        else:
            mv = opp.play(board, chess.engine.Limit(time=sf_movetime)).move
            if mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            board.push(mv)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == model_color:
        score = 1.0
    else:
        score = 0.0
    meta.update({
        "score": score,
        "plies": len(board.move_stack),
        "termination": outcome.termination.name if outcome else "PLY_CAP",
        "final_fen": board.fen(),
    })
    return kept, meta


def pack_cache(rows: list[dict], out_pt: Path) -> int:
    n = len(rows)
    if n == 0:
        return 0
    data = {
        "board_array": torch.from_numpy(np.stack([r["board_array"] for r in rows])),
        "turn": torch.tensor([r["turn"] for r in rows], dtype=torch.int8),
        "castling": torch.tensor([r["castling"] for r in rows], dtype=torch.int8),
        "ep_square": torch.tensor([r["ep_square"] for r in rows], dtype=torch.int8),
        "move_idx": torch.tensor([r["move_idx"] for r in rows], dtype=torch.int64),
        "cp": torch.tensor([r["cp"] for r in rows], dtype=torch.int32),
        "mate": torch.tensor([r["mate"] for r in rows], dtype=torch.int32),
        "soft_indices": torch.tensor([r["soft_indices"] for r in rows], dtype=torch.int64),
        "soft_probs": torch.tensor([r["soft_probs"] for r in rows], dtype=torch.float32),
        "label_depth": torch.tensor([r["label_depth"] for r in rows], dtype=torch.int16),
        "phase": torch.tensor([r["phase"] for r in rows], dtype=torch.int8),
        "source": torch.tensor([r["source"] for r in rows], dtype=torch.int8),
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, out_pt)
    return n


_W_MODEL = None
_W_DEVICE = None
_W_TEACHER = None
_W_SF = None


def _init_worker(checkpoint: str, sf_path: str):
    global _W_MODEL, _W_DEVICE, _W_TEACHER, _W_SF
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    _W_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _W_SF = sf_path
    _W_MODEL = load_checkpoint(checkpoint, _W_DEVICE)
    _W_MODEL.eval()
    _W_TEACHER = chess.engine.SimpleEngine.popen_uci(sf_path)
    _W_TEACHER.configure({"Threads": 1, "Hash": 96})


def _worker_game(spec_and_cfg: tuple) -> dict:
    spec, cfg = spec_and_cfg
    opp = chess.engine.SimpleEngine.popen_uci(_W_SF)
    opp.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": int(spec["sf_elo"]),
        "Threads": 1,
    })
    kept, meta = play_one(
        _W_MODEL, _W_DEVICE, opp, _W_TEACHER,
        model_color=chess.BLACK if spec["as_black"] else chess.WHITE,
        opening=spec["opening"],
        sf_elo=int(spec["sf_elo"]),
        teacher_depth=int(cfg["teacher_depth"]),
        teacher_nodes=int(cfg["teacher_nodes"]),
        teacher_movetime=float(cfg["teacher_movetime"]),
        tau=float(cfg["tau"]),
        ply_cap=int(cfg["ply_cap"]),
        sf_movetime=float(cfg["sf_movetime"]),
        blunder_cp=int(cfg["blunder_cp"]),
        inacc_cp=int(cfg["inacc_cp"]),
    )
    opp.quit()
    lost = meta["score"] == 0.0
    rows = []
    seen: set[str] = set()
    for r in kept:
        fen = r.pop("_fen")
        tag = r.pop("_tag")
        keep = (lost and cfg["keep_all_from_losses"]) or tag in ("blunder", "inaccuracy")
        if not keep:
            continue
        key = " ".join(fen.split()[:4])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    meta["added"] = len(rows)
    meta["lost"] = lost
    meta["game_idx"] = spec["game_idx"]
    line = (
        f"g{spec['game_idx']} SF{spec['sf_elo']} "
        f"{'b' if spec['as_black'] else 'w'} {meta['opening'][:16]:<16} "
        f"{'L' if lost else ('W' if meta['score']==1 else 'D')} +{len(rows)} "
        f"bl={meta['n_blunders']}"
    )
    return {"rows": rows, "metas": [meta], "logs": [line]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--checkpoint", default="outputs/exp191_soft_ft3h_edge_end/best.pt")
    ap.add_argument("--out-dir", default="outputs/policy_loss_soft")
    ap.add_argument("--games", type=int, default=480)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--black-frac", type=float, default=0.78)
    ap.add_argument("--sf-elos", type=int, nargs="+", default=[1750, 1900, 2050])
    ap.add_argument("--teacher-depth", type=int, default=0)
    ap.add_argument("--teacher-nodes", type=int, default=1_200_000)
    ap.add_argument("--teacher-movetime", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--sf-movetime", type=float, default=0.08)
    ap.add_argument("--ply-cap", type=int, default=160)
    ap.add_argument("--blunder-cp", type=int, default=75)
    ap.add_argument("--inacc-cp", type=int, default=25)
    ap.add_argument("--keep-all-from-losses", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-after", action="store_true")
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("pass --go")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    if log_path.exists():
        log_path.unlink()
    if (out / "DONE").exists():
        (out / "DONE").unlink()

    sf_path = resolve_sf()
    rng = random.Random(args.seed)
    n_workers = max(1, args.workers)

    cfg = {
        "teacher_depth": args.teacher_depth,
        "teacher_nodes": args.teacher_nodes,
        "teacher_movetime": args.teacher_movetime,
        "tau": args.tau,
        "ply_cap": args.ply_cap,
        "sf_movetime": args.sf_movetime,
        "blunder_cp": args.blunder_cp,
        "inacc_cp": args.inacc_cp,
        "keep_all_from_losses": args.keep_all_from_losses,
    }
    jobs = []
    for gi in range(args.games):
        as_black = rng.random() < args.black_frac
        opening = WEAK_OPENINGS[rng.randrange(0, 4 if rng.random() < 0.55 else len(WEAK_OPENINGS))]
        spec = {
            "game_idx": gi,
            "sf_elo": args.sf_elos[gi % len(args.sf_elos)],
            "as_black": as_black,
            "opening": list(opening),
        }
        jobs.append((spec, cfg))

    log(f"checkpoint={args.checkpoint}", log_path)
    log(
        f"PARALLEL workers={n_workers} games={args.games} "
        f"nodes={args.teacher_nodes} sf={sf_path}",
        log_path,
    )

    t0 = time.time()
    rows: list[dict] = []
    game_metas: list[dict] = []
    seen: set[bytes] = set()

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.checkpoint, sf_path),
    ) as pool:
        for result in pool.imap_unordered(_worker_game, jobs, chunksize=1):
            for line in result["logs"]:
                log(line, log_path)
            game_metas.extend(result["metas"])
            for r in result["rows"]:
                key = r["board_array"].tobytes() + bytes([int(r["turn"]), int(r["castling"])])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
            if len(game_metas) % 8 == 0 or len(game_metas) == args.games:
                pack_cache(rows, out / "soft_cache.pt")
                (out / "games_metas.json").write_text(json.dumps(game_metas, indent=2))
                sc = sum(m["score"] for m in game_metas) / max(1, len(game_metas))
                log(
                    f"progress games={len(game_metas)}/{args.games} cache={len(rows):,} "
                    f"losses={sum(1 for m in game_metas if m.get('lost'))} "
                    f"rate={sc:.2f} [{time.time()-t0:.0f}s]",
                    log_path,
                )

    n = pack_cache(rows, out / "soft_cache.pt")
    report = {
        "n": n,
        "games": len(game_metas),
        "loss_games": sum(1 for m in game_metas if m.get("lost")),
        "score_rate": sum(m["score"] for m in game_metas) / max(1, len(game_metas)),
        "workers": n_workers,
        "teacher_nodes": args.teacher_nodes,
        "checkpoint": args.checkpoint,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "DONE").write_text(json.dumps(report, indent=2))
    log(f"DONE n={n:,} {report}", log_path)

    if args.train_after and n > 0:
        log("starting FT on loss soft cache…", log_path)
        import subprocess
        subprocess.Popen(
            ["bash", str(ROOT / "scripts/run_ft_policy_loss.sh")],
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
