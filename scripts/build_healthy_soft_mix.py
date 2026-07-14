#!/usr/bin/env python3
"""Build a healthy soft-training mix with game-stage metadata.

Sources
  - deep: Lichess cloud evals soft (thorough SF PVs)
  - puzzle: high-rated Lichess puzzles (tactics / edge)
  - syzygy: perfect ≤5-piece endgames soft (capped)
  - optional harvests: edge / phase-deep extras

Every row gets:
  n_pieces  — pieces on board (int8)
  ply       — fullmove-derived ply estimate (int16); estimated if FEN lacks it
  phase     — 0 open / 1 mid / 2 end
  source    — 0 deep / 1 puzzle / 2 syzygy / 3 harvest
  label_depth

Final mix is phase-capped AND source-capped so we don't overfit openings
or drown midgame with tablebases/puzzles.

Writes:
  <out>/soft_cache.pt
  <out>/mix_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.syzygy
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import _fast_parse_fen  # noqa: E402
from move_vocab import UCI_TO_IDX  # noqa: E402

SOFT_K = 8
SRC = {"deep": 0, "puzzle": 1, "syzygy": 2, "harvest": 3}
PHASE_ID = {"opening": 0, "middlegame": 1, "endgame": 2}
# Healthy defaults — mid-heavy, syzygy/puzzle capped
DEFAULT_PHASE = {"opening": 0.22, "middlegame": 0.48, "endgame": 0.30}
DEFAULT_SOURCE = {"deep": 0.70, "puzzle": 0.18, "syzygy": 0.07, "harvest": 0.05}
CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)
META = ("phase", "label_depth", "n_pieces", "ply", "source")


def log(msg: str) -> None:
    print(msg, flush=True)


def row_key(data, i):
    return (
        bytes(data["board_array"][i].tolist()),
        int(data["turn"][i]),
        int(data["castling"][i]),
        int(data["ep_square"][i]),
    )


def n_pieces_from_board(ba: np.ndarray) -> int:
    return int(np.count_nonzero(ba))


def phase_from_pieces(n: int) -> int:
    if n >= 26:
        return 0
    if n >= 14:
        return 1
    return 2


def ply_from_fen(fen: str, n_pieces: int) -> int:
    """Prefer FEN fullmove; else rough estimate from material."""
    parts = fen.split(" ")
    if len(parts) >= 6:
        try:
            fullmove = max(1, int(parts[5]))
            turn_black = parts[1] == "b"
            return (fullmove - 1) * 2 + (1 if turn_black else 0)
        except ValueError:
            pass
    # estimate: each side traded ~ (32-n)/2 pieces over ~2 plies each
    missing = max(0, 32 - n_pieces)
    return int(min(120, 8 + missing * 3))


def annotate(data: dict, source: int, default_depth: int = 20) -> dict:
    """Ensure phase/n_pieces/ply/source/label_depth on a soft cache (vectorized)."""
    n = int(data["board_array"].shape[0])
    ba = data["board_array"]
    # n_pieces: count nonzeros per row
    n_pieces = (ba != 0).sum(dim=1).to(torch.int8)
    np_i = n_pieces.to(torch.int16)
    if "phase" in data:
        phase = data["phase"].to(torch.int8)
    else:
        phase = torch.where(
            np_i >= 26, torch.zeros(n, dtype=torch.int8),
            torch.where(np_i >= 14, torch.ones(n, dtype=torch.int8),
                        torch.full((n,), 2, dtype=torch.int8)),
        )
    # ply estimate from material + phase nudge
    ply = (8 + (32 - np_i).clamp(min=0) * 3).to(torch.int16).clamp(max=120)
    ply = torch.where(phase == 0, ply.clamp(max=24), ply)
    ply = torch.where(phase == 2, torch.maximum(ply, torch.full_like(ply, 40)), ply)

    out = {k: data[k] for k in CORE if k in data}
    out["phase"] = phase
    if "label_depth" in data:
        out["label_depth"] = data["label_depth"].to(torch.int16)
    else:
        out["label_depth"] = torch.full((n,), default_depth, dtype=torch.int16)
    out["n_pieces"] = n_pieces
    out["ply"] = ply
    out["source"] = torch.full((n,), source, dtype=torch.int8)
    return out


def load_cache(path: Path) -> dict | None:
    if not path.exists():
        log(f"  miss {path}")
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    log(f"  load {path}: {d['board_array'].shape[0]:,}")
    return d


def merge_dedupe(chunks: list[dict]) -> dict:
    """Fast path: concat + hash-bucket dedupe on a subsampled key fingerprint."""
    # Concatenate first
    keys = [k for k in list(CORE) + list(META) if all(k in c for c in chunks)]
    cat = {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}
    n = cat["board_array"].shape[0]
    # Fingerprint: hash of first 16 board bytes + turn/castling/ep (vectorized-ish)
    ba = cat["board_array"].numpy()
    turn = cat["turn"].numpy().astype(np.int64)
    cast = cat["castling"].numpy().astype(np.int64)
    ep = cat["ep_square"].numpy().astype(np.int64)
    # cheap fingerprint
    fp = (
        ba[:, 0].astype(np.int64) * 1_000_003
        + ba[:, 7].astype(np.int64) * 91_337
        + ba[:, 32].astype(np.int64) * 7_919
        + ba[:, 63].astype(np.int64) * 1_009
        + turn * 17
        + cast * 257
        + (ep + 1) * 4099
        + ba.sum(axis=1).astype(np.int64) * 13
    )
    # keep last occurrence of each fp
    order = np.arange(n)
    # stable: later wins — reverse unique
    _, inv = np.unique(fp[::-1], return_index=True)
    keep = n - 1 - inv
    keep.sort()
    log(f"  dedupe {n:,} → {len(keep):,}")
    return {k: v[keep].contiguous() for k, v in cat.items()}


def subsample_quotas(
    data: dict,
    n_target: int,
    phase_frac: dict[str, float],
    source_frac: dict[str, float],
    rng: random.Random,
) -> dict:
    """Two-level quota: source budgets, then phase within each source, then global phase repair."""
    n = data["board_array"].shape[0]
    ph = data["phase"].numpy()
    src = data["source"].numpy()

    # source budgets
    src_want = {k: int(round(n_target * f)) for k, f in source_frac.items()}
    # fix sum
    while sum(src_want.values()) > n_target:
        src_want["deep"] -= 1
    while sum(src_want.values()) < n_target:
        src_want["deep"] += 1

    chosen: list[int] = []
    for sname, sid in SRC.items():
        budget = src_want.get(sname, 0)
        if budget <= 0:
            continue
        idx = np.nonzero(src == sid)[0].tolist()
        rng.shuffle(idx)
        # phase split within source
        by_ph = {0: [], 1: [], 2: []}
        for i in idx:
            by_ph[int(ph[i])].append(i)
        take = []
        for pname, pid in PHASE_ID.items():
            need = int(round(budget * phase_frac[pname]))
            pool = by_ph[pid]
            take.extend(pool[:need])
        # fill remainder from any phase in this source
        if len(take) < budget:
            rest = [i for i in idx if i not in set(take)]
            take.extend(rest[: budget - len(take)])
        chosen.extend(take[:budget])
        log(f"  source {sname}: want={budget:,} got={len(take[:budget]):,} avail={len(idx):,}")

    rng.shuffle(chosen)
    # global phase repair: drop surplus of overweight phases
    chosen = _repair_phase(chosen, ph, n_target, phase_frac, rng)
    keep = [k for k in list(CORE) + list(META) if k in data]
    return {k: data[k][chosen].contiguous() for k in keep}


def _repair_phase(idxs, ph, n_target, phase_frac, rng):
    want = {pid: int(round(n_target * phase_frac[name])) for name, pid in PHASE_ID.items()}
    while sum(want.values()) > n_target:
        want[1] -= 1
    buckets = {0: [], 1: [], 2: []}
    for i in idxs:
        buckets[int(ph[i])].append(i)
    out = []
    for pid in range(3):
        rng.shuffle(buckets[pid])
        out.extend(buckets[pid][: want[pid]])
    # if short, fill from leftovers
    have = set(out)
    if len(out) < n_target:
        rest = [i for i in idxs if i not in have]
        rng.shuffle(rest)
        out.extend(rest[: n_target - len(out)])
    rng.shuffle(out)
    return out[:n_target]


def wdl_to_score(wdl: int) -> float:
    return {2: 400.0, 1: 200.0, 0: 0.0, -1: -200.0, -2: -400.0}.get(int(wdl), 0.0)


def build_syzygy(n: int, syzygy_dir: Path, seed: int, tau: float) -> dict | None:
    if n <= 0 or not syzygy_dir.exists():
        return None
    try:
        tb = chess.syzygy.open_tablebase(str(syzygy_dir))
    except Exception as e:
        log(f"syzygy open fail: {e}")
        return None

    rng = random.Random(seed)
    pieces_pool = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    rows = []
    arr_buf = np.zeros(64, dtype=np.int8)
    seen = set()
    tries = 0
    t0 = time.time()
    while len(rows) < n and tries < n * 50:
        tries += 1
        board = chess.Board(None)
        board.clear()
        wk, bk = rng.randrange(64), rng.randrange(64)
        if chess.square_distance(wk, bk) <= 1:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        n_extra = rng.randint(1, 3)
        sqs = [i for i in range(64) if i not in (wk, bk)]
        rng.shuffle(sqs)
        ok = True
        for j in range(n_extra):
            pt = rng.choice(pieces_pool)
            color = chess.WHITE if rng.random() < 0.5 else chess.BLACK
            sq = sqs[j]
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(sq, chess.Piece(pt, color))
        if not ok or not board.is_valid():
            continue
        board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        if board.is_game_over() or len(board.piece_map()) > 5:
            continue
        fen = board.fen()
        key = " ".join(fen.split(" ")[:4])
        if key in seen:
            continue
        scored = []
        for mv in board.legal_moves:
            uci = mv.uci()
            if uci not in UCI_TO_IDX:
                continue
            board.push(mv)
            try:
                wdl = -tb.probe_wdl(board)
                scored.append((uci, wdl_to_score(wdl)))
            except Exception:
                board.pop()
                continue
            board.pop()
        if not scored:
            continue
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:SOFT_K]
        scores = [s for _, s in scored]
        mx = max(scores)
        exps = [math.exp((s - mx) / tau) for s in scores]
        z = sum(exps) or 1.0
        probs = [e / z for e in exps]
        t, c, e = _fast_parse_fen(" ".join(fen.split(" ")[:4]), arr_buf)
        np_ = n_pieces_from_board(arr_buf)
        si = [-1] * SOFT_K
        sp = [0.0] * SOFT_K
        for k, (uci, _) in enumerate(scored):
            si[k] = UCI_TO_IDX[uci]
            sp[k] = probs[k]
        rows.append({
            "ba": arr_buf.copy(), "t": t, "c": c, "e": e,
            "midx": UCI_TO_IDX[scored[0][0]], "cp": int(scores[0]),
            "si": si, "sp": sp, "np": np_, "ply": ply_from_fen(fen, np_),
        })
        seen.add(key)
        if len(rows) % 5000 == 0:
            log(f"  syzygy {len(rows):,}/{n:,}")

    if not rows:
        return None
    log(f"  syzygy done {len(rows):,} in {time.time()-t0:.1f}s")
    return {
        "board_array": torch.tensor(np.stack([r["ba"] for r in rows]), dtype=torch.int8),
        "turn": torch.tensor([r["t"] for r in rows], dtype=torch.int8),
        "castling": torch.tensor([r["c"] for r in rows], dtype=torch.int8),
        "ep_square": torch.tensor([r["e"] for r in rows], dtype=torch.int8),
        "move_idx": torch.tensor([r["midx"] for r in rows], dtype=torch.int64),
        "cp": torch.tensor([r["cp"] for r in rows], dtype=torch.int32),
        "mate": torch.zeros(len(rows), dtype=torch.int32),
        "soft_indices": torch.tensor([r["si"] for r in rows], dtype=torch.int64),
        "soft_probs": torch.tensor([r["sp"] for r in rows], dtype=torch.float32),
        "label_depth": torch.full((len(rows),), 999, dtype=torch.int16),
        "phase": torch.full((len(rows),), 2, dtype=torch.int8),
        "n_pieces": torch.tensor([r["np"] for r in rows], dtype=torch.int8),
        "ply": torch.tensor([r["ply"] for r in rows], dtype=torch.int16),
        "source": torch.full((len(rows),), SRC["syzygy"], dtype=torch.int8),
    }


def build_puzzles(n: int, min_rating: int, seed: int) -> dict | None:
    """High-rated Lichess puzzles → peaked soft on solution first move."""
    log(f"puzzles: downloading/reading HF (min_rating={min_rating}, target={n:,})")
    paths = []
    for i in range(3):
        paths.append(hf_hub_download(
            "Lichess/chess-puzzles", f"data/train-{i:05d}-of-00003.parquet", repo_type="dataset"
        ))
    rng = random.Random(seed)
    rows = []
    arr_buf = np.zeros(64, dtype=np.int8)
    seen = set()
    for path in paths:
        if len(rows) >= n:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=100_000, columns=["FEN", "Moves", "Rating", "Themes"]):
            fens = batch.column("FEN").to_pylist()
            moves = batch.column("Moves").to_pylist()
            ratings = batch.column("Rating").to_numpy()
            for fen, mvline, rating in zip(fens, moves, ratings):
                if len(rows) >= n:
                    break
                if int(rating) < min_rating or not fen or not mvline:
                    continue
                # subsample high-rated to keep variety (don't take ALL 2700+)
                if int(rating) >= 2400 and rng.random() > 0.7:
                    continue
                uci = mvline.split(" ", 1)[0]
                if uci not in UCI_TO_IDX:
                    continue
                parts = fen.split(" ")
                key = " ".join(parts[:4])
                if key in seen:
                    continue
                try:
                    t, c, e = _fast_parse_fen(" ".join(parts[:4]), arr_buf)
                except Exception:
                    continue
                np_ = n_pieces_from_board(arr_buf)
                ph = phase_from_pieces(np_)
                # theme override if present later — piece count is fine
                si = [-1] * SOFT_K
                sp = [0.0] * SOFT_K
                si[0] = UCI_TO_IDX[uci]
                sp[0] = 1.0
                rows.append({
                    "ba": arr_buf.copy(), "t": t, "c": c, "e": e,
                    "midx": UCI_TO_IDX[uci], "si": si, "sp": sp,
                    "np": np_, "ph": ph, "ply": ply_from_fen(fen, np_),
                    "rating": int(rating),
                })
                seen.add(key)
        log(f"  puzzles collected {len(rows):,}")

    if not rows:
        return None
    rng.shuffle(rows)
    rows = rows[:n]
    log(f"  puzzles kept {len(rows):,} rating_mean={np.mean([r['rating'] for r in rows]):.0f}")
    return {
        "board_array": torch.tensor(np.stack([r["ba"] for r in rows]), dtype=torch.int8),
        "turn": torch.tensor([r["t"] for r in rows], dtype=torch.int8),
        "castling": torch.tensor([r["c"] for r in rows], dtype=torch.int8),
        "ep_square": torch.tensor([r["e"] for r in rows], dtype=torch.int8),
        "move_idx": torch.tensor([r["midx"] for r in rows], dtype=torch.int64),
        "cp": torch.zeros(len(rows), dtype=torch.int32),
        "mate": torch.zeros(len(rows), dtype=torch.int32),
        "soft_indices": torch.tensor([r["si"] for r in rows], dtype=torch.int64),
        "soft_probs": torch.tensor([r["sp"] for r in rows], dtype=torch.float32),
        "label_depth": torch.full((len(rows),), 40, dtype=torch.int16),  # puzzle SF ~40M nodes
        "phase": torch.tensor([r["ph"] for r in rows], dtype=torch.int8),
        "n_pieces": torch.tensor([r["np"] for r in rows], dtype=torch.int8),
        "ply": torch.tensor([r["ply"] for r in rows], dtype=torch.int16),
        "source": torch.full((len(rows),), SRC["puzzle"], dtype=torch.int8),
        "puzzle_rating": torch.tensor([r["rating"] for r in rows], dtype=torch.int16),
    }


def report(data: dict) -> dict:
    n = data["board_array"].shape[0]
    ph = data["phase"].numpy()
    src = data["source"].numpy()
    np_ = data["n_pieces"].numpy()
    ply = data["ply"].numpy()
    depth = data["label_depth"].numpy()
    inv_src = {v: k for k, v in SRC.items()}
    inv_ph = {v: k for k, v in PHASE_ID.items()}
    rep = {
        "n": int(n),
        "phase": {inv_ph[i]: float((ph == i).mean()) for i in range(3)},
        "phase_counts": {inv_ph[i]: int((ph == i).sum()) for i in range(3)},
        "source": {inv_src[i]: float((src == i).mean()) for i in range(4) if (src == i).any()},
        "source_counts": {inv_src[i]: int((src == i).sum()) for i in range(4) if (src == i).any()},
        "n_pieces": {
            "mean": float(np_.mean()), "p10": float(np.percentile(np_, 10)),
            "p50": float(np.percentile(np_, 50)), "p90": float(np.percentile(np_, 90)),
            "hist": {str(k): int(v) for k, v in zip(*np.unique(np_, return_counts=True))},
        },
        "ply": {
            "mean": float(ply.mean()), "p10": float(np.percentile(ply, 10)),
            "p50": float(np.percentile(ply, 50)), "p90": float(np.percentile(ply, 90)),
        },
        "label_depth": {
            "mean": float(depth.mean()), "p50": float(np.percentile(depth, 50)),
        },
        "soft_width_mean": float((data["soft_indices"] >= 0).sum(1).float().mean()),
    }
    # cross: source × phase
    rep["source_x_phase"] = {}
    for sname, sid in SRC.items():
        if not (src == sid).any():
            continue
        mask = src == sid
        rep["source_x_phase"][sname] = {
            inv_ph[i]: float(((ph == i) & mask).sum() / max(mask.sum(), 1)) for i in range(3)
        }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/healthy_soft_mix")
    ap.add_argument("--deep-cache", default="outputs/lichess_evals_soft/soft_cache_8m.pt")
    ap.add_argument("--deep-fallback", default="outputs/lichess_evals_soft/soft_cache.pt")
    ap.add_argument("--target", type=int, default=4_000_000)
    ap.add_argument("--min-puzzle-rating", type=int, default=1800)
    ap.add_argument("--syzygy-dir", default="syzygy")
    ap.add_argument("--tau", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-syzygy", action="store_true")
    ap.add_argument("--syzygy-n", type=int, default=0, help="If >0, exact syzygy count (else from frac)")
    ap.add_argument("--skip-puzzles", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Budgets from source frac (over-generate then quota)
    src_frac = dict(DEFAULT_SOURCE)
    phase_frac = dict(DEFAULT_PHASE)
    # oversample pool 1.4x then subsample
    pool_n = int(args.target * 1.5)

    chunks = []

    deep = load_cache(Path(args.deep_cache)) or load_cache(Path(args.deep_fallback))
    if deep is None:
        raise SystemExit("need a deep Lichess soft cache")
    deep = annotate(deep, SRC["deep"], default_depth=30)
    # Subsample deep early (avoid O(N) python dedupe over full 8M)
    n_deep_pool = min(deep["board_array"].shape[0], int(args.target * src_frac["deep"] * 1.35))
    if deep["board_array"].shape[0] > n_deep_pool:
        perm = torch.randperm(deep["board_array"].shape[0])[:n_deep_pool]
        deep = {k: v[perm].contiguous() for k, v in deep.items()}
        log(f"deep subsampled → {n_deep_pool:,}")
    chunks.append(deep)
    log(f"deep annotated n={deep['board_array'].shape[0]:,}")

    if not args.skip_puzzles:
        n_puz = int(pool_n * src_frac["puzzle"] * 1.3)
        puz = build_puzzles(n_puz, args.min_puzzle_rating, args.seed)
        if puz is not None:
            # drop puzzle_rating before merge (optional meta)
            puz_core = {k: v for k, v in puz.items() if k != "puzzle_rating"}
            chunks.append(puz_core)
            # save puzzle rating sidecar stats
            torch.save({"puzzle_rating": puz["puzzle_rating"]}, out_dir / "puzzle_ratings.pt")

    if not args.skip_syzygy:
        syz_path = out_dir / "syzygy_soft.pt"
        if syz_path.exists():
            syz = load_cache(syz_path)
            log(f"reusing cached syzygy {syz_path}")
        else:
            n_syz = args.syzygy_n if args.syzygy_n > 0 else int(pool_n * src_frac["syzygy"] * 1.3)
            syz = build_syzygy(n_syz, Path(args.syzygy_dir), args.seed + 1, args.tau)
            if syz is not None:
                torch.save(syz, syz_path)
                log(f"cached syzygy → {syz_path}")
        if syz is not None:
            chunks.append(syz)

    # optional harvests as spice
    for path in [
        Path("outputs/exp192_edge_soft/soft_cache.pt"),
        Path("outputs/exp190_phase_deep_continue/soft_cache.pt"),
    ]:
        h = load_cache(path)
        if h is not None:
            chunks.append(annotate(h, SRC["harvest"], default_depth=16))

    mixed = merge_dedupe(chunks)
    final = subsample_quotas(mixed, args.target, phase_frac, src_frac, rng)
    rep = report(final)
    log("=== MIX REPORT ===")
    log(json.dumps(rep, indent=2))

    out_pt = out_dir / "soft_cache.pt"
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(final, tmp)
    os.replace(tmp, out_pt)
    (out_dir / "mix_report.json").write_text(json.dumps(rep, indent=2))
    log(f"wrote {out_pt} ({out_pt.stat().st_size/1e6:.1f} MB)")
    log(f"report {out_dir / 'mix_report.json'}")


if __name__ == "__main__":
    main()
