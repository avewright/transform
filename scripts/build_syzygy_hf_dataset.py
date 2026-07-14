#!/usr/bin/env python3
"""Build a Syzygy soft-label dataset and export HF-compatible parquet.

Train-compatible with exp191 soft caches, plus endgame metadata:

  board_array[64], turn, castling, ep_square,
  move_idx, cp, mate, soft_indices[8], soft_probs[8],
  label_depth, phase, source,
  n_pieces, ply, wdl, dtz, cache_name

Usage:
  MOVE_VOCAB_VERSION=compact python scripts/build_syzygy_hf_dataset.py \\
    --target 500000 --push --repo avewright/chess-soft-syzygy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.syzygy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import _fast_parse_fen  # noqa: E402
from move_vocab import UCI_TO_IDX  # noqa: E402

SOFT_K = 8
SRC_SYZYGY = 2
PHASE_ENDGAME = 2

SCHEMA = pa.schema([
    ("board_array", pa.list_(pa.int8(), 64)),
    ("turn", pa.int8()),
    ("castling", pa.int8()),
    ("ep_square", pa.int8()),
    ("move_idx", pa.int64()),
    ("cp", pa.int32()),
    ("mate", pa.int32()),
    ("soft_indices", pa.list_(pa.int64(), 8)),
    ("soft_probs", pa.list_(pa.float32(), 8)),
    ("label_depth", pa.int16()),
    ("phase", pa.int8()),
    ("source", pa.int8()),
    ("n_pieces", pa.int8()),
    ("ply", pa.int16()),
    ("wdl", pa.int8()),
    ("dtz", pa.int16()),
    ("cache_name", pa.string()),
])


def log(msg: str) -> None:
    print(msg, flush=True)


def wdl_to_cp(wdl: int) -> int:
    return {2: 400, 1: 200, 0: 0, -1: -200, -2: -400}.get(int(wdl), 0)


def ply_estimate(n_pieces: int) -> int:
    missing = max(0, 32 - n_pieces)
    return int(min(120, max(40, 8 + missing * 3)))


def _fixed_list(arr: np.ndarray, value_type: pa.DataType, width: int) -> pa.Array:
    flat = pa.array(arr.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def generate_syzygy(
    n: int,
    syzygy_dir: Path,
    seed: int,
    tau: float,
    piece_balance: dict[int, float] | None = None,
) -> dict:
    """Sample unique ≤5-piece positions; soft labels from TB WDL over legal moves."""
    tb = chess.syzygy.open_tablebase(str(syzygy_dir))
    rng = random.Random(seed)
    pieces_pool = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    # Quotas by piece count (3/4/5)
    bal = piece_balance or {3: 0.30, 4: 0.40, 5: 0.30}
    quotas = {k: int(round(n * v)) for k, v in bal.items()}
    # fix rounding
    quotas[5] = n - quotas.get(3, 0) - quotas.get(4, 0)
    counts = {3: 0, 4: 0, 5: 0}

    rows: list[dict] = []
    arr_buf = np.zeros(64, dtype=np.int8)
    seen: set[str] = set()
    tries = 0
    max_tries = max(n * 80, 100_000)
    t0 = time.time()

    while len(rows) < n and tries < max_tries:
        tries += 1
        # pick under-filled piece bucket
        need = [p for p, q in quotas.items() if counts[p] < q]
        if not need:
            break
        n_pieces_target = rng.choice(need)
        n_extra = n_pieces_target - 2

        board = chess.Board(None)
        board.clear()
        wk, bk = rng.randrange(64), rng.randrange(64)
        if chess.square_distance(wk, bk) <= 1:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
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
        if board.is_game_over() or len(board.piece_map()) != n_pieces_target:
            continue

        fen = board.fen()
        key = " ".join(fen.split(" ")[:4])
        if key in seen:
            continue

        try:
            pos_wdl = int(tb.probe_wdl(board))
        except Exception:
            continue
        try:
            pos_dtz = int(tb.probe_dtz(board))
        except Exception:
            pos_dtz = 0

        scored: list[tuple[str, int, int, int]] = []  # uci, cp, wdl, dtz
        for mv in board.legal_moves:
            uci = mv.uci()
            if uci not in UCI_TO_IDX:
                continue
            board.push(mv)
            try:
                # after move: opponent to move → negate for our STM eval of the move
                wdl = -int(tb.probe_wdl(board))
                try:
                    dtz = -int(tb.probe_dtz(board))
                except Exception:
                    dtz = 0
                scored.append((uci, wdl_to_cp(wdl), wdl, dtz))
            except Exception:
                board.pop()
                continue
            board.pop()
        if not scored:
            continue

        scored.sort(key=lambda x: (x[1], -abs(x[3]) if x[2] != 0 else 0), reverse=True)
        scored = scored[:SOFT_K]
        cps = [s[1] for s in scored]
        mx = max(cps)
        exps = [math.exp((s - mx) / tau) for s in cps]
        z = sum(exps) or 1.0
        probs = [e / z for e in exps]

        t, c, e = _fast_parse_fen(" ".join(fen.split(" ")[:4]), arr_buf)
        np_ = int(np.count_nonzero(arr_buf))
        si = [-1] * SOFT_K
        sp = [0.0] * SOFT_K
        for k, (uci, _, _, _) in enumerate(scored):
            si[k] = UCI_TO_IDX[uci]
            sp[k] = probs[k]

        best_wdl = scored[0][2]
        best_dtz = scored[0][3]
        # mate distance from DTZ when decisive; else 0 and use cp
        mate = 0
        cp = int(scored[0][1])
        if abs(best_wdl) == 2 and best_dtz != 0:
            # DTZ plies ≈ mate distance proxy (sign = STM win/loss)
            mate = int(np.sign(best_wdl) * max(1, (abs(best_dtz) + 1) // 2))
            cp = 0

        rows.append({
            "ba": arr_buf.copy(),
            "t": t, "c": c, "e": e,
            "midx": UCI_TO_IDX[scored[0][0]],
            "cp": cp, "mate": mate,
            "si": si, "sp": sp,
            "np": np_,
            "ply": ply_estimate(np_),
            "wdl": int(pos_wdl),
            "dtz": int(np.clip(pos_dtz, -32767, 32767)),
        })
        seen.add(key)
        counts[np_] = counts.get(np_, 0) + 1
        if len(rows) % 5000 == 0:
            log(
                f"  syzygy {len(rows):,}/{n:,} "
                f"pieces={dict(sorted(counts.items()))} "
                f"({time.time()-t0:.0f}s)"
            )

    if not rows:
        raise RuntimeError("no syzygy rows generated — check tablebase path")

    log(f"  done {len(rows):,} in {time.time()-t0:.1f}s tries={tries:,} pieces={dict(sorted(counts.items()))}")
    return {
        "board_array": torch.tensor(np.stack([r["ba"] for r in rows]), dtype=torch.int8),
        "turn": torch.tensor([r["t"] for r in rows], dtype=torch.int8),
        "castling": torch.tensor([r["c"] for r in rows], dtype=torch.int8),
        "ep_square": torch.tensor([r["e"] for r in rows], dtype=torch.int8),
        "move_idx": torch.tensor([r["midx"] for r in rows], dtype=torch.int64),
        "cp": torch.tensor([r["cp"] for r in rows], dtype=torch.int32),
        "mate": torch.tensor([r["mate"] for r in rows], dtype=torch.int32),
        "soft_indices": torch.tensor([r["si"] for r in rows], dtype=torch.int64),
        "soft_probs": torch.tensor([r["sp"] for r in rows], dtype=torch.float32),
        "label_depth": torch.full((len(rows),), 999, dtype=torch.int16),
        "phase": torch.full((len(rows),), PHASE_ENDGAME, dtype=torch.int8),
        "source": torch.full((len(rows),), SRC_SYZYGY, dtype=torch.int8),
        "n_pieces": torch.tensor([r["np"] for r in rows], dtype=torch.int8),
        "ply": torch.tensor([r["ply"] for r in rows], dtype=torch.int16),
        "wdl": torch.tensor([r["wdl"] for r in rows], dtype=torch.int8),
        "dtz": torch.tensor([r["dtz"] for r in rows], dtype=torch.int16),
    }


KEYS = [
    "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
    "soft_indices", "soft_probs", "label_depth", "phase", "source",
    "n_pieces", "ply", "wdl", "dtz",
]


def annotate_syzygy(d: dict) -> dict:
    """Fill missing metadata columns on an older syzygy soft cache."""
    n = int(d["board_array"].shape[0])
    if "n_pieces" not in d:
        d["n_pieces"] = (d["board_array"] != 0).sum(dim=1).to(torch.int8)
    if "ply" not in d:
        np_i = d["n_pieces"].to(torch.int16)
        d["ply"] = (8 + (32 - np_i).clamp(min=0) * 3).clamp(max=120).to(torch.int16)
    if "wdl" not in d:
        d["wdl"] = torch.zeros(n, dtype=torch.int8)
    if "dtz" not in d:
        d["dtz"] = torch.zeros(n, dtype=torch.int16)
    if "source" not in d:
        d["source"] = torch.full((n,), SRC_SYZYGY, dtype=torch.int8)
    if "phase" not in d:
        d["phase"] = torch.full((n,), PHASE_ENDGAME, dtype=torch.int8)
    if "label_depth" not in d:
        d["label_depth"] = torch.full((n,), 999, dtype=torch.int16)
    return {k: d[k] for k in KEYS if k in d}


def merge_existing(base: dict, extra_path: Path | None) -> dict:
    if extra_path is None or not extra_path.exists():
        return base
    log(f"merge existing {extra_path}")
    from scripts.extract_unseen_soft_cache import pack_keys

    old = annotate_syzygy(torch.load(extra_path, map_location="cpu", weights_only=False))
    n_old = old["board_array"].shape[0]
    cat = {k: torch.cat([old[k], base[k]], dim=0) for k in KEYS}
    pk = pack_keys(cat)
    _, uniq_idx = np.unique(pk, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    out = {k: v[uniq_idx].contiguous() for k, v in cat.items()}
    log(f"  merged {n_old:,}+{base['board_array'].shape[0]:,} → {out['board_array'].shape[0]:,} unique")
    return out

def export_parquet(data: dict, local_dir: Path, shard_size: int, cache_name: str) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    n = int(data["board_array"].shape[0])
    shard = 0
    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        m = end - start
        ba = data["board_array"][start:end].numpy().astype(np.int8, copy=False)
        si = data["soft_indices"][start:end].numpy().astype(np.int64, copy=False)
        sp = data["soft_probs"][start:end].numpy().astype(np.float32, copy=False)
        cols = [
            _fixed_list(ba, pa.int8(), 64),
            pa.array(data["turn"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["castling"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["ep_square"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["move_idx"][start:end].numpy().astype(np.int64, copy=False)),
            pa.array(data["cp"][start:end].numpy().astype(np.int32, copy=False)),
            pa.array(data["mate"][start:end].numpy().astype(np.int32, copy=False)),
            _fixed_list(si, pa.int64(), 8),
            _fixed_list(sp, pa.float32(), 8),
            pa.array(data["label_depth"][start:end].numpy().astype(np.int16, copy=False)),
            pa.array(data["phase"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["source"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["n_pieces"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["ply"][start:end].numpy().astype(np.int16, copy=False)),
            pa.array(data["wdl"][start:end].numpy().astype(np.int8, copy=False)),
            pa.array(data["dtz"][start:end].numpy().astype(np.int16, copy=False)),
            pa.array([cache_name] * m, type=pa.string()),
        ]
        table = pa.Table.from_arrays(cols, schema=SCHEMA)
        path = local_dir / f"data-{shard:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        log(f"wrote {path} n={m:,}")
        shard += 1

    readme = f"""---
license: mit
task_categories:
- other
tags:
- chess
- syzygy
- tablebase
- soft-labels
- endgame
---

# Syzygy soft MultiPV endgame dataset

Perfect ≤5-piece endgame positions labeled via Syzygy WDL (soft top-8 moves).
Compatible with ChessTransformer soft-cache training (compact move vocab).

## Columns
- `board_array` (64 int8), `turn`, `castling`, `ep_square`
- `move_idx`, `cp`, `mate` — hard best + eval
- `soft_indices[8]`, `soft_probs[8]` — TB soft policy
- `n_pieces` — pieces on board (3–5)
- `ply` — estimated ply
- `wdl` — position WDL from STM (−2…2)
- `dtz` — distance-to-zero (signed)
- `phase` = 2 (endgame), `source` = 2 (syzygy), `label_depth` = 999
"""
    (local_dir / "README.md").write_text(readme)
    return shard


def push_hf(local_dir: Path, repo: str, private: bool, token: str) -> None:
    from huggingface_hub import HfApi, create_repo

    create_repo(repo, repo_type="dataset", private=private, exist_ok=True, token=token)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["*.parquet", "README.md", "*.json"],
    )
    log(f"pushed https://huggingface.co/datasets/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500_000)
    ap.add_argument("--syzygy-dir", default="syzygy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=80.0, help="soft softmax temperature on WDL-cp")
    ap.add_argument("--merge-existing", default="outputs/healthy_soft_mix/syzygy_soft.pt")
    ap.add_argument("--output-pt", default="outputs/syzygy_hf/soft_cache.pt")
    ap.add_argument("--local-dir", default="outputs/hf_syzygy_export")
    ap.add_argument("--shard-size", type=int, default=100_000)
    ap.add_argument("--repo", default="avewright/chess-soft-syzygy")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--skip-gen", action="store_true", help="export existing --output-pt only")
    args = ap.parse_args()

    out_pt = Path(args.output_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    local_dir = Path(args.local_dir)

    if args.skip_gen and out_pt.exists():
        data = annotate_syzygy(torch.load(out_pt, map_location="cpu", weights_only=False))
        log(f"loaded {out_pt} n={data['board_array'].shape[0]:,}")
    else:
        merge_path = Path(args.merge_existing) if args.merge_existing else None
        already = 0
        if merge_path and merge_path.exists():
            already = int(torch.load(merge_path, map_location="cpu", weights_only=False)["board_array"].shape[0])
            log(f"will merge {already:,} existing syzygy rows")
        gen_n = max(0, args.target - already)
        if gen_n > 0:
            log(f"generate {gen_n:,} new syzygy rows (target={args.target:,})")
            data = generate_syzygy(gen_n, Path(args.syzygy_dir), args.seed, args.tau)
            data = merge_existing(data, merge_path)
        elif merge_path and merge_path.exists():
            data = annotate_syzygy(torch.load(merge_path, map_location="cpu", weights_only=False))
        else:
            raise SystemExit("nothing to generate and no --merge-existing / --output-pt")

        n = data["board_array"].shape[0]
        if n > args.target:
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(n, size=args.target, replace=False))
            data = {k: v[idx].contiguous() for k, v in data.items()}
            log(f"subsample → {args.target:,}")

        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(data, tmp)
        os.replace(tmp, out_pt)
        log(f"wrote {out_pt} n={data['board_array'].shape[0]:,} ({out_pt.stat().st_size/1e6:.1f} MB)")
    # report
    np_c = Counter(data["n_pieces"].tolist())
    report = {
        "n": int(data["board_array"].shape[0]),
        "n_pieces": {str(k): int(v) for k, v in sorted(np_c.items())},
        "mate_nonzero": int((data["mate"] != 0).sum()),
        "wdl_hist": {str(k): int(v) for k, v in sorted(Counter(data["wdl"].tolist()).items())},
        "source": SRC_SYZYGY,
        "phase": PHASE_ENDGAME,
    }
    (out_pt.parent / "report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report, indent=2))

    # wipe/export
    if local_dir.exists():
        for p in local_dir.glob("data-*.parquet"):
            p.unlink()
    n_shards = export_parquet(data, local_dir, args.shard_size, cache_name="syzygy_soft")
    (local_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"exported {n_shards} shards → {local_dir}")

    if args.push:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token and Path(".env").exists():
            for line in Path(".env").read_text().splitlines():
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if not token:
            raise SystemExit("HF_TOKEN required for --push")
        push_hf(local_dir, args.repo, args.private, token)


if __name__ == "__main__":
    main()
