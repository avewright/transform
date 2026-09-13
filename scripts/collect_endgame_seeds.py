#!/usr/bin/env python3
"""Pull <14-piece FENs from avewright + Lichess HF packs for endgame harvest seeds.

Writes outputs/endgame_dataset/extra_seeds.jsonl (unique 4-field FEN).
Does not stop the live harvest.

  python3 -u scripts/collect_endgame_seeds.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harvest_exp201_lapses import board_array_to_fen  # noqa: E402
from upload_exp201_hf import load_hf_token  # noqa: E402

OUT = ROOT / "outputs" / "endgame_dataset"
SEED_PATH = OUT / "extra_seeds.jsonl"
SUMMARY_PATH = OUT / "extra_seeds.json"
LO, HI = 6, 13  # n_pieces < 14

# (repo, max unique to keep from that repo)
SOURCES = (
    ("avewright/chess-soft-multipv-lichess", 30_000),
    ("avewright/chess-soft-100m-disagreements", 20_000),
    ("avewright/chess-soft-100m-swa-mistakes", 20_000),
    ("avewright/stockfish-19-soft-targets", 15_000),
    ("avewright/chess-positions-sf-labeled", 20_000),
    ("avewright/chess-positions-lichess-sf", 20_000),
    ("Lichess/chess-puzzles", 30_000),
)


def log(msg: str) -> None:
    print(msg, flush=True)


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def pieces_fen(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def add_fen(seen: set[str], rows: list[dict], fen: str, source: str, n_pcs: int) -> bool:
    if not (LO <= n_pcs <= HI):
        return False
    try:
        board = chess.Board(fen)
    except ValueError:
        return False
    if board.is_game_over(claim_draw=True):
        return False
    key = fen_key(board.fen())
    if key in seen:
        return False
    seen.add(key)
    rows.append({"fen": board.fen(), "source": source, "n_pieces": int(n_pcs)})
    return True


def _board_matrix(col) -> np.ndarray:
    chunk = col.combine_chunks()
    if hasattr(chunk, "values"):
        flat = np.asarray(chunk.values.to_numpy(zero_copy_only=False), dtype=np.int8)
        n = len(chunk)
        return flat.reshape(n, -1)[:, :64]
    raw = chunk.to_pylist()
    return np.stack([np.asarray(x, dtype=np.int8).reshape(-1)[:64] for x in raw])


def scan_parquet(path: Path, repo: str, seen: set[str], rows: list[dict], cap: int) -> tuple[int, int]:
    names = pq.ParquetFile(path).schema_arrow.names
    scanned = 0
    kept = 0
    if "board_array" in names:
        cols = ["board_array", "turn", "castling", "ep_square"]
        if "phase" in names:
            cols.append("phase")
        table = pq.read_table(path, columns=cols)
        n = table.num_rows
        scanned = n
        ba = _board_matrix(table.column("board_array"))
        pcs = (ba != 0).sum(axis=1)
        mask = (pcs >= LO) & (pcs <= HI)
        idx = np.nonzero(mask)[0]
        turn = np.asarray(table.column("turn").to_numpy())
        castle = np.asarray(table.column("castling").to_numpy())
        ep = np.asarray(table.column("ep_square").to_numpy())
        for i in idx.tolist():
            if len(rows) >= cap:
                break
            n_pcs = int(pcs[i])
            fen = board_array_to_fen(ba[i], int(turn[i]), int(castle[i]), int(ep[i]))
            if add_fen(seen, rows, fen, repo, n_pcs):
                kept += 1
        return scanned, kept
    fen_col = None
    for cand in ("FEN", "fen", "fen_4"):
        if cand in names:
            fen_col = cand
            break
    if fen_col is None:
        return 0, 0
    table = pq.read_table(path, columns=[fen_col])
    scanned = table.num_rows
    for raw in table.column(fen_col).to_pylist():
        if len(rows) >= cap:
            break
        if not raw:
            continue
        fen = str(raw)
        if fen_col == "fen_4" and len(fen.split()) == 4:
            fen = fen + " 0 1"
        n_pcs = pieces_fen(fen)
        if add_fen(seen, rows, fen, repo, n_pcs):
            kept += 1
    return scanned, kept


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    token = load_hf_token()
    os.environ["HF_TOKEN"] = token
    api = HfApi(token=token)
    OUT.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []
    if SEED_PATH.exists():
        for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            seen.add(fen_key(rec["fen"]))
            rows.append(rec)
        log(f"resume extra_seeds={len(rows):,}")
    stats: dict[str, dict] = {}
    dest = ROOT / ".hf_cache" / "endgame_seeds"
    dest.mkdir(parents=True, exist_ok=True)
    for repo, cap in SOURCES:
        before = len(rows)
        target = before + cap
        files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        files.sort()
        scanned = 0
        log(f"{repo} parquet={len(files)} want+={cap:,}")
        for name in files:
            if len(rows) >= target:
                break
            try:
                path = hf_hub_download(repo, name, repo_type="dataset", token=token, local_dir=str(dest / repo.replace("/", "_")))
                s, k = scan_parquet(Path(path), repo, seen, rows, target)
                scanned += s
            except Exception as exc:
                log(f"  skip {name}: {type(exc).__name__}: {exc}")
                continue
            if k:
                log(f"  {name} scanned={scanned:,} kept_repo={len(rows) - before:,} total={len(rows):,}")
        stats[repo] = {"parquet": len(files), "scanned": scanned, "kept": len(rows) - before}
        tmp = SEED_PATH.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, SEED_PATH)
        SUMMARY_PATH.write_text(json.dumps({"n": len(rows), "lo": LO, "hi": HI, "by_repo": stats}, indent=2), encoding="utf-8")
        log(f"  done {repo} +{len(rows) - before:,} total={len(rows):,}")
    log(f"wrote {SEED_PATH} n={len(rows):,}")


if __name__ == "__main__":
    main()
