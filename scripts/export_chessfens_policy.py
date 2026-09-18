#!/usr/bin/env python3
"""Map existing soft-cache HF datasets into ChessFENS policy rows.

Writes a *new* dataset only. Never uploads into the source repos.
Each row is {fen, policy[1858]} with -1 on unused LC0 slots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, create_repo

from chess_chessbot import CHESSBOT_UCI_TO_IDX, CHESSFENS_POLICY_SIZE
from data_loader import _hf_token, _maybe_load_hf_token_from_env
from move_vocab import IDX_TO_UCI, _CASTLE_960_TO_STD
from scripts.build_organized_chess_mix import pack_puzzle
from value99_sfwdl import board_array_to_fen

DEFAULT_SOURCES = (
    "avewright/local-soft-positions",
    "avewright/chess-soft-sf19",
    "avewright/stockfish-19-soft-targets",
    "avewright/endgame-dataset",
    "avewright/chess-soft-multipv-lichess",
    "avewright/lichess-opening-bestline",
    "avewright/lichess-middlegame-bestline",
    "Lichess/chess-puzzles",
)
PUZZLE_COLS = ("FEN", "Moves", "Rating")
NEW_REPO = "avewright/chessfens-policy"
SHARD = 5000
COLUMNS = ("board_array", "turn", "castling", "ep_square", "move_idx", "soft_indices", "soft_probs")


def compact_to_lc0(idx: int, board: chess.Board | None = None) -> int:
    uci = IDX_TO_UCI[int(idx)]
    if uci in _CASTLE_960_TO_STD:
        piece = board.piece_at(chess.parse_square(uci[:2])) if board is not None else None
        if piece is None or piece.piece_type == chess.KING:
            uci = _CASTLE_960_TO_STD[uci]
    if uci.endswith("n"):
        uci = uci[:-1]
    j = CHESSBOT_UCI_TO_IDX.get(uci, -1)
    return j if 0 <= j < CHESSFENS_POLICY_SIZE else -1


def soft_to_chessfens_policy(move_idx, soft_indices, soft_probs, board: chess.Board | None = None):
    mass: dict[int, float] = {}
    for i, p in zip(list(soft_indices), list(soft_probs)):
        if p is None or float(p) <= 0 or int(i) < 0:
            continue
        j = compact_to_lc0(int(i), board)
        if j >= 0:
            mass[j] = mass.get(j, 0.0) + float(p)
    hard = compact_to_lc0(int(move_idx), board)
    if hard >= 0 and hard not in mass:
        mass[hard] = 0.0
    if not mass:
        return None
    total = sum(mass.values())
    pol = [-1.0] * CHESSFENS_POLICY_SIZE
    if total <= 0:
        pol[hard] = 1.0
        return pol
    for j, p in mass.items():
        pol[j] = p / total
    return pol


def convert_row(rec: dict, source: str):
    try:
        fen = board_array_to_fen(rec["board_array"], rec["turn"], rec["castling"], rec["ep_square"])
        board = chess.Board(fen)
    except Exception:
        return None, "bad_fen"
    policy = soft_to_chessfens_policy(rec["move_idx"], rec["soft_indices"], rec["soft_probs"], board)
    if policy is None:
        return None, "unmapped_policy"
    return dict(fen=fen, policy=policy, source=source), None


def list_parquets(api: HfApi, repo: str) -> list[str]:
    return sorted(p for p in api.list_repo_files(repo, repo_type="dataset") if p.endswith(".parquet"))


def puzzle_to_soft(puzzle: dict) -> dict | None:
    packed, _ = pack_puzzle(puzzle)
    if packed is None:
        return None
    return dict(
        board_array=packed["board_array"].numpy(),
        turn=int(packed["turn"]),
        castling=int(packed["castling"]),
        ep_square=int(packed["ep_square"]),
        move_idx=int(packed["move_idx"]),
        soft_indices=packed["soft_indices"].tolist(),
        soft_probs=packed["soft_probs"].tolist(),
    )


def iter_source_rows(fs: HfFileSystem, repo: str, files: list[str]):
    for name in files:
        uri = f"datasets/{repo}/{name}"
        with fs.open(uri, "rb", block_size=1 << 20) as handle:
            pf = pq.ParquetFile(handle)
            names = set(pf.schema_arrow.names)
            if set(COLUMNS) <= names:
                cols = list(COLUMNS)
                mode = "soft"
            elif {"FEN", "Moves"} <= names:
                cols = [c for c in PUZZLE_COLS if c in names]
                mode = "puzzle"
            else:
                continue
            for batch in pf.iter_batches(batch_size=256, columns=cols):
                for rec in batch.to_pylist():
                    if mode == "puzzle":
                        rec = puzzle_to_soft(rec)
                        if rec is None:
                            continue
                    yield rec


def write_shard(rows: list[dict], path: Path):
    table = pa.table({
        "fen": [r["fen"] for r in rows],
        "policy": [r["policy"] for r in rows],
        "source": [r["source"] for r in rows],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def readme(n: int, sources: list[str]) -> str:
    src = "\n".join(f"- `{s}`" for s in sources)
    return f"""---
license: mit
task_categories: [other]
tags: [chess, policy, chessfens, lc0, compact-vocab]
size_categories: 1M<n<10M
---

# ChessFENS-format policy (mapped)

Projection of existing Avewright soft-cache datasets onto the
[Maxlegrec/ChessFENS](https://huggingface.co/datasets/Maxlegrec/ChessFENS) policy layout.

**This repo is new.** Source datasets are not modified.

## Columns

| column | meaning |
| --- | --- |
| `fen` | 6-field FEN (clocks unknown → `0 1`) |
| `policy` | 1858 LC0 slots. Unused = `-1`. Mass renormalized onto mapped moves. |
| `source` | originating HF dataset id |

Compact-1968 `soft_indices` / `move_idx` are mapped through UCI into the ChessBot/LC0 table.
Knight promotions drop the trailing `n`. Chess960 castling codes become standard UCI.
Moves that land past 1858 (ChessBot black-underpromo suffix) are dropped.

No WDL column. Do not treat these SF MultiPV labels as LC0 search.

## Sources

{src}

`{n:,}` rows. `data/shard_XXXXXX.parquet`, 5,000 rows each.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=NEW_REPO)
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    p.add_argument("--out", type=Path, default=ROOT / "outputs/chessfens_policy")
    p.add_argument("--limit", type=int, default=0, help="Stop after N converted rows (0 = all)")
    p.add_argument("--push", action="store_true")
    p.add_argument("--every", type=int, default=10, help="Upload every N new shards")
    a = p.parse_args()
    out = a.out
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    state_path = out / "export.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "done": [], "shard": 0, "rows": 0, "skipped": {},
    }
    done = set(state["done"])
    _maybe_load_hf_token_from_env()
    token = _hf_token() if a.push else None
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    if a.push:
        create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True, token=token)
    rows: list[dict] = []
    pending_uploads: list[Path] = []

    def flush():
        if not rows:
            return
        path = data / f"shard_{state['shard']:06d}.parquet"
        write_shard(rows, path)
        state["rows"] += len(rows)
        state["shard"] += 1
        rows.clear()
        pending_uploads.append(path)
        state_path.write_text(json.dumps(state))
        print(json.dumps({"wrote": path.name, "rows": state["rows"]}), flush=True)

    def push_pending(force=False):
        if not a.push or not pending_uploads:
            return
        if not force and len(pending_uploads) < a.every:
            return
        extras = []
        readme_path = out / "README.md"
        readme_path.write_text(readme(state["rows"], a.sources))
        extras.append(readme_path)
        for path in pending_uploads + extras:
            dest = f"data/{path.name}" if path.suffix == ".parquet" else path.name
            api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                            repo_id=a.repo, repo_type="dataset", token=token)
            print(json.dumps({"uploaded": dest}), flush=True)
        pending_uploads.clear()

    for source in a.sources:
        files = list_parquets(api, source)
        for name in files:
            key = f"{source}:{name}"
            if key in done:
                continue
            skipped = 0
            kept = 0
            for rec in iter_source_rows(fs, source, [name]):
                converted, reason = convert_row(rec, source)
                if converted is None:
                    skipped += 1
                    state["skipped"][reason] = state["skipped"].get(reason, 0) + 1
                    continue
                rows.append(converted)
                kept += 1
                if len(rows) >= SHARD:
                    flush()
                    push_pending()
                if a.limit and state["rows"] + len(rows) >= a.limit:
                    flush()
                    state_path.write_text(json.dumps(state))
                    push_pending(force=True)
                    print(json.dumps({"done": True, "rows": state["rows"], "skipped": state["skipped"]}))
                    return
            done.add(key)
            state["done"] = sorted(done)
            state_path.write_text(json.dumps(state))
            print(json.dumps({"src": key, "kept": kept, "skipped": skipped, "total": state["rows"]}), flush=True)
    flush()
    push_pending(force=True)
    print(json.dumps({"done": True, "rows": state["rows"], "shards": state["shard"], "skipped": state["skipped"]}))


if __name__ == "__main__":
    main()
