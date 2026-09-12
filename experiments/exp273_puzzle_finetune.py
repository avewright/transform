#!/usr/bin/env python3
"""exp273: 99M squares64 finetune on Lichess/chess-puzzles (80/20).

Loads avewright/chess-transformer-100m-squares64 as a weights-only warm
start. Packs official Lichess puzzles (solver plies after the opponent
setup move) with a PuzzleId-level 80/20 split.

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp273_puzzle_finetune.py --pull
  MOVE_VOCAB_VERSION=compact python experiments/exp273_puzzle_finetune.py --pack
  MOVE_VOCAB_VERSION=compact python experiments/exp273_puzzle_finetune.py --go
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64, count_parameters
from move_vocab import UCI_TO_IDX, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp273_puzzle_finetune"
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"
PUZZLE_REPO = "Lichess/chess-puzzles"
PUZZLE_REVISION = "479ea9bc9f681385f5adb23fa27a96c2dc8ae599"
SOURCE_PUZZLE = 5
SPLIT_SEED = 273
TRAIN_PCT = 80
EXPECTED_99M_PARAMS = 98_971_224

KEYS = (
    "board_array", "turn", "castling", "ep_square", "move_idx",
    "cp", "mate", "soft_indices", "soft_probs", "source", "value_valid",
    "label_depth", "phase", "split",
)


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(
            f"Expected compact vocab 1968, got {VOCAB_SIZE}. "
            "Export MOVE_VOCAB_VERSION=compact."
        )


def split_of(puzzle_id: str, *, seed: int = SPLIT_SEED, train_pct: int = TRAIN_PCT) -> int:
    """0 = train, 1 = eval. Stable across processes; PuzzleId-level."""
    raw = f"{int(seed)}:{puzzle_id}".encode()
    n = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")
    return 0 if (n % 100) < int(train_pct) else 1


def board_to_arr(board) -> list[int]:
    arr = [0] * 64
    for sq, piece in board.piece_map().items():
        arr[sq] = piece.piece_type if piece.color else piece.piece_type + 6
    return arr


def castling_byte(board) -> int:
    c = 0
    if board.has_kingside_castling_rights(True):
        c |= 1
    if board.has_queenside_castling_rights(True):
        c |= 2
    if board.has_kingside_castling_rights(False):
        c |= 4
    if board.has_queenside_castling_rights(False):
        c |= 8
    return c


def phase_id(board) -> int:
    pieces = sum(1 for p in board.piece_map().values() if p.piece_type != 6)
    if pieces >= 20:
        return 0
    if pieces >= 10:
        return 1
    return 2


def play_puzzle(puzzle: dict, *, min_rating: int = 0, max_rating: int = 4000) -> list[tuple]:
    """Label every solver ply after the opponent setup move.

    Official FEN is before that setup move. Returns tuples:
    (board[64], turn, castling, ep, move_idx, phase)
    """
    rating = int(puzzle.get("Rating") or 0)
    if rating < min_rating or rating > max_rating:
        return []
    fen = puzzle.get("FEN")
    moves = (puzzle.get("Moves") or "").split()
    if not fen or len(moves) < 2:
        return []
    import chess
    try:
        board = chess.Board(fen)
    except Exception:
        return []
    rows: list[tuple] = []
    for i, uci in enumerate(moves):
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            break
        if mv not in board.legal_moves:
            break
        if i % 2 == 1:
            mid = UCI_TO_IDX.get(uci)
            if mid is None:
                break
            ep = -1 if board.ep_square is None else int(board.ep_square)
            rows.append((
                board_to_arr(board),
                0 if board.turn else 1,
                castling_byte(board),
                ep,
                int(mid),
                phase_id(board),
            ))
        board.push(mv)
        if board.is_game_over(claim_draw=True):
            break
    return rows


def _stack_rows(rows: list[tuple], split: int) -> dict[str, torch.Tensor] | None:
    n = len(rows)
    if n == 0:
        return None
    boards = np.asarray([r[0] for r in rows], dtype=np.int8)
    turn = np.asarray([r[1] for r in rows], dtype=np.int8)
    castle = np.asarray([r[2] for r in rows], dtype=np.int8)
    ep = np.asarray([r[3] for r in rows], dtype=np.int8)
    mid = np.asarray([r[4] for r in rows], dtype=np.int64)
    phase = np.asarray([r[5] for r in rows], dtype=np.int8)
    si = np.full((n, 8), -1, dtype=np.int64)
    sp = np.zeros((n, 8), dtype=np.float32)
    si[:, 0] = mid
    sp[:, 0] = 1.0
    return {
        "board_array": torch.from_numpy(boards),
        "turn": torch.from_numpy(turn),
        "castling": torch.from_numpy(castle),
        "ep_square": torch.from_numpy(ep),
        "move_idx": torch.from_numpy(mid),
        "cp": torch.zeros(n, dtype=torch.int32),
        "mate": torch.zeros(n, dtype=torch.int32),
        "soft_indices": torch.from_numpy(si),
        "soft_probs": torch.from_numpy(sp),
        "source": torch.full((n,), SOURCE_PUZZLE, dtype=torch.int8),
        "value_valid": torch.zeros(n, dtype=torch.int8),
        "label_depth": torch.zeros(n, dtype=torch.int16),
        "phase": torch.from_numpy(phase),
        "split": torch.full((n,), int(split), dtype=torch.int8),
    }


def cat_tables(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = [k for k in KEYS if k in parts[0]]
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in keys}


def drop_eval_overlap(train: dict, ev: dict) -> tuple[dict, int]:
    """Drop train rows whose position hash appears in eval."""
    from autoresearch_8gb.pipeline import position_hashes

    th = position_hashes(train)
    eh = np.unique(position_hashes(ev))
    mask = torch.from_numpy(~np.isin(th, eh))
    dropped = int((~mask).sum())
    if dropped == 0:
        return train, 0
    return {k: v[mask] for k, v in train.items()}, dropped


def _worker_init() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _pack_record_batch(payload: tuple) -> tuple[list, list, int, int]:
    records, min_rating, max_rating, seed, train_pct = payload
    train_rows: list[tuple] = []
    eval_rows: list[tuple] = []
    skipped = 0
    for rec in records:
        rows = play_puzzle(rec, min_rating=min_rating, max_rating=max_rating)
        if not rows:
            skipped += 1
            continue
        dest = train_rows if split_of(str(rec.get("PuzzleId") or ""), seed=seed, train_pct=train_pct) == 0 else eval_rows
        dest.extend(rows)
    return train_rows, eval_rows, skipped, len(records)


def pack_puzzles(
    out: Path,
    *,
    max_puzzles: int | None = None,
    min_rating: int = 0,
    max_rating: int = 4000,
    workers: int = 16,
    revision: str = PUZZLE_REVISION,
) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    files = [
        f for f in list_repo_files(PUZZLE_REPO, repo_type="dataset", revision=revision)
        if f.endswith(".parquet")
    ]
    files.sort()
    print(f"pack {PUZZLE_REPO}@{revision} files={len(files)} dest={out}", flush=True)

    train_parts: list[dict] = []
    eval_parts: list[dict] = []
    scanned = skipped = 0
    pending: list[dict] = []
    chunk = 1024
    n_workers = max(1, int(workers))
    pool = None
    if n_workers > 1:
        pool = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init)

    def _flush(batch: list[dict]) -> None:
        nonlocal skipped, scanned
        if not batch:
            return
        chunks = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]
        payloads = [(c, min_rating, max_rating, SPLIT_SEED, TRAIN_PCT) for c in chunks]
        if pool is None:
            results = [_pack_record_batch(p) for p in payloads]
        else:
            results = list(pool.map(_pack_record_batch, payloads, chunksize=1))
        tr: list[tuple] = []
        ev: list[tuple] = []
        for t, e, sk, n in results:
            tr.extend(t)
            ev.extend(e)
            skipped += sk
            scanned += n
        stacked_t = _stack_rows(tr, 0)
        stacked_e = _stack_rows(ev, 1)
        if stacked_t is not None:
            train_parts.append(stacked_t)
        if stacked_e is not None:
            eval_parts.append(stacked_e)
        print(
            f"  scanned={scanned:,} train_rows={sum(int(p['turn'].shape[0]) for p in train_parts):,} "
            f"eval_rows={sum(int(p['turn'].shape[0]) for p in eval_parts):,} skipped={skipped:,}",
            flush=True,
        )

    try:
        for fname in files:
            if max_puzzles is not None and scanned >= max_puzzles:
                break
            local = Path(hf_hub_download(PUZZLE_REPO, fname, repo_type="dataset", revision=revision))
            pf = pq.ParquetFile(local)
            cols = ["PuzzleId", "FEN", "Moves", "Rating"]
            for batch in pf.iter_batches(batch_size=8192, columns=cols):
                if max_puzzles is not None and scanned + len(pending) >= max_puzzles:
                    break
                d = batch.to_pydict()
                n = len(d["FEN"])
                for i in range(n):
                    if max_puzzles is not None and scanned + len(pending) >= max_puzzles:
                        break
                    pending.append({
                        "PuzzleId": d["PuzzleId"][i],
                        "FEN": d["FEN"][i],
                        "Moves": d["Moves"][i],
                        "Rating": d["Rating"][i],
                    })
                    if len(pending) >= chunk * n_workers:
                        _flush(pending)
                        pending = []
                if max_puzzles is not None and scanned >= max_puzzles:
                    break
            print(f"  file {fname} scanned={scanned:,}", flush=True)
        _flush(pending)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if not train_parts or not eval_parts:
        raise SystemExit("pack produced empty train or eval")
    train = cat_tables(train_parts)
    ev = cat_tables(eval_parts)
    train, n_overlap = drop_eval_overlap(train, ev)
    train_path = out / "puzzle_train.pt"
    eval_path = out / "puzzle_eval.pt"
    torch.save(train, train_path)
    torch.save(ev, eval_path)
    report = {
        "status": "packed",
        "repo": PUZZLE_REPO,
        "revision": revision,
        "train_n": int(train["turn"].shape[0]),
        "eval_n": int(ev["turn"].shape[0]),
        "scanned_puzzles": scanned,
        "skipped_puzzles": skipped,
        "overlap_dropped": n_overlap,
        "train_pct": TRAIN_PCT,
        "split_seed": SPLIT_SEED,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "note": "PuzzleId-level 80/20. Solver plies after opponent setup. value_valid=0.",
    }
    (out / "pack.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACK", json.dumps(report, indent=2), flush=True)
    return report


def pull_99m(dest: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    dest = dest or (ROOT / "outputs" / "hf_models" / "99m")
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = Path(hf_hub_download(TEACHER_REPO, "latest.pt", local_dir=str(dest)))
    Path(hf_hub_download(TEACHER_REPO, "model_config.json", local_dir=str(dest)))
    print(f"pulled {TEACHER_REPO} -> {ckpt}", flush=True)
    return ckpt


def write_init_ckpt(src: Path, dest: Path) -> Path:
    from autoresearch_8gb.pipeline import load_model_state

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = load_model_state(ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt})
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "config": ckpt.get("config") if isinstance(ckpt, dict) else None,
            "eval_only": True,
            "steps": 0,
            "note": "99M weights-only warm start for exp273 puzzle finetune",
            "source": str(src),
        },
        dest,
    )
    print(f"init ckpt {dest} keys={len(state)}", flush=True)
    return dest


def trial_config() -> dict:
    model = DEFAULT_100M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp273_puzzle_finetune",
        "arch": "squares64",
        "desc": "99M squares64 weights-only FT on Lichess/chess-puzzles, PuzzleId 80/20.",
        "init": {
            "repo": TEACHER_REPO,
            "params": EXPECTED_99M_PARAMS,
        },
        "data": {
            "repo": PUZZLE_REPO,
            "revision": PUZZLE_REVISION,
            "train_pct": TRAIN_PCT,
            "split_seed": SPLIT_SEED,
        },
        "model": model,
        "train": {
            "batch_size": 384,
            "min_batch_size": 32,
            "max_batch_size": 1280,
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.0,
            "soft_temp": 0.0,
            "soft_temp_weight": 0.0,
            "deep_mix_frac": 0.0,
            "bonus_mix_frac": 0.0,
            "quality_mix_frac": 0.0,
            "puzzle_mix_frac": 0.0,
            "use_swa": False,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
            "optimizer": "polar_normuon",
            "compile_polar": True,
            "force_lr": True,
            "muon_lr": 0.002,
            "adam_lr": 3e-5,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup": 200,
            "min_lr_frac": 0.1,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": True,
            "max_vram_gb": 40.0,
            "save_every_steps": 250,
            "keep_step_every": 1000,
            "keep_last_ckpts": 4,
            "val_every_steps": 250,
            "val_eval_n": 2048,
            "elo_every_steps": 0,
        },
    }


def train(args: argparse.Namespace) -> dict:
    from autoresearch_8gb.train_trial import train_trial

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_src = Path(args.checkpoint) if args.checkpoint else ROOT / "outputs" / "hf_models" / "99m" / "latest.pt"
    if not ckpt_src.exists():
        ckpt_src = pull_99m(ckpt_src.parent)
    init_path = out / "init.pt"
    if not init_path.exists() or args.refresh_init:
        write_init_ckpt(ckpt_src, init_path)

    train_cache = Path(args.soft_cache)
    eval_cache = Path(args.eval_cache)
    if not train_cache.exists() or not eval_cache.exists():
        pack_puzzles(
            out,
            max_puzzles=args.max_puzzles,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            workers=args.workers,
        )
        train_cache = out / "puzzle_train.pt"
        eval_cache = out / "puzzle_eval.pt"
    if not train_cache.exists() or not eval_cache.exists():
        raise SystemExit(f"missing caches train={train_cache} eval={eval_cache}")

    trial = trial_config()
    cfg = trial["train"]
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
        cfg["max_batch_size"] = int(args.batch_size)
        cfg["fill_vram"] = False
    if args.muon_lr is not None:
        cfg["muon_lr"] = float(args.muon_lr)
    if args.adam_lr is not None:
        cfg["adam_lr"] = float(args.adam_lr)
    if args.warmup is not None:
        cfg["warmup"] = int(args.warmup)
    if args.max_steps is not None:
        pass
    if args.val_every is not None:
        cfg["val_every_steps"] = int(args.val_every)
    if args.save_every is not None:
        cfg["save_every_steps"] = int(args.save_every)
    if args.fill_vram is not None:
        cfg["fill_vram"] = bool(args.fill_vram)
    cfg["external_eval"] = {"puzzles": str(eval_cache.resolve())}

    result = train_trial(
        trial,
        out,
        soft_cache=train_cache,
        deep_cache=None,
        max_steps=args.max_steps,
        max_minutes=args.train_minutes,
        smoke=False,
        resume_ckpt=init_path,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="Pull 99M if needed, pack if needed, finetune")
    ap.add_argument("--pull", action="store_true", help="Download 99M checkpoint")
    ap.add_argument("--pack", action="store_true", help="Pack Lichess/chess-puzzles 80/20")
    ap.add_argument("--refresh-init", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="99M latest.pt (default outputs/hf_models/99m/latest.pt)")
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--soft-cache", default=str(OUT_DIR / "puzzle_train.pt"))
    ap.add_argument("--eval-cache", default=str(OUT_DIR / "puzzle_eval.pt"))
    ap.add_argument("--max-puzzles", type=int, default=None)
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--max-rating", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=25_000)
    ap.add_argument("--train-minutes", type=float, default=720.0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--muon-lr", type=float, default=None)
    ap.add_argument("--adam-lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--fill-vram", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_100M_SQUARES64_CONFIG
    print(
        f"exp273 99M puzzle FT  {cfg.hidden_dim}d/{cfg.num_heads}H "
        f"effective={cfg.effective_depth}  split={TRAIN_PCT}/{100 - TRAIN_PCT}",
        flush=True,
    )
    if args.pull:
        pull_99m()
    if args.pack:
        pack_puzzles(
            Path(args.output_dir),
            max_puzzles=args.max_puzzles,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            workers=args.workers,
        )
    if args.go:
        n = count_parameters(build_squares64(cfg))
        print(f"params={n:,} expected={EXPECTED_99M_PARAMS:,}", flush=True)
        train(args)
        return
    if not args.pull and not args.pack:
        n = count_parameters(build_squares64(cfg))
        print(f"params≈{n:,} — pass --go to finetune, --pack to build caches", flush=True)


if __name__ == "__main__":
    main()
