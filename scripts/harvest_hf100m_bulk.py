#!/usr/bin/env python3
"""Filter existing HF soft MultiPV rows through the 100M policy. No Stockfish.

This is the only practical path to ~10M lapse rows on a laptop. Playing games
at 600k-node MultiPV is ~2 kept pos/s. This path is model-forward bound.

Keeps rows where greedy policy != teacher best (move_idx). Soft targets stay
the HF teacher distribution.

Usage:
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_hf100m_bulk.py --go
  MOVE_VOCAB_VERSION=compact python -u scripts/harvest_hf100m_bulk.py --go --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_inference import load_checkpoint  # noqa: E402
from data_loader import _hf_token, board_array_to_fused, ep_square_to_file  # noqa: E402
from move_vocab import VOCAB_SIZE  # noqa: E402

SOURCE_HARVEST = 3
DEFAULT_REPO = "avewright/chess-soft-multipv-lichess"


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _list_parquets(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    token = _hf_token()
    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def _download(repo: str, name: str, token: str | None, attempts: int = 8) -> str:
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for i in range(attempts):
        try:
            return hf_hub_download(repo, name, repo_type="dataset", token=token)
        except Exception as exc:
            last = exc
            wait = min(120.0, 2.0 ** i)
            log(f"download retry {i + 1}/{attempts} {name}: {exc}; sleep {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {name} after {attempts} tries") from last


def _iter_tables(repo: str, files: list[str], rows_per_batch: int):
    import pyarrow.parquet as pq

    token = _hf_token()
    for name in files:
        local = _download(repo, name, token)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=rows_per_batch):
            yield batch


def _col(batch, name: str):
    return batch.column(name).to_numpy(zero_copy_only=False)


def _as_fixed(batch, name: str, width: int, dtype):
    """Arrow list/fixed-list → (N, width). Respects RecordBatch.slice offsets."""
    col = batch.column(name)
    n = len(col)
    if n == 0:
        return np.zeros((0, width), dtype=dtype)
    chunk = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    # flatten() honors offset/length. raw .values is the parent buffer and
    # will silently return the unsliced chunk after skip_seen resumes.
    if hasattr(chunk, "flatten"):
        flat = chunk.flatten().to_numpy(zero_copy_only=False)
    elif hasattr(chunk, "values") and chunk.values is not None:
        raw = chunk.values.to_numpy(zero_copy_only=False)
        off = int(getattr(chunk, "offset", 0))
        flat = raw[off * width:(off + n) * width]
    else:
        raise TypeError(f"{name}: cannot flatten Arrow column {type(chunk)}")
    out = np.ascontiguousarray(np.asarray(flat, dtype=dtype).reshape(n, width))
    if out.shape != (n, width):
        raise ValueError(f"{name}: expected {(n, width)}, got {out.shape}")
    return out


def _inbox_tally(out_dir: Path) -> tuple[int, int]:
    """Return (rows_already_written, next_shard_index)."""
    inbox = out_dir / "inbox"
    if not inbox.exists():
        return 0, 0
    n = 0
    nxt = 0
    for sh in inbox.glob("shard_*"):
        try:
            nxt = max(nxt, int(sh.name.split("_", 1)[1]) + 1)
        except ValueError:
            continue
        meta = sh / "meta.json"
        if meta.exists():
            n += int(json.loads(meta.read_text())["n"])
            continue
        cache = sh / "soft_cache.pt"
        if cache.exists():
            data = torch.load(cache, map_location="cpu", weights_only=False)
            n += int(data["move_idx"].shape[0])
    return n, nxt


@torch.no_grad()
def _predict(model, fused, turn, castling, ep_file, device, micro: int) -> torch.Tensor:
    preds = []
    n = fused.shape[0]
    for i in range(0, n, micro):
        sl = slice(i, i + micro)
        inp = {
            "fused_ids": fused[sl].to(device, non_blocking=True),
            "turn": turn[sl].to(device, non_blocking=True),
            "castling": castling[sl].to(device, non_blocking=True),
            "ep_file": ep_file[sl].to(device, non_blocking=True),
        }
        logits = model(inp)["policy_logits"].float()
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0)


def _pack(keep: dict) -> dict:
    return {
        "board_array": torch.from_numpy(keep["board_array"]),
        "turn": torch.from_numpy(keep["turn"]),
        "castling": torch.from_numpy(keep["castling"]),
        "ep_square": torch.from_numpy(keep["ep_square"]),
        "move_idx": torch.from_numpy(keep["move_idx"]),
        "cp": torch.from_numpy(keep["cp"]),
        "mate": torch.from_numpy(keep["mate"]),
        "soft_indices": torch.from_numpy(keep["soft_indices"]),
        "soft_probs": torch.from_numpy(keep["soft_probs"]),
        "label_depth": torch.from_numpy(keep["label_depth"]),
        "phase": torch.from_numpy(keep["phase"]),
        "source": torch.full((keep["move_idx"].shape[0],), SOURCE_HARVEST, dtype=torch.int8),
    }


def _write_shard(data: dict, out_dir: Path, shard_i: int) -> Path:
    sh = out_dir / "inbox" / f"shard_{shard_i:06d}"
    sh.mkdir(parents=True, exist_ok=True)
    path = sh / "soft_cache.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    n = int(data["move_idx"].shape[0])
    (sh / "meta.json").write_text(json.dumps({"n": n}, indent=2), encoding="utf-8")
    (sh / "READY").write_text(f"n={n}\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="outputs/hf100m_bulk")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--target", type=int, default=10_000_000)
    ap.add_argument("--batch-rows", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=100_000)
    ap.add_argument("--keep-all", action="store_true", help="Write every scored row (not just disagreements)")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--skip-seen", type=int, default=0,
                    help="Skip this many leading stream rows (resume after a prior scan).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (avoids long MPS stalls).")
    args = ap.parse_args()
    if not args.go and not args.smoke:
        raise SystemExit("pass --go or --smoke")
    if VOCAB_SIZE != 1968:
        raise SystemExit(f"need compact vocab, got {VOCAB_SIZE}")
    if args.smoke:
        args.target = min(args.target, 2048)
        args.batch_rows = min(args.batch_rows, 512)
        args.shard_size = min(args.shard_size, 2048)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    device = torch.device("cpu") if args.cpu else _device()
    already, shard_i = _inbox_tally(out)
    log(f"ckpt={args.ckpt} device={device} repo={args.repo} target={args.target:,} "
        f"keep_all={args.keep_all} micro={args.micro_batch} already={already:,} "
        f"next_shard={shard_i} skip_seen={args.skip_seen:,}", log_path)

    model = load_checkpoint(args.ckpt, device=device)
    files = _list_parquets(args.repo)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"no parquet files in {args.repo}")
    log(f"parquet files={len(files)}", log_path)

    buf = {k: [] for k in (
        "board_array", "turn", "castling", "ep_square", "move_idx",
        "cp", "mate", "soft_indices", "soft_probs", "label_depth", "phase",
    )}
    n_buf = 0
    n_keep = already
    n_seen = 0
    n_disagree = 0
    t0 = time.time()
    stop = False

    def _ask_stop(*_):
        nonlocal stop
        stop = True
        log("stop requested; finishing current batch then flush", log_path)

    signal.signal(signal.SIGINT, _ask_stop)
    signal.signal(signal.SIGTERM, _ask_stop)

    def flush(force: bool = False) -> None:
        nonlocal n_buf, shard_i
        if n_buf == 0 or (n_buf < args.shard_size and not force):
            return
        keep = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        path = _write_shard(_pack(keep), out, shard_i)
        log(f"wrote {path} n={n_buf:,} total_keep={n_keep:,}", log_path)
        (out / "cursor.json").write_text(json.dumps({
            "seen": n_seen, "kept": n_keep, "shard_i": shard_i + 1,
        }), encoding="utf-8")
        for v in buf.values():
            v.clear()
        n_buf = 0
        shard_i += 1
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    try:
        for batch in _iter_tables(args.repo, files, args.batch_rows):
            if stop or n_keep >= args.target:
                break
            n_batch = batch.num_rows
            if n_seen + n_batch <= args.skip_seen:
                n_seen += n_batch
                continue
            drop = max(0, args.skip_seen - n_seen)
            if drop:
                batch = batch.slice(drop, n_batch - drop)
                n_seen += drop
            if batch.num_rows == 0:
                continue
            ba = _as_fixed(batch, "board_array", 64, np.int8)
            turn = _col(batch, "turn").astype(np.int64, copy=False)
            if ba.shape[0] != len(turn):
                raise ValueError(
                    f"row mismatch after slice: board_array={ba.shape[0]} turn={len(turn)}"
                )
            castling = _col(batch, "castling").astype(np.int64, copy=False)
            ep = _col(batch, "ep_square").astype(np.int64, copy=False)
            move_idx = _col(batch, "move_idx").astype(np.int64, copy=False)
            fused = board_array_to_fused(torch.from_numpy(np.ascontiguousarray(ba)))
            pred = _predict(
                model, fused,
                torch.from_numpy(np.ascontiguousarray(turn)),
                torch.from_numpy(np.ascontiguousarray(castling)),
                ep_square_to_file(torch.from_numpy(np.ascontiguousarray(ep))),
                device, args.micro_batch,
            ).numpy()
            teacher = move_idx
            mask = np.ones(len(teacher), dtype=bool) if args.keep_all else (pred != teacher)
            n_seen += len(teacher)
            n_disagree += int((pred != teacher).sum())
            if not mask.any():
                continue
            si = _as_fixed(batch, "soft_indices", 8, np.int64)
            sp = _as_fixed(batch, "soft_probs", 8, np.float32)
            names = {
                "board_array": ba, "turn": _col(batch, "turn").astype(np.int8, copy=False),
                "castling": _col(batch, "castling").astype(np.int8, copy=False),
                "ep_square": _col(batch, "ep_square").astype(np.int8, copy=False),
                "move_idx": teacher, "cp": _col(batch, "cp").astype(np.int32, copy=False),
                "mate": _col(batch, "mate").astype(np.int32, copy=False),
                "soft_indices": si, "soft_probs": sp,
                "label_depth": _col(batch, "label_depth").astype(np.int16, copy=False)
                if "label_depth" in batch.schema.names
                else np.zeros(len(teacher), dtype=np.int16),
                "phase": _col(batch, "phase").astype(np.int8, copy=False)
                if "phase" in batch.schema.names
                else np.zeros(len(teacher), dtype=np.int8),
            }
            take = int(mask.sum())
            for k, v in names.items():
                buf[k].append(np.ascontiguousarray(v[mask]))
            n_buf += take
            n_keep += take
            elapsed = max(time.time() - t0, 1e-6)
            if n_seen % max(args.batch_rows, 1) < len(teacher):
                log(
                    f"seen={n_seen:,} keep={n_keep:,} disagree={n_disagree / max(n_seen - args.skip_seen, 1):.1%} "
                    f"scan={(n_seen - args.skip_seen) / elapsed:.0f}/s "
                    f"keep={(n_keep - already) / elapsed:.0f}/s "
                    f"[{elapsed:.0f}s]",
                    log_path,
                )
            flush(False)
            if n_keep >= args.target or stop:
                break
            if args.smoke and n_seen >= args.target:
                break
    finally:
        flush(True)

    summary = {
        "ckpt": args.ckpt,
        "repo": args.repo,
        "seen": n_seen,
        "kept": n_keep,
        "disagree": n_disagree,
        "disagree_rate": n_disagree / max(n_seen, 1),
        "secs": time.time() - t0,
        "keep_all": args.keep_all,
        "shards": shard_i,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary), log_path)


if __name__ == "__main__":
    main()
