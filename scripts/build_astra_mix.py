#!/usr/bin/env python3
"""Build Astra's hypothesized next mix. Does not start GPU training.

Drawn-sample hypothesis (not a proven optimum):
  40% fresh Lichess MultiPV  |  30% existing SF19  |  15% SWA mistakes
  10% official Lichess puzzles  |  5% Syzygy

The trainer still has three slots. Assemble writes:
  soft  = Lichess + SF19 + puzzles, sized 50 / 37.5 / 12.5 of the soft pool
          so uniform soft draws are 40 / 30 / 10 of all samples
  bonus = verified mistakes at --bonus-mix-frac 0.15
  deep  = Syzygy at --deep-mix-frac 0.05

Frozen overnight holdouts and their safe flips stay out of every train pool.
Fresh Lichess also excludes positions already trained overnight.

Usage:
  MOVE_VOCAB_VERSION=compact python3 -u scripts/build_astra_mix.py --audit
  MOVE_VOCAB_VERSION=compact python3 -u scripts/build_astra_mix.py --lichess
  MOVE_VOCAB_VERSION=compact python3 -u scripts/build_astra_mix.py --puzzles
  MOVE_VOCAB_VERSION=compact python3 -u scripts/build_astra_mix.py --assemble
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

from autoresearch_8gb.pipeline import (  # noqa: E402
    attach_static_targets,
    audit_soft_targets,
    concat_soft_tables,
    filter_disjoint,
    position_hashes,
)
from build_hf_elo_mix import (  # noqa: E402
    build_soft,
    filter_excluded,
    save_hashes,
)
from exp193_puzzle_soft_harvest import (  # noqa: E402
    board_to_arr,
    castling_byte,
    phase_id,
    phase_name,
    puzzle_to_record,
)
from harvest_swa_mistakes import I_TO_TAG, TAG_TO_I  # noqa: E402
from move_vocab import UCI_TO_IDX  # noqa: E402

import chess  # noqa: E402

SUBSTANTIAL_TAGS = frozenset({
    TAG_TO_I["inaccuracy"],
    TAG_TO_I["blunder"],
    TAG_TO_I["conversion"],
    TAG_TO_I["major"],
})

OVERNIGHT = ROOT / "outputs/sf19_ft/overnight_20260908"
DEFAULT_OUT = ROOT / "outputs/astra_mix"
SOURCE_LICHESS = 1
SOURCE_SYZYGY = 2
SOURCE_MISTAKE = 3
SOURCE_SF19 = 4
SOURCE_PUZZLE = 5
RATING_BUCKETS = (
    (600, 1199),
    (1200, 1599),
    (1600, 1999),
    (2000, 2399),
    (2400, 3500),
)
SHARES = {
    "lichess": 0.40,
    "sf19": 0.30,
    "mistakes": 0.15,
    "puzzles": 0.10,
    "syzygy": 0.05,
}


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def load_hf_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'").strip('"')
            break


def shares_for_pool(pool_n: int) -> dict[str, int]:
    """Integer row targets for a working pool. Rounding residue goes to Lichess."""
    out = {k: int(round(pool_n * f)) for k, f in SHARES.items()}
    out["lichess"] += pool_n - sum(out.values())
    if out["lichess"] < 0:
        raise ValueError(f"pool_n={pool_n} too small for share rounding")
    return out


def soft_internal_counts(pool_n: int) -> dict[str, int]:
    """Row counts inside the composed soft cache (80% of draws)."""
    t = shares_for_pool(pool_n)
    return {"lichess": t["lichess"], "sf19": t["sf19"], "puzzles": t["puzzles"]}


def puzzle_rating_bucket(rating: int) -> int | None:
    for i, (lo, hi) in enumerate(RATING_BUCKETS):
        if lo <= rating <= hi:
            return i
    return None


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_cache(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def load_cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def hashes_of(path: Path) -> np.ndarray:
    data = load_cache(path)
    hs = np.unique(position_hashes(data).astype(np.uint64))
    del data
    return hs


def load_blocked(out: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for p in (
        OVERNIGHT / "val_manifest_soft.json",
        OVERNIGHT / "val_manifest_deep.json",
        OVERNIGHT / "val_manifest_replay.json",
        ROOT / "outputs/sf19_ft/run2/val_manifest_soft.json",
        ROOT / "outputs/sf19_ft/run2/val_manifest_deep.json",
        out / "blocked_hashes.npy",
    ):
        if p.suffix == ".npy" and p.exists():
            chunks.append(np.asarray(np.load(p), dtype=np.uint64))
            continue
        if not p.exists():
            continue
        raw = json.loads(p.read_text()).get("blocked_hashes") or []
        if raw:
            chunks.append(np.asarray(raw, dtype=np.uint64))
    if not chunks:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(chunks))


def used_cache_paths() -> list[Path]:
    paths = [
        OVERNIGHT / "soft_cache.pt",
        OVERNIGHT / "replay_cache.pt",
        OVERNIGHT / "deep_cache.pt",
        ROOT / "outputs/sf19_ft/soft_cache.pt",
        ROOT / "outputs/hf_elo_mix/soft_cache.pt",
        ROOT / "outputs/hf_elo_mix/deep_cache.pt",
    ]
    gen = OVERNIGHT / "generated_verified"
    if gen.is_dir():
        paths.extend(sorted(gen.glob("*.pt")))
    return [p for p in paths if p.exists()]


def collect_used_hashes(log_path: Path | None = None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for p in used_cache_paths():
        hs = hashes_of(p)
        log(f"  used {p.name} unique={hs.size:,}", log_path)
        chunks.append(hs)
    if not chunks:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(chunks))


def tag_source(data: dict, source: int) -> dict:
    n = int(data["board_array"].shape[0])
    data["source"] = torch.full((n,), source, dtype=torch.int8)
    return data


def subsample(data: dict, n_take: int, rng: random.Random) -> dict:
    n = int(data["board_array"].shape[0])
    if n_take >= n:
        return data
    idx = list(range(n))
    rng.shuffle(idx)
    take = torch.tensor(idx[:n_take], dtype=torch.long)
    return {k: v[take].contiguous() if torch.is_tensor(v) else v for k, v in data.items()}


def drop_blocked(data: dict, blocked: np.ndarray) -> tuple[dict, int]:
    if blocked.size == 0:
        return data, 0
    exclude = {int(x) for x in blocked.tolist()}
    before = int(data["board_array"].shape[0])
    data = filter_excluded(data, exclude)
    return data, before - int(data["board_array"].shape[0])


def record_to_tensors(rec: dict) -> dict | None:
    fen, best = rec.get("fen"), rec.get("best_move")
    soft = rec.get("soft_targets") or []
    if not fen or not best or best not in UCI_TO_IDX or not soft:
        return None
    try:
        board = chess.Board(fen)
        mv = chess.Move.from_uci(best)
    except Exception:
        return None
    if mv not in board.legal_moves:
        return None
    idx, pr = [], []
    for it in soft[:8]:
        u = it.get("uci")
        if u and u in UCI_TO_IDX:
            idx.append(UCI_TO_IDX[u])
            pr.append(float(it.get("prob", 0)))
    if not idx:
        return None
    z = sum(pr) or 1.0
    pr = [p / z for p in pr]
    while len(idx) < 8:
        idx.append(-1)
        pr.append(0.0)
    return {
        "board_array": torch.tensor(board_to_arr(board), dtype=torch.int8),
        "turn": torch.tensor(0 if board.turn else 1, dtype=torch.int8),
        "castling": torch.tensor(castling_byte(board), dtype=torch.int8),
        "ep_square": torch.tensor(
            board.ep_square if board.ep_square is not None else 0, dtype=torch.int8
        ),
        "move_idx": torch.tensor(UCI_TO_IDX[best], dtype=torch.int64),
        "cp": torch.tensor(int(rec.get("best_cp", 500) or 500), dtype=torch.int32),
        "mate": torch.tensor(int(rec.get("mate", 0) or 0), dtype=torch.int32),
        "soft_indices": torch.tensor(idx, dtype=torch.int64),
        "soft_probs": torch.tensor(pr, dtype=torch.float32),
        "phase": torch.tensor(phase_id(rec.get("phase") or phase_name(board)), dtype=torch.int8),
        "label_depth": torch.tensor(0, dtype=torch.int16),
        "puzzle_rating": torch.tensor(int(rec.get("puzzle_rating", 0) or 0), dtype=torch.int16),
        "source": torch.tensor(SOURCE_PUZZLE, dtype=torch.int8),
        "value_valid": torch.tensor(0, dtype=torch.int8),
    }


def stack_rows(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}


def run_audit(out: Path) -> dict:
    log_path = out / "audit.log"
    log("audit: uniqueness and overlap", log_path)
    blocked = load_blocked(out)
    report: dict = {"blocked": int(blocked.size), "sources": {}, "overlap": {}}
    named: dict[str, np.ndarray] = {}
    specs = [
        ("sf19_overnight", OVERNIGHT / "soft_cache.pt"),
        ("lichess_replay", OVERNIGHT / "replay_cache.pt"),
        ("syzygy", OVERNIGHT / "deep_cache.pt"),
        ("sf19_raw", ROOT / "outputs/sf19_ft/soft_cache.pt"),
        ("hf_replay", ROOT / "outputs/hf_elo_mix/soft_cache.pt"),
        ("hf_syzygy", ROOT / "outputs/hf_elo_mix/deep_cache.pt"),
    ]
    for name, path in specs:
        if not path.exists():
            log(f"  miss {name}", log_path)
            continue
        data = load_cache(path)
        n = int(data["board_array"].shape[0])
        hs = position_hashes(data).astype(np.uint64)
        unique = np.unique(hs)
        dups = n - int(unique.size)
        vs_block = int(np.isin(hs, blocked).sum()) if blocked.size else 0
        depth = {}
        if "label_depth" in data:
            d = data["label_depth"]
            depth = {
                "min": int(d.min()),
                "p50": int(d.float().median()),
                "max": int(d.max()),
                "ge12": int((d >= 12).sum()),
            }
        named[name] = unique
        report["sources"][name] = {
            "path": str(path),
            "rows": n,
            "unique": int(unique.size),
            "duplicate_rows": dups,
            "overlap_blocked": vs_block,
            "depth": depth,
        }
        log(
            f"  {name} rows={n:,} unique={unique.size:,} dups={dups:,} "
            f"blocked={vs_block:,} depth={depth}",
            log_path,
        )
        del data
    keys = list(named)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            both = int(np.intersect1d(named[a], named[b], assume_unique=True).size)
            report["overlap"][f"{a}∩{b}"] = both
            log(f"  overlap {a} ∩ {b} = {both:,}", log_path)
    gen = OVERNIGHT / "generated_verified"
    if gen.is_dir():
        gen_u = np.zeros(0, dtype=np.uint64)
        for p in sorted(gen.glob("*.pt")):
            hs = hashes_of(p)
            gen_u = np.unique(np.concatenate([gen_u, hs])) if gen_u.size else hs
        report["sources"]["sf19_generated"] = {"unique": int(gen_u.size)}
        for name, hs in named.items():
            report["overlap"][f"generated∩{name}"] = int(
                np.intersect1d(gen_u, hs, assume_unique=True).size
            )
    write_json(out / "audit.json", report)
    save_hashes(out / "blocked_hashes.npy", {int(x) for x in blocked.tolist()})
    log(f"wrote {out / 'audit.json'}", log_path)
    return report


def run_lichess(out: Path, n_target: int, seed: int) -> dict:
    log_path = out / "lichess.log"
    load_hf_token()
    blocked = load_blocked(out)
    log("lichess: collecting overnight used hashes", log_path)
    used = collect_used_hashes(log_path)
    exclude = {int(x) for x in np.unique(np.concatenate([blocked, used])).tolist()}
    log(f"lichess: exclude={len(exclude):,} target={n_target:,} seed={seed}", log_path)
    soft = build_soft(n_target, seed, max_shards=None, exclude=exclude)
    soft, dropped = drop_blocked(soft, blocked)
    soft, hs, stats = filter_disjoint(soft, used)
    if int(soft["board_array"].shape[0]) > n_target:
        soft = subsample(soft, n_target, random.Random(seed))
        hs = position_hashes(soft).astype(np.uint64)
    tag_source(soft, SOURCE_LICHESS)
    overlap = int(np.isin(hs, used).sum()) if used.size else 0
    if overlap:
        raise SystemExit(f"fresh Lichess still overlaps used set: {overlap}")
    path = out / "lichess_cache.pt"
    save_cache(soft, path)
    save_hashes(out / "lichess_hashes.npy", {int(x) for x in np.unique(hs).tolist()})
    report = {
        "n": int(soft["board_array"].shape[0]),
        "dropped_blocked": dropped,
        "disjoint": stats,
        "depth": {
            "min": int(soft["label_depth"].min()),
            "p50": int(soft["label_depth"].float().median()),
            "max": int(soft["label_depth"].max()),
        },
        "phase": {str(i): int((soft["phase"] == i).sum()) for i in range(3)},
        "path": str(path),
        "excluded_prior": len(exclude),
        "seed": seed,
    }
    write_json(out / "lichess_report.json", report)
    log(f"wrote {path} n={report['n']:,} {report['depth']}", log_path)
    return report


def run_puzzles(out: Path, n_target: int, seed: int) -> dict:
    log_path = out / "puzzles.log"
    load_hf_token()
    blocked = load_blocked(out)
    used = np.zeros(0, dtype=np.uint64)
    for p in (out / "lichess_hashes.npy", out / "blocked_hashes.npy"):
        if p.exists():
            extra = np.asarray(np.load(p), dtype=np.uint64)
            used = np.unique(np.concatenate([used, extra])) if used.size else extra
    per_bucket = max(1, n_target // len(RATING_BUCKETS))
    buckets: list[list[dict]] = [[] for _ in RATING_BUCKETS]
    skipped = 0
    scanned = 0
    t0 = time.time()
    log(
        f"puzzles: target={n_target:,} per_bucket={per_bucket:,} "
        f"buckets={list(RATING_BUCKETS)}",
        log_path,
    )
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    files = [
        f for f in list_repo_files("Lichess/chess-puzzles", repo_type="dataset")
        if f.endswith(".parquet")
    ]
    rng = random.Random(seed)
    rng.shuffle(files)
    done = False
    for fname in files:
        if done:
            break
        local = Path(hf_hub_download("Lichess/chess-puzzles", fname, repo_type="dataset"))
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=4096, columns=[
            "PuzzleId", "FEN", "Moves", "Rating", "Themes",
        ]):
            cols = batch.to_pydict()
            n = len(cols["FEN"])
            for i in range(n):
                scanned += 1
                themes = cols["Themes"][i]
                puzzle = {
                    "PuzzleId": cols["PuzzleId"][i],
                    "FEN": cols["FEN"][i],
                    "Moves": cols["Moves"][i],
                    "Rating": int(cols["Rating"][i] or 0),
                    "Themes": themes,
                }
                rec = puzzle_to_record(puzzle, 0, 99_999)
                if rec is None:
                    skipped += 1
                    continue
                b = puzzle_rating_bucket(int(rec["puzzle_rating"]))
                if b is None or len(buckets[b]) >= per_bucket:
                    skipped += 1
                    continue
                buckets[b].append(rec)
            if scanned % 50_000 < 4096:
                filled = [len(x) for x in buckets]
                log(
                    f"  scanned={scanned:,} filled={filled} skipped={skipped:,} "
                    f"{scanned / max(time.time() - t0, 1e-6):.0f}/s",
                    log_path,
                )
            if all(len(x) >= per_bucket for x in buckets):
                done = True
                break
    rows: list[dict] = []
    for recs in buckets:
        rng.shuffle(recs)
        for rec in recs[:per_bucket]:
            packed = record_to_tensors(rec)
            if packed is not None:
                rows.append(packed)
    if not rows:
        raise SystemExit("no puzzle rows survived the adapter")
    data = stack_rows(rows)
    data, dropped = drop_blocked(data, blocked)
    data, hs, stats = filter_disjoint(data, used if used.size else None)
    tag_source(data, SOURCE_PUZZLE)
    path = out / "puzzles_cache.pt"
    save_cache(data, path)
    ratings = data["puzzle_rating"].numpy()
    report = {
        "n": int(data["board_array"].shape[0]),
        "scanned": scanned,
        "skipped": skipped,
        "dropped_blocked": dropped,
        "disjoint": stats,
        "rating": {
            "min": int(ratings.min()),
            "p50": int(np.median(ratings)),
            "max": int(ratings.max()),
            "buckets": {
                f"{lo}-{hi}": int(((ratings >= lo) & (ratings <= hi)).sum())
                for lo, hi in RATING_BUCKETS
            },
        },
        "path": str(path),
        "note": "FEN is after the opponent setup move; target is the solver move",
    }
    write_json(out / "puzzles_report.json", report)
    log(f"wrote {path} n={report['n']:,} rating={report['rating']}", log_path)
    return report


def apply_value_valid(data: dict) -> dict:
    n = int(data["board_array"].shape[0])
    if "source" in data:
        data["value_valid"] = (data["source"] != SOURCE_PUZZLE).to(torch.int8)
    else:
        data["value_valid"] = torch.ones(n, dtype=torch.int8)
    return data


def priority_dedupe(ordered: list[tuple[str, dict]]) -> tuple[dict[str, dict], dict]:
    """First source keeps its rows. Later sources lose overlapping hashes."""
    seen = None
    out: dict[str, dict] = {}
    stats: dict = {}
    for name, data in ordered:
        data, hs, st = filter_disjoint(data, seen)
        seen = hs if seen is None else np.unique(np.concatenate([seen, hs]))
        out[name] = data
        stats[name] = st
    return out, stats


def _iter_mistake_shards(root: Path) -> list[Path]:
    """READY shards, including analyzed/inbox written by _write_shard."""
    if not root.exists():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(root.rglob("soft_cache.pt")):
        if p.parent.name.startswith("shard_") and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _load_mistake_shards(root: Path, *, in_pv_only: bool, substantial_only: bool) -> list[dict]:
    chunks = []
    for p in _iter_mistake_shards(root):
        ready = (p.parent / "READY").exists()
        analyzed = "analyzed" in p.parts
        if not ready and not analyzed:
            continue
        d = load_cache(p)
        n = int(d["board_array"].shape[0])
        keep = np.ones(n, dtype=bool)
        if in_pv_only and "needs_sf" in d:
            keep &= d["needs_sf"].numpy() == 0
            if "tag" in d:
                keep &= d["tag"].numpy() != TAG_TO_I["off_pv"]
        if substantial_only and "tag" in d:
            keep &= np.isin(d["tag"].numpy(), np.fromiter(SUBSTANTIAL_TAGS, dtype=np.int8))
        if not keep.any():
            continue
        chunks.append({k: v[keep] for k, v in d.items() if torch.is_tensor(v)})
    return chunks


def _tag_hist(data: dict | None) -> dict[str, int]:
    if data is None or "tag" not in data:
        return {}
    counts = Counter(int(x) for x in data["tag"].tolist())
    return {I_TO_TAG.get(k, str(k)): v for k, v in sorted(counts.items())}


def filter_substantial(data: dict) -> dict:
    if "tag" not in data:
        return data
    keep = np.isin(data["tag"].numpy(), np.fromiter(SUBSTANTIAL_TAGS, dtype=np.int8))
    return {k: v[keep] for k, v in data.items() if torch.is_tensor(v)}


def load_verified_mistakes(limit: int, blocked: np.ndarray, *, pad_ok: bool = False) -> dict | None:
    """Analyzed SF relabels win. Do not pad with OK / harmless disagreements."""
    analyzed = _load_mistake_shards(
        ROOT / "outputs/swa_mistakes/analyzed", in_pv_only=False, substantial_only=not pad_ok,
    )
    inbox_inpv = _load_mistake_shards(
        ROOT / "outputs/swa_mistakes/inbox", in_pv_only=True, substantial_only=not pad_ok,
    )
    if not analyzed and not inbox_inpv:
        return None
    data = None
    if analyzed:
        data = concat_soft_tables(analyzed)
        data, _ = drop_blocked(data, blocked)
        data, _, _ = filter_disjoint(data, None)
        if not pad_ok:
            data = filter_substantial(data)
    if inbox_inpv:
        extra = concat_soft_tables(inbox_inpv)
        extra, _ = drop_blocked(extra, blocked)
        extra, _, _ = filter_disjoint(extra, position_hashes(data) if data is not None else None)
        if not pad_ok:
            extra = filter_substantial(extra)
        if extra and int(extra["board_array"].shape[0]) > 0:
            data = extra if data is None else concat_soft_tables([data, extra])
    if data is None or int(data["board_array"].shape[0]) == 0:
        return None
    n = int(data["board_array"].shape[0])
    if n > limit:
        if "drop_cp" in data:
            order = torch.argsort(data["drop_cp"], descending=True)[:limit]
            data = {k: v[order].contiguous() if torch.is_tensor(v) else v for k, v in data.items()}
        else:
            data = subsample(data, limit, random.Random(19))
    tag_source(data, SOURCE_MISTAKE)
    return data


def audit_existing_bonus(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    data = load_cache(path)
    n = int(data["board_array"].shape[0])
    tags = _tag_hist(data)
    substantial = 0
    if "tag" in data:
        substantial = int(np.isin(data["tag"].numpy(), np.fromiter(SUBSTANTIAL_TAGS, dtype=np.int8)).sum())
    drop = data["drop_cp"].numpy() if "drop_cp" in data else None
    return {
        "exists": True,
        "path": str(path),
        "n": n,
        "tags": tags,
        "substantial_n": substantial,
        "ok_n": tags.get("ok", 0),
        "ok_frac": tags.get("ok", 0) / n if n else None,
        "needs_sf_all_zero": bool("needs_sf" in data and int(data["needs_sf"].sum()) == 0),
        "drop_cp": {
            "min": int(drop.min()) if drop is not None and drop.size else None,
            "p50": int(np.median(drop)) if drop is not None and drop.size else None,
            "mean": float(drop.mean()) if drop is not None and drop.size else None,
            "max": int(drop.max()) if drop is not None and drop.size else None,
        },
        "label_depth_mean": float(data["label_depth"].float().mean()) if "label_depth" in data else None,
        "provenance": "inbox in-PV teacher labels ranked by drop_cp; 100k analyzed shards were not on the load path",
    }


def run_assemble(out: Path, pool_n: int, allow_partial: bool) -> dict:
    log_path = out / "assemble.log"
    targets = shares_for_pool(pool_n)
    blocked = load_blocked(out)
    log(f"assemble pool={pool_n:,} targets={targets}", log_path)

    bonus_audit = audit_existing_bonus(DEFAULT_OUT / "bonus_cache.pt")
    write_json(out / "mistakes_audit.json", bonus_audit)
    log(
        f"prior bonus n={bonus_audit.get('n')} ok_frac={bonus_audit.get('ok_frac')} "
        f"substantial={bonus_audit.get('substantial_n')}",
        log_path,
    )

    mistakes = load_verified_mistakes(targets["mistakes"], blocked, pad_ok=False)
    if mistakes is None:
        log("assemble: no substantial verified corrections", log_path)
        if not allow_partial:
            write_json(out / "assemble_pending.json", {
                "ready": False,
                "reason": "no substantial SWA corrections after excluding OK/holdouts",
                "targets": targets,
                "prior_bonus_audit": bonus_audit,
            })
            raise SystemExit("mistakes missing after OK filter. Re-run harvest/analyze or --allow-partial.")
        mistake_hs = np.zeros(0, dtype=np.uint64)
    else:
        tag_source(mistakes, SOURCE_MISTAKE)
        mistake_hs = position_hashes(mistakes).astype(np.uint64)
        n_m = int(mistakes["board_array"].shape[0])
        drawn_8k = int(round(8000 * 64 * 0.15))
        log(
            f"assemble: corrections n={n_m:,} unique={int(np.unique(mistake_hs).size):,} "
            f"tags={_tag_hist(mistakes)} cap={targets['mistakes']:,} "
            f"repeat_if_8k={drawn_8k / max(n_m, 1):.3f}",
            log_path,
        )

    def _ordinary(name: str, data: dict, n_take: int, seed: int, source: int) -> dict:
        data, dropped = drop_blocked(data, blocked)
        data, _, st = filter_disjoint(data, mistake_hs if mistake_hs.size else None)
        if st.get("vs_seen"):
            log(f"  {name} dropped {st['vs_seen']:,} rows already covered by corrections", log_path)
        data = subsample(data, n_take, random.Random(seed))
        return tag_source(data, source)

    lich_path = out / "lichess_cache.pt"
    if not lich_path.exists():
        raise SystemExit("run --lichess first")
    puz_path = out / "puzzles_cache.pt"
    if not puz_path.exists():
        raise SystemExit("run --puzzles first")
    sf19_path = OVERNIGHT / "soft_cache.pt"
    if not sf19_path.exists():
        raise SystemExit(f"missing {sf19_path}")
    syz_path = OVERNIGHT / "deep_cache.pt"
    if not syz_path.exists():
        syz_path = ROOT / "outputs/hf_elo_mix/deep_cache.pt"

    ordinary = [
        ("lichess", _ordinary("lichess", load_cache(lich_path), targets["lichess"], 1, SOURCE_LICHESS)),
        ("sf19", _ordinary("sf19", load_cache(sf19_path), targets["sf19"], 2, SOURCE_SF19)),
        ("puzzles", _ordinary("puzzles", load_cache(puz_path), targets["puzzles"], 3, SOURCE_PUZZLE)),
        ("syzygy", _ordinary("syzygy", load_cache(syz_path), targets["syzygy"], 4, SOURCE_SYZYGY)),
    ]
    ordered = ([("mistakes", mistakes)] if mistakes is not None else []) + ordinary
    sources, pstats = priority_dedupe(ordered)
    actual = {}
    for name, data in sources.items():
        apply_value_valid(data)
        attach_static_targets(data)
        audit = audit_soft_targets(data, max_rows=len(data["board_array"]))
        if not audit["ok"]:
            raise SystemExit(f"{name} target audit failed: {audit}")
        sources[name] = data
        actual[name] = int(data["board_array"].shape[0])
        save_cache(data, out / f"{name}_ready.pt")
        log(f"  {name} n={actual[name]:,} audit={audit['ok']} dedupe={pstats.get(name)}", log_path)

    soft = concat_soft_tables([sources["lichess"], sources["sf19"], sources["puzzles"]])
    attach_static_targets(soft)
    save_cache(soft, out / "soft_cache.pt")
    save_cache(sources["syzygy"], out / "deep_cache.pt")
    if "mistakes" in sources:
        save_cache(sources["mistakes"], out / "bonus_cache.pt")

    for name in ("soft", "replay", "deep"):
        src = OVERNIGHT / f"eval_{name}.pt"
        if src.exists():
            dest = out / f"eval_{name}.pt"
            if not dest.exists():
                dest.write_bytes(src.read_bytes())

    total = sum(actual.values())
    drawn = {k: actual.get(k, 0) / total if total else 0.0 for k in SHARES}
    n_m = actual.get("mistakes", 0)
    drawn_8k = int(round(8000 * 64 * 0.15))
    report = {
        "pool_n_requested": pool_n,
        "targets": targets,
        "actual": actual,
        "drawn_if_uniform_concat": drawn,
        "corrections": {
            "n": n_m,
            "unique": n_m,
            "tags": _tag_hist(sources["mistakes"]) if "mistakes" in sources else {},
            "padded_with_ok": False,
            "target_was_not_forced": n_m < targets["mistakes"],
            "repeat_exposure_if_8000x64x0.15": drawn_8k / max(n_m, 1),
            "prior_bonus_audit": bonus_audit,
        },
        "dedupe": pstats,
        "trainer": {
            "soft": str(out / "soft_cache.pt"),
            "bonus": str(out / "bonus_cache.pt") if "mistakes" in sources else None,
            "deep": str(out / "deep_cache.pt"),
            "bonus_mix_frac": 0.15 if "mistakes" in sources else 0.0,
            "deep_mix_frac": 0.05,
            "note": "soft is Lichess+SF19+puzzles sized so uniform soft draws match 40/30/10 of all samples when bonus=0.15 and deep=0.05. Correction bucket is substantial tags only.",
        },
        "holdouts": "trainer must use overnight eval_*.pt via --external-eval; blocked hashes excluded from train",
        "ready": "mistakes" in sources,
    }
    write_json(out / "mix_report.json", report)
    log(f"assemble done actual={actual}", log_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--lichess", action="store_true")
    ap.add_argument("--puzzles", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--audit-mistakes", action="store_true")
    ap.add_argument("--go", action="store_true", help="audit + lichess + puzzles")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--pool-n", type=int, default=4_000_000)
    ap.add_argument("--lichess-n", type=int, default=0, help="Override Lichess rows (0 = share of --pool-n)")
    ap.add_argument("--puzzles-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    targets = shares_for_pool(args.pool_n)
    lich_n = args.lichess_n or targets["lichess"]
    puz_n = args.puzzles_n or targets["puzzles"]
    ran = False
    if args.audit_mistakes:
        audit = audit_existing_bonus(Path(args.output_dir) if Path(args.output_dir).is_absolute() else ROOT / args.output_dir)
        if not Path(str(audit.get("path", ""))).exists():
            audit = audit_existing_bonus(DEFAULT_OUT / "bonus_cache.pt")
        dest = out / "mistakes_audit.json"
        write_json(dest, audit)
        print(json.dumps(audit, indent=2))
        ran = True
    if args.audit or args.go:
        run_audit(out)
        ran = True
    if args.lichess or args.go:
        run_lichess(out, lich_n, args.seed)
        ran = True
    if args.puzzles or args.go:
        run_puzzles(out, puz_n, args.seed + 1)
        ran = True
    if args.assemble:
        run_assemble(out, args.pool_n, args.allow_partial)
        ran = True
    if not ran:
        print("Pass --audit, --lichess, --puzzles, --assemble, or --go")
        print(f"default pool {args.pool_n:,}: {targets}")


if __name__ == "__main__":
    main()
