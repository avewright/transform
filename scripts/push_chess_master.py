#!/usr/bin/env python3
"""Upload chess_master_v1 without saturating the LAN.

One file at a time, hf_transfer off, hard byte-rate cap, pause between files.
Already-uploaded paths are skipped.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import time
from pathlib import Path

# Kill the fast parallel uploader before huggingface_hub imports.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from chess_master.io_util import json_read, json_write, load_hf_token

DEFAULT_REPO = "avewright/chess-master-v1"
SRC = ROOT / "outputs/chess_master_v1"
IGNORE_NAMES = {"progress.json", ".DS_Store"}


class RateLimitedFile(io.BufferedIOBase):
    """Binary reader that sleeps to keep average throughput under a cap."""

    def __init__(self, path: Path, bytes_per_sec: int):
        super().__init__()
        self._f = path.open("rb")
        self._bps = max(16_384, int(bytes_per_sec))
        self._allowance = float(self._bps)
        self._last = time.monotonic()
        self.name = str(path)
        self.mode = "rb"

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, n: int = -1):
        data = self._f.read(n)
        self._pace(len(data))
        return data

    def readinto(self, b):
        n = self._f.readinto(b)
        self._pace(n or 0)
        return n

    def _pace(self, n: int) -> None:
        if n <= 0:
            return
        now = time.monotonic()
        self._allowance += (now - self._last) * self._bps
        self._last = now
        cap = self._bps * 1.5
        if self._allowance > cap:
            self._allowance = cap
        if self._allowance >= n:
            self._allowance -= n
            return
        time.sleep((n - self._allowance) / self._bps)
        self._allowance = 0.0
        self._last = time.monotonic()

    def seek(self, offset, whence=0):
        return self._f.seek(offset, whence)

    def tell(self):
        return self._f.tell()

    def close(self):
        if getattr(self, "_f", None):
            self._f.close()
        super().close()


def write_card(src: Path, repo: str) -> None:
    quality = json_read(src / "quality_report.json", {})
    mix = json_read(src / "mix_ingest_report.json", {})
    recipe_src = ROOT / "chess_master/recipes/pilot_45_35_15_5.json"
    recipe_dest = src / "recipes" / "pilot_45_35_15_5.json"
    recipe_dest.parent.mkdir(parents=True, exist_ok=True)
    if recipe_src.exists():
        shutil.copy2(recipe_src, recipe_dest)
    schema_src = ROOT / "docs/chess_master_v1.md"
    if schema_src.exists():
        shutil.copy2(schema_src, src / "SCHEMA.md")
    (src / "README.md").write_text(
        f"""---
license: mit
pretty_name: Chess master v1
task_categories:
- other
tags:
- chess
- stockfish
- soft-labels
- tablebase
- puzzles
- multipv
configs:
- config_name: positions
  data_files: positions/*.parquet
- config_name: annotations
  data_files: annotations/*.parquet
- config_name: membership
  data_files: membership/*.parquet
- config_name: quarantine
  data_files: quarantine/*.parquet
---

# {repo}

Linked master chess tables: **positions**, **annotations**, and **membership**.
One position can keep SF19, Lichess MultiPV, a puzzle line, Syzygy, and a model
mistake at the same time. Flat training mixes are recipe exports, not the source
of truth.

Join on `position_id`. Absent metadata is `null` / `unknown`, never coerced to 0
or "verified".

## Load

```python
from datasets import load_dataset
positions = load_dataset("{repo}", "positions", split="train")
annotations = load_dataset("{repo}", "annotations", split="train")
membership = load_dataset("{repo}", "membership", split="train")
```

## Counts

| Table | Rows |
|---|---|
| Mix positions / annotations / membership | {mix.get("counts", {}).get("positions", 1_004_000):,} each |
| Unique `position_id` | {quality.get("unique_positions", 0):,} |
| Train ∩ eval | {quality.get("split_overlaps", {}).get("organized_v1_train_eval_position_overlap", 0)} |
| Value-eligible | {quality.get("annotation_coverage", {}).get("value_eligible", 0):,} (SF19 only) |
| Depth sentinels | {quality.get("annotation_coverage", {}).get("depth_sentinels", 0):,} |
| Disagreeing best-move positions | {quality.get("conflicts", {}).get("positions_with_disagreeing_best_moves", 0)} |

See `quality_report.json` and `SCHEMA.md`.

## Label contracts

- SF19 CP/mate/WDL are White-absolute. Lichess perspective is unverified.
- `soft_probs` are **probabilities**, not logits. The transform is recorded.
- Puzzle FEN is before the opponent setup move. Keep the full line. No invented CP/WDL.
- Syzygy DTZ is not mate. Value stays masked until conversion is verified.
- Off-PV model moves have unknown regret, not an assumed penalty.
- Recipes must not silently relax quality gates.

## Pinned sources

| Source | Revision | License |
|---|---|---|
| `avewright/chess-soft-sf19` | `68cef6c9ba62c62f904f25305f1a7489dab825e0` | MIT |
| `avewright/chess-soft-multipv-lichess` | `291acea2a2ce6fc2aba4b289110d1920cf969afe` | MIT |
| `avewright/chess-soft-syzygy` | `3889985c482837aa3082e182d8f123e005e2c0d4` | MIT |
| `Lichess/chess-puzzles` | `479ea9bc9f681385f5adb23fa27a96c2dc8ae599` | CC0-1.0 |
| `avewright/chess-soft-100m-swa-mistakes` | `69b75e12ab2867c8b9a63d385065dfe646cf86e1` | MIT |

Baseline recipe: `recipes/pilot_45_35_15_5.json` (45% SF19 / 35% Lichess / 15% puzzles / 5% Syzygy).
That mix is a baseline, not an optimum.

This pack is **not** a claim that eval rows are unseen by older checkpoints.
""",
        encoding="utf-8",
    )


def local_files(src: Path) -> list[Path]:
    out = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        if p.name in IGNORE_NAMES or p.suffix == ".tmp":
            continue
        out.append(p)
    return out


def remote_paths(api, repo: str) -> set[str]:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as exc:
        print(f"list_repo_files: {type(exc).__name__}: {exc}", flush=True)
        return set()
    return set(files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--private", action="store_true")
    p.add_argument("--kbps", type=int, default=0, help="Upload cap in KiB/s (0 = no cap)")
    p.add_argument("--pause", type=float, default=0.0, help="Seconds to wait between files")
    p.add_argument("--force", action="store_true", help="Re-upload files already on the hub")
    args = p.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    if not (src / "quality_report.json").exists():
        raise SystemExit(f"missing master tables at {src}")

    load_hf_token()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")

    write_card(src, args.repo)
    man = json_read(src / "input_manifest.json", {})
    man["do_not_upload"] = False
    man["hf_repo"] = args.repo
    json_write(src / "input_manifest.json", man)

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    already = remote_paths(api, args.repo)
    files = local_files(src)
    bps = args.kbps * 1024 if args.kbps > 0 else 0
    cap = f"{args.kbps} KiB/s" if args.kbps > 0 else "uncapped"
    print(
        f"upload {len(files)} files, cap={cap}, pause={args.pause}s, "
        f"already_on_hub={len(already)} -> https://huggingface.co/datasets/{args.repo}",
        flush=True,
    )

    sent = skipped = 0
    for i, path in enumerate(files, 1):
        rel = path.relative_to(src).as_posix()
        if not args.force and rel in already:
            skipped += 1
            print(f"[{i}/{len(files)}] skip {rel}", flush=True)
            continue
        size = path.stat().st_size
        print(f"[{i}/{len(files)}] {rel} ({size / 1e6:.1f} MB)", flush=True)
        last_err = None
        for attempt in range(1, 6):
            try:
                if bps > 0:
                    with RateLimitedFile(path, bps) as fh:
                        api.upload_file(
                            path_or_fileobj=fh,
                            path_in_repo=rel,
                            repo_id=args.repo,
                            repo_type="dataset",
                            token=token,
                        )
                else:
                    api.upload_file(
                        path_or_fileobj=str(path),
                        path_in_repo=rel,
                        repo_id=args.repo,
                        repo_type="dataset",
                        token=token,
                    )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                wait = min(60, 8 * (2 ** (attempt - 1)))
                print(
                    f"  retry {attempt}/5 {rel} wait={wait}s err={type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(wait)
        if last_err is not None:
            raise SystemExit(f"gave up on {rel}: {last_err}")
        already.add(rel)
        sent += 1
        time.sleep(args.pause)

    print(
        f"done sent={sent} skipped={skipped} "
        f"https://huggingface.co/datasets/{args.repo}",
        flush=True,
    )


if __name__ == "__main__":
    main()
