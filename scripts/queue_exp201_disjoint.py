#!/usr/bin/env python3
"""Build / consume disjoint 1.5M soft shards for exp201.

  # keep building unused shards (CPU, no GPU):
  python3 scripts/queue_exp201_disjoint.py --build --max-ready 6

  # merge any READY shards into the live mix (call before a continue segment):
  python3 scripts/queue_exp201_disjoint.py --consume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_hf_elo_mix import concat, position_hashes, save_hashes  # noqa: E402

LIVE = ROOT / "outputs" / "hf_elo_mix"
QUEUE = ROOT / "outputs" / "hf_elo_mix_queue"
HASHES = LIVE / "used_hashes.npy"
BUILDER = ROOT / "scripts" / "build_hf_elo_mix.py"
SOFT_N = 1_500_000


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_used_hashes() -> int:
    if HASHES.exists():
        arr = __import__("numpy").load(HASHES, allow_pickle=False)
        return int(arr.size)
    live = LIVE / "soft_cache.pt"
    if not live.exists():
        raise SystemExit(f"missing live mix {live}")
    data = torch.load(live, map_location="cpu", weights_only=False)
    hs = position_hashes(data)
    save_hashes(HASHES, hs)
    log(f"seeded {HASHES} n={len(hs):,} from live mix")
    return int(len(hs))


def next_shard_id() -> int:
    existing = [p.name for p in QUEUE.glob("shard_*") if p.is_dir()]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return (max(nums) + 1) if nums else 2  # shard_001 is the live mix


def ready_count() -> int:
    return sum(1 for p in QUEUE.glob("shard_*/READY") if p.is_file())


def build_one(soft_n: int) -> Path:
    QUEUE.mkdir(parents=True, exist_ok=True)
    seed_used_hashes()
    sid = next_shard_id()
    out = QUEUE / f"shard_{sid:03d}"
    if out.exists() and (out / "READY").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    seed = 1000 + sid  # not 42 — that is the live mix seed
    cmd = [
        sys.executable, "-u", str(BUILDER),
        "--go", "--soft-only",
        "--output-dir", str(out),
        "--soft-n", str(soft_n),
        "--seed", str(seed),
        "--exclude-hashes", str(HASHES),
        "--write-hashes", str(HASHES),
    ]
    log(f"building {out.name} seed={seed} n={soft_n:,}")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(f"build failed rc={rc} dir={out}")
    (out / "READY").write_text(f"n={soft_n} seed={seed}\n", encoding="utf-8")
    log(f"READY {out}")
    return out


def consume() -> dict:
    live_path = LIVE / "soft_cache.pt"
    shards = sorted(p.parent for p in QUEUE.glob("shard_*/READY"))
    if not shards:
        log("consume: no READY shards")
        return {"merged": 0, "soft_n": _soft_n(live_path)}
    live = torch.load(live_path, map_location="cpu", weights_only=False)
    chunks = [live]
    used = []
    for sh in shards:
        cache = sh / "soft_cache.pt"
        if not cache.exists():
            log(f"skip {sh}: missing soft_cache.pt")
            continue
        chunks.append(torch.load(cache, map_location="cpu", weights_only=False))
        used.append(sh)
    if len(chunks) == 1:
        return {"merged": 0, "soft_n": _soft_n(live_path)}
    merged = concat(chunks)
    n = int(merged["board_array"].shape[0])
    perm = torch.randperm(n)
    merged = {k: v[perm].contiguous() for k, v in merged.items()}
    tmp = live_path.with_suffix(".pt.next")
    torch.save(merged, tmp)
    tmp.replace(live_path)
    report = {
        "soft_n": n,
        "merged_shards": [p.name for p in used],
        "live": str(live_path),
    }
    (LIVE / "mix_report.json").write_text(json.dumps(report, indent=2))
    for sh in used:
        (sh / "READY").unlink(missing_ok=True)
        (sh / "CONSUMED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    log(f"consume: merged {len(used)} shards → live n={n:,}")
    return {"merged": len(used), "soft_n": n, "shards": [p.name for p in used]}


def _soft_n(path: Path) -> int:
    if not path.exists():
        return 0
    data = torch.load(path, map_location="cpu", weights_only=False)
    return int(data["board_array"].shape[0])


def build_loop(max_ready: int, soft_n: int, stop_file: Path) -> None:
    seed_used_hashes()
    while True:
        if stop_file.exists():
            log(f"stop file {stop_file}; exiting")
            return
        n_ready = ready_count()
        if n_ready >= max_ready:
            log(f"queue full ({n_ready}/{max_ready}); sleeping")
            time.sleep(120)
            continue
        build_one(soft_n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--consume", action="store_true")
    ap.add_argument("--max-ready", type=int, default=6, help="Unused shards to keep queued")
    ap.add_argument("--soft-n", type=int, default=SOFT_N)
    ap.add_argument("--stop-file", default=str(QUEUE / "STOP_QUEUE"))
    args = ap.parse_args()
    if args.consume:
        print(json.dumps(consume()), flush=True)
        return
    if args.build:
        build_loop(args.max_ready, args.soft_n, Path(args.stop_file))
        return
    raise SystemExit("pass --build or --consume")


if __name__ == "__main__":
    main()
