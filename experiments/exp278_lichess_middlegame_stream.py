#!/usr/bin/env python3
"""exp278: 99M FT on streaming Lichess 16–26-piece best-line (one-hot).

Harvest writes READY inbox shards. This packs a frozen hash-holdout val,
then trains while absorbing new shards into train.

Warm start: avewright/chess-transformer-100m-squares64.
Upload target: avewright/middlegame-model.

  MOVE_VOCAB_VERSION=compact python experiments/exp278_lichess_middlegame_stream.py --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64, count_parameters
from exp273_puzzle_finetune import drop_eval_overlap, pull_99m, write_init_ckpt
from exp275_endgame_finetune import n_pieces, split_of
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp278_lichess_middlegame_stream"
INBOX = ROOT / "outputs" / "lichess_middlegame_bestline" / "inbox"
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"
EXPECTED_99M_PARAMS = 98_971_224
SPLIT_SEED = 278
TRAIN_PCT = 80
VAL_N = 8_192
MIN_PCS, MAX_PCS = 16, 26
SOURCE_LICHESS_MID = 8
CHUNK_STEPS = 200
MIN_TRAIN_START = 80_000
MAX_TRAIN_N = 45_000_000
PACK_SHARDS = 64


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit("Expected compact vocab 1968. Export MOVE_VOCAB_VERSION=compact.")


def ready_shards(inbox: Path) -> list[Path]:
    return [p.parent for p in sorted(inbox.glob("shard_*/READY")) if (p.parent / "soft_cache.pt").exists()]


def mark_attached(sh: Path, note: str) -> None:
    (sh / "READY").unlink(missing_ok=True)
    (sh / "ATTACHED").write_text(note + "\n", encoding="utf-8")


def load_shard(sh: Path) -> dict:
    return torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)


def tag_rows(table: dict) -> dict:
    n = int(table["turn"].shape[0])
    table = dict(table)
    table["source"] = torch.full((n,), SOURCE_LICHESS_MID, dtype=torch.int8)
    table["value_valid"] = torch.ones(n, dtype=torch.int8)
    return table


def keep_train_split(table: dict) -> dict:
    from autoresearch_8gb.pipeline import position_hashes

    hs = position_hashes(table)
    mask = np.fromiter(
        (split_of(str(int(h)), seed=SPLIT_SEED, train_pct=TRAIN_PCT) == 0 for h in hs.tolist()),
        dtype=np.bool_,
        count=int(hs.shape[0]),
    )
    if bool(mask.all()):
        return table
    idx = torch.from_numpy(np.flatnonzero(mask))
    return {k: v[idx] for k, v in table.items() if torch.is_tensor(v) and int(v.shape[0]) == int(mask.shape[0])}


def stratified_val_idx(pcs: np.ndarray, n_take: int, rng: np.random.Generator) -> np.ndarray:
    n_take = min(int(n_take), int(pcs.shape[0]))
    buckets = list(range(MIN_PCS, MAX_PCS + 1))
    per = max(1, n_take // len(buckets))
    chosen: list[np.ndarray] = []
    used = np.zeros(pcs.shape[0], dtype=bool)
    for p in buckets:
        idx = np.flatnonzero(pcs == p)
        rng.shuffle(idx)
        take = idx[: min(per, idx.size)]
        chosen.append(take)
        used[take] = True
    take = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    if take.size < n_take:
        rest = np.flatnonzero(~used)
        rng.shuffle(rest)
        take = np.concatenate([take, rest[: n_take - take.size]])
    return take.astype(np.int64)


def concat_tables(chunks: list[dict]) -> dict:
    keys = [k for k in chunks[0] if torch.is_tensor(chunks[0][k]) and all(k in c for c in chunks)]
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def pack_initial(out: Path, inbox: Path, *, val_n: int = VAL_N) -> dict:
    from autoresearch_8gb.pipeline import position_hashes

    shards = ready_shards(inbox)
    if not shards:
        raise SystemExit(f"no READY shards in {inbox}")
    if len(shards) > PACK_SHARDS:
        shards = shards[:PACK_SHARDS]
    chunks = [tag_rows(load_shard(sh)) for sh in shards]
    table = concat_tables(chunks)
    pcs = n_pieces(table)
    keep = (pcs >= MIN_PCS) & (pcs <= MAX_PCS)
    if int((~keep).sum()):
        idx = torch.from_numpy(np.flatnonzero(keep))
        table = {k: v[idx] for k, v in table.items() if torch.is_tensor(v) and int(v.shape[0]) == int(keep.shape[0])}
    hs = position_hashes(table)
    split = np.fromiter(
        (split_of(str(int(h)), seed=SPLIT_SEED, train_pct=TRAIN_PCT) for h in hs.tolist()),
        dtype=np.int8,
        count=int(hs.shape[0]),
    )
    table["split"] = torch.from_numpy(split)
    train = {k: v[table["split"] == 0] for k, v in table.items()}
    ev = {k: v[table["split"] != 0] for k, v in table.items()}
    train, n_overlap = drop_eval_overlap(train, ev)
    rng = np.random.default_rng(SPLIT_SEED)
    val_idx = stratified_val_idx(n_pieces(ev), val_n, rng)
    val = {k: v[torch.from_numpy(val_idx)] for k, v in ev.items()}
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "lichess_train.pt"
    eval_path = out / "lichess_eval.pt"
    torch.save(train, train_path)
    torch.save(val, eval_path)
    for sh in shards:
        mark_attached(sh, "initial_pack")
    report = {
        "status": "packed",
        "scanned": int(table["turn"].shape[0]),
        "train_n": int(train["turn"].shape[0]),
        "eval_full_n": int(ev["turn"].shape[0]),
        "val_n": int(val["turn"].shape[0]),
        "overlap_dropped": n_overlap,
        "val_method": "piece_stratified_hash_holdout_frozen",
        "piece_min": MIN_PCS,
        "piece_max": MAX_PCS,
        "val_piece_hist": {str(p): int((n_pieces(val) == p).sum()) for p in range(MIN_PCS, MAX_PCS + 1)},
        "note": "Frozen val. Later shards: train-hash only, one-hot best line, value_valid=1.",
        "hf_model": "avewright/middlegame-model",
        "hf_data": "avewright/lichess-middlegame-bestline",
    }
    (out / "pack.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACK", json.dumps(report, indent=2), flush=True)
    return report


def absorb_ready(out: Path, inbox: Path) -> int:
    from autoresearch_8gb.pipeline import concat_soft_tables, filter_disjoint, position_hashes

    train_path = out / "lichess_train.pt"
    eval_path = out / "lichess_eval.pt"
    if not train_path.exists() or not eval_path.exists():
        return 0
    shards = ready_shards(inbox)
    if not shards:
        return 0
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    val = torch.load(eval_path, map_location="cpu", weights_only=False)
    seen = np.unique(np.concatenate([
        position_hashes(train).astype(np.uint64, copy=False),
        position_hashes(val).astype(np.uint64, copy=False),
    ]))
    added = 0
    for sh in shards:
        data = tag_rows(load_shard(sh))
        data = keep_train_split(data)
        if int(data["turn"].shape[0]) == 0:
            mark_attached(sh, "absorbed kept=0 empty_train_split")
            print(f"absorb {sh.name} in=0 keep=0", flush=True)
            continue
        data, new_h, rep = filter_disjoint(data, seen)
        if rep["n_out"] > 0:
            train = concat_soft_tables([train, data])
            seen = np.unique(np.concatenate([seen, new_h.astype(np.uint64, copy=False)]))
            added += int(rep["n_out"])
        mark_attached(sh, f"absorbed kept={rep['n_out']}")
        print(f"absorb {sh.name} in={rep['n_in']:,} keep={rep['n_out']:,}", flush=True)
    if added:
        tmp = train_path.with_suffix(".pt.tmp")
        torch.save(train, tmp)
        os.replace(tmp, train_path)
    print(f"absorb_total +{added:,} train_n={int(train['turn'].shape[0]):,}", flush=True)
    return added


def inbox_n(inbox: Path) -> int:
    n = 0
    for sh in ready_shards(inbox):
        meta = sh / "meta.json"
        if meta.exists():
            n += int(json.loads(meta.read_text(encoding="utf-8")).get("n") or 0)
        else:
            n += int(load_shard(sh)["turn"].shape[0])
    return n


def inbox_cache_shards(inbox: Path) -> list[Path]:
    found: list[Path] = []
    for sh in sorted(inbox.glob("shard_*"), reverse=True):
        if (sh / "soft_cache.pt").exists() and (
            (sh / "READY").exists() or (sh / "ATTACHED").exists()
        ):
            found.append(sh)
    return found


def reload_train_capped(out: Path, inbox: Path, *, cap: int) -> int:
    from autoresearch_8gb.pipeline import concat_soft_tables, filter_disjoint, position_hashes

    eval_path = out / "lichess_eval.pt"
    seen = None
    if eval_path.exists():
        val = torch.load(eval_path, map_location="cpu", weights_only=False)
        seen = np.unique(position_hashes(val).astype(np.uint64, copy=False))
    chunks: list[dict] = []
    n = 0
    used = 0
    for sh in inbox_cache_shards(inbox):
        data = tag_rows(load_shard(sh))
        data, new_h, rep = filter_disjoint(data, seen)
        if rep["n_out"] <= 0:
            continue
        room = cap - n
        if room <= 0:
            break
        if rep["n_out"] > room:
            data = {
                k: (v[:room] if torch.is_tensor(v) and int(v.shape[0]) == rep["n_out"] else v)
                for k, v in data.items()
            }
            new_h = new_h[:room]
            rep = {**rep, "n_out": room}
        chunks.append(data)
        n += int(rep["n_out"])
        used += 1
        if seen is None:
            seen = np.unique(new_h.astype(np.uint64, copy=False))
        else:
            seen = np.unique(np.concatenate([seen, new_h.astype(np.uint64, copy=False)]))
        if used % 50 == 0:
            print(f"  reload shards={used} n={n:,}", flush=True)
    if not chunks:
        raise SystemExit("inbox empty; cannot reload train cache")
    train = concat_soft_tables(chunks)
    train_path = out / "lichess_train.pt"
    tmp = train_path.with_suffix(".pt.tmp")
    torch.save(train, tmp)
    os.replace(tmp, train_path)
    print(f"reloaded train_n={int(train['turn'].shape[0]):,} from {used} newest shards cap={cap:,}", flush=True)
    return int(train["turn"].shape[0])


def harvest_alive() -> bool:
    import subprocess

    r = subprocess.run(["pgrep", "-af", "build_lichess_middlegame_bestline"], capture_output=True, text=True)
    return any("build_lichess_middlegame_bestline" in ln and "cursor" not in ln for ln in (r.stdout or "").splitlines())


def ckpt_step(path: Path) -> int:
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("steps") or ckpt.get("step") or 0)


def trial_config() -> dict:
    model = DEFAULT_100M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp278_lichess_middlegame_stream",
        "arch": "squares64",
        "desc": "99M FT on streaming Lichess 16-26 best-line one-hot, frozen stratified val.",
        "init": {"repo": TEACHER_REPO, "params": EXPECTED_99M_PARAMS},
        "data": {"train_pct": TRAIN_PCT, "split_seed": SPLIT_SEED, "val_n": VAL_N},
        "model": model,
        "train": {
            "batch_size": 528,
            "min_batch_size": 32,
            "max_batch_size": 528,
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
            "warmup": 80,
            "min_lr_frac": 0.1,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": False,
            "max_vram_gb": 40.0,
            "save_every_steps": 150,
            "keep_step_every": 300,
            "keep_last_ckpts": 8,
            "val_every_steps": 100,
            "val_eval_n": VAL_N,
            "elo_every_steps": 0,
            "soft_inbox": str(INBOX),
            "max_soft_n": MAX_TRAIN_N,
        },
    }


def train_loop(args: argparse.Namespace) -> None:
    from autoresearch_8gb.train_trial import train_trial

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_src = Path(args.checkpoint) if args.checkpoint else ROOT / "outputs" / "hf_models" / "99m" / "latest.pt"
    if not ckpt_src.exists():
        ckpt_src = pull_99m(ckpt_src.parent)
    init_path = out / "init.pt"
    if not init_path.exists() or args.refresh_init:
        write_init_ckpt(ckpt_src, init_path)

    inbox = Path(args.inbox)
    print(f"waiting for READY inbox >= {MIN_TRAIN_START:,} in {inbox}", flush=True)
    while inbox_n(inbox) < MIN_TRAIN_START:
        if not harvest_alive() and inbox_n(inbox) == 0:
            raise SystemExit("harvest dead and inbox empty")
        time.sleep(3)
        print(f"  inbox_ready_n={inbox_n(inbox):,}", flush=True)

    if not (out / "lichess_eval.pt").exists() or args.repack:
        pack_initial(out, inbox, val_n=args.val_n)
    if args.reload_inbox or int(torch.load(out / "lichess_train.pt", map_location="cpu", weights_only=False)["turn"].shape[0]) < int(args.max_train_n) // 2:
        reload_train_capped(out, inbox, cap=int(args.max_train_n))

    trial = trial_config()
    cfg = trial["train"]
    cfg["soft_inbox"] = str(inbox)
    cfg["max_soft_n"] = int(args.max_train_n)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
        cfg["max_batch_size"] = int(args.batch_size)
    cfg["external_eval"] = {"lichess_middlegame": str((out / "lichess_eval.pt").resolve())}
    latest = out / "latest.pt"
    resume = latest if latest.exists() else init_path
    n_train = int(torch.load(out / "lichess_train.pt", map_location="cpu", weights_only=False)["turn"].shape[0])
    print(
        f"stream train_n={n_train:,} resume={resume} max_steps={args.max_steps} "
        f"inbox={inbox}",
        flush=True,
    )
    result = train_trial(
        trial,
        out,
        soft_cache=out / "lichess_train.pt",
        deep_cache=None,
        max_steps=int(args.max_steps),
        max_minutes=float(args.train_minutes),
        smoke=False,
        resume_ckpt=resume,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--repack", action="store_true")
    ap.add_argument("--refresh-init", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--inbox", default=str(INBOX))
    ap.add_argument("--max-steps", type=int, default=8_000)
    ap.add_argument("--train-minutes", type=float, default=180.0)
    ap.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    ap.add_argument("--chunk-minutes", type=float, default=25.0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--val-n", type=int, default=VAL_N)
    ap.add_argument("--max-train-n", type=int, default=MAX_TRAIN_N)
    ap.add_argument("--reload-inbox", action="store_true")
    args = ap.parse_args()
    _assert_compact()
    print(
        f"exp278 99M Lichess 16-26 one-hot stream  split={TRAIN_PCT}/{100 - TRAIN_PCT} val_n={args.val_n} "
        f"→ avewright/middlegame-model",
        flush=True,
    )
    if args.pack and not args.go:
        pack_initial(Path(args.output_dir), Path(args.inbox), val_n=args.val_n)
        return
    if args.go:
        n = count_parameters(build_squares64(DEFAULT_100M_SQUARES64_CONFIG))
        print(f"params={n:,} expected={EXPECTED_99M_PARAMS:,}", flush=True)
        train_loop(args)
        return
    print("pass --pack or --go", flush=True)


if __name__ == "__main__":
    main()
