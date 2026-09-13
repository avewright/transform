#!/usr/bin/env python3
"""exp275: 99M squares64 finetune on the 6–13 piece SF19 endgame pack.

Position-hash 80/20. Eval is a piece-stratified holdout, not the first
N train rows. Soft MultiPV is real SF19 (tau=120), so we keep it.

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp275_endgame_finetune.py --pack
  MOVE_VOCAB_VERSION=compact python experiments/exp275_endgame_finetune.py --go --no-push
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64, count_parameters
from exp273_puzzle_finetune import drop_eval_overlap, pull_99m, write_init_ckpt
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp275_endgame_finetune"
INBOX = ROOT / "outputs" / "endgame_dataset" / "inbox"
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"
SOURCE_SF19 = 4
SPLIT_SEED = 275
TRAIN_PCT = 80
EXPECTED_99M_PARAMS = 98_971_224
VAL_N = 8_192
MIN_PCS, MAX_PCS = 6, 13

PACK_KEYS = (
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


def split_of(key: str, *, seed: int = SPLIT_SEED, train_pct: int = TRAIN_PCT) -> int:
    """0 = train, 1 = eval. Position-hash level."""
    raw = f"{int(seed)}:{key}".encode()
    n = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")
    return 0 if (n % 100) < int(train_pct) else 1


def n_pieces(table: dict) -> np.ndarray:
    return (table["board_array"] != 0).sum(dim=1).cpu().numpy().astype(np.int16)


def stratified_indices(pcs: np.ndarray, n_take: int, rng: np.random.Generator) -> np.ndarray:
    """Equal share per piece count in [6, 13]. Fill leftovers at random."""
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


def load_inbox(inbox: Path) -> dict[str, torch.Tensor]:
    chunks: list[dict] = []
    for cache in sorted(inbox.glob("shard_*/soft_cache.pt")):
        if not (cache.parent / "READY").exists():
            continue
        data = torch.load(cache, map_location="cpu", weights_only=False)
        if "policy_mask" in data:
            ok = data["policy_mask"].view(-1) != 0
            data = {k: v[ok] for k, v in data.items() if torch.is_tensor(v) and int(v.shape[0]) == int(ok.shape[0])}
        chunks.append(data)
    if not chunks:
        raise SystemExit(f"no READY rows in {inbox}")
    keys = [k for k in chunks[0] if torch.is_tensor(chunks[0][k]) and all(k in c for c in chunks)]
    stacked = {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}
    n = int(stacked["turn"].shape[0])
    stacked["source"] = torch.full((n,), SOURCE_SF19, dtype=torch.int8)
    stacked["value_valid"] = torch.ones(n, dtype=torch.int8)
    print(f"inbox shards={len(chunks)} rows={n:,}", flush=True)
    return stacked


def pack_endgame(out: Path, inbox: Path = INBOX, *, val_n: int = VAL_N) -> dict:
    from autoresearch_8gb.pipeline import position_hashes

    out.mkdir(parents=True, exist_ok=True)
    table = load_inbox(inbox)
    pcs = n_pieces(table)
    keep = (pcs >= MIN_PCS) & (pcs <= MAX_PCS)
    if int((~keep).sum()):
        idx = torch.from_numpy(np.flatnonzero(keep))
        table = {k: v[idx] for k, v in table.items()}
        pcs = pcs[keep]
    hs = position_hashes(table)
    split = np.fromiter(
        (split_of(str(int(h))) for h in hs.tolist()),
        dtype=np.int8,
        count=int(hs.shape[0]),
    )
    table["split"] = torch.from_numpy(split)
    train = {k: v[table["split"] == 0] for k, v in table.items()}
    ev = {k: v[table["split"] != 0] for k, v in table.items()}
    train, n_overlap = drop_eval_overlap(train, ev)
    rng = np.random.default_rng(SPLIT_SEED)
    ev_pcs = n_pieces(ev)
    val_idx = stratified_indices(ev_pcs, val_n, rng)
    val = {k: v[torch.from_numpy(val_idx)] for k, v in ev.items()}
    train_path = out / "endgame_train.pt"
    eval_path = out / "endgame_eval.pt"
    eval_full = out / "endgame_eval_full.pt"
    torch.save(train, train_path)
    torch.save(val, eval_path)
    torch.save(ev, eval_full)
    report = {
        "status": "packed",
        "inbox": str(inbox),
        "scanned": int(table["turn"].shape[0]),
        "train_n": int(train["turn"].shape[0]),
        "eval_n": int(ev["turn"].shape[0]),
        "val_n": int(val["turn"].shape[0]),
        "overlap_dropped": n_overlap,
        "train_pct": TRAIN_PCT,
        "split_seed": SPLIT_SEED,
        "val_method": "piece_stratified_hash_holdout",
        "piece_min": MIN_PCS,
        "piece_max": MAX_PCS,
        "val_piece_hist": {str(p): int((n_pieces(val) == p).sum()) for p in range(MIN_PCS, MAX_PCS + 1)},
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "eval_full_path": str(eval_full),
        "note": "Hash 80/20. Stratified val over 6–13 pieces. SF19 soft MultiPV kept. value_valid=1.",
    }
    (out / "pack.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACK", json.dumps(report, indent=2), flush=True)
    return report


def trial_config() -> dict:
    model = DEFAULT_100M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp275_endgame_finetune",
        "arch": "squares64",
        "desc": "99M squares64 FT on 6–13 piece SF19 endgame, hash 80/20, stratified val.",
        "init": {"repo": TEACHER_REPO, "params": EXPECTED_99M_PARAMS},
        "data": {"train_pct": TRAIN_PCT, "split_seed": SPLIT_SEED, "val_n": VAL_N},
        "model": model,
        "train": {
            "batch_size": 528,
            "min_batch_size": 32,
            "max_batch_size": 528,
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.55,
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
        },
    }


def pick_resume(out: Path, init_path: Path, resume_arg: str | None) -> Path:
    if resume_arg:
        p = Path(resume_arg)
        if not p.exists():
            raise SystemExit(f"resume ckpt missing: {p}")
        return p
    latest = out / "latest.pt"
    if latest.exists():
        return latest
    return init_path


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
    if not train_cache.exists() or not eval_cache.exists() or args.repack:
        pack_endgame(out, Path(args.inbox), val_n=args.val_n)
        train_cache = out / "endgame_train.pt"
        eval_cache = out / "endgame_eval.pt"

    trial = trial_config()
    cfg = trial["train"]
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
        cfg["max_batch_size"] = int(args.batch_size)
        cfg["fill_vram"] = False
    if args.muon_lr is not None:
        cfg["muon_lr"] = float(args.muon_lr)
    if args.val_every is not None:
        cfg["val_every_steps"] = int(args.val_every)
    if args.save_every is not None:
        cfg["save_every_steps"] = int(args.save_every)
    if args.fill_vram is not None:
        cfg["fill_vram"] = bool(args.fill_vram)
    n_train = int(torch.load(train_cache, map_location="cpu", weights_only=False)["turn"].shape[0])
    if args.one_epoch or args.epochs:
        epochs = 1 if args.one_epoch else max(1, int(args.epochs))
        bs = int(cfg["batch_size"])
        args.max_steps = max(1, epochs * ((n_train + bs - 1) // bs))
        print(f"epochs={epochs} n={n_train:,} bs={bs} max_steps={args.max_steps}", flush=True)
    cfg["external_eval"] = {"endgame": str(eval_cache.resolve())}

    resume_ckpt = pick_resume(out, init_path, args.resume)
    print(f"resume_ckpt={resume_ckpt}", flush=True)
    result = train_trial(
        trial,
        out,
        soft_cache=train_cache,
        deep_cache=None,
        max_steps=args.max_steps,
        max_minutes=args.train_minutes,
        smoke=False,
        resume_ckpt=resume_ckpt,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--repack", action="store_true")
    ap.add_argument("--refresh-init", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--inbox", default=str(INBOX))
    ap.add_argument("--soft-cache", default=str(OUT_DIR / "endgame_train.pt"))
    ap.add_argument("--eval-cache", default=str(OUT_DIR / "endgame_eval.pt"))
    ap.add_argument("--max-steps", type=int, default=2_000)
    ap.add_argument("--train-minutes", type=float, default=180.0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--muon-lr", type=float, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--val-n", type=int, default=VAL_N)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--one-epoch", action="store_true")
    ap.add_argument("--fill-vram", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    _assert_compact()
    cfg = DEFAULT_100M_SQUARES64_CONFIG
    print(
        f"exp275 99M endgame FT  {cfg.hidden_dim}d/{cfg.num_heads}H "
        f"split={TRAIN_PCT}/{100 - TRAIN_PCT} val_n={args.val_n}",
        flush=True,
    )
    if args.pack and not args.go:
        pack_endgame(Path(args.output_dir), Path(args.inbox), val_n=args.val_n)
        return
    if args.go:
        n = count_parameters(build_squares64(cfg))
        print(f"params={n:,} expected={EXPECTED_99M_PARAMS:,}", flush=True)
        train(args)
        return
    print("pass --pack or --go", flush=True)


if __name__ == "__main__":
    main()
