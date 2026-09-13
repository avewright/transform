#!/usr/bin/env python3
"""exp274: 99M squares64 finetune on avewright/chess-soft-syzygy (80/20).

Weights-only warm start from the public 99M incumbent. Position-hash
80/20. Value stays masked (DTZ is not mate / WDL is unverified).

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp274_syzygy_finetune.py --pull
  MOVE_VOCAB_VERSION=compact python experiments/exp274_syzygy_finetune.py --pack
  MOVE_VOCAB_VERSION=compact python experiments/exp274_syzygy_finetune.py --go
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
from exp273_puzzle_finetune import (  # noqa: E402
    drop_eval_overlap,
    pull_99m,
    write_init_ckpt,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp274_syzygy_finetune"
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"
SYZYGY_REPO = "avewright/chess-soft-syzygy"
SYZYGY_REVISION = "3889985c482837aa3082e182d8f123e005e2c0d4"
SOURCE_SYZYGY = 2
SPLIT_SEED = 274
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


def split_of(key: str, *, seed: int = SPLIT_SEED, train_pct: int = TRAIN_PCT) -> int:
    """0 = train, 1 = eval. Stable; position-hash level."""
    raw = f"{int(seed)}:{key}".encode()
    n = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")
    return 0 if (n % 100) < int(train_pct) else 1


def _to_1d(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().to(dtype).reshape(-1).contiguous()


def table_from_hf(ds) -> dict[str, torch.Tensor]:
    from autoresearch_8gb.hf_soft_to_cache import _to_tensor

    n = len(ds)
    out = {
        "board_array": _to_tensor(ds["board_array"], torch.int8),
        "turn": _to_1d(_to_tensor(ds["turn"], torch.int8), torch.int8),
        "castling": _to_1d(_to_tensor(ds["castling"], torch.int8), torch.int8),
        "ep_square": _to_1d(_to_tensor(ds["ep_square"], torch.int8), torch.int8),
        "move_idx": _to_1d(_to_tensor(ds["move_idx"], torch.int64), torch.int64),
        "cp": _to_1d(_to_tensor(ds["cp"], torch.int32), torch.int32),
        "mate": _to_1d(_to_tensor(ds["mate"], torch.int32), torch.int32),
        "soft_indices": _to_tensor(ds["soft_indices"], torch.int64),
        "soft_probs": _to_tensor(ds["soft_probs"], torch.float32),
        "source": torch.full((n,), SOURCE_SYZYGY, dtype=torch.int8),
        "value_valid": torch.zeros(n, dtype=torch.int8),
    }
    if "label_depth" in ds.column_names:
        out["label_depth"] = _to_1d(_to_tensor(ds["label_depth"], torch.int16), torch.int16)
    else:
        out["label_depth"] = torch.full((n,), 999, dtype=torch.int16)
    if "phase" in ds.column_names:
        out["phase"] = _to_1d(_to_tensor(ds["phase"], torch.int8), torch.int8)
    else:
        out["phase"] = torch.full((n,), 2, dtype=torch.int8)
    return out


def to_onehot(table: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace WDL-soup soft targets with a one-hot on move_idx (DTZ-best)."""
    n = int(table["move_idx"].shape[0])
    mid = table["move_idx"].to(torch.int64).reshape(-1)
    si = torch.full((n, 8), -1, dtype=torch.int64)
    sp = torch.zeros(n, 8, dtype=torch.float32)
    si[:, 0] = mid
    sp[:, 0] = 1.0
    table["soft_indices"] = si
    table["soft_probs"] = sp
    return table


def onehot_cache(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    data = to_onehot(data)
    torch.save(data, path)
    p1 = float(data["soft_probs"][:, 0].mean())
    print(f"onehot {path.name} n={int(data['turn'].shape[0]):,} top1={p1:.3f}", flush=True)
    return data


def pack_syzygy(
    out: Path,
    *,
    max_rows: int | None = None,
    revision: str = SYZYGY_REVISION,
) -> dict:
    from datasets import load_dataset
    from autoresearch_8gb.pipeline import position_hashes

    out.mkdir(parents=True, exist_ok=True)
    print(f"pack {SYZYGY_REPO}@{revision} dest={out}", flush=True)
    ds = load_dataset(SYZYGY_REPO, split="train", revision=revision)
    n_all = len(ds)
    if max_rows is not None and max_rows < n_all:
        ds = ds.select(range(int(max_rows)))
    ds = ds.with_format("numpy")
    table = table_from_hf(ds)
    n = int(table["turn"].shape[0])
    hs = position_hashes(table)
    split = np.fromiter(
        (split_of(str(int(h))) for h in hs.tolist()),
        dtype=np.int8,
        count=n,
    )
    table["split"] = torch.from_numpy(split)
    table = to_onehot(table)
    train_mask = table["split"] == 0
    eval_mask = ~train_mask
    train = {k: v[train_mask] for k, v in table.items()}
    ev = {k: v[eval_mask] for k, v in table.items()}
    train, n_overlap = drop_eval_overlap(train, ev)
    train_path = out / "syzygy_train.pt"
    eval_path = out / "syzygy_eval.pt"
    torch.save(train, train_path)
    torch.save(ev, eval_path)
    report = {
        "status": "packed",
        "repo": SYZYGY_REPO,
        "revision": revision,
        "scanned": n,
        "repo_n": n_all,
        "train_n": int(train["turn"].shape[0]),
        "eval_n": int(ev["turn"].shape[0]),
        "overlap_dropped": n_overlap,
        "train_pct": TRAIN_PCT,
        "split_seed": SPLIT_SEED,
        "value_valid": 0,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "note": "Position-hash 80/20. One-hot on DTZ-best WDL move. value_valid=0.",
    }
    (out / "pack.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACK", json.dumps(report, indent=2), flush=True)
    return report


def trial_config() -> dict:
    model = DEFAULT_100M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp274_syzygy_finetune",
        "arch": "squares64",
        "desc": "99M squares64 weights-only FT on chess-soft-syzygy one-hot, hash 80/20.",
        "init": {"repo": TEACHER_REPO, "params": EXPECTED_99M_PARAMS},
        "data": {
            "repo": SYZYGY_REPO,
            "revision": SYZYGY_REVISION,
            "train_pct": TRAIN_PCT,
            "split_seed": SPLIT_SEED,
        },
        "model": model,
        "train": {
            "batch_size": 536,
            "min_batch_size": 32,
            "max_batch_size": 536,
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
            "warmup": 50,
            "min_lr_frac": 0.1,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": False,
            "max_vram_gb": 40.0,
            "save_every_steps": 250,
            "keep_step_every": 500,
            "keep_last_ckpts": 8,
            "val_every_steps": 250,
            "val_eval_n": 2048,
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
    if not train_cache.exists() or not eval_cache.exists():
        pack_syzygy(out, max_rows=args.max_rows)
        train_cache = out / "syzygy_train.pt"
        eval_cache = out / "syzygy_eval.pt"
    if not train_cache.exists() or not eval_cache.exists():
        raise SystemExit(f"missing caches train={train_cache} eval={eval_cache}")
    if args.onehot:
        onehot_cache(train_cache)
        onehot_cache(eval_cache)

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
    if args.val_every is not None:
        cfg["val_every_steps"] = int(args.val_every)
    if args.save_every is not None:
        cfg["save_every_steps"] = int(args.save_every)
    if args.fill_vram is not None:
        cfg["fill_vram"] = bool(args.fill_vram)
    if args.one_epoch:
        n_train = int(torch.load(train_cache, map_location="cpu", weights_only=False)["turn"].shape[0])
        bs = int(cfg["batch_size"])
        args.max_steps = max(1, (n_train + bs - 1) // bs)
        print(f"one epoch n={n_train:,} bs={bs} max_steps={args.max_steps}", flush=True)
    cfg["external_eval"] = {"syzygy": str(eval_cache.resolve())}

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
    if not args.no_push:
        latest = out / "latest.pt"
        if latest.exists():
            from upload_exp274_hf import upload as upload_syzygy

            print(f"pushing {latest} → {args.hf_repo}", flush=True)
            upload_syzygy(args.hf_repo, latest, private=bool(args.hf_private))
        else:
            print(f"skip HF push: missing {latest}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--refresh-init", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--soft-cache", default=str(OUT_DIR / "syzygy_train.pt"))
    ap.add_argument("--eval-cache", default=str(OUT_DIR / "syzygy_eval.pt"))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=8_000)
    ap.add_argument("--train-minutes", type=float, default=240.0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--muon-lr", type=float, default=None)
    ap.add_argument("--adam-lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--fill-vram", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--hf-repo", default="avewright/syzygy-model")
    ap.add_argument("--hf-private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--onehot", action="store_true", help="Rewrite caches to one-hot on move_idx")
    ap.add_argument("--one-epoch", action="store_true", help="Set max_steps to one pass over train")
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_100M_SQUARES64_CONFIG
    print(
        f"exp274 99M syzygy FT  {cfg.hidden_dim}d/{cfg.num_heads}H "
        f"effective={cfg.effective_depth}  split={TRAIN_PCT}/{100 - TRAIN_PCT}",
        flush=True,
    )
    if args.pull:
        pull_99m()
    if args.pack:
        pack_syzygy(Path(args.output_dir), max_rows=args.max_rows)
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
