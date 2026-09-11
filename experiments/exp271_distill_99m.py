#!/usr/bin/env python3
"""exp271: distill 99M squares64 logits into the 270M student on SF19 boards.

Pulls:
  avewright/chess-transformer-100m-squares64   (teacher)
  avewright/chess-transformer-270m-squares64   (student, step 43000 warm start)
  avewright/chess-soft-sf19                    (positions + SF19 hard/WDL)

Policy soft target is Hinton KD on 99M logits. Hard CE and WDL stay on the
Stockfish 19 labels. split!=0 is held out for eval.

Usage:
  MOVE_VOCAB_VERSION=compact python experiments/exp271_distill_99m.py --pull
  MOVE_VOCAB_VERSION=compact python experiments/exp271_distill_99m.py --pack
  MOVE_VOCAB_VERSION=compact python experiments/exp271_distill_99m.py --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chess_squares64 import (
    DEFAULT_270M_SQUARES64_CONFIG,
    EXPECTED_270M_PARAMS,
    build_squares64,
    count_parameters,
)
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp271_distill_99m"
MIX_DIR = ROOT / "outputs" / "exp271_mix"
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"
STUDENT_REPO = "avewright/chess-transformer-270m-squares64"
TEACHER_CKPT = ROOT / "outputs" / "hf_models" / "99m" / "latest.pt"
STUDENT_CKPT = ROOT / "outputs" / "hf_models" / "270m" / "latest.pt"


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit(
            f"Expected compact vocab 1968, got {VOCAB_SIZE}. "
            "Export MOVE_VOCAB_VERSION=compact."
        )


def pull_models(*, teacher: Path = TEACHER_CKPT, student: Path = STUDENT_CKPT) -> dict:
    from huggingface_hub import hf_hub_download

    teacher.parent.mkdir(parents=True, exist_ok=True)
    student.parent.mkdir(parents=True, exist_ok=True)
    t = Path(hf_hub_download(TEACHER_REPO, "latest.pt", local_dir=str(teacher.parent)))
    Path(hf_hub_download(TEACHER_REPO, "model_config.json", local_dir=str(teacher.parent)))
    s = Path(hf_hub_download(STUDENT_REPO, "latest.pt", local_dir=str(student.parent)))
    Path(hf_hub_download(STUDENT_REPO, "model_config.json", local_dir=str(student.parent)))
    info = {"teacher": str(t), "student": str(s)}
    print(json.dumps(info, indent=2), flush=True)
    return info


def _pack_dir(raw: Path, source_id: int, value_valid: int, name: str, max_files: int | None = None):
    from pack_exp270_all import as_tensors, cat_dicts, policy_ok, read_parquet

    files = sorted(p for p in raw.rglob("*.parquet"))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    if not files:
        return None, {"files": 0, "train": 0}
    parts = []
    stats = {"files": 0, "rows": 0, "train": 0, "dropped_policy": 0}
    for fp in files:
        raw_rows = read_parquet(fp)
        n = int(raw_rows["turn"].shape[0])
        ok = policy_ok(raw_rows["move_idx"], raw_rows["soft_indices"], raw_rows["soft_probs"])
        stats["files"] += 1
        stats["rows"] += n
        stats["dropped_policy"] += int((~ok).sum())
        if not ok.any():
            continue
        sl = {k: v[ok] for k, v in raw_rows.items()}
        parts.append(as_tensors(sl, source_id, value_valid))
        stats["train"] += int(ok.sum())
        print(f"  {name} {fp.relative_to(raw)} rows={n} keep={int(ok.sum())}", flush=True)
    if not parts:
        return None, stats
    return cat_dicts(parts), stats


def pack_mix(out: Path = MIX_DIR, max_lichess_files: int | None = None) -> dict:
    """Pack Stockfish 19 soft targets. Holdout = split!=0. Lichess arg unused."""
    del max_lichess_files
    from pack_exp270_all import pack_repo

    out.mkdir(parents=True, exist_ok=True)
    raw_sf19 = out / "raw_sf19"
    if not raw_sf19.exists() or not any(raw_sf19.rglob("*.parquet")):
        raise SystemExit(f"missing SF19 parquets in {raw_sf19}")

    print("pack avewright/chess-soft-sf19 (train=split==0, eval=split!=0)", flush=True)
    train, ev, stats = pack_repo("sf19", "avewright/chess-soft-sf19", 4, raw_sf19)
    torch.save(train, out / "soft_cache.pt")
    if ev is not None:
        torch.save(ev, out / "sf19_eval.pt")
    report = {
        "status": "complete",
        "source": "avewright/chess-soft-sf19",
        "sf19": stats,
        "soft_n": int(train["turn"].shape[0]),
        "eval_n": int(ev["turn"].shape[0]) if ev is not None else 0,
        "note": "SF19 boards + hard/WDL. 99M logits replace MultiPV for policy soft.",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def ensure_raw_datasets(out: Path = MIX_DIR) -> None:
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    dest = out / "raw_sf19"
    if dest.exists() and any(dest.rglob("*.parquet")):
        print(f"have {dest}", flush=True)
        return
    print("download avewright/chess-soft-sf19", flush=True)
    snapshot_download("avewright/chess-soft-sf19", repo_type="dataset", local_dir=str(dest))


def trial_config() -> dict:
    model = DEFAULT_270M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp271_distill_99m",
        "arch": "squares64",
        "desc": (
            "270M warm-start from HF step 43000 on SF19 boards. "
            "Policy soft = 99M logit KD (Hinton T=2). Hard + WDL stay SF19."
        ),
        "teacher": {
            "repo": TEACHER_REPO,
            "ckpt": str(TEACHER_CKPT),
            "params": 98_971_224,
        },
        "student": {
            "repo": STUDENT_REPO,
            "ckpt": str(STUDENT_CKPT),
            "params": EXPECTED_270M_PARAMS,
        },
        "model": model,
        "train": {
            "batch_size": 32,
            "min_batch_size": 8,
            "max_batch_size": 64,
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.85,
            "soft_temp": 0.0,
            "soft_temp_weight": 0.0,
            "teacher_temp": 2.0,
            "teacher_topk": 32,
            "teacher_kd_frac": 1.0,
            "teacher_replace_hard": False,
            "teacher_value": False,
            "deep_mix_frac": 0.0,
            "deep_in_each_batch": False,
            "bonus_mix_frac": 0.0,
            "quality_mix_frac": 0.0,
            "puzzle_mix_frac": 0.0,
            "use_swa": False,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
            "optimizer": "polar_normuon",
            "compile_polar": True,
            "force_lr": False,
            "muon_lr": 0.002,
            "adam_lr": 3e-5,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup": 100,
            "min_lr_frac": 0.1,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": True,
            "max_vram_gb": 40.0,
            "save_every_steps": 250,
            "keep_step_every": 1000,
            "keep_last_ckpts": 4,
            "val_every_steps": 500,
            "val_eval_n": 256,
            "elo_every_steps": 0,
            "keep_step_every": 1000,
        },
    }


def train(args: argparse.Namespace) -> dict:
    from autoresearch_8gb.train_trial import train_trial

    _assert_compact()
    if not TEACHER_CKPT.exists() or not STUDENT_CKPT.exists():
        pull_models()
    mix = Path(args.mix_dir)
    soft = mix / "soft_cache.pt"
    eval_pt = mix / "sf19_eval.pt"
    if not soft.exists():
        ensure_raw_datasets(mix)
        pack_mix(mix)
    if not soft.exists():
        raise SystemExit(f"missing {soft}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trial = trial_config()
    train_cfg = trial["train"]
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
        train_cfg["max_batch_size"] = max(
            int(train_cfg.get("max_batch_size", 0)), int(args.batch_size),
        )
    if args.max_vram_gb is not None:
        train_cfg["max_vram_gb"] = float(args.max_vram_gb)
    if eval_pt.exists():
        train_cfg["external_eval"] = {"sf19": str(eval_pt.resolve())}

    resume = Path(args.resume) if args.resume else STUDENT_CKPT
    result = train_trial(
        trial,
        out,
        soft_cache=soft,
        deep_cache=None,
        max_steps=args.max_steps,
        max_minutes=args.train_minutes,
        smoke=False,
        resume_ckpt=resume if resume.exists() else None,
        teacher_ckpt=TEACHER_CKPT,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull", action="store_true", help="Download 99M + 270M checkpoints")
    ap.add_argument("--pack", action="store_true", help="Download + pack position mix")
    ap.add_argument(
        "--max-lichess-files",
        type=int,
        default=None,
        help="Unused. Mix is Stockfish 19 only.",
    )
    ap.add_argument("--go", action="store_true", help="Start 270M KD training")
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--mix-dir", default=str(MIX_DIR))
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--train-minutes", type=float, default=720.0)
    ap.add_argument("--resume", default=None, help="Override student init (default: HF 270M)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-vram-gb", type=float, default=None)
    args = ap.parse_args()
    _assert_compact()

    cfg = DEFAULT_270M_SQUARES64_CONFIG
    print(
        f"exp271 distill 99M -> 270M  hidden={cfg.hidden_dim} heads={cfg.num_heads} "
        f"expected_params={EXPECTED_270M_PARAMS:,}",
        flush=True,
    )
    if args.pull:
        pull_models()
    if args.pack:
        ensure_raw_datasets(Path(args.mix_dir))
        pack_mix(Path(args.mix_dir), max_lichess_files=args.max_lichess_files)
    if args.go:
        train(args)
        return
    if not args.pull and not args.pack:
        n = count_parameters(build_squares64(cfg))
        print(f"  student params={n:,} — pass --pull / --pack / --go")


if __name__ == "__main__":
    main()
