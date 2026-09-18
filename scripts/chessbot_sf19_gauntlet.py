#!/usr/bin/env python3
"""Clean paired gauntlet vs published ChessBot. No training process on the GPU.

  python3 scripts/chessbot_sf19_gauntlet.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from chess_chessbot_recurrent import wrap_published
from experiments.chessbot_sf19_recurrent import (
    DEFAULT_OPENINGS,
    evaluate_pair,
    identity_check,
    load_openings,
    load_student,
)


def emit(out: Path, row: dict) -> None:
    row = dict(time=time.time(), **row)
    print(json.dumps(row), flush=True)
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    (out / "status.json").write_text(json.dumps(row, indent=2))


def write_leg(out: Path, result: dict) -> dict:
    summary = result["summary"]
    tag = summary["tag"]
    (out / f"eval_{tag}.json").write_text(json.dumps({**result["match"], **summary}, indent=2))
    slim = {k: summary[k] for k in (
        "tag", "step", "wins", "draws", "losses", "unknown", "n", "score",
        "paired_ci_95", "verdict", "pairs", "gate", "elapsed_s", "unrolls",
        "teacher_unrolls", "question", "opponent",
    ) if k in summary}
    emit(out, dict(stage="match", **slim))
    return slim


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/chessbot_sf19_gauntlet")
    p.add_argument("--wrecked", type=Path, default=ROOT / "outputs/chessbot_sf19_n3/latest.pt")
    p.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    p.add_argument("--pairs", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ply-cap", type=int, default=400)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {a.out}")
    a.out.mkdir(parents=True)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    openings = load_openings(a.openings, a.pairs)
    dest = a.out / "development_openings.json"
    dest.write_text(json.dumps(openings, indent=2))

    published = wrap_published(device, default_unrolls=1, gate_extra=True)
    n3 = wrap_published(device, default_unrolls=3, gate_extra=True)
    identity = identity_check(n3, device, 3)
    emit(out := a.out, dict(
        stage="loaded", device=str(device), pairs=len(openings),
        wrecked=str(a.wrecked) if a.wrecked.exists() else None, **identity,
    ))
    if identity["train_depth_max"] > 1e-4:
        raise SystemExit(f"gated N=3 is not identity: {identity}")

    legs = []
    emit(out, dict(stage="match_start", tag="init_n3_vs_published", pairs=len(openings)))
    legs.append(write_leg(out, evaluate_pair(
        n3, published, openings, device,
        student_unrolls=3, teacher_unrolls=1, opponent="original",
        tag="init_n3_vs_published", ply_cap=a.ply_cap,
        extra=dict(note="frozen published trunk, alpha=0"),
    )))

    emit(out, dict(stage="match_start", tag="init_n3_vs_self_n1", pairs=len(openings)))
    legs.append(write_leg(out, evaluate_pair(
        n3, n3, openings, device,
        student_unrolls=3, teacher_unrolls=1, opponent="self_n1", question="recurrence",
        tag="init_n3_vs_self_n1", ply_cap=a.ply_cap,
        extra=dict(note="same frozen weights, N=3 vs N=1"),
    )))

    if a.wrecked.exists():
        wrecked, meta = load_student(a.wrecked, device)
        emit(out, dict(stage="match_start", tag="wrecked_n3_vs_published",
                       step=int(meta.get("step") or 0), pairs=len(openings)))
        legs.append(write_leg(out, evaluate_pair(
            wrecked, published, openings, device,
            student_unrolls=3, teacher_unrolls=1, opponent="original",
            step=int(meta.get("step") or 0), tag="wrecked_n3_vs_published",
            ply_cap=a.ply_cap,
            extra=dict(checkpoint=str(a.wrecked.resolve()), note="unfrozen SL checkpoint"),
        )))
        emit(out, dict(stage="match_start", tag="wrecked_n3_vs_self_n1",
                       step=int(meta.get("step") or 0), pairs=len(openings)))
        legs.append(write_leg(out, evaluate_pair(
            wrecked, wrecked, openings, device,
            student_unrolls=3, teacher_unrolls=1, opponent="self_n1", question="recurrence",
            step=int(meta.get("step") or 0), tag="wrecked_n3_vs_self_n1",
            ply_cap=a.ply_cap,
            extra=dict(note="does the tiny gate change the damaged net"),
        )))
        del wrecked

    (out / "gauntlet.json").write_text(json.dumps(dict(identity=identity, legs=legs), indent=2))
    emit(out, dict(stage="complete", legs=[{k: x[k] for k in ("tag", "verdict", "score", "wins", "draws", "losses") if k in x} for x in legs]))
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
