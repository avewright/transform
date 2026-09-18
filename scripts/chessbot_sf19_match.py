#!/usr/bin/env python3
"""Occasional paired matches: N=3 student vs published ChessBot.

Watches checkpoints and appends match events. Training stays on the GPU.

  python3 scripts/chessbot_sf19_match.py --once
  python3 scripts/chessbot_sf19_match.py
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

from experiments.chessbot_sf19_recurrent import (
    DEFAULT_OPENINGS,
    evaluate_vs_chessbot,
    load_openings,
)


DEFAULT_OUT = ROOT / "outputs/chessbot_sf19_n3"
DEFAULT_CFG = ROOT / "configs/chessbot_sf19_n3.json"


def emit(out: Path, row: dict) -> None:
    row = dict(time=time.time(), **row)
    print(json.dumps(row), flush=True)
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def ckpt_step(path: Path) -> int | None:
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    step = ckpt.get("step")
    return int(step) if step is not None else None


def matched_path(out: Path) -> Path:
    return out / "matched_steps.json"


def already_matched(out: Path) -> set[int]:
    path = matched_path(out)
    if not path.exists():
        return set()
    try:
        return {int(x) for x in json.loads(path.read_text())}
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


def remember(out: Path, step: int) -> None:
    done = already_matched(out)
    done.add(int(step))
    matched_path(out).write_text(json.dumps(sorted(done)))


def play(out: Path, ckpt: Path, device, openings, unrolls: int, ply_cap: int) -> dict | None:
    step = ckpt_step(ckpt)
    if step is None:
        return None
    if step in already_matched(out):
        return None
    dest = out / "development_openings.json"
    if not dest.exists() and DEFAULT_OPENINGS.exists():
        dest.write_text(DEFAULT_OPENINGS.read_text())
    emit(out, dict(stage="match_start", step=step, checkpoint=str(ckpt), pairs=len(openings)))
    summary = evaluate_vs_chessbot(
        ckpt, out, device, openings=openings, unrolls=unrolls, ply_cap=ply_cap,
    )
    slim = {k: summary[k] for k in (
        "step", "wins", "draws", "losses", "unknown", "n", "score",
        "paired_ci_95", "verdict", "pairs", "gate", "elapsed_s", "unrolls",
    ) if k in summary}
    emit(out, dict(stage="match", opponent="original", **slim))
    remember(out, int(summary["step"]))
    return summary


def latest_ckpt(out: Path) -> Path | None:
    path = out / "latest.pt"
    return path if path.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--every", type=int, default=0, help="Override match_every")
    p.add_argument("--pairs", type=int, default=0, help="Override match_pairs")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll", type=float, default=30.0)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()) if a.config.exists() else {}
    every = int(a.every or cfg.get("match_every") or 2500)
    pairs = int(a.pairs or cfg.get("match_pairs") or 16)
    ply_cap = int(cfg.get("match_ply_cap") or 400)
    unrolls = int(cfg.get("unrolls") or 3)
    import torch
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    openings = load_openings(a.openings, pairs)
    a.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(
        stage="match_watch", out=str(a.out), every=every, pairs=len(openings),
        device=str(device), once=a.once,
    )), flush=True)

    def maybe_play():
        ckpt = latest_ckpt(a.out)
        if ckpt is None:
            return None
        step = ckpt_step(ckpt)
        if step is None:
            return None
        done = already_matched(a.out)
        if step in done:
            return None
        if done and step - max(done) < every:
            return None
        return play(a.out, ckpt, device, openings, unrolls, ply_cap)

    if a.once:
        ckpt = latest_ckpt(a.out)
        if ckpt is None:
            raise SystemExit("no latest.pt")
        play(a.out, ckpt, device, openings, unrolls, ply_cap)
        return
    maybe_play()
    while True:
        time.sleep(a.poll)
        if (a.out / "STOP").exists():
            print(json.dumps(dict(stage="match_watch_stop")), flush=True)
            return
        maybe_play()


if __name__ == "__main__":
    main()
