#!/usr/bin/env python3
"""Elo screen for each new exp270 step_*.pt. Logs to elo_gauntlet.jsonl
so http://localhost:8090/ picks it up. Model on CUDA; Stockfish stays CPU.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Empty string hides every GPU; treat it as unset.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "exp270_squares64_pretrain"
CKPT_DIR = OUT
HIST = OUT / "elo_gauntlet.jsonl"
LOG = OUT / "train.log"
WATCH_LOG = OUT / "elo_watch.log"
PROTOCOL = OUT / "elo_screen_protocol.json"
STEP_RE = re.compile(r"step_(\d+)\.pt$")
ELOS = ["2050"]


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_steps() -> set[int]:
    seen: set[int] = set()
    if not HIST.exists():
        return seen
    for line in HIST.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("step") is not None:
            seen.add(int(row["step"]))
    return seen


def list_ready() -> list[tuple[int, Path]]:
    found = []
    for p in CKPT_DIR.glob("step_*.pt"):
        m = STEP_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def freeze_pending(seen: set[int]) -> None:
    """Copy new step ckpts immediately; trainer only keeps the last 4."""
    for step, src in list_ready():
        if step in seen:
            continue
        dest = OUT / f"elo_frozen_{step}.pt"
        if dest.exists():
            continue
        try:
            shutil.copy2(src, dest)
            log(f"froze {src.name} → {dest.name}")
        except Exception as e:
            log(f"freeze failed {src.name}: {e}")


def queued() -> list[tuple[int, Path]]:
    found = []
    for p in OUT.glob("elo_frozen_*.pt"):
        m = re.search(r"elo_frozen_(\d+)\.pt$", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def append_result(step: int, estimate: dict | None, json_path: Path | None, rc: int) -> None:
    elo = None
    if estimate:
        elo = estimate.get("estimated_elo")
    row = {
        "step": step,
        "elo": elo,
        "estimate": estimate,
        "rc": rc,
        "json_path": str(json_path) if json_path else None,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    with HIST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] elo@{step} estimate={elo} rc={rc}\n")
    log(f"logged elo@{step} estimate={elo} rc={rc}")


def run_one(step: int, frozen: Path) -> None:
    prefix = f"exp270_step{step}"
    cmd = [
        sys.executable, "-u", "-m", "harness.elo",
        "--ckpt", str(frozen),
        "--protocol", str(PROTOCOL),
        "--device", "cuda",
        "--no-book",
        "--no-syzygy",
        "--elos", *ELOS,
        "--stop-after-bracket",
        "--out-prefix", prefix,
    ]
    log(f"start step={step} ckpt={frozen.name} device=cuda")
    env = os.environ.copy()
    env["MOVE_VOCAB_VERSION"] = "compact"
    if env.get("CUDA_VISIBLE_DEVICES") == "":
        del env["CUDA_VISIBLE_DEVICES"]
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    json_path = ROOT / "outputs" / f"elo_eval_{prefix}.json"
    estimate = None
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        estimate = payload.get("estimate")
    append_result(step, estimate, json_path if json_path.exists() else None, rc)
    if rc != 0:
        log(f"gauntlet rc={rc} step={step}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("watch start")
    while True:
        seen = done_steps()
        freeze_pending(seen)
        pending = [(s, p) for s, p in queued() if s not in seen]
        if pending:
            step, path = pending[0]
            try:
                run_one(step, path)
            except Exception as e:
                log(f"failed step={step}: {e}")
                append_result(step, None, None, 1)
        else:
            time.sleep(30)


if __name__ == "__main__":
    main()
