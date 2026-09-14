#!/usr/bin/env python3
"""Wait for exp278 step 8000, upload middlegame-model, train the router.

  MOVE_VOCAB_VERSION=compact python3 -u scripts/run_exp279_after_middlegame.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from upload_exp201_hf import ckpt_steps  # noqa: E402

MG = ROOT / "outputs" / "exp278_lichess_middlegame_stream"
CKPT = MG / "latest.pt"
LOG = MG / "train.log"
NEED = 8000


def log(msg: str) -> None:
    print(msg, flush=True)


def current_steps() -> int:
    if not CKPT.exists():
        return 0
    try:
        return int(ckpt_steps(CKPT))
    except Exception:
        return 0


def wait_for_8000() -> int:
    last = -1
    while True:
        steps = current_steps()
        trained = LOG.exists() and "status=trained" in LOG.read_text(errors="replace")[-4000:]
        if steps >= NEED or trained:
            log(f"middlegame ready steps={steps} trained={trained}")
            return steps
        if steps != last:
            log(f"waiting for middlegame step {NEED}; now {steps}")
            last = steps
        time.sleep(20)


def main() -> None:
    steps = wait_for_8000()
    env = os.environ.copy()
    env["MOVE_VOCAB_VERSION"] = "compact"
    log(f"upload middlegame-model steps={steps}")
    subprocess.check_call(
        [sys.executable, "-u", str(ROOT / "scripts" / "upload_exp278_hf.py")],
        cwd=str(ROOT),
        env=env,
    )
    stale = ROOT / "outputs" / "hf_models" / "experts" / "middlegame" / "latest.pt"
    stale.unlink(missing_ok=True)
    out = ROOT / "outputs" / "exp279_moe_router"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    log_path.touch()
    log("start exp279 loss server :8092")
    subprocess.Popen(
        [
            sys.executable, "-u", str(ROOT / "scripts" / "exp201_loss_server.py"),
            "--log", str(log_path), "-p", "8092",
        ],
        cwd=str(ROOT),
        env=env,
    )
    log("pull / score / train router")
    subprocess.check_call(
        [
            sys.executable, "-u", str(ROOT / "experiments" / "exp279_moe_router.py"),
            "--pull", "--score", "--go",
        ],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
