#!/usr/bin/env python3
"""When scoring writes router_labels.pt, stop the old --go and train the new router."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs" / "exp279_moe_router" / "router_labels.pt"


def log(msg: str) -> None:
    print(msg, flush=True)


def score_pids() -> list[int]:
    out = subprocess.check_output(["pgrep", "-f", "exp279_moe_router.py --score --go"], text=True)
    return [int(p) for p in out.split() if p.strip()]


def main() -> None:
    while not LABELS.exists():
        log(f"waiting for {LABELS}")
        time.sleep(30)
    log(f"labels ready {LABELS.stat().st_size}")
    for pid in score_pids():
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"stopped stale train pid={pid}")
        except ProcessLookupError:
            pass
    time.sleep(3)
    env = os.environ.copy()
    env["MOVE_VOCAB_VERSION"] = "compact"
    subprocess.check_call(
        [sys.executable, "-u", str(ROOT / "experiments" / "exp279_moe_router.py"), "--go", "--play"],
        cwd=str(ROOT),
        env=env,
    )


if __name__ == "__main__":
    main()
