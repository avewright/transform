#!/usr/bin/env python3
"""Mirror new harvest READY shards into run2/late_inbox without touching harvest flags."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "outputs/sf19_soft/expand1/inbox"
LATE = ROOT / "outputs/sf19_ft/run2/late_inbox"
MANIFEST = ROOT / "outputs/sf19_ft/run2/dataset_manifest.json"
STOP = ROOT / "outputs/sf19_ft/run2/STOP_WATCH"
LOG = ROOT / "outputs/sf19_ft/run2/late_watch.log"


def say(msg: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    skip: set[str] = set()
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        skip = {Path(p).parent.name for p in man.get("shards") or []}
    LATE.mkdir(parents=True, exist_ok=True)
    say(f"watch skip={len(skip)}")
    while True:
        if STOP.exists():
            say("STOP")
            return
        for sh in sorted(INBOX.glob("shard_*")):
            if sh.name in skip:
                continue
            if not ((sh / "READY").exists() and (sh / "soft_cache.pt").exists()):
                continue
            dest = LATE / sh.name
            if (dest / "READY").exists() or (dest / "ATTACHED").exists():
                skip.add(sh.name)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            src = (sh / "soft_cache.pt").resolve()
            link = dest / "soft_cache.pt"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(src, link)
            (dest / "READY").write_text("ok\n")
            skip.add(sh.name)
            say(f"late {sh.name}")
        time.sleep(30)


if __name__ == "__main__":
    main()
