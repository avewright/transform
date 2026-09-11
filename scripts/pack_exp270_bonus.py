#!/usr/bin/env python3
"""Pack 100M-policy disagreements as the exp270 correction / bonus cache."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from pack_exp270_all import as_tensors, cat_dicts, policy_ok, read_parquet

REPO = "avewright/chess-soft-100m-disagreements"
OUT = ROOT / "outputs" / "exp270_mix_v1" / "bonus_cache.pt"
SOURCE_ID = 5


def main() -> None:
    local = Path(snapshot_download(REPO, repo_type="dataset"))
    files = sorted(local.rglob("*.parquet"))
    print(f"files={len(files)} dest={OUT}", flush=True)
    parts = []
    total = 0
    dropped = 0
    for fp in files:
        raw = read_parquet(fp)
        ok = policy_ok(raw["move_idx"], raw["soft_indices"], raw["soft_probs"])
        keep = int(ok.sum())
        dropped += int((~ok).sum())
        if keep <= 0:
            print(f"  {fp.name} keep=0", flush=True)
            continue
        sl = {k: v[ok] for k, v in raw.items()}
        parts.append(as_tensors(sl, SOURCE_ID, 0))
        total += keep
        print(f"  {fp.name} keep={keep}", flush=True)
    if not parts:
        raise SystemExit("no disagreement rows")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cat_dicts(parts), OUT)
    report = {"status": "bonus_complete", "n": total, "dropped": dropped, "files": len(files), "path": str(OUT)}
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
