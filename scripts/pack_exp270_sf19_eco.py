#!/usr/bin/env python3
"""Pack avewright/stockfish-19-soft-targets. Honor split==1 holdout."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from pack_exp270_all import as_tensors, cat_dicts, policy_ok, read_parquet

REPO = "avewright/stockfish-19-soft-targets"
OUT = ROOT / "outputs" / "exp270_mix_v1"
SOURCE_ID = 4


def main() -> None:
    local = Path(snapshot_download(REPO, repo_type="dataset"))
    files = sorted(local.rglob("*.parquet"))
    print(f"files={len(files)} repo={REPO}", flush=True)
    train_parts, eval_parts = [], []
    n_train = n_eval = n_drop = 0
    for fp in files:
        raw = read_parquet(fp)
        ok = policy_ok(raw["move_idx"], raw["soft_indices"], raw["soft_probs"])
        n_drop += int((~ok).sum())
        split = raw.get("split")
        if split is None:
            split = np.zeros(int(raw["turn"].shape[0]), dtype=np.int8)
        split = np.asarray(split).reshape(-1)
        tr = ok & (split == 0)
        ev = ok & (split != 0)
        if int(tr.sum()):
            sl = {k: v[tr] for k, v in raw.items()}
            train_parts.append(as_tensors(sl, SOURCE_ID, 1))
            n_train += int(tr.sum())
        if int(ev.sum()):
            sl = {k: v[ev] for k, v in raw.items()}
            eval_parts.append(as_tensors(sl, SOURCE_ID, 1))
            n_eval += int(ev.sum())
        print(f"  {fp.name} train={int(tr.sum())} eval={int(ev.sum())}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    train_p = OUT / "sf19_eco_train.pt"
    eval_p = OUT / "sf19_eco_eval.pt"
    if not train_parts:
        raise SystemExit("no train rows")
    torch.save(cat_dicts(train_parts), train_p)
    if eval_parts:
        torch.save(cat_dicts(eval_parts), eval_p)
    report = {
        "status": "sf19_eco_complete",
        "repo": REPO,
        "train": n_train,
        "eval": n_eval,
        "dropped_policy": n_drop,
        "train_path": str(train_p),
        "eval_path": str(eval_p) if eval_parts else None,
    }
    (OUT / "sf19_eco.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
