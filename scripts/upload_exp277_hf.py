#!/usr/bin/env python3
"""Upload exp277 Lichess >=26 FT checkpoint to avewright/opening-model.

Never writes the 99M incumbent, puzzle, syzygy, or endgame experts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from upload_exp201_hf import ckpt_steps, load_hf_token  # noqa: E402

DEFAULT_REPO = "avewright/opening-model"
BLOCKED = {
    "avewright/chess-transformer-100m-squares64",
    "avewright/puzzle-model",
    "avewright/syzygy-model",
    "avewright/endgame-model",
    "avewright/endgame-dataset",
    "avewright/lichess-endgame-bestline",
    "avewright/middlegame-model",
}
OUT = ROOT / "outputs" / "exp277_lichess_opening_stream"


def refuse_blocked(repo: str) -> None:
    if repo.strip() in BLOCKED:
        raise SystemExit(f"refusing to upload opening FT over {repo}")


def write_card(repo: str, steps: int, extra: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    loss = extra.get("recent_loss", "—")
    val = extra.get("recent_val", extra.get("best_val", "—"))
    pack = extra.get("pack") or {}
    card = Path(extra.get("card_path") or (OUT / "HF_README.md"))
    card.write_text(
        f"""---
license: mit
tags:
  - chess
  - transformer
  - recurrent
  - policy
  - opening
  - pytorch
library_name: pytorch
---

# 99M opening specialist (squares64)

Same **99M** squares64 architecture as
[`avewright/chess-transformer-100m-squares64`](https://huggingface.co/avewright/chess-transformer-100m-squares64),
finetuned on `≥26`-piece positions (one-hot best first move) from
[`Lichess/chess-position-evaluations`](https://huggingface.co/datasets/Lichess/chess-position-evaluations)
via [`avewright/lichess-opening-bestline`](https://huggingface.co/datasets/avewright/lichess-opening-bestline).

This file is **`latest.pt` at opening-FT step {steps}** ({stamp}).
Train loss ~{loss}. Frozen holdout hard CE ~{val}.

Not the generalist incumbent, the puzzle expert, the Syzygy expert, or the endgame expert.

## Training

- Warm start: public 99M `latest.pt` (weights only), then full resume.
- Split: position-hash 80/20 (seed 277). Frozen piece-stratified val {pack.get("val_n", 8192)}.
- One-hot PV1 (`soft_alpha=0`). Pieces 26–32.
- Polar-NorMuon, bs=528. Best disk ckpt at upload: step {steps}.

## Files

- `latest.pt`
- `step_{steps:06d}.pt`
- `model_config.json`
- `train.log`
- `pack.json`
""",
        encoding="utf-8",
    )
    return card


def recent_metrics(log: Path) -> dict:
    extra: dict = {}
    if not log.exists():
        return extra
    text = log.read_text(errors="replace")
    losses = re.findall(r"step \d+/\d+ \| loss=([\d.]+)", text)
    if losses:
        extra["recent_loss"] = losses[-1]
    step_loss = dict(re.findall(r"step (\d+)/\d+ \| loss=([\d.]+)", text))
    extra["loss_by_step"] = step_loss
    vals = re.findall(r"val/lichess_opening hard_ce=([\d.]+)", text)
    if vals:
        extra["recent_val"] = vals[-1]
        extra["best_val"] = min(vals, key=float)
    return extra


def upload(repo: str, ckpt: Path, *, private: bool = False) -> dict:
    refuse_blocked(repo)
    token = load_hf_token()
    from huggingface_hub import HfApi, create_repo, whoami

    user = whoami(token=token).get("name", "?")
    print(f"hf user={user} repo={repo}", flush=True)
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    steps = ckpt_steps(ckpt)
    out = ckpt.parent
    extra = recent_metrics(out / "train.log")
    pack_path = out / "pack.json"
    extra["pack"] = json.loads(pack_path.read_text(encoding="utf-8")) if pack_path.exists() else {}
    by_step = extra.pop("loss_by_step", {})
    if str(steps) in by_step:
        extra["recent_loss"] = by_step[str(steps)]
    extra["card_path"] = str(out / "HF_README.md")
    card = write_card(repo, steps, extra)
    create_repo(repo, repo_type="model", exist_ok=True, private=private, token=token)
    api = HfApi(token=token)
    files = [
        (ckpt, "latest.pt", f"exp277 opening FT latest.pt step {steps}"),
        (ckpt, f"step_{steps:06d}.pt", f"exp277 opening FT step {steps}"),
        (out / "model_config.json", "model_config.json", "model_config.json"),
        (card, "README.md", "opening expert model card"),
        (out / "train.log", "train.log", "train.log"),
        (pack_path, "pack.json", "pack.json"),
    ]
    for src, dest, msg in files:
        if not src.exists():
            print(f"skip missing {src.name}", flush=True)
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=repo,
            repo_type="model",
            commit_message=msg,
        )
        print(f"uploaded {dest}", flush=True)
    url = f"https://huggingface.co/{repo}"
    print(f"uploaded {repo} latest.pt steps={steps}", flush=True)
    print(url, flush=True)
    return {"repo": repo, "steps": steps, "url": url}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ckpt", default=str(OUT / "latest.pt"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    refuse_blocked(args.repo)
    upload(args.repo, Path(args.ckpt), private=args.private)


if __name__ == "__main__":
    main()
