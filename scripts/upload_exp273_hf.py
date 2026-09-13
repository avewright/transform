#!/usr/bin/env python3
"""Upload exp273 puzzle-FT checkpoint to avewright/puzzle-model.

Never writes to the 99M incumbent repo.
Token from HF_TOKEN / .env (same loader as upload_exp201_hf).

  python3 scripts/upload_exp273_hf.py
  python3 scripts/upload_exp273_hf.py --ckpt outputs/exp273_puzzle_finetune/latest.pt
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

DEFAULT_REPO = "avewright/puzzle-model"
INCUMBENT_REPO = "avewright/chess-transformer-100m-squares64"
OUT = ROOT / "outputs" / "exp273_puzzle_finetune"


def refuse_incumbent(repo: str) -> None:
    if repo.strip() == INCUMBENT_REPO:
        raise SystemExit(f"refusing to upload puzzle FT over incumbent {INCUMBENT_REPO}")


def write_card(repo: str, steps: int, extra: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    loss = extra.get("recent_loss", "—")
    val = extra.get("recent_val", "—")
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
  - puzzles
  - pytorch
library_name: pytorch
---

# 99M puzzle expert (squares64)

Tactics specialist. Same **99M** squares64 architecture as
[`avewright/chess-transformer-100m-squares64`](https://huggingface.co/avewright/chess-transformer-100m-squares64),
finetuned on official Lichess puzzles (solver plies after the opponent setup move).

This file is **`latest.pt` at puzzle-FT step {steps}** ({stamp}).
Recent train loss ~{loss}. Last puzzle holdout hard CE ~{val}.

This repo is **not** the generalist incumbent. Do not overwrite that model
with these weights.

## Architecture

| | |
|---|---|
| Params | 98.97M |
| Hidden / heads | 736d / 8 |
| Encoder dim | 256 |
| Trunk | prefix 4 + bank 7×3 unrolls + suffix 4 |
| Effective depth | 29 (15 unique layer modules) |
| Attention | 64×64 squares only |
| Vocab | **1968** compact |

Config is in `model_config.json`.

## Training

- Warm start: public 99M `latest.pt` (weights only; optimizer reset).
- Data: [`Lichess/chess-puzzles`](https://huggingface.co/datasets/Lichess/chess-puzzles)
  `{pack.get("revision", "479ea9bc9f681385f5adb23fa27a96c2dc8ae599")}`
- Split: PuzzleId-level 80/20 (seed 273). Train rows {pack.get("train_n", "—")}; eval {pack.get("eval_n", "—")}.
- Labels: one-hot solver moves. `value_valid=0`.
- Optimizer: Polar-NorMuon + AdamW aux. `muon_lr=0.002`, `adam_lr=3e-5`.

## Inference

Requires `avewright/transform` (`chess_inference.py`, `chess_squares64.py`) and compact vocab 1968.

```python
import os
os.environ["MOVE_VOCAB_VERSION"] = "compact"

import chess
from huggingface_hub import hf_hub_download
from chess_inference import load_checkpoint, get_model_move

path = hf_hub_download("{repo}", "latest.pt")
model = load_checkpoint(path, device="cpu")
board = chess.Board()
move, info = get_model_move(model, board, device="cpu")
print(move, info["top_moves"], info["wdl"])
```

## Files

- `latest.pt` — full train checkpoint
- `model_config.json`
- `train.log`
- `pack.json` — puzzle split counts
""",
        encoding="utf-8",
    )
    return card


def recent_metrics(log: Path) -> dict:
    extra: dict = {}
    if not log.exists():
        return extra
    text = log.read_text(errors="replace")
    losses = re.findall(r"loss=([\d.]+)", text)
    if losses:
        extra["recent_loss"] = losses[-1]
    vals = re.findall(r"val/puzzles hard_ce=([\d.]+)", text)
    if vals:
        extra["recent_val"] = vals[-1]
    return extra


def upload(repo: str, ckpt: Path, *, private: bool = False) -> dict:
    refuse_incumbent(repo)
    token = load_hf_token()
    from huggingface_hub import HfApi, create_repo, whoami

    user = whoami(token=token).get("name", "?")
    print(f"hf user={user} repo={repo}", flush=True)
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    steps = ckpt_steps(ckpt)
    out = ckpt.parent
    cfg = out / "model_config.json"
    log = out / "train.log"
    pack_path = out / "pack.json"
    pack = {}
    if pack_path.exists():
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
    extra = recent_metrics(log)
    extra["pack"] = pack
    extra["card_path"] = str(out / "HF_README.md")
    card = write_card(repo, steps, extra)

    create_repo(repo, repo_type="model", exist_ok=True, private=private, token=token)
    api = HfApi(token=token)
    files = [
        (ckpt, "latest.pt", f"exp273 puzzle FT latest.pt step {steps}"),
        (ckpt, f"step_{steps:06d}.pt", f"exp273 puzzle FT step {steps}"),
        (cfg, "model_config.json", "model_config.json"),
        (card, "README.md", "puzzle expert model card"),
        (log, "train.log", "train.log"),
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
    url = f"https://huggingface.co/{repo}"
    print(f"uploaded {repo} latest.pt steps={steps}", flush=True)
    print(url, flush=True)
    (out / "hf_upload.json").write_text(
        json.dumps({"repo": repo, "steps": steps, "url": url, "ckpt": str(ckpt)}, indent=2),
        encoding="utf-8",
    )
    return {"repo": repo, "steps": steps, "url": url}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ckpt", default=str(OUT / "latest.pt"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    refuse_incumbent(args.repo)
    upload(args.repo, Path(args.ckpt), private=args.private)


if __name__ == "__main__":
    main()
