#!/usr/bin/env python3
"""Upload exp201 squares64 checkpoint to Hugging Face. Token from .env / env."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "avewright/chess-transformer-100m-squares64"
OUT = ROOT / "outputs" / "exp201_recurrent_64"


def load_hf_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                value = value.strip().strip("'").strip('"')
                if value:
                    os.environ["HF_TOKEN"] = value
                    return value
    raise SystemExit("HF_TOKEN not found in env or .env")


def ckpt_steps(path: Path) -> int:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("steps", 0))


def write_card(repo: str, steps: int, extra: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    loss = extra.get("recent_loss", "—")
    card = OUT / "HF_README.md"
    card.write_text(
        f"""---
license: mit
tags:
  - chess
  - transformer
  - recurrent
  - policy
  - pytorch
library_name: pytorch
---

# Chess transformer 100M (squares64 recurrent)

~**99M** parameter recurrent chess policy/value net. Attention is only over the
64 board squares. Turn, castling, and en passant are FiLM on the square stream,
not extra tokens.

This file is **`latest.pt` at step {steps}** ({stamp}). Recent train loss ~{loss}.

## Architecture

| | |
|---|---|
| Params | 98.97M |
| Hidden / heads | 736d / 8 |
| Encoder dim | 256 |
| Trunk | prefix 4 + bank 7×3 unrolls + suffix 4 |
| Effective depth | 29 (15 unique layer modules) |
| Attention | 64×64 squares only |
| Embeds | fused piece×color (13) |
| Side info | FiLM (turn / castling / EP), zero-init |
| Heads | spatial policy (vocab **1968** compact) + 3-way WDL |
| Recurrent grads | bank grads divided by 3 after backward |

Config is in `model_config.json` (`Squares64RecurrentConfig`).

## Training data

Soft labels from public Hugging Face packs (depth ≥ 12, phase-balanced open/mid/end):

- [`avewright/chess-soft-multipv-lichess`](https://huggingface.co/datasets/avewright/chess-soft-multipv-lichess) — 8-wide MultiPV (~91M rows in the full pack)
- [`avewright/chess-soft-syzygy`](https://huggingface.co/datasets/avewright/chess-soft-syzygy) — tablebase WDL/policy (~498k rows)

**Mix at this checkpoint**

- Soft cache: **10.5M** rows (**10.30M unique** boards). First 16.5k steps used a 1.5M slice; at step 16488 six disjoint 1.5M shards were merged in.
- Deep cache: **400k** Syzygy rows, mixed into **40%** of batches.
- Soft objective: α=0.55 toward the MultiPV distribution, T=4, temp aux weight 0.4.
- Horizontal flip aug 50%. Min label depth 12.
- Value: 3-way WDL, loss weight 0.15.

The 91M MultiPV pack is not fully consumed. More disjoint shards are queued for later segments.

## Training config

Hardware: 1× RTX 2000 Ada 16GB. Batch **192** (fill-VRAM probe; 256 OOM), accum 1.

| | steps 1–16488 | steps 16488+ (this ckpt) |
|---|---|---|
| Soft unique | 1.5M | 10.5M |
| Optimizer | Polar NorMuon (97.7M) + AdamW aux (1.3M) | same, **floor LR** |
| muon_lr / adam_lr | 0.02 / 3e-4 | 0.001 / 1.5e-5 |
| warmup | 500 | 0 |
| min_lr_frac | 0.05 cosine | 1.0 (held) |
| Syzygy mix | 0.4 | 0.4 |
| compile | `torch.compile` + `compile_polar` | same |
| SWA | from 75% of the *segment* | next SWA at step 34866 |

Other: weight decay 0.01, grad clip 1.0, no grad checkpoint. Checkpoints every 250 steps.

## Inference

Requires this repo (`chess_inference.py`, `chess_squares64.py`) and **compact vocab 1968**.

```bash
pip install torch huggingface_hub python-chess
export MOVE_VOCAB_VERSION=compact
```

```python
import os
os.environ["MOVE_VOCAB_VERSION"] = "compact"

import chess
from huggingface_hub import hf_hub_download
from chess_inference import load_checkpoint, get_model_move

path = hf_hub_download("{repo}", "latest.pt")
device = "cpu"  # or "cuda"
model = load_checkpoint(path, device=device)

board = chess.Board()
move, info = get_model_move(model, board, device)
print(move, info["top_moves"], info["wdl"])
```

Local play GUI (policy argmax, no search):

```bash
export MOVE_VOCAB_VERSION=compact
python play_factory_gui.py -c latest.pt --policy-only --device cpu -p 8080
```

## Rough Elo (noisy)

Greedy policy, no book, no Syzygy. 18 games vs Stockfish 14.1 `UCI_Elo` (3 openings × both colors), CPU, step 30750:

- SF 1350: 5.5/6
- SF 1450: 4/6
- SF 1600: 3/6

Treat as **~1500–1650** policy Elo, not a rating.

## Files

- `latest.pt` — full train checkpoint (`model_state_dict`, `config`, `steps`, optimizer)
- `model_config.json` — architecture
- `train.log` — step/loss log
""",
        encoding="utf-8",
    )
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ckpt", default=str(OUT / "latest.pt"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = load_hf_token()
    from huggingface_hub import HfApi, create_repo, whoami

    user = whoami(token=token).get("name", "?")
    print(f"hf user={user} repo={args.repo}", flush=True)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    steps = ckpt_steps(ckpt)
    cfg = OUT / "model_config.json"
    log = OUT / "train.log"
    if not cfg.exists():
        sys.path.insert(0, str(ROOT))
        from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG

        cfg.write_text(
            json.dumps(DEFAULT_100M_SQUARES64_CONFIG.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    extra = {"steps": steps, "ckpt": ckpt.name, "arch": "squares64", "params_m": 98.97}
    extra["config"] = json.loads(cfg.read_text(encoding="utf-8"))
    if log.exists():
        import re

        losses = re.findall(r"loss=([\d.]+)", log.read_text(errors="replace"))
        if losses:
            extra["recent_loss"] = losses[-1]
    card = write_card(args.repo, steps, extra)

    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private, token=token)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="latest.pt",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"exp201 squares64 step {steps}",
    )
    if cfg.exists():
        api.upload_file(
            path_or_fileobj=str(cfg),
            path_in_repo="model_config.json",
            repo_id=args.repo,
            repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    if log.exists():
        api.upload_file(
            path_or_fileobj=str(log),
            path_in_repo="train.log",
            repo_id=args.repo,
            repo_type="model",
        )
    print(f"uploaded {args.repo} latest.pt steps={steps}", flush=True)
    print(f"https://huggingface.co/{args.repo}", flush=True)


if __name__ == "__main__":
    main()
