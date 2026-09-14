#!/usr/bin/env python3
"""Upload the jointly trained MoE (router + five 99M experts).

Default repo: avewright/chess-moe
Also refreshes avewright/chess-moe-router with the joint-train router head.

Never writes the original specialist repos
(chess-transformer-100m-squares64 / puzzle-model / endgame-model /
opening-model / middlegame-model).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chess_moe import EXPERTS, N_EXPERTS, TRUNK_HIDDEN_DIM, router_param_count  # noqa: E402
from upload_exp201_hf import format_elo_md, load_hf_token  # noqa: E402

DEFAULT_REPO = "avewright/chess-moe"
ROUTER_REPO = "avewright/chess-moe-router"
BLOCKED = {
    "avewright/chess-transformer-100m-squares64",
    "avewright/puzzle-model",
    "avewright/syzygy-model",
    "avewright/endgame-model",
    "avewright/opening-model",
    "avewright/middlegame-model",
}
OUT = ROOT / "outputs" / "exp280_moe_full"


def refuse_blocked(repo: str) -> None:
    if repo.strip() in BLOCKED:
        raise SystemExit(f"refusing to upload joint MoE over {repo}")


def recent_metrics(log: Path) -> dict:
    extra: dict = {}
    if not log.exists():
        return extra
    text = log.read_text(errors="replace")
    losses = re.findall(r"step \d+/\d+ \| loss=([\d.]+)", text)
    if losses:
        extra["recent_loss"] = losses[-1]
    vals = re.findall(r"val/lichess hard_ce=([\d.]+)", text)
    if vals:
        extra["recent_val"] = vals[-1]
        extra["best_val"] = min(vals, key=float)
    return extra


def write_card(repo: str, steps: int, extra: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    loss = extra.get("recent_loss", "—")
    val = extra.get("best_val", extra.get("recent_val", "—"))
    elo_md = extra.get("elo_md") or "Search-free greedy policy. Elo pending."
    experts_md = "\n".join(f"- `{n}`" for n, _ in EXPERTS)
    card = Path(extra.get("card_path") or (OUT / "HF_README.md"))
    card.write_text(
        f"""---
license: mit
tags:
  - chess
  - mixture-of-experts
  - transformer
  - recurrent
  - policy
  - pytorch
library_name: pytorch
---

# Chess MoE (five 99M experts + router)

Jointly trained Switch MoE. Five **99M** squares64 specialists plus a small
MLP router on the incumbent `global_hidden`. Compact vocab 1968.

This file is **`latest.pt` at step {steps}** ({stamp}).
Train loss ~{loss}. Val hard CE ~{val}.

## Elo

{elo_md}

## Experts

{experts_md}

Router: LayerNorm → 736→256→256→5 (~{router_param_count():,} params).
Hard argmax dispatch. Incumbent encode is reused if the gate picks it.

## Files

- `latest.pt` — full bundle (router + all five expert state dicts)
- `router.pt` — router head only
- `experts/<name>.pt` — each 99M specialist
- `router_config.json`
- `train.log`
- `elo_eval.json` (if present)

Load with this repo and `MOVE_VOCAB_VERSION=compact`:

```python
from chess_inference import load_checkpoint, get_model_move
model = load_checkpoint("latest.pt", device="cuda")
```

Not a drop-in replacement for the standalone incumbent / puzzle / endgame /
opening / middlegame repos. Those stay as the pre-joint specialists.
""",
        encoding="utf-8",
    )
    return card


def split_bundle(src: Path, dest_dir: Path) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    blob = torch.load(src, map_location="cpu", weights_only=False)
    steps = int(blob.get("steps") or 0)
    router = {
        "arch": "frozen_moe_router",
        "vocab_version": "compact",
        "model_state_dict": blob["model_state_dict"],
        "experts": list(EXPERTS),
        "steps": steps,
    }
    torch.save(router, dest_dir / "router.pt")
    exp_dir = dest_dir / "experts"
    exp_dir.mkdir(exist_ok=True)
    states = blob.get("expert_state_dicts") or []
    for i, (name, _) in enumerate(EXPERTS):
        if i >= len(states) or states[i] is None:
            continue
        torch.save(
            {
                "arch": "squares64",
                "vocab_version": "compact",
                "model_state_dict": states[i],
                "steps": steps,
                "expert": name,
            },
            exp_dir / f"{name}.pt",
        )
    cfg = {
        "arch": "chess_moe_full",
        "hidden_dim": TRUNK_HIDDEN_DIM,
        "n_experts": N_EXPERTS,
        "experts": [{"name": n, "repo": r} for n, r in EXPERTS],
        "vocab_version": "compact",
        "steps": steps,
        "val_soft_ce": blob.get("val_soft_ce"),
    }
    (dest_dir / "router_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return {"steps": steps, "blob": blob}


def upload(repo: str, ckpt: Path, *, private: bool = False, elo_json: Path | None = None) -> dict:
    refuse_blocked(repo)
    token = load_hf_token()
    from huggingface_hub import HfApi, create_repo, whoami

    user = whoami(token=token).get("name", "?")
    print(f"hf user={user} repo={repo}", flush=True)
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    extra = recent_metrics(OUT / "train.log")
    if elo_json and elo_json.exists():
        extra["elo_md"] = format_elo_md(elo_json)
    split = split_bundle(ckpt, OUT)
    steps = split["steps"]
    extra["card_path"] = str(OUT / "HF_README.md")
    card = write_card(repo, steps, extra)
    create_repo(repo, repo_type="model", exist_ok=True, private=private, token=token)
    api = HfApi(token=token)
    files = [
        (ckpt, "latest.pt", f"exp280 full MoE latest.pt step {steps}"),
        (OUT / "router.pt", "router.pt", f"exp280 router step {steps}"),
        (OUT / "router_config.json", "router_config.json", "router_config.json"),
        (card, "README.md", "full MoE model card"),
        (OUT / "train.log", "train.log", "train.log"),
    ]
    for name, _ in EXPERTS:
        p = OUT / "experts" / f"{name}.pt"
        files.append((p, f"experts/{name}.pt", f"exp280 expert {name} step {steps}"))
    if elo_json and elo_json.exists():
        files.append((elo_json, "elo_eval.json", "elo gauntlet"))
    for src, dest, msg in files:
        if not src.exists():
            print(f"skip missing {src}", flush=True)
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=repo,
            repo_type="model",
            commit_message=msg,
        )
        print(f"uploaded {repo} {dest}", flush=True)
    if repo != ROUTER_REPO:
        refuse_blocked(ROUTER_REPO)
        create_repo(ROUTER_REPO, repo_type="model", exist_ok=True, private=private, token=token)
        api.upload_file(
            path_or_fileobj=str(OUT / "router.pt"),
            path_in_repo="latest.pt",
            repo_id=ROUTER_REPO,
            repo_type="model",
            commit_message=f"joint-train router step {steps}",
        )
        print(f"uploaded {ROUTER_REPO} latest.pt", flush=True)
    url = f"https://huggingface.co/{repo}"
    print(f"uploaded {repo} steps={steps}", flush=True)
    print(url, flush=True)
    return {"repo": repo, "steps": steps, "url": url}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ckpt", default=str(OUT / "best.pt"))
    ap.add_argument("--elo-json", default="")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    refuse_blocked(args.repo)
    elo = Path(args.elo_json) if args.elo_json else None
    upload(args.repo, Path(args.ckpt), private=args.private, elo_json=elo)


if __name__ == "__main__":
    main()
