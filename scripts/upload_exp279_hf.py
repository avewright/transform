#!/usr/bin/env python3
"""Upload the exp279 MoE router head to avewright/chess-moe-router.

Never writes the 99M incumbent or any specialist repo. The uploaded
checkpoint is the router MLP only; experts stay in their own repos.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chess_moe import EXPERTS, N_EXPERTS, TRUNK_HIDDEN_DIM, router_param_count  # noqa: E402
from upload_exp201_hf import ckpt_steps, load_hf_token  # noqa: E402

DEFAULT_REPO = "avewright/chess-moe-router"
BLOCKED = {
    "avewright/chess-transformer-100m-squares64",
    "avewright/puzzle-model",
    "avewright/syzygy-model",
    "avewright/endgame-model",
    "avewright/opening-model",
    "avewright/middlegame-model",
}
OUT = ROOT / "outputs" / "exp279_moe_router"


def refuse_blocked(repo: str) -> None:
    if repo.strip() in BLOCKED:
        raise SystemExit(f"refusing to upload router over {repo}")


def recent_metrics(log: Path) -> dict:
    extra: dict = {}
    if not log.exists():
        return extra
    text = log.read_text(errors="replace")
    losses = re.findall(r"step \d+/\d+ \| loss=([\d.]+)", text)
    if losses:
        extra["recent_loss"] = losses[-1]
    vals = re.findall(r"val_cls_ce=([\d.]+)", text)
    if vals:
        extra["recent_val"] = vals[-1]
        extra["best_val"] = min(vals, key=float)
    accs = re.findall(r"best_acc=([\d.]+)", text)
    if accs:
        extra["recent_acc"] = accs[-1]
    return extra


def write_card(repo: str, steps: int, extra: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    loss = extra.get("recent_loss", "—")
    val = extra.get("best_val", extra.get("recent_val", "—"))
    acc = extra.get("recent_acc", "—")
    experts_md = "\n".join(f"- `{name}` → [`{hf}`](https://huggingface.co/{hf})" for name, hf in EXPERTS)
    card = Path(extra.get("card_path") or (OUT / "HF_README.md"))
    card.write_text(
        f"""---
license: mit
tags:
  - chess
  - mixture-of-experts
  - router
  - pytorch
library_name: pytorch
---

# Frozen-expert MoE router

Small MLP gate over five frozen **99M** squares64 specialists. Only this
router is trained; the experts stay in their own repos.

This file is **`latest.pt` at router step {steps}** ({stamp}).
Train loss ~{loss}. Val source-cls CE ~{val}. Val acc ~{acc}.

Not a 99M policy. Not the incumbent, puzzle, endgame, opening, or middlegame expert.

## Experts

{experts_md}

## Architecture

- Stem: frozen incumbent `global_hidden` (736d)
- Head: LayerNorm → Linear(736,256) → GELU → Linear(256,256) → GELU → Linear(256,5)
- Params: {router_param_count():,}
- Vocab: compact 1968
- Dispatch: hard argmax, one specialist forward (incumbent encode is reused if chosen)

## Files

- `latest.pt` — router weights + expert repo list
- `step_{steps:06d}.pt`
- `router_config.json`
- `train.log`
""",
        encoding="utf-8",
    )
    return card


def tagged_ckpt(src: Path, dest: Path) -> dict:
    import torch

    blob = torch.load(src, map_location="cpu", weights_only=False)
    blob["arch"] = "frozen_moe_router"
    blob["vocab_version"] = "compact"
    blob["experts"] = list(EXPERTS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, dest)
    return blob


def upload(repo: str, ckpt: Path, *, private: bool = False) -> dict:
    refuse_blocked(repo)
    token = load_hf_token()
    from huggingface_hub import HfApi, create_repo, whoami

    user = whoami(token=token).get("name", "?")
    print(f"hf user={user} repo={repo}", flush=True)
    if not ckpt.exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    extra = recent_metrics(OUT / "train.log")
    tagged = OUT / "latest.pt"
    blob = tagged_ckpt(ckpt, tagged)
    steps = int(blob.get("steps") or ckpt_steps(tagged))
    extra["card_path"] = str(OUT / "HF_README.md")
    card = write_card(repo, steps, extra)
    cfg = {
        "arch": "frozen_moe_router",
        "hidden_dim": TRUNK_HIDDEN_DIM,
        "mlp_hidden": 256,
        "n_experts": N_EXPERTS,
        "experts": [{"name": n, "repo": r} for n, r in EXPERTS],
        "vocab_version": "compact",
        "steps": steps,
        "val_cls_ce": blob.get("val_cls_ce"),
        "val_acc": (blob.get("val") or {}).get("best_acc"),
    }
    cfg_path = OUT / "router_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    step_copy = OUT / f"step_{steps:06d}.pt"
    if not step_copy.exists() or step_copy.resolve() != tagged.resolve():
        import shutil

        shutil.copy2(tagged, step_copy)
    create_repo(repo, repo_type="model", exist_ok=True, private=private, token=token)
    api = HfApi(token=token)
    files = [
        (tagged, "latest.pt", f"exp279 MoE router latest.pt step {steps}"),
        (step_copy, f"step_{steps:06d}.pt", f"exp279 MoE router step {steps}"),
        (cfg_path, "router_config.json", "router_config.json"),
        (card, "README.md", "MoE router model card"),
        (OUT / "train.log", "train.log", "train.log"),
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
    ap.add_argument("--ckpt", default=str(OUT / "router_best.pt"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    refuse_blocked(args.repo)
    upload(args.repo, Path(args.ckpt), private=args.private)


if __name__ == "__main__":
    main()
