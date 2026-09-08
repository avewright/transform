#!/usr/bin/env python3
"""Evacuate the incumbent overnight SWA and pod-unique data to Hugging Face.

Does not overwrite avewright/chess-transformer-100m-squares64.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import _hf_token  # noqa: E402

OVERNIGHT = ROOT / "outputs/sf19_ft/overnight_20260908"
MODEL_REPO = "avewright/chess-transformer-100m-overnight_20260908"
GEN_REPO = "avewright/chess-soft-sf19"
CORR_REPO = "avewright/chess-soft-100m-swa-mistakes"
STAGE = ROOT / "outputs/hf_pod_snapshot"


def log(msg: str) -> None:
    print(msg, flush=True)


def write_model_card(dest: Path) -> None:
    dest.write_text(
        """---
license: mit
tags:
  - chess
  - transformer
  - recurrent
  - policy
  - pytorch
library_name: pytorch
---

# Chess transformer 100M — overnight SF19 SWA (2026-09-08)

Searchless greedy policy. Incumbent after a matched 128-game SF19 2050/2200
screen against two 8k fine-tunes. Those mixes did not beat this checkpoint
on a paired-opening bootstrap.

**This is not FIDE or Lichess Elo.** Local UCI_Elo screen ≈ **2150**
(2050 score 0.602, 2200 score 0.445, 128 games, 8000 nodes, no book / no Syzygy).
A same-protocol re-run scored 0.504 overall (estimate 2132). Treat the band,
not a single number.

The public repo `avewright/chess-transformer-100m-squares64` was **not** overwritten.

## Files

- `eval_swa.pt` / `latest.pt` — evaluation weights (same SWA snapshot)
- `training_resume.pt` — full overnight live checkpoint (optimizer + RNG)
- `model_config.json` — Squares64RecurrentConfig
- `evaluation/` — gauntlet, holdouts, mix audits

## Inference

```python
import os
os.environ["MOVE_VOCAB_VERSION"] = "compact"
import chess
from huggingface_hub import hf_hub_download
from chess_inference import load_checkpoint, get_model_move

path = hf_hub_download("avewright/chess-transformer-100m-overnight_20260908", "eval_swa.pt")
model = load_checkpoint(path, device="cuda")
move, info = get_model_move(model, chess.Board(), "cuda")
```

Compact vocab **1968**. No search at inference.
""",
        encoding="utf-8",
    )


def stage_model() -> Path:
    d = STAGE / "model"
    ev = d / "evaluation"
    if d.exists():
        shutil.rmtree(d)
    ev.mkdir(parents=True)
    shutil.copy2(OVERNIGHT / "eval_swa.pt", d / "eval_swa.pt")
    shutil.copy2(OVERNIGHT / "eval_swa.pt", d / "latest.pt")
    shutil.copy2(OVERNIGHT / "latest.pt", d / "training_resume.pt")
    shutil.copy2(OVERNIGHT / "model_config.json", d / "model_config.json")
    write_model_card(d / "README.md")
    copies = {
        ROOT / "outputs/searchless_gauntlet/report.json": ev / "gauntlet_report.json",
        ROOT / "outputs/elo_eval_overnight_swa_2050_2200_n8000.json": ev / "elo_reference_2050_2200.json",
        ROOT / "outputs/elo_eval_searchless_incumbent_n8000_r4.json": ev / "elo_incumbent_recheck.json",
        ROOT / "outputs/elo_eval_searchless_control_n8000_r4.json": ev / "elo_control.json",
        ROOT / "outputs/elo_eval_searchless_astra_n8000_r4.json": ev / "elo_astra.json",
        ROOT / "outputs/astra_compare/compare_report.json": ev / "astra_compare.json",
        ROOT / "outputs/astra_compare/holdout.json": ev / "astra_holdout.json",
        ROOT / "outputs/astra_mix_v2/mix_report.json": ev / "mix_v2_report.json",
        ROOT / "outputs/astra_mix_v2/mistakes_audit.json": ev / "mistakes_audit.json",
        OVERNIGHT / "data_audit.json": ev / "data_audit.json",
        OVERNIGHT / "promotion.json": ev / "promotion.json",
        OVERNIGHT / "finished.json": ev / "finished.json",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src, dst)
    return d


def upload_model(api, token: str, folder: Path) -> None:
    from huggingface_hub import create_repo

    create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True, token=token)
    log(f"upload model {MODEL_REPO}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token,
        commit_message="overnight SWA incumbent + evaluation snapshot",
    )
    log(f"https://huggingface.co/{MODEL_REPO}")


def upload_generated(api, token: str) -> None:
    import pyarrow.parquet as pq
    from export_soft_caches_to_hf import sf19_chunk_table

    gen = OVERNIGHT / "generated_verified"
    shards = sorted(gen.glob("shard_*.pt"))
    if not shards:
        log("no overnight generated shards")
        return
    staging = STAGE / "sf19_gen"
    staging.mkdir(parents=True, exist_ok=True)
    existing = set(api.list_repo_files(GEN_REPO, repo_type="dataset"))
    for p in shards:
        remote = f"overnight_20260908/{p.name.replace('.pt', '.parquet')}"
        if remote in existing:
            log(f"skip {remote}")
            continue
        d = __import__("torch").load(p, map_location="cpu", weights_only=False)
        n = int(d["move_idx"].shape[0])
        local = staging / p.with_suffix(".parquet").name
        pq.write_table(sf19_chunk_table(d, p.stem, 0, n), local, compression="zstd")
        log(f"upload {remote} n={n:,}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=GEN_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"overnight generated {p.name} n={n:,}",
        )
        local.unlink(missing_ok=True)
    extra = OVERNIGHT / "generation" / "summary.json"
    if extra.exists():
        api.upload_file(
            path_or_fileobj=str(extra),
            path_in_repo="overnight_20260908/summary.json",
            repo_id=GEN_REPO,
            repo_type="dataset",
            token=token,
            commit_message="overnight generation summary",
        )
    log(f"https://huggingface.co/datasets/{GEN_REPO} overnight_20260908/")


def upload_corrections(api, token: str) -> None:
    import pyarrow.parquet as pq
    from export_soft_caches_to_hf import cache_chunk_table

    bonus = ROOT / "outputs/astra_mix_v2/bonus_cache.pt"
    if not bonus.exists():
        log("no mix v2 bonus")
        return
    import torch

    d = torch.load(bonus, map_location="cpu", weights_only=False)
    n = int(d["board_array"].shape[0])
    staging = STAGE / "corrections"
    staging.mkdir(parents=True, exist_ok=True)
    local = staging / "verified_corrections.parquet"
    pq.write_table(cache_chunk_table(d, "verified_corrections", 0, n), local, compression="zstd")
    remote = "data/corrections/verified_246k.parquet"
    log(f"upload {CORR_REPO} {remote} n={n:,}")
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=CORR_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"verified substantial corrections n={n:,}",
    )
    report = ROOT / "outputs/astra_mix_v2/mix_report.json"
    if report.exists():
        api.upload_file(
            path_or_fileobj=str(report),
            path_in_repo="data/corrections/mix_v2_report.json",
            repo_id=CORR_REPO,
            repo_type="dataset",
            token=token,
            commit_message="mix v2 report",
        )
    log(f"https://huggingface.co/datasets/{CORR_REPO}")


def main() -> None:
    token = _hf_token()
    if not token:
        raise SystemExit("HF_TOKEN missing")
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    STAGE.mkdir(parents=True, exist_ok=True)
    folder = stage_model()
    upload_model(api, token, folder)
    upload_generated(api, token)
    upload_corrections(api, token)
    log("snapshot uploads done")


if __name__ == "__main__":
    main()
