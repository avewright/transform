#!/usr/bin/env python3
"""Auto-push daemon for exp075 training.

Watches for:
1. New best_model.pt → uploads to HuggingFace with updated README + logs
2. Periodically pushes code/logs to GitHub remote

Run in tmux: tmux new -s push 'python3 auto_push.py'
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Config ──
OUTPUT_DIR = Path("outputs/exp075_ddp_4gpu")
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
TRAINING_LOG_PATH = OUTPUT_DIR / "training_log.json"
HF_REPO = "avewright/chess-transformer-200m-v2"
HF_DATASET = "avewright/chess-positions-lichess-sf"
GIT_PUSH_INTERVAL = 1800     # push to GitHub every 30 min
HF_CHECK_INTERVAL = 120      # check for new best model every 2 min

# ── Load tokens from .env ──
def load_env():
    env = {}
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

ENV = load_env()
HF_TOKEN = ENV.get("HF_TOKEN", "")
GH_PAT = ENV.get("GH_PAT", "")

if not HF_TOKEN:
    print("[ERROR] No HF_TOKEN in .env", flush=True)
    sys.exit(1)
if not GH_PAT:
    print("[ERROR] No GH_PAT in .env", flush=True)
    sys.exit(1)


# ── State tracking ──
last_best_model_mtime = 0.0
last_git_push = 0.0
last_hf_step = -1


def get_latest_worker_status():
    """Read latest step/loss from all worker logs."""
    statuses = {}
    for i in range(4):
        log_path = OUTPUT_DIR / f"worker{i}" / "train.log"
        if not log_path.exists():
            continue
        try:
            lines = log_path.read_text().splitlines()
            for line in reversed(lines):
                if f"[W{i}][" in line and "pl=" in line:
                    statuses[f"worker{i}"] = line.strip()
                    break
        except Exception:
            pass
    return statuses


def get_training_summary():
    """Build a training summary from worker logs and training_log.json."""
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp075_ddp_4gpu",
        "parent_model": HF_REPO,
        "dataset": HF_DATASET,
        "strategy": "Local SGD, 4xA40, weight averaging every 500 steps",
        "architecture": "ChessTransformer200M (FusedBoardEncoder 256d→1024d, 16L 16H FFN4x, SpatialPolicyHead)",
        "params": "204M",
        "training_config": {
            "batch_size": 256,
            "accum_steps": 4,
            "effective_batch": 1024,
            "lr": "1e-4 cosine → 5% floor",
            "warmup": "1%",
            "grad_clip": 0.5,
            "gpus": "4x NVIDIA A40 46GB",
        },
    }

    # Read training log for eval results
    if TRAINING_LOG_PATH.exists():
        try:
            log = json.loads(TRAINING_LOG_PATH.read_text())
            summary["eval_history"] = log
            # Find best result
            best = max(
                (e for e in log if "accuracy" in e),
                key=lambda e: e["accuracy"],
                default=None,
            )
            if best:
                summary["best_accuracy"] = best["accuracy"]
                summary["best_step"] = best.get("step", 0)
        except Exception:
            pass

    # Aggregate worker status
    worker_status = get_latest_worker_status()
    if worker_status:
        summary["worker_status"] = worker_status

        # Parse total positions from worker logs
        total_pos = 0
        for i in range(4):
            log_path = OUTPUT_DIR / f"worker{i}" / "train.log"
            if not log_path.exists():
                continue
            try:
                lines = log_path.read_text().splitlines()
                for line in reversed(lines):
                    if f"[W{i}]" in line and "pos " in line:
                        # Extract positions from "pos     435,200/207,772,000"
                        import re
                        m = re.search(r'pos\s+([\d,]+)/', line)
                        if m:
                            total_pos += int(m.group(1).replace(",", ""))
                            break
            except Exception:
                pass
        summary["total_positions_trained"] = total_pos

    # Data split info
    summary["data_splits"] = {
        "worker0": "files [0:818) of 3275 src parquets",
        "worker1": "files [818:1636)",
        "worker2": "files [1636:2454)",
        "worker3": "files [2454:3275)",
        "total_files": 3275,
        "est_total_positions": "~832M",
    }

    return summary


def build_hf_readme(summary):
    """Generate README.md for the HuggingFace model repo."""
    best_acc = summary.get("best_accuracy", "N/A")
    best_step = summary.get("best_step", "N/A")
    total_pos = summary.get("total_positions_trained", 0)
    ts = summary.get("timestamp", "")

    eval_table = ""
    if "eval_history" in summary:
        eval_table = "\n| Step | Positions | Accuracy | Top-3 | SF Rank | Value Acc |\n"
        eval_table += "|------|-----------|----------|-------|---------|----------|\n"
        for e in summary["eval_history"]:
            if "accuracy" in e:
                eval_table += (
                    f"| {e.get('step', 0):,} "
                    f"| {e.get('positions_seen', 0):,} "
                    f"| {e['accuracy']:.1%} "
                    f"| {e.get('top3_accuracy', 0):.1%} "
                    f"| {e.get('mean_sf_rank', 0):.1f} "
                    f"| {e.get('value_accuracy', 0):.1%} |\n"
                )

    readme = f"""---
license: mit
tags:
- chess
- transformer
- policy-value
datasets:
- avewright/chess-positions-lichess-sf
---

# Chess Transformer 200M v2

A 204M parameter chess transformer trained on Stockfish-labeled positions from Lichess games.

## Current Results

- **Best Accuracy**: {best_acc:.1%} (step {best_step:,})
- **Total Positions Trained**: {total_pos:,} across 4 GPUs
- **Last Updated**: {ts}

## Training

- **Experiment**: exp075_ddp_4gpu (Local SGD, 4x NVIDIA A40)
- **Dataset**: [{HF_DATASET}](https://huggingface.co/datasets/{HF_DATASET}) (~832M positions, 3275 source parquets)
- **Architecture**: FusedBoardEncoder 256d → 1024d transformer, 16 layers, 16 heads, FFN 4×, SpatialPolicyHead
- **Strategy**: 4 independent workers each training on 1/4 of data, weights averaged every 500 optimizer steps
- **Batch**: 256 × accum 4 = effective 1024 per worker
- **LR**: 1e-4 cosine schedule → 5% floor, 1% warmup
- **Parent**: Continued from exp074 best checkpoint

## Eval History
{eval_table}
## Architecture

```
ChessTransformer200M (~204M params)
├── FusedBoardEncoder (embed_dim=256)
├── Linear projection (256 → 1024)
├── CLS token + positional embeddings (68 positions)
├── TransformerEncoder (16 layers, 16 heads, FFN 4096, GELU, norm_first)
├── LayerNorm
├── SpatialPolicyHead (head_dim=512) → 1968 moves
└── Value head (1024 → 512 → 3 WDL)
```

## Files

- `best_model.pt` — best checkpoint (state_dict only)
- `training_log.json` — full eval history  
- `config.json` — training configuration
- `train.log` — aggregated worker logs

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download("{HF_REPO}", "best_model.pt")
state_dict = torch.load(path, map_location="cpu", weights_only=True)
# Load into ChessTransformer200M architecture
```
"""
    return readme


def push_to_hf(summary):
    """Upload best model + logs to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print(f"  [HF] Uploading best_model.pt ({BEST_MODEL_PATH.stat().st_size / 1e6:.0f} MB)...",
          flush=True)

    # Upload best model
    api.upload_file(
        path_or_fileobj=str(BEST_MODEL_PATH),
        path_in_repo="best_model.pt",
        repo_id=HF_REPO,
        commit_message=f"exp075: best model step {summary.get('best_step', '?')}, "
                       f"acc={summary.get('best_accuracy', 0):.1%}",
    )
    print("  [HF] best_model.pt uploaded", flush=True)

    # Upload README
    readme = build_hf_readme(summary)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        commit_message=f"Update README: step {summary.get('best_step', '?')}",
    )

    # Upload training log
    if TRAINING_LOG_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(TRAINING_LOG_PATH),
            path_in_repo="training_log.json",
            repo_id=HF_REPO,
            commit_message=f"Update training log: step {summary.get('best_step', '?')}",
        )

    # Upload config
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=HF_REPO,
            commit_message=f"Update config",
        )

    # Upload aggregated worker logs
    combined_log = ""
    for i in range(4):
        log_path = OUTPUT_DIR / f"worker{i}" / "train.log"
        if log_path.exists():
            combined_log += f"\n{'='*72}\n WORKER {i}\n{'='*72}\n"
            combined_log += log_path.read_text()
    if combined_log:
        api.upload_file(
            path_or_fileobj=combined_log.encode("utf-8"),
            path_in_repo="train.log",
            repo_id=HF_REPO,
            commit_message=f"Update worker logs",
        )

    print(f"  [HF] All files uploaded to {HF_REPO}", flush=True)


def push_to_github():
    """Commit and push code + lightweight logs to GitHub."""
    repo_dir = Path(__file__).resolve().parent

    # Set up git auth
    remote_url = f"https://x-access-token:{GH_PAT}@github.com/avewright/transform.git"

    try:
        # Configure git
        subprocess.run(["git", "config", "user.name", "auto-push"], cwd=repo_dir,
                       capture_output=True)
        subprocess.run(["git", "config", "user.email", "auto-push@exp075"],
                       cwd=repo_dir, capture_output=True)

        # Set remote URL with PAT
        subprocess.run(["git", "remote", "set-url", "origin", remote_url],
                       cwd=repo_dir, capture_output=True)

        # Stage experiment files and lightweight logs (not model weights)
        files_to_add = [
            "experiments/exp075_ddp_4gpu.py",
            "watchdog_exp075.sh",
            "auto_push.py",
        ]

        # Also add training log if small enough
        training_log = OUTPUT_DIR / "training_log.json"
        if training_log.exists() and training_log.stat().st_size < 1_000_000:
            files_to_add.append(str(training_log))

        config_json = OUTPUT_DIR / "config.json"
        if config_json.exists():
            files_to_add.append(str(config_json))

        for f in files_to_add:
            if Path(f).exists():
                subprocess.run(["git", "add", f], cwd=repo_dir, capture_output=True)

        # Check if there's anything to commit
        result = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                cwd=repo_dir, capture_output=True)
        if result.returncode == 0:
            print("  [GIT] Nothing new to commit", flush=True)
            return

        # Get latest worker status for commit message
        workers = get_latest_worker_status()
        step_info = ""
        if workers:
            import re
            for w, line in workers.items():
                m = re.search(r'step\s+([\d,]+)', line)
                if m:
                    step_info = f"step ~{m.group(1)}"
                    break

        msg = f"exp075: auto-push {step_info} [{datetime.now().strftime('%H:%M')}]"
        subprocess.run(["git", "commit", "-m", msg], cwd=repo_dir,
                       capture_output=True)

        result = subprocess.run(["git", "push", "origin", "main"],
                                cwd=repo_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [GIT] Pushed to GitHub: {msg}", flush=True)
        else:
            print(f"  [GIT] Push failed: {result.stderr[:200]}", flush=True)

    except Exception as e:
        print(f"  [GIT] Error: {e}", flush=True)


def main():
    global last_best_model_mtime, last_git_push, last_hf_step

    print(f"\n{'='*60}")
    print(f" AUTO-PUSH DAEMON STARTED")
    print(f"{'='*60}")
    print(f"  HF repo:  {HF_REPO}")
    print(f"  Git push: every {GIT_PUSH_INTERVAL}s")
    print(f"  HF check: every {HF_CHECK_INTERVAL}s")
    print(f"  Time:     {datetime.now(timezone.utc).isoformat()}")
    print(flush=True)

    # Initial git push of experiment files
    print("\n[INIT] Pushing experiment files to GitHub...", flush=True)
    push_to_github()
    last_git_push = time.time()

    # Track initial best_model.pt
    if BEST_MODEL_PATH.exists():
        last_best_model_mtime = BEST_MODEL_PATH.stat().st_mtime

    while True:
        try:
            now = time.time()

            # Check for new best model
            if BEST_MODEL_PATH.exists():
                current_mtime = BEST_MODEL_PATH.stat().st_mtime
                if current_mtime > last_best_model_mtime:
                    # Also check training_log to see if there's a new eval
                    current_step = -1
                    if TRAINING_LOG_PATH.exists():
                        try:
                            log = json.loads(TRAINING_LOG_PATH.read_text())
                            best = max(
                                (e for e in log if "accuracy" in e),
                                key=lambda e: e["accuracy"],
                                default=None,
                            )
                            if best:
                                current_step = best.get("step", -1)
                        except Exception:
                            pass

                    if current_step != last_hf_step:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                              f"New best model detected (step {current_step})!",
                              flush=True)
                        summary = get_training_summary()
                        push_to_hf(summary)
                        last_best_model_mtime = current_mtime
                        last_hf_step = current_step

                        # Also push to git when HF updates
                        push_to_github()
                        last_git_push = now

            # Periodic git push
            if now - last_git_push >= GIT_PUSH_INTERVAL:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Periodic git push...", flush=True)
                push_to_github()
                last_git_push = now

            time.sleep(HF_CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n[STOP] Daemon stopped.", flush=True)
            break
        except Exception as e:
            print(f"\n[ERROR] {e}", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
