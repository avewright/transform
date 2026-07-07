#!/usr/bin/env python3
"""Upload the best model checkpoint to HuggingFace Hub.

Usage:
    # First, login:
    huggingface-cli login

    # Then upload:
    python upload_model_hf.py
    python upload_model_hf.py --repo avewright/chess-transformer-200m
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="avewright/chess-transformer-200m",
                        help="HF repo name")
    parser.add_argument("--model", default="outputs/exp073_200m_full_epoch/best_model.pt",
                        help="Path to model weights")
    parser.add_argument("--checkpoint", default="outputs/exp073_200m_full_epoch/checkpoints/step_10000.pt",
                        help="Path to full checkpoint (optional)")
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    print(f"Logged in as: {user}")

    # Create repo if needed
    repo_id = args.repo
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"Repo: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Model card
    model_card = """---
license: mit
tags:
  - chess
  - transformer
  - policy-value
datasets:
  - avewright/chess-positions-lichess-sf
---

# ChessTransformer200M

A 204M parameter chess-native transformer trained on Stockfish-labeled positions.

## Architecture
- **Encoder**: FusedBoardEncoder (256d) — learned piece-color + square + context embeddings
- **Backbone**: 16-layer Transformer (1024d, 16 heads, FFN 4096, GELU, norm_first)
- **Policy Head**: SpatialPolicyHead (from×to square features, 512d)
- **Value Head**: WDL (win/draw/loss) classification

## Training
- **Dataset**: avewright/chess-positions-lichess-sf (10.2M positions seen out of 48M available)
- **Steps**: 10,000 optimizer steps (effective batch 1024)
- **Final Policy Loss**: ~2.5 (estimated from loss curve)
- **Top-1 Accuracy**: 18.4% (on 5K eval positions vs Stockfish best moves)
- **GPU**: NVIDIA A40 46GB, FP16 + torch.compile
- **Training time**: ~6 hours to step 10,000

## Usage

```python
import torch
from play import ChessTransformer200M, load_model, encode_board, get_model_move
import chess

model = load_model("best_model.pt", torch.device("cpu"))
board = chess.Board()
move, info = get_model_move(model, board, torch.device("cpu"))
print(f"Best move: {move.uci()}, Top 5: {info['top_moves']}")
```

## Files
- `best_model.pt` — Model weights only (816 MB)
- `training_log.json` — Loss curve data
- `config.json` — Architecture config

## Known Issues
- Training hit FP16 NaN at step ~13,800. Best checkpoint is step 10,000.
- Model is only ~21% through 1 epoch of the 48M subset dataset.
- Opens with 1.d4 as White. Plays reasonable chess but still early in training.
"""

    config = {
        "architecture": "ChessTransformer200M",
        "params": 204_006_404,
        "encoder_dim": 256,
        "hidden_dim": 1024,
        "num_layers": 16,
        "num_heads": 16,
        "ffn_ratio": 4,
        "policy_head_dim": 512,
        "value_hidden": 512,
        "vocab_size": 5504,
        "training": {
            "steps": 10000,
            "positions_seen": 10_240_256,
            "batch_size": 256,
            "accum_steps": 4,
            "effective_batch": 1024,
            "lr": 2e-4,
            "best_accuracy": 0.184,
            "dataset": "avewright/chess-positions-lichess-sf",
            "gpu": "NVIDIA A40 46GB",
        }
    }

    # Write temp files
    card_path = Path("/tmp/README.md")
    card_path.write_text(model_card)

    config_path = Path("/tmp/config.json")
    config_path.write_text(json.dumps(config, indent=2))

    # Upload files
    print("Uploading model weights...")
    api.upload_file(path_or_fileobj=args.model, path_in_repo="best_model.pt",
                    repo_id=repo_id, repo_type="model")

    print("Uploading README...")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")

    print("Uploading config...")
    api.upload_file(path_or_fileobj=str(config_path), path_in_repo="config.json",
                    repo_id=repo_id, repo_type="model")

    # Upload training log
    log_path = Path("outputs/exp073_200m_full_epoch/training_log.json")
    if log_path.exists():
        print("Uploading training log...")
        api.upload_file(path_or_fileobj=str(log_path), path_in_repo="training_log.json",
                        repo_id=repo_id, repo_type="model")

    print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
