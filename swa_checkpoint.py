"""Stochastic Weight Averaging (SWA) for chess transformer checkpoints.

Averages model weights from multiple checkpoints to produce a smoother model
that generalizes better. Used by ChessFormer (Monroe et al. 2024) and standard
in modern deep learning.

Typical usage:
  # Average last 5 checkpoints (every 1K steps)
  python swa_checkpoint.py outputs/exp149_scratch_204m/step_*.pt -o swa_model.pt

  # Average specific checkpoints
  python swa_checkpoint.py ckpt_100k.pt ckpt_101k.pt ckpt_102k.pt -o swa.pt

  # Average with exponential decay (recent checkpoints weighted more)
  python swa_checkpoint.py outputs/exp149_scratch_204m/step_*.pt -o swa.pt --decay 0.9
"""

import argparse
import sys
from pathlib import Path

import torch


def average_checkpoints(paths, decay=None):
    """Average model_state_dict from multiple checkpoint files.
    
    Args:
        paths: list of checkpoint file paths
        decay: if set, use exponential weighting (most recent = highest weight)
               e.g. decay=0.9 → weights [0.9^(n-1), 0.9^(n-2), ..., 0.9^0]
    
    Returns:
        averaged state_dict, config dict, list of steps
    """
    assert len(paths) >= 2, f"Need at least 2 checkpoints, got {len(paths)}"
    
    # Sort by step number if available
    ckpts = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        step = ckpt.get("step", 0)
        ckpts.append((step, p, ckpt))
    ckpts.sort(key=lambda x: x[0])
    
    steps = [s for s, _, _ in ckpts]
    print(f"Averaging {len(ckpts)} checkpoints: steps {steps}")
    
    # Compute weights
    n = len(ckpts)
    if decay is not None:
        weights = [decay ** (n - 1 - i) for i in range(n)]
    else:
        weights = [1.0] * n
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    print(f"Weights: {[f'{w:.3f}' for w in weights]}")
    
    # Average state dicts
    avg_state = {}
    ref_state = ckpts[0][2]["model_state_dict"]
    
    for key in ref_state:
        tensors = []
        for _, _, ckpt in ckpts:
            t = ckpt["model_state_dict"][key]
            tensors.append(t.float())
        
        avg = sum(w * t for w, t in zip(weights, tensors))
        avg_state[key] = avg.to(ref_state[key].dtype)
    
    config = ckpts[-1][2].get("config", ckpts[0][2].get("config"))
    return avg_state, config, steps


def main():
    ap = argparse.ArgumentParser(description="SWA checkpoint averaging")
    ap.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files to average")
    ap.add_argument("-o", "--output", required=True, help="Output checkpoint path")
    ap.add_argument("--decay", type=float, default=None,
                    help="Exponential decay factor (e.g. 0.9). Default: uniform average")
    args = ap.parse_args()
    
    paths = []
    for p in args.checkpoints:
        pp = Path(p)
        if not pp.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        paths.append(pp)
    
    if len(paths) < 2:
        print("ERROR: Need at least 2 valid checkpoint files")
        sys.exit(1)
    
    avg_state, config, steps = average_checkpoints(paths, decay=args.decay)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "model_state_dict": avg_state,
        "config": config,
        "step": steps[-1],
        "swa_steps": steps,
        "swa_n": len(steps),
    }
    
    torch.save(save_dict, out_path)
    print(f"Saved SWA checkpoint to {out_path}")
    print(f"  Steps averaged: {steps}")
    print(f"  Param count: {sum(p.numel() for p in avg_state.values()):,}")


if __name__ == "__main__":
    main()
