"""Build a larger eval set (20K positions) by extending current 5K eval with 15K from shard_00010.

The current 5K eval set produces ~±1% top-1 noise, making checkpoint ranking unreliable.
20K should reduce noise to ~±0.5%, giving clearer signal on whether training is improving.

Shard 10 is the smallest (110K positions) — taking 15K from it has minimal training impact.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"

# Load current eval
eval_data = torch.load(SHARD_DIR / "eval.pt", map_location="cpu", weights_only=False)
print(f"Current eval: {eval_data['board_array'].shape[0]} positions")

# Load shard 10
shard = torch.load(SHARD_DIR / "shard_00010.pt", map_location="cpu", weights_only=False)
print(f"Shard 10: {shard['board_array'].shape[0]} positions")

# Take first 15K from shard 10
N_EXTRA = 15_000

# Build merged eval
merged = {}
for key in ["board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate"]:
    merged[key] = torch.cat([eval_data[key], shard[key][:N_EXTRA]], dim=0)

# Handle 'fen' list if present
if "fen" in eval_data and isinstance(eval_data["fen"], list):
    # Shard doesn't have FENs, so we drop fen from merged (eval code reconstructs from board_array)
    pass

# Handle 'depth' if shard has it
if "depth" in shard:
    if "depth" in eval_data:
        merged["depth"] = torch.cat([eval_data["depth"], shard["depth"][:N_EXTRA]], dim=0)
    else:
        merged["depth"] = torch.cat([
            torch.zeros(eval_data["board_array"].shape[0], dtype=shard["depth"].dtype),
            shard["depth"][:N_EXTRA]
        ], dim=0)

total = merged["board_array"].shape[0]
print(f"Merged eval: {total} positions")

# Save
out_path = SHARD_DIR / "eval_20k.pt"
torch.save(merged, out_path)
print(f"Saved to {out_path}")
print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

# Also trim shard 10 to exclude the positions we took
remaining = {k: v[N_EXTRA:] for k, v in shard.items()}
out_shard = SHARD_DIR / "shard_00010_trimmed.pt"
torch.save(remaining, out_shard)
print(f"Trimmed shard 10: {remaining['board_array'].shape[0]} positions -> {out_shard}")
