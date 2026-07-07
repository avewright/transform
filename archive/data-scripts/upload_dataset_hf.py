"""Upload the generated 200K chess positions dataset to Hugging Face.

Runs on CPU only — safe to run while GPU training is active.
Uploads to avewright/chess-positions-sf-200k as a new dataset.
"""

import json
import os

from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import login

# Auth
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    # Read directly from .env file
    with open("/root/transform/.env") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.strip().split("=", 1)[1]
                break

login(token=HF_TOKEN)

JSONL_PATH = "outputs/exp059_data_scaling/generated_200k.jsonl"
REPO_ID = "avewright/chess-positions-sf-200k"

print(f"Loading {JSONL_PATH}...")
rows = []
with open(JSONL_PATH) as f:
    for line in f:
        row = json.loads(line)
        # Flatten WDL into separate columns for easier use
        wdl = row["wdl"]
        row["wdl_win"] = wdl[0]
        row["wdl_draw"] = wdl[1]
        row["wdl_loss"] = wdl[2]
        # Flatten top_moves into a JSON string column (nested structs are tricky)
        row["top_moves_json"] = json.dumps(row["top_moves"])
        del row["wdl"]
        del row["top_moves"]
        rows.append(row)

print(f"Loaded {len(rows):,} rows")

# Build HF Dataset
ds = Dataset.from_list(rows)
print(f"Dataset: {ds}")
print(f"Features: {ds.features}")

# Split: 95% train, 5% test
ds_split = ds.train_test_split(test_size=0.05, seed=42)
print(f"Train: {len(ds_split['train']):,}, Test: {len(ds_split['test']):,}")

# Push
print(f"Pushing to {REPO_ID}...")
ds_split.push_to_hub(
    REPO_ID,
    private=False,
    commit_message="Upload 200K SF-labeled chess positions (depth 6, diverse sources)",
)
print("Done!")
