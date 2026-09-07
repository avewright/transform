#!/bin/bash
# Finetune latest 100M HF checkpoint on harvested disagreements + replay.
# Torch MPS (Apple GPU). There is no MLX port of squares64.
#
# Mix: 75% harvest (bonus) / 25% hf_elo_mix (representative replay).
# After a checkpoint exists, compare playing strength with:
#   MOVE_VOCAB_VERSION=compact .venv/bin/python -u scripts/match_two_ckpts.py --go \
#     --a outputs/hf100m_lapse_ft/init_from_hf.pt \
#     --b outputs/hf100m_lapse_ft/latest.pt \
#     --out outputs/hf100m_lapse_ft/match_init_vs_latest.json
set -euo pipefail
cd "$(dirname "$0")/.."
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

INBOX="${INBOX:-outputs/hf100m_bulk/20260905_175038_bulk/inbox}"
HARVEST="${HARVEST:-outputs/hf100m_lapse_ft/soft_cache.pt}"
REPLAY="${REPLAY:-outputs/hf_elo_mix/soft_cache.pt}"
OUT="${OUT:-outputs/hf100m_lapse_ft}"
STEPS="${STEPS:-8000}"
MINUTES="${MINUTES:-720}"
ADAM_LR="${ADAM_LR:-3e-5}"
BONUS_MIX="${BONUS_MIX:-0.75}"

mkdir -p "$OUT"

CKPT=$(.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "avewright/chess-transformer-100m-squares64"
revision = HfApi().model_info(repo).sha
print(hf_hub_download(repo, "latest.pt", revision=revision))
PY
)
echo "Checkpoint: $CKPT"
INIT="$OUT/init_from_hf.pt"

.venv/bin/python - <<PY
from pathlib import Path
import torch
from scripts.autoresearch_8gb.pipeline import concat_soft_tables

inbox = Path("$INBOX")
out = Path("$HARVEST")
force = "${FORCE_REBUILD:-}"
if out.exists() and not force:
    d = torch.load(out, map_location="cpu", weights_only=False)
    print(f"reuse {out} n={int(d['move_idx'].shape[0]):,}", flush=True)
else:
    chunks = []
    for sh in sorted(inbox.glob("shard_*/soft_cache.pt")):
        d = torch.load(sh, map_location="cpu", weights_only=False)
        chunks.append(d)
        print(f"  {sh.parent.name} n={int(d['move_idx'].shape[0]):,}", flush=True)
    if not chunks:
        raise SystemExit(f"no shards in {inbox}")
    merged = concat_soft_tables(chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    print(f"wrote {out} n={int(merged['move_idx'].shape[0]):,}", flush=True)

replay = Path("$REPLAY")
if not replay.exists():
    raise SystemExit(f"missing representative replay cache: {replay}")

# HF latest.pt is at ~61k steps. Trainer treats max-steps as an absolute
# counter, so resume-as-is with --max-steps 8000 exits immediately.
src = Path("$CKPT")
dst = Path("$INIT")
if not dst.exists():
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    for k in ("optimizer_state_dict", "rng"):
        ckpt.pop(k, None)
    ckpt["steps"] = 0
    ckpt["global_step"] = 0
    torch.save(ckpt, dst)
    print(f"warm-start init {dst} from {src}", flush=True)
else:
    print(f"reuse warm-start {dst}", flush=True)
PY

echo "Training $STEPS steps lr=$ADAM_LR bonus=$BONUS_MIX -> $OUT"
exec .venv/bin/python -u experiments/exp201_recurrent_64.py --go --skip-mix \
  --resume "$INIT" \
  --soft-cache "$REPLAY" \
  --deep-cache outputs/hf_elo_mix/deep_cache.pt \
  --bonus-cache "$HARVEST" \
  --bonus-mix-frac "$BONUS_MIX" \
  --output-dir "$OUT" \
  --max-steps "$STEPS" \
  --train-minutes "$MINUTES" \
  --deep-mix-frac 0 \
  --batch-size 48 \
  --optimizer adamw \
  --adam-lr "$ADAM_LR" \
  --val-eval-n 2000 \
  --val-every 250
