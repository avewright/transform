#!/usr/bin/env bash
# Phase 2 (M5): LoRA soft FT on deep HF mix — sanity / small Elo tick.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_ENABLE_MPS_FALLBACK=1

SOFT="${SOFT_CACHE:-outputs/hf_soft_mix/soft_cache.pt}"
export CKPT="${CKPT:-outputs/hf_437m/best_model.pt}"
export OUT="${OUT:-outputs/lora_soft_hfmix}"
STEPS="${STEPS:-3000}"
export RANK="${RANK:-16}"
mkdir -p "$OUT" logs

if [[ ! -f "$SOFT" ]]; then
  echo "missing $SOFT — rebuilding from local HF pts"
  SKIP_DOWNLOAD=1 OUT=outputs/hf_soft_mix TARGET=1500000 DEEP_MAX=2500000 \
    bash scripts/run_path_2500_p1_soft_mix.sh
  SOFT=outputs/hf_soft_mix/soft_cache.pt
fi

python -u experiments/lora_soft.py \
  --ckpt "$CKPT" \
  --soft-cache "$SOFT" \
  --steps "$STEPS" \
  --batch "${BATCH:-16}" \
  --rank "$RANK" \
  --max-train "${MAX_TRAIN:-500000}" \
  --out "$OUT" \
  2>&1 | tee "$OUT/run.log"

python -u - <<'PY'
import os, sys, torch
from pathlib import Path
sys.path.insert(0, ".")
from chess_transformer_factory import build_model, ChessTransformerConfig
from experiments.lora_tac import apply_lora, LoRALinear

ckpt_path = Path(os.environ["CKPT"])
out = Path(os.environ["OUT"])
lora_pt = out / "lora.pt"
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ck.get("config")
cfg = ChessTransformerConfig(**cfg) if not isinstance(cfg, ChessTransformerConfig) else cfg
model = build_model(cfg)
sd = {k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()}
model.load_state_dict(sd, strict=False)
apply_lora(model, rank=int(os.environ.get("RANK", "16")), alpha=32.0)
lora = torch.load(lora_pt, map_location="cpu", weights_only=False)
model.load_state_dict(lora["lora_state"], strict=False)

baked = 0
for name, module in list(model.named_modules()):
    if not isinstance(module, LoRALinear):
        continue
    with torch.no_grad():
        delta = (module.lora_B @ module.lora_A) * module.scaling
        module.original.weight.add_(delta)
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module.original)
    baked += 1

merged = out / "merged_model.pt"
cfg_dict = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
torch.save({
    "model_state_dict": model.state_dict(),
    "config": cfg_dict,
    "vocab_version": ck.get("vocab_version", "compact"),
}, merged)
print(f"baked {baked} LoRA modules → {merged}")
PY

if [[ -f "$OUT/merged_model.pt" ]]; then
  python -u elo_eval_latest.py "$OUT/merged_model.pt" lora_soft_hfmix \
    --elos 1600 1750 1900 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    2>&1 | tee -a "$OUT/elo_probe.log" || true
fi
