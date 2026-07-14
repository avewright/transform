#!/usr/bin/env bash
# FT3f Elo-safe: init from FT3b champion, gentler soft, spice-enriched mix, Elo gate after.
# Does NOT chain off FT3e (soft_loss↑ Elo↓).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
[[ -f .env ]] && source .env
set +a
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT=outputs/exp191_soft_ft3f_elo_safe
MIX_DIR=outputs/unseen_quality_mix_big
MIX="$MIX_DIR/soft_cache.pt"
MIX_SPICED="$MIX_DIR/soft_cache_spiced.pt"
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
# Champion = last Elo winner, not last soft_loss winner
INIT=outputs/exp191_soft_ft3b_unseen/best.pt
LOG=outputs/overnight_maxelo/ft3f_elo_safe.log
ELO_LOG="$OUT/elo_gauntlet.log"

mkdir -p outputs/overnight_maxelo "$OUT" "$MIX_DIR"

echo "[$(date -Is)] FT3f Elo-safe start" | tee -a "$LOG"
echo "[$(date -Is)] init=$INIT (FT3b champion; NOT FT3e)" | tee -a "$LOG"

if [[ ! -f "$MIX" ]]; then
  echo "[$(date -Is)] ERROR missing base mix $MIX — run pool+mix first" | tee -a "$LOG"
  exit 1
fi

# Spice was 0% because harvest/puzzle keys were in exclude via quality_deep_mix.
# Re-add them as regularizers (OK to reuse tactical spice; avoid reusing cloud soft).
if [[ ! -f "$MIX_SPICED" ]]; then
  echo "[$(date -Is)] enriching mix with harvest/puzzle spice…" | tee -a "$LOG"
  python3 -u - <<'PY' | tee -a "$LOG"
import os, json
from pathlib import Path
import torch
import numpy as np
from scripts.extract_unseen_soft_cache import pack_keys

mix_path = Path("outputs/unseen_quality_mix_big/soft_cache.pt")
out_path = Path("outputs/unseen_quality_mix_big/soft_cache_spiced.pt")
mix = torch.load(mix_path, map_location="cpu", weights_only=False)
print(f"base mix n={mix['board_array'].shape[0]:,} sources={ {int(k):int(v) for k,v in zip(*mix['source'].unique(return_counts=True))} if 'source' in mix else 'n/a'}", flush=True)

spice_paths = [
    ("harvest", 3, Path("outputs/exp190_phase_deep/soft_cache.pt")),
    ("harvest", 3, Path("outputs/exp190_phase_deep_continue/soft_cache.pt")),
    ("harvest", 3, Path("outputs/exp192_edge_soft/soft_cache.pt")),
    ("harvest", 3, Path("outputs/exp095_endgame_deep/soft_cache.pt")),
    ("puzzle", 1, Path("outputs/exp193_puzzle_soft/soft_cache.pt")),
]
CORE = ["board_array","turn","castling","ep_square","move_idx","cp","mate",
        "soft_indices","soft_probs","label_depth","phase"]

# Dedup spice against current mix keys only (not full train history)
mix_keys = set(pack_keys(mix).tolist()) if False else None
# use searchsorted for speed
mk = np.unique(pack_keys(mix))

def unseen_vs_mix(d):
    k = pack_keys(d)
    idx = np.searchsorted(mk, k)
    in_b = idx < len(mk)
    matched = np.zeros(len(k), dtype=bool)
    matched[in_b] = mk[idx[in_b]] == k[in_b]
    keep = np.nonzero(~matched)[0]
    return keep

chunks = [mix]
added = {"harvest": 0, "puzzle": 0}
for kind, sid, path in spice_paths:
    if not path.exists():
        print(f"skip missing {path}", flush=True)
        continue
    d = torch.load(path, map_location="cpu", weights_only=False)
    keep = unseen_vs_mix(d)
    # Even if all overlap mix, still allow up to 50% of spice (force) — tactical reuse OK
    if len(keep) < d["board_array"].shape[0] // 2:
        # force-include random half when exclude wiped them
        rng = np.random.default_rng(42 + sid)
        keep = np.sort(rng.choice(d["board_array"].shape[0], size=max(len(keep), d["board_array"].shape[0] // 2), replace=False))
        print(f"  {path.name}: force-include {len(keep):,} (was mostly in exclude)", flush=True)
    else:
        print(f"  {path.name}: add unseen-vs-mix {len(keep):,}/{d['board_array'].shape[0]:,}", flush=True)
    part = {k: d[k][keep].contiguous() for k in CORE if k in d}
    if "phase" not in part and "board_array" in part:
        # leave without phase if missing — mixer had ensure; skip
        pass
    part["source"] = torch.full((part["board_array"].shape[0],), sid, dtype=torch.int8)
    if "phase" not in part:
        # cheap phase proxy: piece count
        ba = part["board_array"]
        n_pieces = (ba != 0).sum(dim=1)
        phase = torch.zeros(ba.shape[0], dtype=torch.int8)
        phase[n_pieces <= 10] = 2
        phase[(n_pieces > 10) & (n_pieces <= 20)] = 1
        part["phase"] = phase
    if "label_depth" not in part:
        part["label_depth"] = torch.full((part["board_array"].shape[0],), 20, dtype=torch.int16)
    chunks.append(part)
    added[kind] += part["board_array"].shape[0]
    del d

keys = [k for k in list(CORE) + ["source"] if all(k in c for c in chunks)]
out = {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}
n = out["board_array"].shape[0]
perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
out = {k: v[perm].contiguous() for k, v in out.items()}
tmp = out_path.with_suffix(".pt.tmp")
torch.save(out, tmp)
os.replace(tmp, out_path)
src = out["source"]
report = {
    "n": int(n),
    "added": added,
    "source": {
        "deep": float((src == 0).float().mean()),
        "puzzle": float((src == 1).float().mean()),
        "syzygy": float((src == 2).float().mean()),
        "harvest": float((src == 3).float().mean()),
    },
}
(out_path.parent / "mix_spiced_report.json").write_text(json.dumps(report, indent=2))
print(f"wrote {out_path} n={n:,} {report}", flush=True)
PY
else
  echo "[$(date -Is)] using existing $MIX_SPICED" | tee -a "$LOG"
fi

DEEP="$MIX_SPICED"
[[ -f "$DEEP" ]] || DEEP="$MIX"

echo "[$(date -Is)] train init=$INIT deep=$DEEP → $OUT" | tee -a "$LOG"
# Gentler than FT3e: less soft mass, lower LR, shorter, select shallow-leaning top1
python -u experiments/exp191_400m_meta_attention.py \
  --go \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --soft-cache "$SOFT" \
  --deep-soft-cache "$DEEP" \
  --deep-mix-frac 0.40 \
  --soft-frac 0.85 \
  --soft-alpha 0.38 \
  --batch-size 256 \
  --accum-steps 1 \
  --steps 7000 \
  --warmup 300 \
  --muon-lr 0.0012 \
  --adam-lr 6e-5 \
  --value-weight 0.08 \
  --grad-clip 0.5 \
  --hflip-p 0.5 \
  --min-depth 15 \
  --hard-cache "$HARD" \
  --hard-n 1000000 \
  --prefetch-depth 3 \
  --log-interval 25 \
  --save-interval 1000 \
  --eval-interval 1000 \
  --eval-n 1024 \
  --select-metric top1 \
  --compile \
  --polar \
  --cautious-wd \
  2>&1 | tee -a "$OUT/run.log" | tee -a "$LOG"

echo "[$(date -Is)] train done; Elo gauntlet on $OUT/best.pt" | tee -a "$LOG"
CKPT="$OUT/best.pt"
[[ -f "$CKPT" ]] || CKPT="$OUT/latest.pt"
set +e
timeout --signal=INT --kill-after=60 7200 \
  python -u elo_eval_latest.py "$CKPT" exp191_soft_ft3f_elo_safe \
    --movetime 0.05 \
    --games-per-opening-per-color 1 \
    --stop-after-bracket \
    --elos 1450 1600 1750 1900 2050 \
    2>&1 | tee -a "$ELO_LOG" | tee -a "$LOG"
echo "[$(date -Is)] Elo finished; compare vs FT3b ~1832 before promoting" | tee -a "$LOG"
