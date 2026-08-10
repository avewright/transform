#!/bin/bash
# LEGACY — soft_loss crowning. Do not use for the max-Elo push.
# Prefer: python -m harness.loop --name ... --soft-cache ...
# Resumable max-Elo pipeline. Safe to re-run: completed stages are skipped via
# marker files under outputs/exp191_soft_merged/stages/.
#
# Stages: merge_v1 → elo_baseline → ft1 → elo_ft1 → wait095 → merge_v2 → ft2 → elo_ft2
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-/root/transform/stockfish/stockfish-latest}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRETRAIN=outputs/exp191_400m_meta_attention
FT1=outputs/exp191_soft_ft
FT2=outputs/exp191_soft_ft2
MERGED_DIR=outputs/exp191_soft_merged
STAGES="$MERGED_DIR/stages"
DEEP1="$MERGED_DIR/deep_v1.pt"
DEEP2="$MERGED_DIR/deep_v2.pt"
SOFT="${SOFT_CACHE:-outputs/exp186_sf_multipv/soft_cache.pt}"
HARD_CACHE="${HARD_CACHE:-outputs/data_cache/hard_ballast_d15_n2000000_s42.pt}"
CKPT_PRE="$PRETRAIN/best.pt"
STATUS="$MERGED_DIR/STATUS.txt"
PIPE_LOG="$MERGED_DIR/pipeline.log"
ELO_TIMEOUT_SEC="${ELO_TIMEOUT_SEC:-5400}"   # 90 min cap per elo pass
FT1_STEPS="${FT1_STEPS:-7000}"
FT2_STEPS="${FT2_STEPS:-4000}"

mkdir -p "$FT1" "$FT2" "$MERGED_DIR" "$STAGES" "$PRETRAIN"

log() {
  local msg="[$(date -Is)] $*"
  echo "$msg" | tee -a "$PIPE_LOG"
  echo "$msg" > "$STATUS"
}

stage_done() { [[ -f "$STAGES/$1.done" ]]; }
mark_done() { date -Is > "$STAGES/$1.done"; log "STAGE DONE: $1"; }
mark_fail() { date -Is > "$STAGES/$1.fail"; log "STAGE FAIL: $1 — $*"; }

need_file() {
  if [[ ! -f "$1" ]]; then
    log "FATAL missing file: $1"
    exit 1
  fi
}

run_elo() {
  # usage: run_elo <ckpt> <out_prefix> <stage_name> [extra elo levels...]
  local ckpt="$1" prefix="$2" stage="$3"
  shift 3
  local elos=("$@")
  if ((${#elos[@]} == 0)); then
    elos=(1450 1600 1750 1900)
  fi
  if stage_done "$stage"; then
    log "skip $stage (already done)"
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    mark_fail "$stage" "missing ckpt $ckpt"
    return 0
  fi
  log "elo $stage ckpt=$ckpt timeout=${ELO_TIMEOUT_SEC}s elos=${elos[*]}"
  set +e
  timeout --signal=INT --kill-after=60 "$ELO_TIMEOUT_SEC" \
    python -u elo_eval_latest.py "$ckpt" "$prefix" \
      --movetime 0.05 \
      --games-per-opening-per-color 1 \
      --stop-after-bracket \
      --elos "${elos[@]}" \
      2>&1 | tee -a "$PIPE_LOG"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    mark_done "$stage"
  elif [[ $rc -eq 124 ]] || [[ $rc -eq 137 ]]; then
    log "WARN: $stage timed out (rc=$rc) — continuing"
    mark_done "$stage"  # don't retry forever
  else
    log "WARN: $stage failed rc=$rc — continuing"
    mark_done "$stage"
  fi
  return 0
}

refresh_cache() {
  local exp="$1" script="$2"
  local ds="outputs/$exp/dataset"
  if [[ -d "$ds" ]] && compgen -G "$ds/positions_*.jsonl" >/dev/null; then
    log "rebuild soft_cache for $exp"
    python -u "$script" --build-cache-only --output-dir "outputs/$exp" 2>&1 | tee -a "$PIPE_LOG" || \
      log "WARN: rebuild $exp failed"
  fi
}

pick_ckpt() {
  # prints best available among args
  local p
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

need_file "$CKPT_PRE"
need_file "$SOFT"
need_file "$HARD_CACHE"

log "=== pipeline start (resumable) pid=$$ ==="
log "pretrain=$CKPT_PRE soft=$SOFT hard=$HARD_CACHE deep1=$DEEP1"

# ── merge_v1 ────────────────────────────────────────────────────────────────
if ! stage_done merge_v1 || [[ ! -f "$DEEP1" ]]; then
  refresh_cache exp190_phase_deep_continue experiments/exp190_phase_deep_harvest.py
  refresh_cache exp192_edge_soft experiments/exp192_edge_soft_harvest.py
  # 193 complete — only rebuild if missing
  if [[ ! -f outputs/exp193_puzzle_soft/soft_cache.pt ]]; then
    refresh_cache exp193_puzzle_soft experiments/exp193_puzzle_soft_harvest.py
  fi
  need_file outputs/exp190_phase_deep/soft_cache.pt
  need_file outputs/exp190_phase_deep_continue/soft_cache.pt
  need_file outputs/exp193_puzzle_soft/soft_cache.pt
  need_file outputs/exp192_edge_soft/soft_cache.pt
  log "merge deep_v1"
  python -u scripts/merge_soft_caches.py \
    outputs/exp190_phase_deep/soft_cache.pt \
    outputs/exp190_phase_deep_continue/soft_cache.pt \
    outputs/exp193_puzzle_soft/soft_cache.pt \
    outputs/exp192_edge_soft/soft_cache.pt \
    -o "$DEEP1" 2>&1 | tee -a "$PIPE_LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]] || [[ ! -f "$DEEP1" ]]; then
    log "FATAL merge_v1 failed rc=$rc"
    exit 1
  fi
  mark_done merge_v1
else
  log "skip merge_v1 (exists)"
fi

# ── elo baseline ────────────────────────────────────────────────────────────
run_elo "$CKPT_PRE" "exp191_pretrain_baseline" elo_baseline 1450 1600 1750 1900

# ── FT1 ─────────────────────────────────────────────────────────────────────
if ! stage_done ft1 || [[ ! -f "$FT1/best.pt" && ! -f "$FT1/latest.pt" ]]; then
  need_file "$DEEP1"
  log "soft FT1 steps=$FT1_STEPS from $CKPT_PRE"
  python -u experiments/exp191_400m_meta_attention.py \
    --go \
    --init-checkpoint "$CKPT_PRE" \
    --output-dir "$FT1" \
    --soft-cache "$SOFT" \
    --deep-soft-cache "$DEEP1" \
    --deep-mix-frac 0.30 \
    --soft-frac 0.95 \
    --soft-alpha 0.55 \
    --batch-size 256 \
    --accum-steps 1 \
    --steps "$FT1_STEPS" \
    --warmup 300 \
    --muon-lr 0.005 \
    --adam-lr 1.5e-4 \
    --value-weight 0.08 \
    --grad-clip 0.5 \
    --hflip-p 0.5 \
    --min-depth 15 \
    --hard-cache "$HARD_CACHE" \
    --hard-n 1000000 \
    --prefetch-depth 3 \
    --log-interval 25 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-n 2048 \
    --select-metric soft_loss \
    2>&1 | tee -a "$FT1/run.log" | tee -a "$PIPE_LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    mark_fail ft1 "train rc=$rc"
    log "FT1 failed — will continue with pretrain weights where needed"
  else
    mark_done ft1
  fi
else
  log "skip ft1 (already done)"
fi

CKPT_FT1="$(pick_ckpt "$FT1/best.pt" "$FT1/latest.pt" "$CKPT_PRE")"
log "CKPT_FT1=$CKPT_FT1"

run_elo "$CKPT_FT1" "exp191_soft_ft1" elo_ft1 1450 1600 1750 1900 2050

# ── wait for 095 (max ~3h), then merge_v2 + FT2 ─────────────────────────────
if ! stage_done wait095; then
  log "waiting for exp095 (max 180 min)"
  for i in $(seq 1 180); do
    if ! pgrep -f '[e]xp095_endgame_harvest.py' >/dev/null 2>&1; then
      log "exp095 process gone after ${i}m"
      break
    fi
    # progress hit target?
    if [[ -f outputs/exp095_endgame_deep/run.log ]] && \
       grep -qE 'written=200000/200000' outputs/exp095_endgame_deep/run.log 2>/dev/null; then
      log "exp095 hit 200k — waiting for clean exit"
      sleep 45
      if ! pgrep -f '[e]xp095_endgame_harvest.py' >/dev/null 2>&1; then
        break
      fi
    fi
    if (( i % 10 == 0 )); then
      tail -1 outputs/exp095_endgame_deep/run.log 2>/dev/null | tee -a "$PIPE_LOG" || true
    fi
    sleep 60
  done
  mark_done wait095
fi

if ! stage_done merge_v2 || [[ ! -f "$DEEP2" ]]; then
  if [[ -d outputs/exp095_endgame_deep/dataset ]] && \
     compgen -G "outputs/exp095_endgame_deep/dataset/positions_*.jsonl" >/dev/null; then
    log "build exp095 soft_cache"
    python - <<'PY' 2>&1 | tee -a outputs/exp191_soft_merged/pipeline.log || true
import os, sys
from pathlib import Path
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
sys.path.insert(0, ".")
from experiments.exp190_phase_deep_harvest import build_cache
out = Path("outputs/exp095_endgame_deep/soft_cache.pt")
n = build_cache(Path("outputs/exp095_endgame_deep/dataset"), out)
print(f"095 soft_cache rows={n} → {out}", flush=True)
PY
  fi
  refresh_cache exp190_phase_deep_continue experiments/exp190_phase_deep_harvest.py
  refresh_cache exp192_edge_soft experiments/exp192_edge_soft_harvest.py

  DEEP_INPUTS=(
    outputs/exp190_phase_deep/soft_cache.pt
    outputs/exp190_phase_deep_continue/soft_cache.pt
    outputs/exp193_puzzle_soft/soft_cache.pt
    outputs/exp192_edge_soft/soft_cache.pt
  )
  if [[ -f outputs/exp095_endgame_deep/soft_cache.pt ]]; then
    DEEP_INPUTS+=(outputs/exp095_endgame_deep/soft_cache.pt)
  fi
  log "merge deep_v2: ${DEEP_INPUTS[*]}"
  python -u scripts/merge_soft_caches.py "${DEEP_INPUTS[@]}" -o "$DEEP2" 2>&1 | tee -a "$PIPE_LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]] || [[ ! -f "$DEEP2" ]]; then
    log "WARN merge_v2 failed — reusing deep_v1"
    cp -f "$DEEP1" "$DEEP2"
  fi
  mark_done merge_v2
else
  log "skip merge_v2"
fi

INIT_FT2="$(pick_ckpt "$FT1/best.pt" "$FT1/latest.pt" "$CKPT_PRE")"

if ! stage_done ft2 || [[ ! -f "$FT2/best.pt" && ! -f "$FT2/latest.pt" ]]; then
  need_file "$DEEP2"
  log "soft FT2 steps=$FT2_STEPS from $INIT_FT2"
  python -u experiments/exp191_400m_meta_attention.py \
    --go \
    --init-checkpoint "$INIT_FT2" \
    --output-dir "$FT2" \
    --soft-cache "$SOFT" \
    --deep-soft-cache "$DEEP2" \
    --deep-mix-frac 0.35 \
    --soft-frac 0.95 \
    --soft-alpha 0.55 \
    --batch-size 256 \
    --accum-steps 1 \
    --steps "$FT2_STEPS" \
    --warmup 200 \
    --muon-lr 0.003 \
    --adam-lr 1e-4 \
    --value-weight 0.08 \
    --grad-clip 0.5 \
    --hflip-p 0.5 \
    --min-depth 15 \
    --hard-cache "$HARD_CACHE" \
    --hard-n 1000000 \
    --prefetch-depth 3 \
    --log-interval 25 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-n 2048 \
    --select-metric soft_loss \
    2>&1 | tee -a "$FT2/run.log" | tee -a "$PIPE_LOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    mark_fail ft2 "train rc=$rc"
  else
    mark_done ft2
  fi
else
  log "skip ft2"
fi

CKPT_FT2="$(pick_ckpt "$FT2/best.pt" "$FT2/latest.pt" "$INIT_FT2")"
log "CKPT_FT2=$CKPT_FT2"

run_elo "$CKPT_FT2" "exp191_soft_ft2" elo_ft2 1450 1600 1750 1900 2050

mark_done all
log "=== pipeline DONE ==="
log "pretrain=$CKPT_PRE"
log "ft1=$CKPT_FT1"
log "ft2=$CKPT_FT2"
log "status file: $STATUS"
log "stages: $STAGES"
printf '%s\n' "$CKPT_FT2" > "$MERGED_DIR/BEST_CKPT.txt"
