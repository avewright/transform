#!/bin/bash
# Overnight max-Elo chain for rented A40.
# Assumes FT3 (run_ft3_lichess.sh) is already running.
# Parallel: build larger Lichess soft cache on CPU, then FT4 → elo → FT5 → elo.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROOT="$(pwd)"
STATUS=outputs/overnight_maxelo/STATUS.txt
LOG=outputs/overnight_maxelo/pipeline.log
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
DEEP3=outputs/lichess_evals_soft/soft_cache.pt          # 3M (FT3)
DEEP8=outputs/lichess_evals_soft/soft_cache_8m.pt       # raw volume
DEEPQ=outputs/lichess_evals_soft/quality_deep_mix.pt    # phase-balanced + edge + syzygy
FT3=outputs/exp191_soft_ft3_lichess
FT4=outputs/exp191_soft_ft4_quality
FT5=outputs/exp191_soft_ft5_quality
STAGES=outputs/overnight_maxelo/stages

mkdir -p outputs/overnight_maxelo "$STAGES" "$FT4" "$FT5" outputs/lichess_evals_soft

log() {
  local msg="[$(date -Is)] $*"
  echo "$msg" | tee -a "$LOG"
  echo "$msg" > "$STATUS"
}
stage_done() { [[ -f "$STAGES/$1.done" ]]; }
mark_done() { date -Is > "$STAGES/$1.done"; log "STAGE DONE: $1"; }

pick_ckpt() {
  local p
  for p in "$@"; do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done
  return 1
}

run_elo() {
  local ckpt="$1" prefix="$2" stage="$3"
  shift 3
  if stage_done "$stage"; then
    log "skip $stage"
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    log "WARN $stage missing ckpt $ckpt"
    return 0
  fi
  log "elo $stage ckpt=$ckpt elos=$*"
  set +e
  timeout --signal=INT --kill-after=60 7200 \
    python -u elo_eval_latest.py "$ckpt" "$prefix" \
      --movetime 0.05 \
      --games-per-opening-per-color 1 \
      --stop-after-bracket \
      --elos "$@" \
      2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  log "elo $stage rc=$rc"
  mark_done "$stage"
}

run_ft() {
  # run_ft <stage> <out> <init> <deep> <steps> <warmup> <muon_lr> <adam_lr> <deep_mix>
  local stage="$1" out="$2" init="$3" deep="$4"
  local steps="$5" warmup="$6" muon_lr="$7" adam_lr="$8" deep_mix="$9"
  if stage_done "$stage" && [[ -f "$out/best.pt" ]]; then
    log "skip $stage"
    return 0
  fi
  need_file "$init"
  need_file "$deep"
  need_file "$SOFT"
  need_file "$HARD"
  log "$stage init=$init deep=$deep steps=$steps deep_mix=$deep_mix muon_lr=$muon_lr"
  set +e
  python -u experiments/exp191_400m_meta_attention.py \
    --go \
    --init-checkpoint "$init" \
    --output-dir "$out" \
    --soft-cache "$SOFT" \
    --deep-soft-cache "$deep" \
    --deep-mix-frac "$deep_mix" \
    --soft-frac 0.95 \
    --soft-alpha 0.55 \
    --batch-size 256 \
    --accum-steps 1 \
    --steps "$steps" \
    --warmup "$warmup" \
    --muon-lr "$muon_lr" \
    --adam-lr "$adam_lr" \
    --value-weight 0.08 \
    --grad-clip 0.5 \
    --hflip-p 0.5 \
    --min-depth 15 \
    --hard-cache "$HARD" \
    --hard-n 1000000 \
    --prefetch-depth 3 \
    --log-interval 25 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-n 2048 \
    --select-metric soft_loss \
    2>&1 | tee -a "$out/run.log" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "FAIL $stage rc=$rc"
    return "$rc"
  fi
  mark_done "$stage"
}

need_file() {
  if [[ ! -f "$1" ]]; then
    log "FATAL missing $1"
    exit 1
  fi
}

log "=== overnight maxelo start pid=$$ ==="
log "FT3 running? $(pgrep -af 'exp191_soft_ft3_lichess' | head -1 || echo no)"

# ── 1) Build larger Lichess cache on CPU while FT3 uses GPU ─────────────────
if ! stage_done deep8 || [[ ! -f "$DEEP8" ]]; then
  log "building $DEEP8 target=8M (CPU, parallel with FT3)"
  set +e
  python -u scripts/build_lichess_evals_soft_cache.py \
    --min-depth 22 \
    --min-knodes 4000 \
    --target 8000000 \
    --tau 120 \
    --batch-rows 200000 \
    --output "$DEEP8" \
    2>&1 | tee -a outputs/lichess_evals_soft/build_8m.log | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 0 && -f "$DEEP8" ]]; then
    mark_done deep8
  else
    log "WARN deep8 build failed rc=$rc — will fall back to 3M cache"
  fi
else
  log "skip deep8 build"
fi

# Phase-balanced quality mix: Lichess + edge/puzzle/190/095 + capped Syzygy
if ! stage_done deepq || [[ ! -f "$DEEPQ" ]]; then
  LICH_SRC="$DEEP8"
  [[ -f "$LICH_SRC" ]] || LICH_SRC="$DEEP3"
  # refresh harvest caches (cheap) for latest edge/190 rows
  for pair in \
    "exp190_phase_deep_continue:experiments/exp190_phase_deep_harvest.py" \
    "exp192_edge_soft:experiments/exp192_edge_soft_harvest.py"; do
    exp="${pair%%:*}"; script="${pair##*:}"
    if [[ -d "outputs/$exp/dataset" ]] && compgen -G "outputs/$exp/dataset/positions_*.jsonl" >/dev/null; then
      log "refresh soft_cache $exp"
      python -u "$script" --build-cache-only --output-dir "outputs/$exp" \
        2>&1 | tee -a "$LOG" || log "WARN refresh $exp failed"
    fi
  done
  log "building quality mix $DEEPQ from $LICH_SRC"
  set +e
  python -u scripts/build_quality_deep_soft_mix.py \
    --lichess "$LICH_SRC" \
    --lichess-fallback "$DEEP3" \
    --target 4000000 \
    --syzygy-frac 0.08 \
    --syzygy-dir syzygy \
    --output "$DEEPQ" \
    2>&1 | tee -a outputs/lichess_evals_soft/quality_mix.log | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 0 && -f "$DEEPQ" ]]; then
    mark_done deepq
  else
    log "WARN quality mix failed — FT4 will use raw lichess"
  fi
else
  log "skip deepq"
fi

# ── 2) Wait for FT3 train+elo wrapper to finish ─────────────────────────────
log "waiting for FT3 wrapper (run_ft3_lichess.sh) / train proc"
while pgrep -f '[r]un_ft3_lichess.sh' >/dev/null 2>&1 \
   || pgrep -f 'output-dir outputs/exp191_soft_ft3_lichess' >/dev/null 2>&1; do
  if (( SECONDS % 300 < 30 )); then
    tail -1 "$FT3/run.log" 2>/dev/null | tee -a "$LOG" || true
  fi
  sleep 30
done
log "FT3 train process gone"

# If FT3's own elo didn't run/mark, do it
CKPT3="$(pick_ckpt "$FT3/best.pt" "$FT3/latest.pt" || true)"
if [[ -n "${CKPT3:-}" ]]; then
  if [[ ! -f outputs/elo_eval_exp191_soft_ft3_lichess.json ]] && ! stage_done elo_ft3; then
    run_elo "$CKPT3" exp191_soft_ft3_lichess elo_ft3 1450 1600 1750 1900 2050
  else
    mark_done elo_ft3 2>/dev/null || true
    log "FT3 elo already present or staged"
  fi
else
  log "FATAL no FT3 checkpoint"
  exit 1
fi

DEEP_USE="$DEEPQ"
if [[ ! -f "$DEEP_USE" ]]; then
  DEEP_USE="$DEEP8"
fi
if [[ ! -f "$DEEP_USE" ]]; then
  DEEP_USE="$DEEP3"
fi
log "FT4/FT5 deep=$DEEP_USE"

# ── 3) FT4: quality-balanced deep soft, longer ───────────────────────────────
run_ft ft4 "$FT4" "$CKPT3" "$DEEP_USE" 12000 500 0.002 7e-5 0.65
CKPT4="$(pick_ckpt "$FT4/best.pt" "$FT4/latest.pt" "$CKPT3")"
run_elo "$CKPT4" exp191_soft_ft4_quality elo_ft4 1450 1600 1750 1900 2050 2200

# ── 4) FT5: cooler LR polish if we still have night left ─────────────────────
hour=$(date -u +%H)
if (( 10#$hour >= 11 )) && stage_done ft4; then
  log "past 11:00 UTC — skip FT5, keep FT4 as morning best"
else
  run_ft ft5 "$FT5" "$CKPT4" "$DEEP_USE" 8000 300 0.0012 5e-5 0.55
  CKPT5="$(pick_ckpt "$FT5/best.pt" "$FT5/latest.pt" "$CKPT4")"
  run_elo "$CKPT5" exp191_soft_ft5_quality elo_ft5 1450 1600 1750 1900 2050 2200
fi

BEST="$(pick_ckpt "$FT5/best.pt" "$FT4/best.pt" "$FT3/best.pt")"
echo "$BEST" > outputs/overnight_maxelo/BEST_CKPT.txt
log "=== overnight DONE best=$BEST ==="
log "status=$STATUS"
printf '%s\n' "$BEST"
