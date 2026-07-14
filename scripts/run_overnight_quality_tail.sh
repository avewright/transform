#!/bin/bash
# Quality-tail overnight: wait for FT3 + (8M or 3M) Lichess cache, build
# phase-balanced mix with edge/puzzle/syzygy, then FT4 → elo → FT5 → elo.
# Safe to run alongside FT3. Does not touch the FT3 process.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export MOVE_VOCAB_VERSION=compact
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STATUS=outputs/overnight_maxelo/STATUS.txt
LOG=outputs/overnight_maxelo/quality_tail.log
SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
DEEP3=outputs/lichess_evals_soft/soft_cache.pt
DEEP8=outputs/lichess_evals_soft/soft_cache_8m.pt
DEEPQ=outputs/lichess_evals_soft/quality_deep_mix.pt
FT3=outputs/exp191_soft_ft3_lichess
FT4=outputs/exp191_soft_ft4_quality
FT5=outputs/exp191_soft_ft5_quality
STAGES=outputs/overnight_maxelo/stages

mkdir -p outputs/overnight_maxelo "$STAGES" "$FT4" "$FT5"

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

need_file() {
  [[ -f "$1" ]] || { log "FATAL missing $1"; exit 1; }
}

run_elo() {
  local ckpt="$1" prefix="$2" stage="$3"
  shift 3
  stage_done "$stage" && { log "skip $stage"; return 0; }
  [[ -f "$ckpt" ]] || { log "WARN no ckpt $ckpt"; return 0; }
  log "elo $stage ckpt=$ckpt"
  set +e
  timeout --signal=INT --kill-after=60 7200 \
    python -u elo_eval_latest.py "$ckpt" "$prefix" \
      --movetime 0.05 --games-per-opening-per-color 1 --stop-after-bracket \
      --elos "$@" 2>&1 | tee -a "$LOG"
  set -e
  mark_done "$stage"
}

run_ft() {
  local stage="$1" out="$2" init="$3" deep="$4"
  local steps="$5" warmup="$6" muon_lr="$7" adam_lr="$8" deep_mix="$9"
  if stage_done "$stage" && [[ -f "$out/best.pt" ]]; then
    log "skip $stage"; return 0
  fi
  need_file "$init"; need_file "$deep"; need_file "$SOFT"; need_file "$HARD"
  log "$stage init=$init deep=$deep steps=$steps deep_mix=$deep_mix"
  set +e
  python -u experiments/exp191_400m_meta_attention.py \
    --go --init-checkpoint "$init" --output-dir "$out" \
    --soft-cache "$SOFT" --deep-soft-cache "$deep" \
    --deep-mix-frac "$deep_mix" --soft-frac 0.95 --soft-alpha 0.55 \
    --batch-size 256 --accum-steps 1 --steps "$steps" --warmup "$warmup" \
    --muon-lr "$muon_lr" --adam-lr "$adam_lr" --value-weight 0.08 \
    --grad-clip 0.5 --hflip-p 0.5 --min-depth 15 \
    --hard-cache "$HARD" --hard-n 1000000 --prefetch-depth 3 \
    --log-interval 25 --save-interval 500 --eval-interval 1000 \
    --eval-n 2048 --select-metric soft_loss \
    2>&1 | tee -a "$out/run.log" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  [[ $rc -eq 0 ]] || { log "FAIL $stage rc=$rc"; return "$rc"; }
  mark_done "$stage"
}

log "=== quality-tail overnight start pid=$$ ==="

# Wait for 8M builder (or timeout → use 3M)
log "waiting for Lichess volume cache (8M preferred)"
for i in $(seq 1 240); do  # up to ~2h
  if [[ -f "$DEEP8" ]]; then
    log "found $DEEP8"
    mark_done deep8
    break
  fi
  # still building?
  if ! pgrep -f 'build_lichess_evals_soft_cache.py' >/dev/null && [[ ! -f "$DEEP8" ]]; then
    log "builder gone without 8M — using 3M"
    break
  fi
  sleep 30
done

LICH="$DEEP8"
[[ -f "$LICH" ]] || LICH="$DEEP3"
need_file "$LICH"

# Refresh harvest caches for edge diversity
for pair in \
  "exp190_phase_deep_continue:experiments/exp190_phase_deep_harvest.py" \
  "exp192_edge_soft:experiments/exp192_edge_soft_harvest.py"; do
  exp="${pair%%:*}"; script="${pair##*:}"
  if [[ -d "outputs/$exp/dataset" ]] && compgen -G "outputs/$exp/dataset/positions_*.jsonl" >/dev/null; then
    log "refresh $exp soft_cache"
    python -u "$script" --build-cache-only --output-dir "outputs/$exp" \
      2>&1 | tee -a "$LOG" || log "WARN refresh $exp failed"
  fi
done

if ! stage_done deepq || [[ ! -f "$DEEPQ" ]]; then
  log "building quality mix (phase 22/48/30, edge+puzzle, syzygy≤8%)"
  python -u scripts/build_quality_deep_soft_mix.py \
    --lichess "$LICH" --lichess-fallback "$DEEP3" \
    --target 4000000 --syzygy-frac 0.08 --syzygy-dir syzygy \
    --output "$DEEPQ" \
    2>&1 | tee -a outputs/lichess_evals_soft/quality_mix.log | tee -a "$LOG"
  [[ -f "$DEEPQ" ]] && mark_done deepq
fi
need_file "$DEEPQ"

# Wait for FT3 train (+ its wrapper elo)
log "waiting for FT3 to finish"
while pgrep -f '[r]un_ft3_lichess.sh' >/dev/null \
   || pgrep -f 'output-dir outputs/exp191_soft_ft3_lichess' >/dev/null; do
  sleep 30
done
# Also wait if old overnight tries to run FT4 on raw 8M — preempt by claiming GPU after FT3
# If a stale overnight FT4 starts, we skip if our stages exist; kill competing FT4 starters:
if pgrep -f 'run_overnight_maxelo.sh' >/dev/null; then
  log "stopping legacy overnight wrapper (quality-tail owns FT4/FT5)"
  pkill -f 'run_overnight_maxelo.sh' || true
  sleep 2
fi

CKPT3="$(pick_ckpt "$FT3/best.pt" "$FT3/latest.pt")"
need_file "$CKPT3"
if [[ ! -f outputs/elo_eval_exp191_soft_ft3_lichess.json ]]; then
  run_elo "$CKPT3" exp191_soft_ft3_lichess elo_ft3 1450 1600 1750 1900 2050
fi

run_ft ft4 "$FT4" "$CKPT3" "$DEEPQ" 12000 500 0.002 7e-5 0.65
CKPT4="$(pick_ckpt "$FT4/best.pt" "$FT4/latest.pt" "$CKPT3")"
run_elo "$CKPT4" exp191_soft_ft4_quality elo_ft4 1450 1600 1750 1900 2050 2200

hour=$(date -u +%H)
if (( 10#$hour >= 11 )); then
  log "past 11:00 UTC — skip FT5"
  BEST="$CKPT4"
else
  run_ft ft5 "$FT5" "$CKPT4" "$DEEPQ" 8000 300 0.0012 5e-5 0.55
  CKPT5="$(pick_ckpt "$FT5/best.pt" "$FT5/latest.pt" "$CKPT4")"
  run_elo "$CKPT5" exp191_soft_ft5_quality elo_ft5 1450 1600 1750 1900 2050 2200
  BEST="$CKPT5"
fi

echo "$BEST" > outputs/overnight_maxelo/BEST_CKPT.txt
log "=== quality-tail DONE best=$BEST ==="
