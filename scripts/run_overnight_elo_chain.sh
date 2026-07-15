#!/usr/bin/env bash
# Overnight Elo-max chain AFTER FT3f (do not interrupt FT3f).
#
# Lessons baked in:
#   - FT3b ~1832 Elo champion; FT3e soft_loss↑ → Elo↓ (~1725). Never chase soft_loss.
#   - Elo-safe recipe: soft_frac~0.85, soft_alpha~0.38, deep_mix~0.40, select top1.
#   - Always init from Elo champion; promote only if Elo estimate improves.
#
# Timeline (~16h from now):
#   FT3f finish+elo (~3h) → FT3g 8k+elo (~5h) → FT3h 6k+elo (~4h) → FT3i 5k+elo (~4h)
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

SOFT=outputs/exp186_sf_multipv/soft_cache.pt
HARD=outputs/data_cache/hard_ballast_d15_n2000000_s42.pt
FT3B=outputs/exp191_soft_ft3b_unseen/best.pt
FT3F_DIR=outputs/exp191_soft_ft3f_elo_safe
FT3G_DIR=outputs/exp191_soft_ft3g_virgin_e
FT3H_DIR=outputs/exp191_soft_ft3h_edge_end
FT3I_DIR=outputs/exp191_soft_ft3i_cool_polish
MIX_G=outputs/unseen_quality_mix_ft3g
MIX_H=outputs/unseen_quality_mix_ft3h
STAGES=outputs/overnight_maxelo/stages_elo_chain
LOG=outputs/overnight_maxelo/elo_chain.log
STATUS=outputs/overnight_maxelo/STATUS.txt
BEST_FILE=outputs/overnight_maxelo/BEST_CKPT.txt
CHAMP_META=outputs/overnight_maxelo/CHAMPION.json

mkdir -p outputs/overnight_maxelo "$STAGES" "$FT3G_DIR" "$FT3H_DIR" "$FT3I_DIR" "$MIX_G" "$MIX_H"

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

elo_estimate() {
  # stdout: estimated_elo integer, or empty
  local json="$1"
  python3 - <<PY
import json, sys
from pathlib import Path
p = Path("$json")
if not p.exists():
    sys.exit(0)
d = json.loads(p.read_text())
est = (d.get("estimate") or {}).get("estimated_elo")
if est is None:
    sys.exit(0)
print(int(est))
PY
}

write_champion() {
  local ckpt="$1" elo="$2" tag="$3"
  printf '%s\n' "$ckpt" > "$BEST_FILE"
  python3 - <<PY
import json
from pathlib import Path
Path("$CHAMP_META").write_text(json.dumps({
    "ckpt": "$ckpt",
    "estimated_elo": int("$elo") if str("$elo").isdigit() else None,
    "tag": "$tag",
    "updated_at": __import__("datetime").datetime.now().isoformat(),
}, indent=2))
PY
  log "CHAMPION tag=$tag elo=$elo ckpt=$ckpt"
}

run_elo() {
  local ckpt="$1" prefix="$2" stage="$3"
  shift 3
  if stage_done "$stage"; then
    log "skip $stage"
    return 0
  fi
  need_file "$ckpt"
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

# Promote only if new Elo estimate >= current champion Elo (ties keep newer).
gate_promote() {
  local new_ckpt="$1" new_json="$2" tag="$3"
  local new_elo champ_elo
  new_elo="$(elo_estimate "$new_json" || true)"
  champ_elo="$(python3 - <<'PY'
import json
from pathlib import Path
p=Path("outputs/overnight_maxelo/CHAMPION.json")
if p.exists():
  d=json.loads(p.read_text())
  e=d.get("estimated_elo")
  print(e if e is not None else "")
PY
)"
  if [[ -z "${new_elo:-}" ]]; then
    log "GATE $tag: no Elo estimate — keep prior champion"
    return 1
  fi
  if [[ -z "${champ_elo:-}" ]] || (( new_elo >= champ_elo )); then
    write_champion "$new_ckpt" "$new_elo" "$tag"
    return 0
  fi
  log "GATE $tag: reject elo=$new_elo < champ=$champ_elo — keep prior"
  return 1
}

refresh_harvests() {
  log "refresh harvest soft_caches (CPU)"
  for pair in \
    "exp190_phase_deep_continue:experiments/exp190_phase_deep_harvest.py" \
    "exp192_edge_soft:experiments/exp192_edge_soft_harvest.py"; do
    local exp="${pair%%:*}" script="${pair##*:}"
    if [[ -d "outputs/$exp/dataset" ]] && compgen -G "outputs/$exp/dataset/positions_*.jsonl" >/dev/null; then
      log "  build-cache-only $exp"
      python -u "$script" --build-cache-only --output-dir "outputs/$exp" \
        2>&1 | tee -a "$LOG" || log "WARN refresh $exp failed"
    fi
  done
}

build_mix() {
  # build_mix <stage> <out_dir> <pool> <target> <deep> <puzzle> <harvest> <syzygy>
  local stage="$1" out="$2" pool="$3" target="$4"
  local deep_f="$5" puz_f="$6" harv_f="$7" syz_f="$8"
  if stage_done "$stage" && [[ -f "$out/soft_cache.pt" ]]; then
    log "skip $stage (mix exists)"
    return 0
  fi
  need_file "$pool"
  log "build mix $stage pool=$pool target=$target → $out"
  python3 -u scripts/build_unseen_quality_mix.py \
    --pool "$pool" \
    --exclude \
      outputs/lichess_evals_soft/soft_cache.pt \
      outputs/lichess_evals_soft/soft_cache_8m.pt \
      outputs/lichess_evals_soft/quality_deep_mix.pt \
      outputs/lichess_evals_soft/soft_cache_unseen_ft3.pt \
      outputs/lichess_evals_soft/soft_cache_shards2to19_12m.pt \
      outputs/lichess_evals_soft/soft_cache_ft3style_remainder.pt \
      outputs/unseen_quality_mix/soft_cache.pt \
      outputs/unseen_quality_mix_big/soft_cache.pt \
      outputs/unseen_quality_mix_big/soft_cache_spiced.pt \
      outputs/unseen_quality_mix_ft3g/soft_cache.pt \
    --output-dir "$out" \
    --target "$target" \
    --deep-frac "$deep_f" \
    --puzzle-frac "$puz_f" \
    --harvest-frac "$harv_f" \
    --syzygy-frac "$syz_f" \
    2>&1 | tee -a "$out/build.log" | tee -a "$LOG"
  need_file "$out/soft_cache.pt"
  mark_done "$stage"
}

run_ft_elo_safe() {
  # run_ft_elo_safe stage out init deep steps warmup muon adam soft_frac soft_alpha deep_mix
  local stage="$1" out="$2" init="$3" deep="$4"
  local steps="$5" warmup="$6" muon_lr="$7" adam_lr="$8"
  local soft_frac="$9" soft_alpha="${10}" deep_mix="${11}"
  if stage_done "$stage" && [[ -f "$out/best.pt" ]]; then
    log "skip $stage"
    return 0
  fi
  need_file "$init"; need_file "$deep"; need_file "$SOFT"; need_file "$HARD"
  log "$stage init=$init deep=$deep steps=$steps soft_frac=$soft_frac alpha=$soft_alpha deep_mix=$deep_mix muon=$muon_lr"
  set +e
  python -u experiments/exp191_400m_meta_attention.py \
    --go \
    --init-checkpoint "$init" \
    --output-dir "$out" \
    --soft-cache "$SOFT" \
    --deep-soft-cache "$deep" \
    --deep-mix-frac "$deep_mix" \
    --soft-frac "$soft_frac" \
    --soft-alpha "$soft_alpha" \
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
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-n 1024 \
    --select-metric top1 \
    --compile \
    --polar \
    --cautious-wd \
    2>&1 | tee -a "$out/run.log" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    log "FAIL $stage rc=$rc"
    return "$rc"
  fi
  # free ~3.4G if best exists
  if [[ -f "$out/best.pt" && -f "$out/latest.pt" ]]; then
    rm -f "$out/latest.pt" || true
  fi
  mark_done "$stage"
}

log "=== overnight Elo chain start pid=$$ ==="
need_file "$SOFT"
need_file "$HARD"
need_file "$FT3B"

# ── 0) CPU prep while FT3f still owns GPU ───────────────────────────────────
if ! stage_done prep_harvests; then
  refresh_harvests
  mark_done prep_harvests
fi

# Mix G: deepest virgin (e) + balanced spice. Build on CPU now.
build_mix mix_g "$MIX_G" \
  outputs/lichess_evals_soft/soft_cache_virgin_e_12m.pt \
  6000000 0.72 0.10 0.12 0.06 || log "WARN mix_g failed — will retry later"

# ── 1) Wait for FT3f wrapper (train + its Elo) ──────────────────────────────
log "waiting for FT3f (run_ft3f_elo_safe.sh / train) to finish"
while pgrep -f '[r]un_ft3f_elo_safe.sh' >/dev/null 2>&1 \
   || pgrep -f 'output-dir outputs/exp191_soft_ft3f_elo_safe' >/dev/null 2>&1; do
  if (( SECONDS % 300 < 30 )); then
    tail -1 "$FT3F_DIR/run.log" 2>/dev/null | tee -a "$LOG" || true
  fi
  sleep 30
done
log "FT3f process gone"

# If FT3f train finished but Elo didn't (crash), run Elo once
FT3F_CKPT="$(pick_ckpt "$FT3F_DIR/best.pt" "$FT3F_DIR/latest.pt" || true)"
if [[ -n "${FT3F_CKPT:-}" ]] && [[ ! -f outputs/elo_eval_exp191_soft_ft3f_elo_safe.json ]]; then
  run_elo "$FT3F_CKPT" exp191_soft_ft3f_elo_safe elo_ft3f 1450 1600 1750 1900 2050
else
  mark_done elo_ft3f 2>/dev/null || true
fi

# ── 2) Seed champion: FT3b vs FT3f ──────────────────────────────────────────
FT3B_ELO="$(elo_estimate outputs/elo_eval_exp191_soft_ft3b_unseen.json || true)"
FT3B_ELO="${FT3B_ELO:-1832}"
write_champion "$FT3B" "$FT3B_ELO" ft3b_seed
if [[ -n "${FT3F_CKPT:-}" ]]; then
  gate_promote "$FT3F_CKPT" outputs/elo_eval_exp191_soft_ft3f_elo_safe.json ft3f || true
fi
CHAMP="$(cat "$BEST_FILE")"
need_file "$CHAMP"
log "post-FT3f champion=$CHAMP"

# Ensure mix_g ready (retry if CPU prep failed)
if [[ ! -f "$MIX_G/soft_cache.pt" ]]; then
  refresh_harvests
  build_mix mix_g "$MIX_G" \
    outputs/lichess_evals_soft/soft_cache_virgin_e_12m.pt \
    6000000 0.72 0.10 0.12 0.06
fi
need_file "$MIX_G/soft_cache.pt"

# ── 3) FT3g: virgin-e quality continue (Elo-safe) ───────────────────────────
run_ft_elo_safe ft3g "$FT3G_DIR" "$CHAMP" "$MIX_G/soft_cache.pt" \
  8000 250 0.0010 5e-5 0.85 0.38 0.40
FT3G_CKPT="$(pick_ckpt "$FT3G_DIR/best.pt" "$CHAMP")"
run_elo "$FT3G_CKPT" exp191_soft_ft3g_virgin_e elo_ft3g 1450 1600 1750 1900 2050 2200
gate_promote "$FT3G_CKPT" outputs/elo_eval_exp191_soft_ft3g_virgin_e.json ft3g || true
CHAMP="$(cat "$BEST_FILE")"

# ── 4) FT3h: edge/endgame-lean mix from virgin_b ─────────────────────────────
refresh_harvests
build_mix mix_h "$MIX_H" \
  outputs/lichess_evals_soft/soft_cache_virgin_b_12m.pt \
  5000000 0.62 0.10 0.18 0.10
need_file "$MIX_H/soft_cache.pt"

run_ft_elo_safe ft3h "$FT3H_DIR" "$CHAMP" "$MIX_H/soft_cache.pt" \
  6000 200 0.0009 4.5e-5 0.85 0.38 0.42
FT3H_CKPT="$(pick_ckpt "$FT3H_DIR/best.pt" "$CHAMP")"
run_elo "$FT3H_CKPT" exp191_soft_ft3h_edge_end elo_ft3h 1450 1600 1750 1900 2050 2200
gate_promote "$FT3H_CKPT" outputs/elo_eval_exp191_soft_ft3h_edge_end.json ft3h || true
CHAMP="$(cat "$BEST_FILE")"

# ── 5) FT3i: cool polish on best mix so far (prefer H if exists else G) ──────
DEEP_I="$MIX_H/soft_cache.pt"
[[ -f "$DEEP_I" ]] || DEEP_I="$MIX_G/soft_cache.pt"
run_ft_elo_safe ft3i "$FT3I_DIR" "$CHAMP" "$DEEP_I" \
  5000 150 0.0006 3e-5 0.80 0.35 0.35
FT3I_CKPT="$(pick_ckpt "$FT3I_DIR/best.pt" "$CHAMP")"
run_elo "$FT3I_CKPT" exp191_soft_ft3i_cool_polish elo_ft3i 1450 1600 1750 1900 2050 2200
gate_promote "$FT3I_CKPT" outputs/elo_eval_exp191_soft_ft3i_cool_polish.json ft3i || true

BEST="$(cat "$BEST_FILE")"
log "=== overnight Elo chain DONE best=$BEST ==="
log "champion meta: $(cat "$CHAMP_META" 2>/dev/null || echo none)"
printf '%s\n' "$BEST"
