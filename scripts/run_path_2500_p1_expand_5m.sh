#!/usr/bin/env bash
# After multipv_lichess_soft_8m.pt is ready, rebuild true 5M mid-balanced mix.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

SRC="${SRC:-outputs/hf_soft/multipv_lichess_soft_8m.pt}"
if [[ ! -f "$SRC" ]]; then
  echo "missing $SRC — wait for HF convert or run hf_soft_to_cache.py"
  exit 1
fi
# Point the builder at the larger deep cache
cp -f "$SRC" outputs/hf_soft/multipv_lichess_soft.pt
SKIP_DOWNLOAD=1 OUT=outputs/hf_soft_mix_5m TARGET=5000000 \
  bash scripts/run_path_2500_p1_soft_mix.sh
echo "5M mix ready at outputs/hf_soft_mix_5m/soft_cache.pt"
