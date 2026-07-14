#!/usr/bin/env bash
# DEPRECATED aggressive FT3f path (init FT3e + soft_frac=0.95).
# Redirects to Elo-safe recipe after FT3e soft_loss↑/Elo↓ regression.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=outputs/overnight_maxelo/ft3f_chain.log
mkdir -p outputs/overnight_maxelo
echo "[$(date -Is)] run_ft3f_when_ready → redirect to run_ft3f_elo_safe.sh" | tee -a "$LOG"
exec bash scripts/run_ft3f_elo_safe.sh
