#!/usr/bin/env bash
# Path-to-2500 orchestrator. Stages can be run independently.
#
#   bash scripts/run_path_2500.sh p0          # baselines
#   bash scripts/run_path_2500.sh p1          # soft mix
#   bash scripts/run_path_2500.sh p2-lora     # M5 LoRA soft
#   bash scripts/run_path_2500.sh p2-ft       # A40 full soft FT
#   bash scripts/run_path_2500.sh p3          # tournament MCTS Elo
#   bash scripts/run_path_2500.sh p4          # KL expert-iter (after ~1900 policy)
#   bash scripts/run_path_2500.sh all-m5      # p0 → p1 → p2-lora → p3 (M5 path)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-all-m5}"

case "$STAGE" in
  p0) bash scripts/run_path_2500_p0_baseline.sh ;;
  p1) bash scripts/run_path_2500_p1_soft_mix.sh ;;
  p2-lora) bash scripts/run_path_2500_p2_lora_soft.sh ;;
  p2-ft) bash scripts/run_exp191_hf437m_soft_ft.sh ;;
  p3) bash scripts/run_path_2500_p3_tournament.sh ;;
  p4) bash scripts/run_path_2500_p4_expert_iter.sh ;;
  all-m5)
    bash scripts/run_path_2500_p0_baseline.sh
    SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}" bash scripts/run_path_2500_p1_soft_mix.sh
    bash scripts/run_path_2500_p2_lora_soft.sh
    bash scripts/run_path_2500_p3_tournament.sh
    ;;
  *)
    echo "unknown stage: $STAGE"
    echo "usage: $0 {p0|p1|p2-lora|p2-ft|p3|p4|all-m5}"
    exit 1
    ;;
esac
