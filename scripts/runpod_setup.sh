#!/bin/bash
# RunPod / Linux: install deps + print commands for A100 training.
# Usage: bash scripts/runpod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== RunPod setup (chess-transformer) ==="

apt-get update -qq && apt-get install -y -qq wget unzip git >/dev/null 2>&1 || true

pip install -q -U pip
pip install -q -e .
pip install -q git+https://github.com/KellerJordan/Muon

# Stockfish binary (for RL / eval)
if ! find stockfish -name 'stockfish*' -type f 2>/dev/null | grep -q .; then
  mkdir -p stockfish && cd stockfish
  wget -q https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar -O sf.tar
  tar xf sf.tar && rm sf.tar
  cd ..
fi
export STOCKFISH_PATH="$(find stockfish -name 'stockfish*' -type f | head -1)"
echo "STOCKFISH_PATH=$STOCKFISH_PATH"

python - <<'PY'
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu {torch.cuda.get_device_name(0)} vram={p.total_memory/1e9:.1f}GB")
PY

cat <<'EOF'

=== Ready. Suggested commands ===

# Smoke test (A100 705M preset)
python experiments/exp182_pretrain_700m.py --go --a100 --smoke

# Full pretrain (705M, tuned for 80GB)
python experiments/exp182_pretrain_700m.py --go --a100 --resume

# Expert-iteration RL / self-play
python experiments/exp183_selfplay.py --preset a100 --go --mode sf

# Inference / UCI engine (after training)
python play.py --checkpoint outputs/exp182_pretrain_a100/latest.pt
python uci_engine.py --checkpoint outputs/exp182_pretrain_a100/latest.pt

Checkpoints are gitignored — copy outputs/ off the pod before terminate.
EOF
