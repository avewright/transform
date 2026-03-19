#!/bin/bash
# setup.sh — RunPod / Linux GPU environment setup for chess-transformer
# Usage: bash setup.sh
set -e

echo "=== Chess-Transformer RunPod Setup ==="

# 1. System dependencies
echo "[1/5] Installing system packages..."
apt-get update -qq && apt-get install -y -qq wget unzip git > /dev/null 2>&1

# 2. Python dependencies
echo "[2/5] Installing Python packages..."
pip install -q torch transformers accelerate datasets python-chess numpy tqdm safetensors huggingface-hub stockfish

# 3. Stockfish binary
echo "[3/5] Installing Stockfish..."
STOCKFISH_DIR="stockfish/stockfish"
if [ -f "$STOCKFISH_DIR/stockfish" ] || [ -f "$STOCKFISH_DIR/stockfish-ubuntu-x86-64-avx2" ]; then
    echo "  Stockfish already installed."
else
    mkdir -p stockfish
    cd stockfish
    # Download latest Stockfish for Linux
    wget -q https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar -O stockfish.tar
    tar xf stockfish.tar
    rm stockfish.tar
    cd ..
    echo "  Stockfish installed."
fi

# Find the actual binary path
SF_BIN=$(find stockfish/ -name "stockfish*" -type f -executable 2>/dev/null | head -1)
if [ -z "$SF_BIN" ]; then
    SF_BIN=$(find stockfish/ -name "stockfish*" -type f 2>/dev/null | head -1)
fi
echo "  Stockfish binary: $SF_BIN"

# 4. Update stockfish path in experiment files for Linux
echo "[4/5] Patching Stockfish path for Linux..."
if [ -n "$SF_BIN" ]; then
    # Replace Windows path with Linux path in experiment files
    find experiments/ -name "*.py" -exec sed -i \
        "s|stockfish/stockfish/stockfish-windows-x86-64-avx2.exe|$SF_BIN|g" {} +
    echo "  Patched experiment files to use: $SF_BIN"
fi

# 5. Verify
echo "[5/5] Verifying installation..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import chess
print(f'  python-chess {chess.__version__}')

from stockfish import Stockfish
print('  stockfish package: OK')

from transformers import AutoModelForCausalLM
print('  transformers: OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  # Generate Stockfish-labeled data and train:"
echo "  python experiments/exp012b_quick_stockfish.py"
echo ""
echo "  # Or run the full pipeline:"
echo "  python experiments/exp012_stockfish_supervised.py"
