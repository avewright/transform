#!/bin/bash
# Create/update the public disagreement dataset and keep uploading new shards.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a
export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

exec .venv/bin/python -u scripts/push_hf100m_disagreement.py --go --watch "$@"
