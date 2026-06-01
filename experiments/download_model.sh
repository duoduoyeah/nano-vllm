#!/usr/bin/env bash
# Download model weights into <repo>/.cache/models/<name> — a clean local dir
# (nano-vLLM's Config asserts os.path.isdir(model), so we use --local-dir, not
# the hashed hub layout). All HF traffic is routed into .cache/ by env.sh, so the
# download is disposable: `rm -rf .cache` reclaims the space.
#
#   bash experiments/download_model.sh                   # Qwen/Qwen3-32B (default)
#   bash experiments/download_model.sh Qwen/Qwen3-0.6B   # small, for quick tests
#
# Requires the project .venv (provides the `hf` CLI). Run setup_env.sh first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"

MODEL_ID="${1:-Qwen/Qwen3-32B}"
DEST="$MODELS_DIR/$(basename "$MODEL_ID")"

echo ">>> downloading $MODEL_ID -> $DEST"
echo "    HF_HOME=$HF_HOME   (token: ${HF_TOKEN:+set})"
hf download "$MODEL_ID" --local-dir "$DEST" --exclude "*.pth" "original/*"

echo "done. Use it with:  MODEL_PATH=$DEST"
