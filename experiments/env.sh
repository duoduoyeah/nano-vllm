#!/usr/bin/env bash
# Central environment for the decode sweep. SOURCE this (do not execute):
#
#   source "$(dirname "$0")/env.sh"      # from another script in experiments/
#
# It does three things:
#   1. resolves the repo root (git, else walk up from this file),
#   2. routes EVERY external download — HuggingFace hub/datasets, torch.hub,
#      triton JIT, and anything XDG — into <repo>/.cache/ so the whole cache
#      is gitignored and disposable (rm -rf .cache to reclaim space),
#   3. loads secrets from <repo>/.env (HF_TOKEN) and activates <repo>/.venv.
#
# Mirrors the resolve_repo_root + `set -a; source .env` convention used in
# the forensics-nanochat slurm scripts.

resolve_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  git -C "$here" rev-parse --show-toplevel 2>/dev/null || (cd "$here/.." && pwd)
}

export REPO_ROOT="${REPO_ROOT:-$(resolve_repo_root)}"

# --- all caches live under <repo>/.cache (gitignored, safe to delete) ---
export CACHE_ROOT="$REPO_ROOT/.cache"
export HF_HOME="$CACHE_ROOT/huggingface"            # HF base: hub + token + datasets
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"        # legacy alias some libs still read
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"           # legacy alias
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"               # torch.hub downloads
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"        # triton JIT kernels
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"             # catch-all for everything else
export PIP_CACHE_DIR="$CACHE_ROOT/pip"              # pip/uv wheel cache
export UV_CACHE_DIR="$CACHE_ROOT/uv"

# Clean local dir where downloaded model weights land (NOT the hashed hub layout).
export MODELS_DIR="$CACHE_ROOT/models"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$MODELS_DIR" \
         "$CACHE_ROOT/slurm_logs"

# --- secrets (HF_TOKEN, ...) ---
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi

# --- project venv (created by setup_env.sh) ---
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.venv/bin/activate"
fi

# Default model path for smoke.py / bench_decode.py / sweep.sh when MODEL_PATH unset.
export MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/Qwen3-32B}"
