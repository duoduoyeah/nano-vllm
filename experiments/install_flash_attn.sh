#!/usr/bin/env bash
# Get flash-attn into .venv. Two paths, auto-selected:
#   FAST — install a prebuilt, cluster-compatible wheel (built once by
#          build_flash_attn_wheel.sh and/or hosted in a public repo). Seconds.
#   SLOW — compile from source. ~10-90 min, RAM-hungry.
#
# WHY a custom wheel at all: every flash-attn wheel on PyPI is linked against
# glibc >= 2.32, but the cluster runs Rocky 8 (glibc 2.28) so those wheels fail to
# import. A wheel built on the cluster links against glibc 2.28 and works everywhere here.
#
#   bash experiments/install_flash_attn.sh
#
# Fast path is taken automatically if a wheel is found via (in order):
#   FLASH_ATTN_WHEEL=/path/to.whl          explicit local wheel
#   $CACHE_ROOT/wheels/flash_attn-*.whl    a locally built wheel
#   FLASH_ATTN_WHEEL_URL=https://...       download it (e.g. a public-repo release asset)
#
# Source-build overrides (env):
#   CUDA_MODULE=cuda/12.8  GCC_MODULE=gcc/11.5.0  FLASH_ATTN_VERSION=2.8.3
#   TORCH_CUDA_ARCH_LIST="8.9;12.0"   # 8.9=ada6000, 12.0=blackwell6000
#   MAX_JOBS=8
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"
cd "$REPO_ROOT"

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "ERROR: $VENV_PY not found — run experiments/setup_env.sh first." >&2; exit 1; }
echo "using $("$VENV_PY" --version) at $VENV_PY"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
# Default hosted wheel (published from this cluster to duoduoyeah/hpcc-flash-attn). Pinned to
# cp312 + torch 2.9.1+cu128 + Rocky 8 (glibc 2.28). Override FLASH_ATTN_WHEEL_URL to point elsewhere,
# or set it empty to force a local source build.
: "${FLASH_ATTN_WHEEL_URL:=https://github.com/duoduoyeah/hpcc-flash-attn/releases/download/v2.8.3-cp312-cu128/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl}"

# ---------- FAST PATH: prebuilt wheel ----------
WHEEL="${FLASH_ATTN_WHEEL:-}"
if [ -z "$WHEEL" ]; then
  WHEEL="$(ls -t "$CACHE_ROOT"/wheels/flash_attn-*.whl 2>/dev/null | head -1 || true)"
fi
if [ -z "$WHEEL" ] && [ -n "$FLASH_ATTN_WHEEL_URL" ]; then
  mkdir -p "$CACHE_ROOT/wheels"
  WHEEL="$CACHE_ROOT/wheels/$(basename "$FLASH_ATTN_WHEEL_URL")"
  echo ">>> fetching prebuilt wheel: $FLASH_ATTN_WHEEL_URL"
  curl -fL "$FLASH_ATTN_WHEEL_URL" -o "$WHEEL" || { echo "WARN: download failed" >&2; WHEEL=""; }
fi
if [ -n "$WHEEL" ] && [ -f "$WHEEL" ]; then
  echo ">>> installing prebuilt flash-attn wheel: $WHEEL"
  uv pip uninstall --python "$VENV_PY" flash-attn 2>/dev/null || true
  uv pip install --python "$VENV_PY" "$WHEEL"
  if "$VENV_PY" -c "import flash_attn" 2>/dev/null; then
    "$VENV_PY" -c "import flash_attn; print('OK prebuilt flash_attn', flash_attn.__version__)"
    exit 0
  fi
  echo "WARN: prebuilt wheel failed to import (glibc/torch/python mismatch?); compiling from source." >&2
  uv pip uninstall --python "$VENV_PY" flash-attn 2>/dev/null || true
fi

# ---------- SLOW PATH: compile from source ----------
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
GCC_MODULE="${GCC_MODULE:-gcc/11.5.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"             # flash-attn CUDA TUs are RAM-hungry; too many -> OOM-kill
export NVCC_THREADS="${NVCC_THREADS:-1}"     # 1 = less RAM per nvcc (multi-arch otherwise multiplies it)
# Force a real source compile — otherwise flash-attn's setup.py downloads a prebuilt
# wheel from Dao-AILab releases (glibc 2.32, fails to import on Rocky 8).
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"   # same FS as .cache (avoid cross-device rename)
mkdir -p "$TMPDIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$GCC_MODULE"
module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
echo "CUDA_HOME=$CUDA_HOME"; { gcc --version || true; } | head -1; { nvcc --version || true; } | tail -1

uv pip install --python "$VENV_PY" setuptools wheel psutil ninja packaging einops
uv pip uninstall --python "$VENV_PY" flash-attn 2>/dev/null || true

echo "compiling flash-attn ${FLASH_ATTN_VERSION} for arch ${TORCH_CUDA_ARCH_LIST} (MAX_JOBS=${MAX_JOBS}) — ~10-90 min"
# --no-binary forces a source build; building on Rocky 8 links against glibc 2.28 so it imports here.
uv pip install --python "$VENV_PY" --no-binary flash-attn \
  "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

"$VENV_PY" -c "import flash_attn; print('OK compiled flash_attn', flash_attn.__version__)"
