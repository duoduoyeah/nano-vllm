#!/usr/bin/env bash
# Build a REUSABLE flash-attn wheel for this cluster's GPUs, ONCE, so future
# installs are a fast `pip install <wheel>` instead of a ~30-90 min source compile.
#
# WHY a custom wheel: every flash-attn wheel published on PyPI is linked against
# glibc >= 2.32, but the cluster runs Rocky 8 (glibc 2.28) — those wheels fail to
# import. A wheel built HERE links against the host glibc 2.28 and works on every
# cluster node (login + GPU). One wheel can embed kernels for several GPU archs.
#
#   bash experiments/build_flash_attn_wheel.sh
#
# Output:  .cache/wheels/flash_attn-<ver>-cp<py>-...-linux_x86_64.whl  (+ .meta.txt)
# It then installs that wheel into .venv and verifies the import.
#
# Overrides (env):
#   FLASH_ATTN_VERSION=2.8.3
#   TORCH_CUDA_ARCH_LIST="8.9;12.0"   # 8.9 = RTX 6000 Ada (ada6000), 12.0 = RTX PRO 6000 Blackwell
#   CUDA_MODULE=cuda/12.8   GCC_MODULE=gcc/11.5.0   MAX_JOBS=16
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"
cd "$REPO_ROOT"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;12.0}"
export MAX_JOBS="${MAX_JOBS:-12}"            # flash-attn CUDA TUs are RAM-hungry; too many -> OOM-kill
export NVCC_THREADS="${NVCC_THREADS:-1}"     # 1 = less RAM per nvcc (multi-arch otherwise multiplies it)
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
GCC_MODULE="${GCC_MODULE:-gcc/11.5.0}"
WHEELHOUSE="${WHEELHOUSE:-$CACHE_ROOT/wheels}"
mkdir -p "$WHEELHOUSE"

# CRITICAL: flash-attn's setup.py otherwise DOWNLOADS a prebuilt wheel from Dao-AILab's
# GitHub releases (linked against glibc 2.32 -> fails on Rocky 8). Force a real compile.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# Keep build tmp on the same filesystem as the wheelhouse/pip cache (the download path
# died on an "Invalid cross-device link" between /scratch tmp and /bigdata .cache).
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
mkdir -p "$TMPDIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$GCC_MODULE"
module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
echo "CUDA_HOME=$CUDA_HOME"; { gcc --version || true; } | head -1; { nvcc --version || true; } | tail -1

VENV_PY="$REPO_ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "ERROR: $VENV_PY missing — run experiments/setup_env.sh first." >&2; exit 1; }
echo "using $("$VENV_PY" --version) at $VENV_PY"

# uv venvs ship without pip; `pip wheel` is the simplest source-wheel builder.
uv pip install --python "$VENV_PY" pip wheel setuptools ninja packaging psutil einops

echo ">>> building flash-attn ${FLASH_ATTN_VERSION} wheel for arch '${TORCH_CUDA_ARCH_LIST}' (MAX_JOBS=${MAX_JOBS}) — this is the long part"
# --no-binary flash-attn forces a source build (ignore the glibc-2.32 PyPI wheels);
# --no-build-isolation lets it see the installed torch; --no-deps keeps it to flash-attn.
"$VENV_PY" -m pip wheel --no-binary flash-attn "flash-attn==${FLASH_ATTN_VERSION}" \
  --no-build-isolation --no-deps -w "$WHEELHOUSE"

WHL="$(ls -t "$WHEELHOUSE"/flash_attn-*.whl | head -1)"
echo ">>> built wheel: $WHL"

# install into the venv (replace any broken prebuilt) and verify
uv pip uninstall --python "$VENV_PY" flash-attn 2>/dev/null || true
uv pip install --python "$VENV_PY" "$WHL"
"$VENV_PY" -c "import flash_attn, torch; print('OK import flash_attn', flash_attn.__version__, '| torch', torch.__version__)"

# metadata sidecar (so the public wheel repo records exactly what this wheel targets)
PYTAG="$("$VENV_PY" -c 'import sys;print("cp%d%d"%sys.version_info[:2])')"
TORCHV="$("$VENV_PY" -c 'import torch;print(torch.__version__)')"
GLIBC="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo '?')"
SHA="$(sha256sum "$WHL" | cut -d' ' -f1)"
{
  echo "wheel=$(basename "$WHL")"
  echo "flash_attn_version=$FLASH_ATTN_VERSION"
  echo "python=$PYTAG"
  echo "torch=$TORCHV"
  echo "cuda_module=$CUDA_MODULE"
  echo "gpu_archs=$TORCH_CUDA_ARCH_LIST   # 8.9=ada6000, 12.0=blackwell6000"
  echo "built_on=Rocky 8 / glibc $GLIBC"
  echo "sha256=$SHA"
} | tee "${WHL}.meta.txt"
echo ">>> wheel + meta ready in $WHEELHOUSE"
