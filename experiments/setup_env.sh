#!/usr/bin/env bash
# One-time provisioning of the project .venv on the UCR HPCC.
#
#   bash experiments/setup_env.sh
#
# Creates .venv (Python 3.12 — satisfies pyproject >=3.10,<3.13), installs torch
# for CUDA 12.8 (Blackwell sm_120) + nano-vLLM's runtime deps, then compiles
# flash-attn from source (no usable prebuilt wheel on Rocky 8 / glibc 2.28).
# Uses uv (mirrors the forensics-nanochat toolchain). Idempotent-ish: re-running
# recreates .venv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"      # repo root + .cache routing (.venv may not exist yet — fine)
cd "$REPO_ROOT"

command -v uv >/dev/null || { echo "ERROR: uv not on PATH (expected ~/.local/bin/uv)" >&2; exit 1; }

# 1. venv on a uv-MANAGED Python 3.12. The system /usr/bin/python3.12 has NO dev
#    headers, so flash-attn fails to compile ("Python.h: No such file or directory").
#    uv's managed standalone CPython ships the headers; only-managed forces it.
uv python install 3.12
uv venv --python 3.12 --python-preference only-managed "$REPO_ROOT/.venv"
VENV_PY="$REPO_ROOT/.venv/bin/python"

# 2. torch for CUDA 12.8 (Blackwell). Pin matches the proven cluster build.
uv pip install --python "$VENV_PY" torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. nano-vLLM's other runtime deps — NOT torch (installed above from the cu128
#    index) and NOT flash-attn (compiled separately in step 5, so pip never tries
#    to build it for the wrong arch).
uv pip install --python "$VENV_PY" "triton>=3.0.0" "transformers>=4.51.0" xxhash tqdm huggingface_hub matplotlib

# 4. the package itself, editable, without re-resolving deps
uv pip install --python "$VENV_PY" -e . --no-deps

# 5. flash-attn from source
bash "$HERE/install_flash_attn.sh"

echo
echo "OK: .venv provisioned at $REPO_ROOT/.venv"
echo "Next:"
echo "  bash experiments/download_model.sh        # -> .cache/models/Qwen3-32B"
echo "  cd experiments && SMOKE=1 sbatch run.slurm # smoke test on blackwell6000"
echo "  cd experiments && sbatch run.slurm         # full K=S=1 sweep"
