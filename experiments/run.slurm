#!/bin/bash -l
#SBATCH --job-name=nanovllm-decode
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:blackwell6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leo2@duyuanyuan.uk

# Vanilla (K=S=1) decode sweep for Qwen3-32B at TP=4 on RTX PRO 6000 Blackwell.
#
# Blackwell6000 lives on nodes gpu13/gpu14, reachable via short_gpu (2h cap) or
# the account-gated `raise` QOS (30d). For the long queue, override at submit:
#   sbatch --partition=raise --qos=raise --time=8:00:00 run.slurm
# Quick GPU helpers (from design/cluster/hpcc.md): black-free / black-who / black-queue.
#
# Modes (env):
#   SMOKE=1        run experiments/smoke.py only (load + decode 16 tokens), skip the sweep.
#   FULL=1         full (K,S) sweep -> results/sweep.csv (else vanilla K=S=1 -> vanilla.csv).
#   OUTPUT_LEN=N   decode length T (default 256).
set -euo pipefail

# Under sbatch, $0 is a copy in /var/spool/slurmd; use SLURM_SUBMIT_DIR (the dir you ran
# `sbatch` from = experiments/). Falls back to $0 for a plain `bash run.slurm`.
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"                       # repo root, .cache routing, .env, .venv

module load cuda/12.8 2>/dev/null || true   # runtime CUDA libs (flash-attn built against this)
rm -f /dev/shm/nanovllm 2>/dev/null || true # clear stale fixed-name shm landmine from a crashed job
cd "$HERE"

echo "=== node $(hostname) ==="; nvidia-smi -L || true
echo "MODEL_PATH=$MODEL_PATH"; python -c "import torch,flash_attn;print('torch',torch.__version__,'flash_attn',flash_attn.__version__)"

if [ "${SMOKE:-0}" = "1" ]; then
  echo ">>> SMOKE: load Qwen3-32B at TP=4 and decode 16 tokens"
  TP=4 python smoke.py
elif [ "${FULL:-0}" = "1" ]; then
  echo ">>> FULL (K,S) sweep -> results/sweep.csv, T=${OUTPUT_LEN:-256}"
  TP=4 OUTPUT_LEN="${OUTPUT_LEN:-256}" OUT="${OUT:-results/sweep.csv}" bash sweep_full.sh
else
  echo ">>> VANILLA K=S=1 sweep -> results/vanilla.csv, T=${OUTPUT_LEN:-256}"
  TP=4 OUTPUT_LEN="${OUTPUT_LEN:-256}" OUT="${OUT:-results/vanilla.csv}" bash sweep.sh
fi
