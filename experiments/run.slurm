#!/usr/bin/env bash
#SBATCH --job-name=nanovllm-decode
#SBATCH --partition=<GPU_PARTITION>      # TODO: verify with `sinfo` (UCR HPCC GPU partition)
#SBATCH --gres=gpu:4                     # TODO: match GPU type if required, e.g. gpu:rtxpro6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # TODO: adjust
#SBATCH --mem=128G                       # TODO: adjust (host RAM, not VRAM)
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# --- environment (TODO: match your HPCC) ---
module load <CUDA_MODULE>                # e.g. cuda/12.x  (check `module avail cuda`)
source <PATH_TO_ENV>/bin/activate        # venv/conda with nano-vLLM installed (pip install -e .)

# --- paths (weights live on scratch/bigdata, NOT home) ---
export MODEL_PATH="${MODEL_PATH:-$SCRATCH/models/Qwen3-32B}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/hf}"

cd "$SLURM_SUBMIT_DIR"                    # submit from the experiments/ dir

# --- vanilla (K=S=1) sweep across prefix x batch ---
TP=4 OUTPUT_LEN=256 OUT=results/vanilla.csv bash sweep.sh
