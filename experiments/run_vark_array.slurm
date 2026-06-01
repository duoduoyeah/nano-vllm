#!/bin/bash -l
#SBATCH --job-name=nv-vark
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:blackwell6000:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0:45:00
#SBATCH --output=slurm-vark-%A_%a.out
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leo2@duyuanyuan.uk
#
# Variable-K (ragged) decode study at B=64. Each array task is ONE fresh process (TP=4) -> a capture
# error / OOM stays isolated and detectable (missing part file), per the multi-job request. Configs:
#   ragged  = variable K in [1,8] per request per step, balanced to a constant per-step total (avg 4),
#             paged VARLEN path + single cudagraph (bench_vark.py --mode ragged).
#   uniform = same average work, K=4 S=1 every step (the raggedness-cost reference).
# Baseline K=1 already lives in results/vanilla.csv (not rerun). 4-GPU assoc cap -> tasks run serially.
#   cd experiments && sbatch run_vark_array.slurm                # all 4
#   cd experiments && sbatch --array=0 run_vark_array.slurm      # one config (validate/retry)
# Then merge: cat results/parts_vark/*.csv | awk 'NR==1||!/^model,/' > results/vark.csv
set -uo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"
module load cuda/12.8 2>/dev/null || true
rm -f /dev/shm/nanovllm 2>/dev/null || true   # clear stale fixed-name shm landmine from a crashed job
cd "$HERE"
mkdir -p results/parts_vark

# index -> "mode:prefix"
CONFIGS=(ragged:1024 ragged:10240 uniform:1024 uniform:10240)
cfg="${CONFIGS[$SLURM_ARRAY_TASK_ID]:-}"
[ -n "$cfg" ] || { echo "no config for array index $SLURM_ARRAY_TASK_ID"; exit 1; }
MODE="${cfg%%:*}"; P="${cfg##*:}"
OUT="results/parts_vark/vark_${MODE}_B64_P${P}.csv"
[ -f "$OUT" ] && { echo "SKIP $OUT (already done)"; exit 0; }

echo ">>> array ${SLURM_ARRAY_TASK_ID} on $(hostname): mode=${MODE} B=64 prefix=${P}"
nvidia-smi -L || true
TP=4 python bench_vark.py --tp 4 --mode "$MODE" --batch 64 --prefix-len "$P" \
    --output-len 256 --num-steps 64 --k-min 1 --k-max 8 --out "$OUT" \
    || echo "!!! crash mode=${MODE} P=${P} (no part written)"
echo ">>> array ${SLURM_ARRAY_TASK_ID} done"
