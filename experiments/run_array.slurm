#!/bin/bash -l
#SBATCH --job-name=nv-sweep
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:blackwell6000:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0:45:00
#SBATCH --output=slurm-sweep-%A_%a.out
#SBATCH --array=0-95%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leo2@duyuanyuan.uk
#
# Decode sweep as a JOB ARRAY: one array task = ONE (K,S,batch,prefix) config = one fresh
# process AND a fresh 4-GPU allocation. This isolates failures (a hang/OOM only loses that task,
# detectable as a missing part file + its own log) and avoids the cross-config GPU-memory cascade
# of a single monolithic job. Each task writes ONE row to results/parts/<config>.csv.
#
#   cd experiments && sbatch run_array.slurm                 # all 96 configs, 4 running at once
#   cd experiments && sbatch --array=16,40 run_array.slurm   # just specific configs (validation/retry)
# Resumable: a config whose part file already exists is skipped. After it finishes:
#   bash experiments/merge_results.sh   # parts -> results/sweep.csv
set -uo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"
module load cuda/12.8 2>/dev/null || true
cd "$HERE"

# The full grid: 12 valid (K,S) pairs x {1024,10240} prefix x {1,4,16,64} batch = 96 configs.
KS_PAIRS=(1:1 2:1 2:2 4:1 4:2 4:4 8:1 8:2 8:4 16:1 16:2 16:4)
PREFIXES=(1024 10240)
BATCHES=(1 4 16 64)
CONFIGS=()
for ks in "${KS_PAIRS[@]}"; do
  for p in "${PREFIXES[@]}"; do
    for b in "${BATCHES[@]}"; do
      CONFIGS+=("${ks}:${p}:${b}")
    done
  done
done

cfg="${CONFIGS[$SLURM_ARRAY_TASK_ID]:-}"
[ -n "$cfg" ] || { echo "no config for array index $SLURM_ARRAY_TASK_ID (have ${#CONFIGS[@]})"; exit 1; }
IFS=: read -r K S P B <<< "$cfg"
mkdir -p results/parts
OUT="results/parts/K${K}_S${S}_P${P}_B${B}.csv"
if [ -f "$OUT" ]; then echo "SKIP (already done): $OUT"; exit 0; fi

echo ">>> array task ${SLURM_ARRAY_TASK_ID} on $(hostname): K=$K S=$S prefix=$P batch=$B -> $OUT"
nvidia-smi -L || true
# bench_decode writes a row even on OOM/error (status column), so a present part file = a recorded
# result; a MISSING part file = a hard failure (hang/kill) -> re-runnable by resubmitting the array.
TP=4 python bench_decode.py --tp 4 --k "$K" --s "$S" --prefix-len "$P" --output-len 256 --batch "$B" --out "$OUT"
echo ">>> array task ${SLURM_ARRAY_TASK_ID} done"
