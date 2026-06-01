#!/bin/bash -l
#SBATCH --job-name=nv-ksweep
#SBATCH --partition=short_gpu
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:blackwell6000:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:45:00
#SBATCH --output=slurm-ksweep-%A_%a.out
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leo2@duyuanyuan.uk
#
# Per-(K,S) sweep: each array task loads Qwen3-32B ONCE and runs all 8 (batch x prefix) configs for
# one (K,S) pair in-process (sweep_ks.py). 12 tasks = 12 model loads (vs 96 fresh processes) ->
# ~8x less reload overhead. Failure-detectable per pair (missing part files); resumable (skips
# existing parts). The 4-GPU assoc cap runs these serially (one TP=4 task at a time).
#   cd experiments && sbatch run_ks_array.slurm                # all 12 pairs
#   cd experiments && sbatch --array=2 run_ks_array.slurm      # one pair (validation/retry)
# Then merge: bash experiments/merge_results.sh
set -uo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"
module load cuda/12.8 2>/dev/null || true
rm -f /dev/shm/nanovllm 2>/dev/null || true   # clear stale fixed-name shm landmine from a crashed job
cd "$HERE"

# Fair grid: baseline + low-K (K=2,4 at S=1 — fills the K=1 -> K=16 gap) + K=16 (S=1,2,16) +
# high-K single-step (K=32,64,256 at S=1; eff=K). Every pair is the UNIFORM "commit exactly K every
# step, all requests equal" decode (not ragged); cudagraph where it fits, very large bs*K falls back
# to eager (negligible for few-forward large K). The ragged variable-K study is run separately (bench_vark.py).
KS_PAIRS=(1:1 16:1 16:2 16:16 32:1 64:1 256:1 2:1 4:1)
cfg="${KS_PAIRS[$SLURM_ARRAY_TASK_ID]:-}"
[ -n "$cfg" ] || { echo "no (K,S) for array index $SLURM_ARRAY_TASK_ID"; exit 1; }
K="${cfg%%:*}"; S="${cfg##*:}"

echo ">>> array task ${SLURM_ARRAY_TASK_ID} on $(hostname): (K,S)=(${K},${S}) — batch x prefix in ONE process"
nvidia-smi -L || true
TP=4 python sweep_ks.py --tp 4 --k "$K" --s "$S" --output-len 256 --batches 1,64,128
echo ">>> array task ${SLURM_ARRAY_TASK_ID} done"
