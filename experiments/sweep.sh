#!/usr/bin/env bash
# Vanilla (K=S=1) decode sweep: prefix-length x batch-size, fixed output length.
# Run on HPCC (4 GPUs). Requires MODEL_PATH set to a local weights dir.
#
#   MODEL_PATH=$SCRATCH/models/Qwen3-32B TP=4 OUTPUT_LEN=256 bash sweep.sh
#
# Each (prefix, batch) is a fresh `python` process -> reloads the model each time
# (clean, but ~1-2 min reload x 8 configs). TODO: optionally sweep inside one process.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the local Qwen3-32B weights dir}"
TP="${TP:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
OUT="${OUT:-results/vanilla.csv}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for prefix in 1024 10240; do
  for batch in 1 4 16 64; do
    echo ">>> prefix=${prefix} batch=${batch}"
    python "${HERE}/bench_decode.py" \
      --tp "${TP}" \
      --prefix-len "${prefix}" \
      --output-len "${OUTPUT_LEN}" \
      --batch "${batch}" \
      --out "${OUT}"
  done
done
echo "done -> ${OUT}"
