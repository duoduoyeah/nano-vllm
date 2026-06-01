#!/usr/bin/env bash
# Vanilla (K=S=1) decode sweep -> results/vanilla.csv  (demand.md priority 1).
# prefix-length x batch-size, fixed output length, on a 4-GPU Blackwell node.
# env.sh sets MODEL_PATH + routes caches to .cache/.
#
#   cd experiments && OUTPUT_LEN=256 bash sweep.sh
#   MODEL_PATH=/path/to/other-model bash sweep.sh
#
# Resumable: a (K=1,S=1,batch,prefix) already present in $OUT is skipped, so re-submitting
# after a short_gpu 2h timeout continues. Each config is a fresh `python` process.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"

: "${MODEL_PATH:?env.sh should set this; or pass MODEL_PATH=/path/to/weights}"
[ -d "$MODEL_PATH" ] || { echo "ERROR: MODEL_PATH is not a dir: $MODEL_PATH (run download_model.sh)" >&2; exit 1; }
TP="${TP:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
OUT="${OUT:-results/vanilla.csv}"

# CSV columns: model,tp,K,S,batch,prefix_len,...  -> K=$3, S=$4, batch=$5, prefix_len=$6
done_already() {
  [ -f "$OUT" ] && awk -F, -v b="$1" -v p="$2" \
    'NR>1 && $3==1 && $4==1 && $5==b && $6==p {h=1} END{exit !h}' "$OUT"
}

for prefix in 1024 10240; do
  for batch in 1 4 16 64; do
    if done_already "$batch" "$prefix"; then
      echo ">>> skip K=1 S=1 batch=${batch} prefix=${prefix} (already in ${OUT})"
      continue
    fi
    echo ">>> K=1 S=1 prefix=${prefix} batch=${batch}"
    python "${HERE}/bench_decode.py" --model "${MODEL_PATH}" --tp "${TP}" --k 1 --s 1 \
      --prefix-len "${prefix}" --output-len "${OUTPUT_LEN}" --batch "${batch}" --out "${OUT}" \
      || echo "!!! config FAILED with no row (hard crash) prefix=${prefix} batch=${batch} — see logs" >&2
  done
done
echo "done -> ${OUT}"
