#!/usr/bin/env bash
# Full (K,S) decode sweep -> results/sweep.csv  (demand.md priority 2).
# 12 valid (K,S) pairs (K/S >= 1) x batch x prefix, on a 4-GPU Blackwell node.
#
#   cd experiments && OUTPUT_LEN=256 bash sweep_full.sh
#
# K=1,S=1 runs the production decode path; K>1 uses the K-over-S engine path (kovers_impl.md) —
# IMPLEMENTED but UNVALIDATED until the first GPU run. Any failure is recorded as `error:...`
# (never silently dropped — demand.md). Resumable: configs already in $OUT are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=experiments/env.sh
source "$HERE/env.sh"

: "${MODEL_PATH:?env.sh should set this; or pass MODEL_PATH=/path/to/weights}"
[ -d "$MODEL_PATH" ] || { echo "ERROR: MODEL_PATH is not a dir: $MODEL_PATH (run download_model.sh)" >&2; exit 1; }
TP="${TP:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
OUT="${OUT:-results/sweep.csv}"
# 12 valid (K,S) pairs with K/S >= 1
KS_PAIRS="${KS_PAIRS:-1:1 2:1 2:2 4:1 4:2 4:4 8:1 8:2 8:4 16:1 16:2 16:4}"

# CSV columns: model,tp,K,S,batch,prefix_len,...  -> K=$3, S=$4, batch=$5, prefix_len=$6
done_already() {
  [ -f "$OUT" ] && awk -F, -v k="$1" -v s="$2" -v b="$3" -v p="$4" \
    'NR>1 && $3==k && $4==s && $5==b && $6==p {h=1} END{exit !h}' "$OUT"
}

for pair in $KS_PAIRS; do
  K="${pair%%:*}"; S="${pair##*:}"
  for prefix in 1024 10240; do
    for batch in 1 4 16 64; do
      if done_already "$K" "$S" "$batch" "$prefix"; then
        echo ">>> skip K=${K} S=${S} batch=${batch} prefix=${prefix} (already in ${OUT})"
        continue
      fi
      echo ">>> K=${K} S=${S} prefix=${prefix} batch=${batch}"
      python "${HERE}/bench_decode.py" --model "${MODEL_PATH}" --tp "${TP}" --k "${K}" --s "${S}" \
        --prefix-len "${prefix}" --output-len "${OUTPUT_LEN}" --batch "${batch}" --out "${OUT}" \
        || echo "!!! config FAILED with no row (hard crash) K=${K} S=${S} prefix=${prefix} batch=${batch} — see logs" >&2
    done
  done
done
echo "done -> ${OUT}"
