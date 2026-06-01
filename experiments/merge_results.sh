#!/usr/bin/env bash
# Merge the per-config part CSVs written by the run_array.slurm job array into one CSV,
# and report which of the 96 configs are still missing (hard failures to re-run).
#
#   bash experiments/merge_results.sh                      # -> results/sweep.csv
#   bash experiments/merge_results.sh results/sweep.csv    # custom output
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PARTS="$HERE/results/parts"
OUT="${1:-$HERE/results/sweep.csv}"

shopt -s nullglob
parts=("$PARTS"/*.csv)
[ "${#parts[@]}" -gt 0 ] || { echo "no part files in $PARTS — run run_array.slurm first"; exit 1; }

head -1 "${parts[0]}" > "$OUT"
for f in "${parts[@]}"; do tail -n +2 "$f"; done >> "$OUT"
rows=$(($(wc -l < "$OUT") - 1))
echo "merged ${#parts[@]} parts -> $OUT (${rows} rows)"

# report missing configs (the 96-cell grid) so hard failures are never silently lost
echo "missing configs (no part file):"
miss=0
for ks in 1:1 2:1 2:2 4:1 4:2 4:4 8:1 8:2 8:4 16:1 16:2 16:4; do
  K="${ks%%:*}"; S="${ks##*:}"
  for P in 1024 10240; do
    for B in 1 4 16 64; do
      [ -f "$PARTS/K${K}_S${S}_P${P}_B${B}.csv" ] || { echo "  K=$K S=$S P=$P B=$B"; miss=$((miss+1)); }
    done
  done
done
[ "$miss" -eq 0 ] && echo "  (none — all 96 present)" || echo "  -> $miss missing; resubmit: cd experiments && sbatch run_array.slurm"
echo "status breakdown:"; tail -n +2 "$OUT" | awk -F, '{print $NF}' | sort | uniq -c
