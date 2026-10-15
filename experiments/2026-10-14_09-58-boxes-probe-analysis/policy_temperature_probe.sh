#!/usr/bin/env bash
# Run from the repository root. Uses the completed probe's frozen inputs.
set -euo pipefail
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
PREV="$EXP/logs/astra-probe-PCcE6W"
RUN=$(mktemp -d "$EXP/logs/astra-policy-temp-XXXXXX")
git rev-parse HEAD > "$RUN/code-commit.txt"
echo "Results directory: $RUN"
CACHE="$PREV/mix/labels.json"
status=0

for TEMP in 1 2; do
  if OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    timeout --signal=TERM --kill-after=30s 1h \
    uv run python -u -m alpha_go.boxes.search_probe \
      --reference "$PREV/reference.pt" --candidate "$PREV/candidate.pt" \
      --positions-file "$PREV/mix/positions.jsonl" --label-cache "$CACHE" \
      --rows 5 --min-undrawn 35 --max-undrawn 38 \
      --mixes reference candidate --simulations 400 --lambdas 0.5 \
      --policy-temperature "$TEMP" \
      --solver 30 --solver-budget 50000 --label-budget 20000000 \
      --seed 20261014 --output "$RUN/t$TEMP" \
      2>&1 | tee "$RUN/t$TEMP.log"; then
    CACHE="$RUN/t$TEMP/labels.json"
  else
    status=$?
    break
  fi
done

# Preserve partial results and logs too, if a bounded run fails or times out.
items=(code-commit.txt)
for item in "$RUN"/t*; do
  if [ -e "$item" ]; then items+=("${item##*/}"); fi
done
tar -czf "$RUN/results.tgz" -C "$RUN" "${items[@]}"
echo "Exit status: $status; send this file: $RUN/results.tgz"
exit "$status"
