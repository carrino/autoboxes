#!/usr/bin/env bash
# Run from the repository root; argument 1 is the completed collection's games directory.
set -euo pipefail

EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
GAMES=${1:?Usage: bash run_pilot.sh /path/to/pilot/games [replay-manifest]}
REPLAY=${2:-$EXP/dataset-5x5-exact-it22.txt}
SOURCE="$EXP/logs/astra-probe-PCcE6W"
for INPUT in "$REPLAY" "$SOURCE/reference.pt" "$SOURCE/mix/positions.jsonl" \
             "$GAMES/collection.json"; do
  if [ ! -f "$INPUT" ]; then
    echo "Missing required input: $INPUT" >&2
    exit 1
  fi
done
test -d "$GAMES"
python3 - "$GAMES/collection.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
assert manifest["status"] == "complete", "wait for the collection pilot to finish"
PY
mkdir -p "$EXP/logs"
RUN=$(mktemp -d "$EXP/logs/policy-target-ab-XXXXXX")
echo "Results directory: $RUN"
git rev-parse HEAD > "$RUN/code-commit.txt"
cp "$SOURCE/reference.pt" "$RUN/reference.pt"

finish() {
  STATUS=$?
  trap - EXIT
  echo "$STATUS" > "$RUN/exit-status.txt"
  # Keep checkpoints and reusable data locally; send diagnostics even after a timeout.
  tar --exclude='*.pt' --exclude='*.npz' --exclude='results.tgz' \
      -czf "$RUN/results.tgz" -C "$RUN" .
  echo "Exit status: $STATUS; send this file: $RUN/results.tgz"
  exit "$STATUS"
}
trap finish EXIT

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=30s 2h \
  uv run python -u -m alpha_go.boxes.policy_ab_data \
    --games "$GAMES" --replay-manifest "$REPLAY" \
    --exclude-positions "$SOURCE/mix/positions.jsonl" \
    --output "$RUN/prepared" --rows 5 --min-undrawn 35 --max-undrawn 38 \
    --node-budget 20000000 --child-node-budget 20000000 \
    --max-per-game 2 --workers 4 \
    2>&1 | tee "$RUN/prepare.log"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
timeout --signal=TERM --kill-after=30s 30m \
  uv run python -u -m alpha_go.boxes.policy_target_ab \
    --reference "$RUN/reference.pt" --prepared "$RUN/prepared" --output "$RUN/train" \
    --epochs 2 --batch-size 256 --min-undrawn 24 --seed 20261015 --split-seed 0 \
    2>&1 | tee "$RUN/train.log"
