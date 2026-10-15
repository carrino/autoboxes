#!/usr/bin/env bash
# Start a fresh, integrated policy-target loop from the surviving iter0 snapshot.
set -euo pipefail
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
SOURCE="$EXP/logs/astra-probe-PCcE6W"
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
for INPUT in "$SOURCE/reference.pt" "$SOURCE/mix/positions.jsonl"; do
    [ -f "$INPUT" ] || { echo "Missing required input: $INPUT" >&2; exit 1; }
done
ITERATIONS=${ITERATIONS:-3}
[ "$ITERATIONS" -gt 0 ]
mkdir -p "$EXP/checkpoints"
CKPT=$(mktemp -d "$EXP/checkpoints/5x5-policy-loop-XXXXXX")
TAG=${CKPT##*/}
LOGS="$EXP/logs/$TAG"
mkdir -p "$LOGS"
cp "$SOURCE/reference.pt" "$CKPT/iter0.pt"
echo '{"champion": 0, "history": []}' > "$EXP/league_state-$TAG.json"
git rev-parse HEAD > "$LOGS/code-commit.txt"

ROWS=5; COLS=5; FEATURES=chains; GATE=0
POLICY_LABELS=1
POLICY_MIN_UNDRAWN=${POLICY_MIN_UNDRAWN:-35}; POLICY_MAX_UNDRAWN=${POLICY_MAX_UNDRAWN:-36}
POLICY_MAX_PER_GAME=${POLICY_MAX_PER_GAME:-2}; POLICY_WORKERS=${POLICY_WORKERS:-4}
POLICY_NODE_BUDGET=${POLICY_NODE_BUDGET:-20000000}; POLICY_CHILD_BUDGET=${POLICY_CHILD_BUDGET:-20000000}
POLICY_TIMEOUT=${POLICY_TIMEOUT:-2h}; POLICY_EXCLUDE_POSITIONS="$SOURCE/mix/positions.jsonl"
RELABEL_N=0  # selected roots and children already receive exact values; avoid a second solve pass
SP_GAMES=${SP_GAMES:-100}; SP_SIMS=${SP_SIMS:-1600}; SP_PROCS=${SP_PROCS:-4}; SP_WORKERS=${SP_WORKERS:-8}
TRAIN_BUDGET=0; TRAIN_EPOCHS=${TRAIN_EPOCHS:-4}; TRAIN_MIN_UNDRAWN=24; TRAIN_ARGS=""
WINDOW=${WINDOW:-8}; EXTRA_DATA=""; START_FROM=""; START_FRACTION=0; STOP_WHEN_SOLVED=1
SOLVER_N=30; ARENA_SOLVER_N=30; SOLVER_BUDGET=50000; MERGE_EQ=0
ARENA_GAMES=${ARENA_GAMES:-60}; BASE_GAMES=${BASE_GAMES:-20}; ARENA_SIMS=400; BASELINES=boxes-classic
SEARCH_ARGS="--policy_temperature 1.0 --prove_terminals 1 --margin_utility_lambda 0.5 --margin_utility_k 6 --shape_terminals 1"

# Save the exact run settings and restart entry point, independent of later shell variables.
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$PWD"
    for VARIABLE in EXP TAG LOGS GAME_DATA_DIR ITERATIONS ROWS COLS FEATURES GATE POLICY_LABELS \
        POLICY_MIN_UNDRAWN POLICY_MAX_UNDRAWN POLICY_MAX_PER_GAME POLICY_WORKERS POLICY_NODE_BUDGET \
        POLICY_CHILD_BUDGET POLICY_TIMEOUT POLICY_EXCLUDE_POSITIONS RELABEL_N SP_GAMES SP_SIMS \
        SP_PROCS SP_WORKERS TRAIN_BUDGET TRAIN_EPOCHS TRAIN_MIN_UNDRAWN TRAIN_ARGS WINDOW EXTRA_DATA \
        START_FROM START_FRACTION STOP_WHEN_SOLVED SOLVER_N ARENA_SOLVER_N SOLVER_BUDGET MERGE_EQ \
        ARENA_GAMES BASE_GAMES ARENA_SIMS BASELINES SEARCH_ARGS; do
        printf 'export %s=%q\n' "$VARIABLE" "${!VARIABLE}"
    done
    cat <<'SH'
finish() {
    STATUS=$?
    trap - EXIT
    echo "$STATUS" > "$LOGS/exit-status.txt"
    tar --exclude='*.pt' --exclude='*.npz' --exclude='results.tgz' \
        -czf "$LOGS/results.tgz" -C "$LOGS" .
    echo "Exit status: $STATUS; send this file: $LOGS/results.tgz"
    exit "$STATUS"
}
trap finish EXIT
START=0
while [ -f "$EXP/timing/$TAG/it$((START + 1)).json" ]; do
    DONE=$(python3 "$EXP/arena_completed.py" --state "$EXP/league_state-$TAG.json" \
        --checkpoint "$EXP/checkpoints/$TAG/iter$((START + 1)).pt" \
        --iteration "$((START + 1))" --timing "$EXP/timing/$TAG/it$((START + 1)).json")
    [ "$DONE" = "1" ] || { echo "Timing exists without a committed arena result" >&2; exit 1; }
    START=$((START + 1))
done
if [ "$START" -lt "$ITERATIONS" ]; then
    bash "$EXP/run_iteration_local.sh" "$START" "$((ITERATIONS - 1))" 2>&1 | tee -a "$LOGS/run.log"
else
    echo "All $ITERATIONS iterations already completed."
fi
SH
} > "$LOGS/resume.sh"
echo "Run tag: $TAG"
echo "Resume the same run after interruption: bash $LOGS/resume.sh"
bash "$LOGS/resume.sh"
