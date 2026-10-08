#!/usr/bin/env bash
# Local loop: bootstrap -> train iter0 -> [collect-it{N} -> train-it{N+1} -> promote] for N in start..end.
# Usage: [ROWS=5 [COLS=5]] bash run_iteration_local.sh <start_iter> <end_iter> [--cpu]
# Everything a run writes is tagged by board size (TAG=<rows>x<cols>): game data under
# $GAME_DATA_DIR/experiments/<this folder>/$TAG/, checkpoints/$TAG/, logs/$TAG/, timing/$TAG/,
# dataset-$TAG-itN.txt and league_state-$TAG.json, so 3x3 and 5x5 runs never collide.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
START=${1:?Usage: run_iteration_local.sh <start_iter> <end_iter> [--cpu]}
END=${2:?Usage: run_iteration_local.sh <start_iter> <end_iter> [--cpu]}
CPU_FLAG=${3:-}
ROWS=${ROWS:-3}; COLS=${COLS:-$ROWS}; TAG="${ROWS}x${COLS}"
SIZE_ARGS="--rows $ROWS --cols $COLS"
CKPT="$EXP_DIR/checkpoints/$TAG"
LOGS="$EXP_DIR/logs/$TAG"
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
mkdir -p "$CKPT" "$LOGS" "$EXP_DIR/timing/$TAG" "$GAME_DATA_DIR"

# GPU budgets (3x3: minutes per iteration; 5x5: read timing/5x5/it1.json and scale these so
# one iteration fits the time you have). The --cpu smoke profile proves the pipeline end to
# end in a few minutes.
BOOT_GAMES=400; SP_GAMES=200; SP_SIMS=200; SP_WORKERS=8; TRAIN_BUDGET=180; ARENA_GAMES=100; BASE_GAMES=40; ARENA_SIMS=200
if [ "$ROWS" -ge 5 ]; then
    # boxes-ab-d4 is pure Python and costs ~2 s/move on 5x5, so the baseline matches stay small.
    BOOT_GAMES=600; SP_GAMES=400; SP_SIMS=400; TRAIN_BUDGET=300; ARENA_GAMES=60; BASE_GAMES=10; ARENA_SIMS=400
fi
if [ "$CPU_FLAG" = "--cpu" ]; then
    BOOT_GAMES=${SMOKE_BOOT_GAMES:-30}; SP_GAMES=${SMOKE_SP_GAMES:-12}; SP_SIMS=${SMOKE_SP_SIMS:-24}; SP_WORKERS=2
    TRAIN_BUDGET=${SMOKE_TRAIN_BUDGET:-20}; ARENA_GAMES=${SMOKE_ARENA_GAMES:-6}; BASE_GAMES=${SMOKE_BASE_GAMES:-4}; ARENA_SIMS=24
fi
DATA="experiments/${EXP_NAME}/${TAG}"
log() { echo; echo "############### [$TAG] $* ###############"; }

if [ ! -f "$CKPT/iter${START}.pt" ]; then
    [ "$START" -eq 0 ] || { echo "ERROR: $CKPT/iter${START}.pt missing" >&2; exit 1; }
    if [ -z "$(ls -A "$GAME_DATA_DIR/$DATA/bootstrap-it0" 2>/dev/null)" ]; then
        log "Bootstrap: collecting $BOOT_GAMES games per matchup without search"
        uv run "$EXP_DIR/pre_collect.py" $SIZE_ARGS --num_games "$BOOT_GAMES" \
            --save-name "${DATA}/bootstrap-it0" 2>&1 | tee "$LOGS/bootstrap.log"
    fi
    echo "${DATA}/bootstrap-it0" > "$EXP_DIR/dataset-${TAG}-it0.txt"
    log "Train iter0 from bootstrap games"
    uv run "$EXP_DIR/train.py" $SIZE_ARGS --dataset-txt "dataset-${TAG}-it0.txt" --iteration 0 \
        --time-budget "$TRAIN_BUDGET" $CPU_FLAG 2>&1 | tee "$LOGS/train-it0.log"
    log "Arena: iter0 becomes the first champion"
    uv run "$EXP_DIR/arena_promote.py" $SIZE_ARGS --iteration 0 --num_games "$ARENA_GAMES" \
        --baseline_games "$BASE_GAMES" --num_simulations "$ARENA_SIMS" $CPU_FLAG 2>&1 | tee "$LOGS/arena-it0.log"
fi

for ITER in $(seq "$START" "$END"); do
    NEXT=$((ITER + 1))
    log "Iter ${ITER}: self-play with $CKPT/iter${ITER}.pt"
    t0=$(date +%s)
    uv run "$EXP_DIR/run_games.py" $SIZE_ARGS --checkpoint "$CKPT/iter${ITER}.pt" \
        --num_games "$SP_GAMES" --num_simulations "$SP_SIMS" --num_workers "$SP_WORKERS" \
        --save-name "${DATA}/selfplay-it${ITER}" --seed "$((ITER * 100000))" $CPU_FLAG \
        2>&1 | tee "$LOGS/collect-it${ITER}.log"
    t1=$(date +%s)
    DS="$EXP_DIR/dataset-${TAG}-it${NEXT}.txt"
    { echo "# ${EXP_NAME} ${TAG} iter${NEXT} dataset (auto-generated): last 4 self-play iterations"
      for K in $(seq $((ITER > 3 ? ITER - 3 : 0)) "$ITER"); do echo "${DATA}/selfplay-it${K}"; done; } > "$DS"
    log "Train iter${NEXT} from $(basename "$DS")"
    uv run "$EXP_DIR/train.py" $SIZE_ARGS --dataset-txt "$(basename "$DS")" --iteration "$NEXT" \
        --resume-from "$CKPT/iter${ITER}.pt" --time-budget "$TRAIN_BUDGET" $CPU_FLAG \
        2>&1 | tee "$LOGS/train-it${NEXT}.log"
    t2=$(date +%s)
    log "Arena: iter${NEXT} vs champion"
    uv run "$EXP_DIR/arena_promote.py" $SIZE_ARGS --iteration "$NEXT" --num_games "$ARENA_GAMES" \
        --baseline_games "$BASE_GAMES" --num_simulations "$ARENA_SIMS" $CPU_FLAG \
        2>&1 | tee "$LOGS/arena-it${NEXT}.log"
    t3=$(date +%s)
    echo "{\"iteration\": $NEXT, \"collect\": $((t1 - t0)), \"train\": $((t2 - t1)), \"arena\": $((t3 - t2))}" \
        > "$EXP_DIR/timing/$TAG/it${NEXT}.json"
done
log "Done. Last trained iter: $((END + 1))"
uv run "$EXP_DIR/analyze.py" "$TAG"
