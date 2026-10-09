#!/usr/bin/env bash
# Local loop: bootstrap -> train iter0 -> [collect-it{N} -> train-it{N+1} -> promote] for N in start..end.
# Usage: [ROWS=5 [COLS=5] [TAG=5x5-probe] [SP_GAMES=.. SP_SIMS=.. ...]] bash run_iteration_local.sh <start_iter> <end_iter> [--cpu]
# Everything a run writes is tagged (TAG defaults to <rows>x<cols>): game data under
# $GAME_DATA_DIR/experiments/<this folder>/$TAG/, checkpoints/$TAG/, logs/$TAG/, timing/$TAG/,
# dataset-$TAG-itN.txt and league_state-$TAG.json, so 3x3 and 5x5 runs never collide.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
START=${1:?Usage: run_iteration_local.sh <start_iter> <end_iter> [--cpu]}
END=${2:?Usage: run_iteration_local.sh <start_iter> <end_iter> [--cpu]}
CPU_FLAG=${3:-}
ROWS=${ROWS:-3}; COLS=${COLS:-$ROWS}; TAG=${TAG:-${ROWS}x${COLS}}
SIZE_ARGS="--rows $ROWS --cols $COLS"
TAG_ARGS="--tag $TAG"
CKPT="$EXP_DIR/checkpoints/$TAG"
LOGS="$EXP_DIR/logs/$TAG"
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
mkdir -p "$CKPT" "$LOGS" "$EXP_DIR/timing/$TAG" "$GAME_DATA_DIR"

# Budget defaults: --cpu smoke (proves the pipeline in minutes), 5x5 overnight, 3x3 (minutes per
# iteration). Any budget can be overridden from the environment, e.g. a 5x5 timing probe:
#   ROWS=5 TAG=5x5-probe BOOT_GAMES=50 SP_GAMES=16 SP_SIMS=100 TRAIN_BUDGET=60 ARENA_GAMES=8 BASE_GAMES=2 bash run_iteration_local.sh 0 0
# boxes-ab-d4 is pure Python and costs ~2 s/move on 5x5, so its baseline matches stay small.
if [ "$CPU_FLAG" = "--cpu" ]; then
    D_BOOT=30; D_SP=12; D_SIMS=24; D_WORKERS=2; D_TRAIN=20; D_ARENA=6; D_BASE=4; D_ASIMS=24
elif [ "$ROWS" -ge 5 ]; then
    D_BOOT=600; D_SP=400; D_SIMS=400; D_WORKERS=8; D_TRAIN=300; D_ARENA=60; D_BASE=10; D_ASIMS=400
else
    D_BOOT=400; D_SP=200; D_SIMS=200; D_WORKERS=8; D_TRAIN=180; D_ARENA=100; D_BASE=40; D_ASIMS=200
fi
BOOT_GAMES=${BOOT_GAMES:-$D_BOOT}; SP_GAMES=${SP_GAMES:-$D_SP}; SP_SIMS=${SP_SIMS:-$D_SIMS}
SP_WORKERS=${SP_WORKERS:-$D_WORKERS}; TRAIN_BUDGET=${TRAIN_BUDGET:-$D_TRAIN}; ARENA_GAMES=${ARENA_GAMES:-$D_ARENA}
BASE_GAMES=${BASE_GAMES:-$D_BASE}; ARENA_SIMS=${ARENA_SIMS:-$D_ASIMS}
echo "[$TAG] budgets: bootstrap $BOOT_GAMES/matchup, self-play $SP_GAMES games x $SP_SIMS sims ($SP_WORKERS workers), "\
     "train ${TRAIN_BUDGET}s, arena $ARENA_GAMES vs champion + $BASE_GAMES vs each baseline at $ARENA_SIMS sims"
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
    uv run "$EXP_DIR/train.py" $SIZE_ARGS $TAG_ARGS --dataset-txt "dataset-${TAG}-it0.txt" --iteration 0 \
        --time-budget "$TRAIN_BUDGET" $CPU_FLAG 2>&1 | tee "$LOGS/train-it0.log"
    log "Arena: iter0 becomes the first champion"
    uv run "$EXP_DIR/arena_promote.py" $SIZE_ARGS $TAG_ARGS --iteration 0 --num_games "$ARENA_GAMES" \
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
    uv run "$EXP_DIR/train.py" $SIZE_ARGS $TAG_ARGS --dataset-txt "$(basename "$DS")" --iteration "$NEXT" \
        --resume-from "$CKPT/iter${ITER}.pt" --time-budget "$TRAIN_BUDGET" $CPU_FLAG \
        2>&1 | tee "$LOGS/train-it${NEXT}.log"
    t2=$(date +%s)
    log "Arena: iter${NEXT} vs champion"
    uv run "$EXP_DIR/arena_promote.py" $SIZE_ARGS $TAG_ARGS --iteration "$NEXT" --num_games "$ARENA_GAMES" \
        --baseline_games "$BASE_GAMES" --num_simulations "$ARENA_SIMS" $CPU_FLAG \
        2>&1 | tee "$LOGS/arena-it${NEXT}.log"
    t3=$(date +%s)
    echo "{\"iteration\": $NEXT, \"collect\": $((t1 - t0)), \"train\": $((t2 - t1)), \"arena\": $((t3 - t2))}" \
        > "$EXP_DIR/timing/$TAG/it${NEXT}.json"
done
log "Done. Last trained iter: $((END + 1))"
uv run "$EXP_DIR/analyze.py" "$TAG"
