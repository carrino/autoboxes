#!/usr/bin/env bash
# Local loop: bootstrap -> train iter0 -> [collect-it{N} -> train-it{N+1} -> promote] for N in start..end.
# Usage: bash run_iteration_local.sh <start_iter> <end_iter> [--cpu]
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
START=${1:?Usage: run_iteration_local.sh <start_iter> <end_iter> [--cpu]}
END=${2:?Usage: run_iteration_local.sh <start_iter> <end_iter> [--cpu]}
CPU_FLAG=${3:-}
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/timing" "$GAME_DATA_DIR"

# GPU budgets; the --cpu smoke profile proves the pipeline end to end in a few minutes.
BOOT_GAMES=400; SP_GAMES=200; SP_SIMS=200; SP_WORKERS=8; TRAIN_BUDGET=180; ARENA_GAMES=100; BASE_GAMES=40; ARENA_SIMS=200
if [ "$CPU_FLAG" = "--cpu" ]; then
    BOOT_GAMES=${SMOKE_BOOT_GAMES:-30}; SP_GAMES=${SMOKE_SP_GAMES:-12}; SP_SIMS=${SMOKE_SP_SIMS:-24}; SP_WORKERS=2
    TRAIN_BUDGET=${SMOKE_TRAIN_BUDGET:-20}; ARENA_GAMES=${SMOKE_ARENA_GAMES:-6}; BASE_GAMES=${SMOKE_BASE_GAMES:-4}; ARENA_SIMS=24
fi
DATA="experiments/${EXP_NAME}"
log() { echo; echo "############### $* ###############"; }

if [ ! -f "$EXP_DIR/checkpoints/iter${START}.pt" ]; then
    [ "$START" -eq 0 ] || { echo "ERROR: checkpoints/iter${START}.pt missing" >&2; exit 1; }
    if [ -z "$(ls -A "$GAME_DATA_DIR/$DATA/bootstrap-it0" 2>/dev/null)" ]; then
        log "Bootstrap: collecting $BOOT_GAMES games per matchup without search"
        uv run "$EXP_DIR/pre_collect.py" --num_games "$BOOT_GAMES" 2>&1 | tee "$EXP_DIR/logs/bootstrap.log"
    fi
    echo "${DATA}/bootstrap-it0" > "$EXP_DIR/dataset-it0.txt"
    log "Train iter0 from bootstrap games"
    uv run "$EXP_DIR/train.py" --dataset-txt dataset-it0.txt --iteration 0 --time-budget "$TRAIN_BUDGET" $CPU_FLAG \
        2>&1 | tee "$EXP_DIR/logs/train-it0.log"
    log "Arena: iter0 becomes the first champion"
    uv run "$EXP_DIR/arena_promote.py" --iteration 0 --num_games "$ARENA_GAMES" --baseline_games "$BASE_GAMES" \
        --num_simulations "$ARENA_SIMS" $CPU_FLAG 2>&1 | tee "$EXP_DIR/logs/arena-it0.log"
fi

for ITER in $(seq "$START" "$END"); do
    NEXT=$((ITER + 1))
    log "Iter ${ITER}: self-play with checkpoints/iter${ITER}.pt"
    t0=$(date +%s)
    uv run "$EXP_DIR/run_games.py" --checkpoint "$EXP_DIR/checkpoints/iter${ITER}.pt" \
        --num_games "$SP_GAMES" --num_simulations "$SP_SIMS" --num_workers "$SP_WORKERS" \
        --save-name "${DATA}/selfplay-it${ITER}" --seed "$((ITER * 100000))" $CPU_FLAG \
        2>&1 | tee "$EXP_DIR/logs/collect-it${ITER}.log"
    t1=$(date +%s)
    DS="$EXP_DIR/dataset-it${NEXT}.txt"
    { echo "# ${EXP_NAME} iter${NEXT} dataset (auto-generated): last 4 self-play iterations"
      for K in $(seq $((ITER > 3 ? ITER - 3 : 0)) "$ITER"); do echo "${DATA}/selfplay-it${K}"; done; } > "$DS"
    log "Train iter${NEXT} from $(basename "$DS")"
    uv run "$EXP_DIR/train.py" --dataset-txt "dataset-it${NEXT}.txt" --iteration "$NEXT" \
        --resume-from "$EXP_DIR/checkpoints/iter${ITER}.pt" --time-budget "$TRAIN_BUDGET" $CPU_FLAG \
        2>&1 | tee "$EXP_DIR/logs/train-it${NEXT}.log"
    t2=$(date +%s)
    log "Arena: iter${NEXT} vs champion"
    uv run "$EXP_DIR/arena_promote.py" --iteration "$NEXT" --num_games "$ARENA_GAMES" \
        --baseline_games "$BASE_GAMES" --num_simulations "$ARENA_SIMS" $CPU_FLAG \
        2>&1 | tee "$EXP_DIR/logs/arena-it${NEXT}.log"
    t3=$(date +%s)
    echo "{\"iteration\": $NEXT, \"collect\": $((t1 - t0)), \"train\": $((t2 - t1)), \"arena\": $((t3 - t2))}" \
        > "$EXP_DIR/timing/it${NEXT}.json"
done
log "Done. Last trained iter: $((END + 1))"
uv run "$EXP_DIR/analyze.py"
