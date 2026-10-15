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
SOLVER_N=${SOLVER_N:-0}  # exact endgame solver at <= N undrawn edges in self-play and arena (0 = off)
SOLVER_BUDGET=${SOLVER_BUDGET:-20000}  # nodes the solver may spend per position before the search takes over
ARENA_SOLVER_N=${ARENA_SOLVER_N:-$SOLVER_N}  # solver depth in the arena only; 0 makes promotion compare the nets
                                            # themselves (with the solver on, two nets of any quality tie)
BASELINES=${BASELINES:-boxes-greedy,boxes-ab-d4}  # arena absolute-scale opponents
MERGE_EQ=${MERGE_EQ:-0}  # 1: one search action per independent chain or loop (equivalent edges)
SP_PROCS=${SP_PROCS:-1}  # self-play processes per iteration; each gets SP_WORKERS/SP_PROCS threads and
                         # its own GPU engine, which sidesteps the interpreter lock across threads
WINDOW=${WINDOW:-4}  # self-play iterations in each training set (replay window)
TRAIN_EPOCHS=${TRAIN_EPOCHS:-20}  # cap on passes over the window per iteration (the time budget usually binds first)
TRAIN_ARGS=${TRAIN_ARGS:-}  # extra train.py flags, e.g. "--q-mix 1.0" or "--channels 192 --n-blocks 12"
TRAIN_MIN_UNDRAWN=${TRAIN_MIN_UNDRAWN:-0}  # train only on positions with at least this many undrawn edges
                                           # (e.g. SOLVER_N - 4: the solver owns the rest in play)
FEATURES=${FEATURES:-basic}  # net input planes: basic (11) or chains (+10 chain / loop planes, PLAN 4.2);
                             # fixed per checkpoint, so a chains run starts from iter0
EXTRA_DATA=${EXTRA_DATA:-}  # extra dataset directories (under $GAME_DATA_DIR, space-separated) added to every
                            # iteration's training set, e.g. solver-labelled midgame positions from solver_label.py
START_FROM=${START_FROM:-}  # branch self-play games from stored ones (a games directory under $GAME_DATA_DIR)
START_UNDRAWN=${START_UNDRAWN:-36 40}  # ... cut where this many edges remain undrawn (min max)
START_FRACTION=${START_FRACTION:-0.75}  # share of games that branch; the rest open from the empty board
STOP_WHEN_SOLVED=${STOP_WHEN_SOLVED:-0}  # 1: a self-play game ends with the exact outcome once the solver settles it
GATE=${GATE:-0}  # 1: self-play with the arena champion instead of the latest checkpoint (AlphaGo Zero's gating);
                 # training still continues from the latest, so a regressed net stops generating the data
RELABEL_N=${RELABEL_N:-0}  # > 0: after self-play, give every position with <= N undrawn edges its exact final
                           # margin from the solver as the value target (relabel_exact.py), replacing the outcome
RELABEL_WORKERS=${RELABEL_WORKERS:-4}
POLICY_LABELS=${POLICY_LABELS:-0}  # 1: label fresh mistakes before training; replay exact policies + child values
POLICY_MIN_UNDRAWN=${POLICY_MIN_UNDRAWN:-35}; POLICY_MAX_UNDRAWN=${POLICY_MAX_UNDRAWN:-38}
POLICY_MAX_PER_GAME=${POLICY_MAX_PER_GAME:-2}; POLICY_WORKERS=${POLICY_WORKERS:-$RELABEL_WORKERS}
POLICY_NODE_BUDGET=${POLICY_NODE_BUDGET:-20000000}
POLICY_CHILD_BUDGET=${POLICY_CHILD_BUDGET:-$POLICY_NODE_BUDGET}
POLICY_TIMEOUT=${POLICY_TIMEOUT:-2h}  # completed game labels survive this wall-time cap
POLICY_EXCLUDE_POSITIONS=${POLICY_EXCLUDE_POSITIONS:-}  # optional frozen diagnostic JSONL
SEARCH_ARGS=${SEARCH_ARGS:-}  # extra search flags for self-play and the arena, e.g.
                              # "--policy_temperature 0.7 --margin_utility_lambda 0.5" (see nn_agent.add_search_flags)
SIZE_ARGS="--rows $ROWS --cols $COLS"
TAG_ARGS="--tag $TAG"
CKPT="$EXP_DIR/checkpoints/$TAG"
LOGS="$EXP_DIR/logs/$TAG"
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
mkdir -p "$CKPT" "$LOGS" "$EXP_DIR/timing/$TAG" "$GAME_DATA_DIR"
if [ "$POLICY_LABELS" = "1" ]; then
    exec 9> "$LOGS/loop.lock"
    flock -n 9 || { echo "Another loop is running for $TAG" >&2; exit 1; }
fi
PIDS=()
cleanup_collectors() {
    for PID in "${PIDS[@]}"; do kill -TERM -- "-$PID" 2>/dev/null || true; done
    for PID in "${PIDS[@]}"; do wait "$PID" 2>/dev/null || true; done
}
trap cleanup_collectors EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
if [ "$POLICY_LABELS" = "1" ]; then
    [ "$SP_PROCS" -gt 0 ] && [ "$SP_GAMES" -gt 0 ] && [ "$((SP_GAMES % SP_PROCS))" -eq 0 ] || {
        echo "POLICY_LABELS requires positive SP_GAMES divisible by positive SP_PROCS" >&2; exit 1; }
    [ -z "$POLICY_EXCLUDE_POSITIONS" ] || [ -f "$POLICY_EXCLUDE_POSITIONS" ] || {
        echo "Missing POLICY_EXCLUDE_POSITIONS: $POLICY_EXCLUDE_POSITIONS" >&2; exit 1; }
fi
echo "[$TAG] budgets: bootstrap $BOOT_GAMES/matchup, self-play $SP_GAMES games x $SP_SIMS sims ($SP_WORKERS workers), "\
     "train ${TRAIN_BUDGET}s / <= $TRAIN_EPOCHS epochs over the last $WINDOW iterations (>= $TRAIN_MIN_UNDRAWN undrawn), "\
     "arena $ARENA_GAMES vs champion + $BASE_GAMES vs each baseline at $ARENA_SIMS sims, "\
     "solver N=$SOLVER_N (budget $SOLVER_BUDGET nodes; arena N=$ARENA_SOLVER_N), merge equivalent $MERGE_EQ, relabel from $RELABEL_N, "\
     "self-play processes $SP_PROCS, "\
     "search args '$SEARCH_ARGS', "\
     "features $FEATURES, train args '$TRAIN_ARGS', extra data '$EXTRA_DATA', branch from '$START_FROM' at $START_UNDRAWN undrawn "\
     "($START_FRACTION of games), stop when solved $STOP_WHEN_SOLVED, self-play from the champion $GATE"
if [ "$POLICY_LABELS" = "1" ]; then
    echo "[$TAG] exact move targets: $POLICY_MIN_UNDRAWN..$POLICY_MAX_UNDRAWN undrawn, "\
         "at most $POLICY_MAX_PER_GAME mistakes/game, $POLICY_WORKERS CPU workers, "\
         "budgets $POLICY_NODE_BUDGET/$POLICY_CHILD_BUDGET, cap $POLICY_TIMEOUT"
fi
BRANCH_ARGS=""
if [ -n "$START_FROM" ]; then
    set -- $START_UNDRAWN
    BRANCH_ARGS="--start-positions $START_FROM --start-undrawn-min $1 --start-undrawn-max $2 --start-fraction $START_FRACTION"
fi
[ "$STOP_WHEN_SOLVED" = "1" ] && BRANCH_ARGS="$BRANCH_ARGS --stop-when-solved"
DATA="experiments/${EXP_NAME}/${TAG}"
log() { echo; echo "############### [$TAG] $* ###############"; }

if [ ! -f "$CKPT/iter${START}.pt" ]; then
    [ "$START" -eq 0 ] || { echo "ERROR: $CKPT/iter${START}.pt missing" >&2; exit 1; }
    if [ -z "$(ls -A "$GAME_DATA_DIR/$DATA/bootstrap-it0" 2>/dev/null)" ]; then
        log "Bootstrap: collecting $BOOT_GAMES games per matchup without search"
        uv run "$EXP_DIR/pre_collect.py" $SIZE_ARGS --num_games "$BOOT_GAMES" \
            --save-name "${DATA}/bootstrap-it0" 2>&1 | tee "$LOGS/bootstrap.log"
    fi
    { echo "${DATA}/bootstrap-it0"; for X in $EXTRA_DATA; do echo "$X"; done; } > "$EXP_DIR/dataset-${TAG}-it0.txt"
    log "Train iter0 from bootstrap games"
    uv run "$EXP_DIR/train.py" $SIZE_ARGS $TAG_ARGS --dataset-txt "dataset-${TAG}-it0.txt" --iteration 0 \
        --time-budget "$TRAIN_BUDGET" --max-epochs "$TRAIN_EPOCHS" --min-undrawn "$TRAIN_MIN_UNDRAWN" \
        --features "$FEATURES" $TRAIN_ARGS $CPU_FLAG 2>&1 | tee "$LOGS/train-it0.log"
    log "Arena: iter0 becomes the first champion"
    uv run "$EXP_DIR/arena_promote.py" $SIZE_ARGS $TAG_ARGS --iteration 0 --num_games "$ARENA_GAMES" \
        --baseline_games "$BASE_GAMES" --num_simulations "$ARENA_SIMS" --solver_max_undrawn "$ARENA_SOLVER_N" --solver_node_budget "$SOLVER_BUDGET" \
        --baselines "$BASELINES" --merge_equivalent "$MERGE_EQ" $SEARCH_ARGS $CPU_FLAG 2>&1 | tee "$LOGS/arena-it0.log"
fi

for ITER in $(seq "$START" "$END"); do
    NEXT=$((ITER + 1))
    if [ "$POLICY_LABELS" = "1" ]; then
        ARENA_DONE=$(uv run "$EXP_DIR/arena_completed.py" --state "$EXP_DIR/league_state-${TAG}.json" \
            --checkpoint "$CKPT/iter${NEXT}.pt" --iteration "$NEXT" \
            --timing "$EXP_DIR/timing/$TAG/it${NEXT}.json")
        if [ "$ARENA_DONE" = "1" ]; then
            log "Iteration $NEXT already committed to the league; reusing checkpoint and results"
            continue
        fi
    fi
    SP_ITER=$ITER
    [ "$GATE" = "1" ] && SP_ITER=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['champion'])" \
        "$EXP_DIR/league_state-${TAG}.json")
    log "Iter ${ITER}: self-play with $CKPT/iter${SP_ITER}.pt"
    t0=$(date +%s)
    # SP_PROCS processes play disjoint game-index ranges into the same directory (NPZ names carry
    # the game index), each with its own seed, threads and GPU engine.
    PER_PROC=$((SP_GAMES / SP_PROCS)); THREADS=$(( SP_WORKERS / SP_PROCS > 0 ? SP_WORKERS / SP_PROCS : 1 ))
    COLLECT_NAME="${DATA}/selfplay-it${ITER}"
    COLLECT_ACTION=collect
    if [ "$POLICY_LABELS" = "1" ]; then
        # Publish a full collection atomically; restarting the labeller reuses these games.
        COLLECT_SIGNATURE="$SIZE_ARGS sims=$SP_SIMS workers=$SP_WORKERS procs=$SP_PROCS solver=$SOLVER_N budget=$SOLVER_BUDGET merge=$MERGE_EQ $SEARCH_ARGS $BRANCH_ARGS seed=$ITER $CPU_FLAG"
        CACHE_ARGS=(--games "$GAME_DATA_DIR/$COLLECT_NAME" --checkpoint "$CKPT/iter${SP_ITER}.pt"
                    --signature "$COLLECT_SIGNATURE" --num-games "$SP_GAMES" --archive "$LOGS")
        COLLECT_ACTION=$(uv run "$EXP_DIR/collection_cache.py" begin "${CACHE_ARGS[@]}")
        COLLECT_NAME="${COLLECT_NAME}.collecting"
    fi
    if [ "$COLLECT_ACTION" = "collect" ]; then
      PIDS=()
      for P in $(seq 0 $((SP_PROCS - 1))); do
        setsid uv run "$EXP_DIR/run_games.py" $SIZE_ARGS --checkpoint "$CKPT/iter${SP_ITER}.pt" \
            --num_games "$PER_PROC" --game_index_offset "$((P * PER_PROC))" \
            --num_simulations "$SP_SIMS" --num_workers "$THREADS" --solver_max_undrawn "$SOLVER_N" --solver_node_budget "$SOLVER_BUDGET" \
            --merge_equivalent "$MERGE_EQ" $SEARCH_ARGS $BRANCH_ARGS \
            --save-name "$COLLECT_NAME" --seed "$((ITER * 100000 + P * 1000))" $CPU_FLAG \
            > "$LOGS/collect-it${ITER}-p${P}.log" 2>&1 &
        PIDS+=("$!")
      done
      COLLECT_OK=1
      for PID in "${PIDS[@]}"; do
          if ! wait "$PID"; then COLLECT_OK=0; fi
      done
      PIDS=()
      [ "$COLLECT_OK" = "1" ] || { echo "A self-play worker failed; training stopped" >&2; exit 1; }
      tail -n 4 "$LOGS"/collect-it${ITER}-p*.log
      if [ "$POLICY_LABELS" = "1" ]; then
          uv run "$EXP_DIR/collection_cache.py" finish "${CACHE_ARGS[@]}"
      fi
    else
      log "Reusing completed self-play iteration $ITER"
    fi
    if [ "$RELABEL_N" -gt 0 ]; then
      VALUE_DONE="$LOGS/relabel-it${ITER}.done"
      if [ "$POLICY_LABELS" = "1" ] && [ -f "$VALUE_DONE" ]; then
        [ "$(cat "$VALUE_DONE")" = "$RELABEL_N" ] || {
            echo "RELABEL_N changed for cached iteration $ITER; use a fresh TAG" >&2; exit 1; }
        log "Reusing exact value labels for iteration $ITER"
      else
        log "Relabel iter${ITER}: exact values from $RELABEL_N undrawn edges"
        uv run "$EXP_DIR/relabel_exact.py" --games "$GAME_DATA_DIR/${DATA}/selfplay-it${ITER}" --rows "$ROWS" \
            --cols "$COLS" --max-undrawn "$RELABEL_N" --num-workers "$RELABEL_WORKERS" 2>&1 \
            | tee "$LOGS/relabel-it${ITER}.log"
        if [ "$POLICY_LABELS" = "1" ]; then echo "$RELABEL_N" > "$VALUE_DONE"; fi
      fi
    fi
    if [ "$POLICY_LABELS" = "1" ]; then
        log "Teach iter${ITER} mistakes: exact move targets and successor values"
        POLICY_ARGS=()
        if [ -n "$POLICY_EXCLUDE_POSITIONS" ]; then
            POLICY_ARGS+=(--exclude-positions "$POLICY_EXCLUDE_POSITIONS")
        fi
        timeout --signal=TERM --kill-after=30s "$POLICY_TIMEOUT" \
          uv run python -u -m alpha_go.boxes.policy_loop_data \
            --games "$GAME_DATA_DIR/${DATA}/selfplay-it${ITER}" \
            --output "$GAME_DATA_DIR/${DATA}/policy-it${ITER}" \
            --rows "$ROWS" --cols "$COLS" --min-undrawn "$POLICY_MIN_UNDRAWN" \
            --max-undrawn "$POLICY_MAX_UNDRAWN" --max-per-game "$POLICY_MAX_PER_GAME" \
            --node-budget "$POLICY_NODE_BUDGET" --child-node-budget "$POLICY_CHILD_BUDGET" \
            --workers "$POLICY_WORKERS" "${POLICY_ARGS[@]}" \
            2>&1 | tee "$LOGS/policy-it${ITER}.log"
    fi
    t1=$(date +%s)
    DS="$EXP_DIR/dataset-${TAG}-it${NEXT}.txt"
    { echo "# ${EXP_NAME} ${TAG} iter${NEXT} dataset (auto-generated): last $WINDOW self-play iterations"
      for K in $(seq $((ITER - WINDOW + 1 > 0 ? ITER - WINDOW + 1 : 0)) "$ITER"); do
          if [ "$POLICY_LABELS" = "1" ] && [ -f "$GAME_DATA_DIR/${DATA}/policy-it${K}/dataset.txt" ]; then
              while IFS= read -r DIRECTORY; do
                  [ -z "$DIRECTORY" ] || echo "${DATA}/policy-it${K}/$DIRECTORY"
              done < "$GAME_DATA_DIR/${DATA}/policy-it${K}/dataset.txt"
          else
              echo "${DATA}/selfplay-it${K}"
          fi
      done
      for X in $EXTRA_DATA; do echo "$X"; done; } > "$DS"
    log "Train iter${NEXT} from $(basename "$DS")"
    uv run "$EXP_DIR/train.py" $SIZE_ARGS $TAG_ARGS --dataset-txt "$(basename "$DS")" --iteration "$NEXT" \
        --resume-from "$CKPT/iter${ITER}.pt" --time-budget "$TRAIN_BUDGET" --max-epochs "$TRAIN_EPOCHS" \
        --min-undrawn "$TRAIN_MIN_UNDRAWN" --features "$FEATURES" $TRAIN_ARGS $CPU_FLAG \
        2>&1 | tee "$LOGS/train-it${NEXT}.log"
    t2=$(date +%s)
    log "Arena: iter${NEXT} vs champion"
    uv run "$EXP_DIR/arena_promote.py" $SIZE_ARGS $TAG_ARGS --iteration "$NEXT" --num_games "$ARENA_GAMES" \
        --baseline_games "$BASE_GAMES" --num_simulations "$ARENA_SIMS" $CPU_FLAG \
        --solver_max_undrawn "$ARENA_SOLVER_N" --solver_node_budget "$SOLVER_BUDGET" --baselines "$BASELINES" --merge_equivalent "$MERGE_EQ" $SEARCH_ARGS \
        2>&1 | tee "$LOGS/arena-it${NEXT}.log"
    t3=$(date +%s)
    echo "{\"iteration\": $NEXT, \"collect\": $((t1 - t0)), \"train\": $((t2 - t1)), \"arena\": $((t3 - t2))}" \
        > "$EXP_DIR/timing/$TAG/it${NEXT}.tmp"
    mv "$EXP_DIR/timing/$TAG/it${NEXT}.tmp" "$EXP_DIR/timing/$TAG/it${NEXT}.json"
done
log "Done. Last trained iter: $((END + 1))"
uv run "$EXP_DIR/analyze.py" "$TAG"
