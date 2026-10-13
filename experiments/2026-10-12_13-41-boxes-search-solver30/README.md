# Task 4: search tuning with solver 30

Goal: make exact and network leaves use the same optional margin utility, tune on
`5x5-chains`, then freeze the settings and evaluate on 300 clean `5x5-mid` positions
with both uploaded checkpoints. The held-out set does not select any settings.

Base: `70558b73a181c3b29a246979d507f335e913e8c1` on `main`. Branch:
`astra/search-solver30`. The initial environment had an older checkout and two local
build portability edits; this task uses an isolated worktree and preserves those edits.
The initial snapshot was fetched through the GitHub connector and verified against
Git blob/tree/commit hashes, then its history was fetched with `git fetch --refetch
origin main` once shell network access was enabled.

Machine: five available AMD EPYC 9V74 cores, about 17 GiB RAM, no CUDA device.
Python 3.12.14, PyTorch 2.11.0+cu130, one Torch/OMP/MKL thread per measurement,
at most four concurrent jobs. Oracle construction uses about 475 MiB RSS per job;
network inference also needs activation memory. See `data/environment.json` for
machine metadata and upload hashes. Data and checkpoints are excluded from Git.

## Build and test

The workspace already supplied the Python dependencies, CMake 3.31.10, and cached
pybind11/Catch2 sources. Its Python library is under the interpreter's `lib/`, and
the existing upstream MCTS test needs `<algorithm>` supplied by a compiler flag.
Use the normal setup elsewhere:

```bash
uv sync
bash scripts/build_cpp.sh
```

Actual offline build used here, from the worktree:

```bash
PYTHON_ROOT=$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')
PYTHON_LIBRARY=$(.venv/bin/python -c 'import sys,sysconfig,pathlib; print(pathlib.Path(sys.base_prefix)/"lib"/sysconfig.get_config_var("LDLIBRARY"))')
/workspace/autoboxes-tools/bin/cmake -S src/alpha_go/cpp -B src/alpha_go/cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/workspace/autoboxes/.venv/bin/python \
  -DPython3_ROOT_DIR="$PYTHON_ROOT" -DPython3_LIBRARY="$PYTHON_LIBRARY" \
  -DCMAKE_CXX_FLAGS='-include algorithm' \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=/workspace/autoboxes-tools/cmake-fetchcontent/pybind11-src \
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=/workspace/autoboxes-tools/cmake-fetchcontent/catch2-src
/workspace/autoboxes-tools/bin/cmake --build src/alpha_go/cpp/build -j4
export PYTHONPATH=src:src/alpha_go/cpp/build
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
src/alpha_go/cpp/build/go_game_test
src/alpha_go/cpp/build/boxes_game_test
src/alpha_go/cpp/build/mcts_test
.venv/bin/ruff check src/alpha_go/boxes tests/test_boxes_utility.py \
  tests/test_boxes_terminal_utility.py tests/test_boxes_nn_agent.py \
  experiments/2026-10-12_13-41-boxes-search-solver30/*.py
.venv/bin/mypy --follow-imports=silent src/alpha_go/boxes \
  tests/test_boxes_terminal_utility.py tests/test_boxes_utility.py \
  experiments/2026-10-12_13-41-boxes-search-solver30/*.py
```

Baseline: 351 passed, 24 skipped. With the fix: 382 passed, 24 skipped.
Catch2: 143 assertions/18 cases (Go), 42/7 (Boxes), 34/12 (MCTS).
Strict mypy follows imports silently to isolate the new code from existing Go and
generated-protobuf errors (112 errors in 12 unchanged files without that option).
The instructions' `src/alpha_go/game.py` path is stale; the game API is a package.

## Data and metric

```bash
export GAME_DATA_DIR=/workspace/work/task4-input
tar xzf boxes-games-slim.tgz -C "$GAME_DATA_DIR"
tar xzf boxes-ckpts.tgz  # checkpoint paths are relative to repository root
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
TASK=experiments/2026-10-12_13-41-boxes-search-solver30
.venv/bin/python "$TASK/bench.py" sample --tag 5x5-chains --output "$TASK/data/development.json"
# Only after freezing the development settings:
.venv/bin/python "$TASK/bench.py" sample --tag 5x5-mid --output "$TASK/data/heldout.json"
```

The sampler matches `oracle_eval.late_positions`: distinct (mask, player, margin),
no forced-capture prefix, 29..34 undrawn, sorted keys shuffled with seed 0. Stored
move prefixes make each sample reproducible. It includes branched-game `start_moves`.
The development sample contains 300 of 103,589 positions from 24,474 games.

`bench.py evaluate` always uses CPU, 400 simulations, solver 30, 100,000 solver
nodes, a 1,048,576-entry agent table, temperature-zero selection, the same seed per
position as `oracle_eval`, and a fresh agent per configuration. Its searched-move
metric is the share of moves in the exact margin-optimal set. The agent's solver
table persists across positions, just as in the original evaluator. Positions are
not sharded, because sharding changes the table's history.

The development oracle cache is gitignored and only speeds repeated measurements.
Clean oracle values exist only in memory; the held-out CSVs contain chosen moves,
correctness and timings, not training labels. A separate exact solver constructs
reference moves. Its cost is logged separately from the measured agent time.

Each run writes `data/<name>.json` with every flag, command, checkpoint/sample hash,
counts and timing, plus `data/<name>.csv` with one row per position. Logs are under
`logs/`. Initial configurations compare lambda 0, 0.25, 0.5 and 1 at k=6. All other
knobs begin at task-command defaults: c_puct=1.5, policy temperature=1, leaf batch=16,
proof=1. Subsequent candidates change one knob relative to the retained setting.
Treat improvements under 3 percentage points as noise; the largest observed score is retained when multiple candidates clear that threshold. Small differences among those candidates do not establish a precise optimum.
The complete decisions and final settings are in `report.md` and `data/runs.csv`.

## Implementation

`--shape_terminals 1` copies the evaluator's nonnegative lambda and positive k into
the C++ Boxes search. Terminal boards, forced-collapse terminals, solver-settled
leaves, and direct solver results use `win + lambda*tanh(final_margin/k)`.
The pure-Python formula is in `boxes/utility.py`. Lambda zero retains the old search.
Go ignores the new Boxes utility fields, including when explicitly set.

With shaping, one proven win cannot establish the optimal margin: proofs require
every legal child. An unproven root also retains proven losses because losses have
different margins. A fully proven root retains only maximum-utility children, even
when every child has a negative utility. Perspective flips remain `1-u`, including
extra turns after captures. Defaults and the exact solver API are unchanged.


## Development rounds

The first four measurements (baseline, lambda 0.25/0.5/1) use `bench.py evaluate`
with `--prove_terminals 1`; nonzero lambdas additionally use `--shape_terminals 1`.
Every actual command is preserved in the run JSON. The subsequent round commands are:

```bash
.venv/bin/python "$TASK/run_round.py" --parent dev-lambda05 --name round2 \
  --candidates '{"dev-cpuct3":{"c_puct":3.0},"dev-policy07":{"policy_temperature":0.7},"dev-batch4":{"leaf_batch_size":4},"dev-unshaped05":{"shape_terminals":false}}' \
  --diagnostics dev-unshaped05
.venv/bin/python "$TASK/run_round.py" --parent dev-lambda05 --name round3 \
  --candidates '{"dev-k3":{"margin_utility_k":3.0},"dev-k12":{"margin_utility_k":12.0},"dev-proof0":{"prove_terminals":false},"dev-batch8":{"leaf_batch_size":8}}'
.venv/bin/python "$TASK/summarize.py"
```

`run_round.py` enforces one changed flag per candidate and at most four workers.
The inconsistent-terminal ablation is diagnostic only. It cannot become the winner.
`data/round*.json` records each keep/discard decision. `summarize.py` writes the
run ledger, the 31..34 sub-band results, and paired bootstrap intervals on final
held-out comparisons. Intervals resample positions and do not account for
correlation between positions from the same game.


## Frozen validation

```bash
.venv/bin/python "$TASK/validate.py"
.venv/bin/python "$TASK/summarize.py"
```

`validate.py` writes `data/frozen.json` before accessing the held-out games,
checks its sample against the original oracle evaluator, and launches only the
original and frozen settings on iterations 54 and 15. Each is a complete 300-position
run. A rerun refuses a changed frozen configuration. Both development and held-out
sample keys/order matched the original `oracle_eval.late_positions` exactly.

To use the selected configuration with the original evaluation command:

```bash
uv run "$EXP/oracle_eval.py" --tag 5x5-chains --positions-tag 5x5-mid \
  --min-undrawn 29 --max-undrawn 34 --solver_max_undrawn 30 \
  --solver_node_budget 100000 --num_simulations 400 --prove_terminals 1 \
  --iterations 54 15 --cpu --shape_terminals 1 --margin_utility_lambda 0.5 \
  --margin_utility_k 6 --c_puct 1.5 --policy_temperature 1 --leaf_batch_size 16
```
