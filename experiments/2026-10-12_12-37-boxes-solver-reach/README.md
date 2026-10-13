# Exact solver reach (ASTRA task 1)

Goal: extend the exact 5x5 solver's practical reach without changing its Python API,
values, threading, or default 16 MiB transposition table. Base: `70558b73a181c3b29a246979d507f335e913e8c1`.

Machine: Intel Xeon Platinum 8573C, x86-64; `nproc` = 5, cgroup CPU quota = 4 cores;
16 GiB cgroup memory limit, about 16 GiB available at setup. Python 3.12.14, GCC 14.2.0,
CMake 3.31.10, Release `-O3 -DNDEBUG`. At most four computation workers run together.
Dates and folder names use America/Los_Angeles.

The upload contained 24,474 `5x5-chains` games and 1,714 `5x5-mid` games. Development
uses chains only; mid is reserved for final accuracy/performance evaluation. No
training data or labels are generated from mid.

## Setup and baseline

```bash
git clone https://github.com/carrino/autoboxes
cd autoboxes
git checkout -b astra/solver-reach 70558b73a181c3b29a246979d507f335e913e8c1
uv sync
bash scripts/build_cpp.sh
export GAME_DATA_DIR=/workspace/work/game_data_root
mkdir -p "$GAME_DATA_DIR"
tar xzf boxes-games-slim.tgz -C "$GAME_DATA_DIR"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
src/alpha_go/cpp/build/go_game_test
src/alpha_go/cpp/build/boxes_game_test
src/alpha_go/cpp/build/mcts_test
```

This cloud image had cached Python packages and CMake sources, but its network proxy
was unavailable. The current main files were read through the GitHub connector;
the complete working-tree hash and signed commit object matched GitHub exactly.
`uv sync --frozen --offline --python python3.12` used `/workspace/.uv-cache`.
The build used a local CMake wrapper pointing Python3_LIBRARY at the actual bundled
libpython, FetchContent at the existing pybind11/Catch2 sources, and `-include algorithm`
for an existing upstream MCTS test include omission. These environment workarounds
do not modify repository Go/shared sources. The pre-existing `/workspace/autoboxes`
checkout and its uncommitted changes were left intact.

Baseline checks: 351 passed, 24 skipped; Catch2 143/42/34 assertions in 18/7/12 cases.

## Sampling and measurements

```bash
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
TASK=experiments/2026-10-12_12-37-boxes-solver-reach
for TAG in 5x5-chains 5x5-mid; do
  uv run "$EXP/region_stats.py" --positions-tag "$TAG" --min-undrawn 30 --max-undrawn 34 --num-positions 3000 --save-masks
  cp "$EXP/data/positions-$TAG-30-34.txt" "$TASK/data/"
done
# Keep a copy of the baseline extension before rebuilding candidate code.
mkdir -p /workspace/work/baseline-extension
cp src/alpha_go/cpp/build/alpha_go_cpp*.so /workspace/work/baseline-extension/
PYTHONPATH=/workspace/work/baseline-extension uv run "$TASK/bench.py" "$TASK/data/positions-5x5-chains-30-34.txt" "$TASK/data/baseline-chains.csv" --workers 2 --resume
PYTHONPATH=/workspace/work/baseline-extension uv run "$TASK/bench.py" "$TASK/data/positions-5x5-mid-30-34.txt" "$TASK/data/baseline-mid.csv" --workers 2 --resume
# After each candidate build, select its extension using PYTHONPATH and compare values:
uv run "$TASK/bench.py" "$TASK/data/positions-5x5-chains-30-34.txt" "$TASK/data/candidate-chains.csv" --reference "$TASK/data/baseline-chains.csv" --workers 2 --resume
uv run "$TASK/bench.py" "$TASK/data/positions-5x5-mid-30-34.txt" "$TASK/data/candidate-mid.csv" --reference "$TASK/data/baseline-mid.csv" --workers 2 --resume
PYTHONPATH=/workspace/work/baseline-extension uv run "$EXP/region_stats.py" --positions-tag 5x5-mid --min-undrawn 30 --max-undrawn 32 --solver-nodes 30 32
```

Each measurement uses a new solver for the unlimited solve and for each of the
100,000- and 1,000,000-node probes. Allocation is excluded from solve milliseconds.
Budgeted answers are checked against the unlimited answer. Candidate exact answers
must match the baseline by input index and mask. CSV rows checkpoint independently;
`--resume` validates them against the mask file and skips completed positions.
The initial baseline ran with three workers, then was checkpointed to leave room
for the prescribed four-process labelling pilot.
The remaining baseline used one worker. Candidate 1 used one development worker,
then resumed from 1,695 validated rows with two after the arena finished; its
held-out run used two. Candidate 2 used four workers after labeling completed.
Per-position solve timings and answers survive these resumes unchanged. The
timing comparisons share a machine with the other tasks; use node counts and
budget success rates when assessing search-tree improvements.

The first candidate is commit `b525d22b3a95b5573b0f3483a60847964c73b692`.
After it passed both complete mask files, the second trial applied
`data/candidate2.patch`, rebuilt, ran the full suite and Catch2 binaries, then
measured both files with the same baseline references. It keeps lower and upper
bounds in the existing 16-byte table entry. The fixed selection rule requires
at least a 1.02 geometric-mean speedup over candidate 1 on development positions
and no increase in their total nodes. `data/candidate2-decision.json` records
the calculation. The held-out measurements do not select the candidate.

Predeclared reach criterion: at least 80% settled within 100k nodes, with median
time at most 110% of baseline at 30, on both position files. The full tables remain
the primary result; this operational definition does not turn budget misses into solves.

Profiling: compile `profile.cpp` with `g++ -O3 -pg -std=c++17` and the three Boxes
source files (game, chains, solver), solve the first five development masks with a
fresh solver each, then `gprof <binary> gmon.out`. A larger 100-position instrumented
probe was stopped before completion; it contributes no reported performance numbers.
See `data/profile-flat.txt` and `profile.cpp` for the retained profile. Before applying
candidate edits, the retained profile can be reproduced with:

```bash
mkdir -p work/profile
head -n 5 "$TASK/data/positions-5x5-chains-30-34.txt" > work/profile/five.txt
g++ -O3 -pg -std=c++17 -I src/alpha_go/cpp "$TASK/profile.cpp" \
  src/alpha_go/cpp/boxes/boxes_game.cpp src/alpha_go/cpp/boxes/boxes_chains.cpp \
  src/alpha_go/cpp/boxes/boxes_solver.cpp -o work/profile/baseline-profile
(cd work/profile && ./baseline-profile five.txt > values.txt && gprof baseline-profile gmon.out > profile.txt)
```

Relevant checks after solver edits:

```bash
uv run -m pytest tests/test_boxes_cpp_solver.py tests/test_boxes_solver.py tests/test_boxes_solver_bounds.py tests/test_boxes_forced.py tests/test_boxes_chains.py -q
uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
src/alpha_go/cpp/build/go_game_test
src/alpha_go/cpp/build/boxes_game_test
src/alpha_go/cpp/build/mcts_test
uv run ruff check "$TASK/bench.py" "$TASK/report.py" tests/test_boxes_solver_bounds.py
MYPYPATH=src uv run -m mypy "$TASK/bench.py" "$TASK/report.py" tests/test_boxes_solver_bounds.py --follow-imports=silent
uv run "$TASK/report.py"  # requires all six completed benchmark CSVs and the candidate decision
```
