# Classical Boxes yardstick (ASTRA task 3)

Base: `70558b73a181c3b29a246979d507f335e913e8c1`. This optional engine is selected
explicitly by a `boxes-classic*` agent name. Go and existing agents are unchanged.

| Agent | Search time per move | Probe eligibility | Tables per agent |
|---|---:|---:|---:|
| `boxes-classic` | 1.0 s | ≤26 undrawn | 4 MiB search + 16 MiB solver |
| `boxes-classic-fast` | 0.1 s | ≤26 undrawn | same |
| `boxes-classic-tiny` | 0.02 s | ≤26 undrawn | same |

The C++ search performs iterative-deepening alpha-beta, keeps the last completed
iteration on timeout, orders a remembered move then safe/history-ranked moves, and
uses exact forced-capture/handout and equivalent-edge reductions. Each exact-solver
probe has a cumulative 16,384-node budget in 1,024-node slices, retaining its table
between slices and checking the deadline before and after each one.
Cached exact results are reused; an unfinished probe is never treated as a proof.
All solver and search work is single-threaded per agent, and the binding releases
the GIL. A new game resets its tables to avoid carrying information across arena pairs.

The initial 1,024-node/30-edge setting completed only 5.3% of fresh probes at 30 in
a separate development sample. Lowering the threshold to 26 alone completed 45.0%.
The selected sliced 16,384-node/26-edge setting completed 98.2% at 26, with at least
98.2% at each sampled count from 24 through 26. This calibration uses 1,400
`5x5-chains` positions, seed 73; no held-out games or arena outcomes selected the
setting. It measures fresh-probe coverage, not the hit rate of the engine's shared
table during play. The deadline can stop a sliced probe before its node budget.

The Python reference was written and tested before the C++ port. Simple loony
endings use the existing exact recurrence. Else the evaluation estimates control
from long-chain parity and controlled value, reducing confidence with the number
of junction boxes. This estimate is heuristic: it supplies no bound or pruning
proof to the exact solver. Search stores remaining margin in hundredths of a box;
the current score margin is added only when reporting the final-margin evaluation.

## Reproduce

Machine: Intel Xeon Platinum 8573C, five visible cores/four-core quota, 16 GiB memory
limit, Python 3.12.14, CMake 3.31.10, C++ Release `-O3 -DNDEBUG`. The shared setup
used cached dependencies and a CMake wrapper for the image's bundled libpython and
existing FetchContent source directories; an upstream test's missing `<algorithm>`
include was supplied through `-include algorithm`, without modifying Go sources.

```bash
uv sync && bash scripts/build_cpp.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
uv run ruff check src/alpha_go/boxes/agents.py src/alpha_go/boxes/classic.py tests/test_boxes_classic_reference.py tests/test_boxes_cpp_classic.py
uv run -m mypy src/alpha_go/boxes/agents.py src/alpha_go/boxes/classic.py tests/test_boxes_classic_reference.py tests/test_boxes_cpp_classic.py --follow-imports=silent
TASK=experiments/2026-10-12_12-37-boxes-classical-engine
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
# GAME_DATA_DIR points to the uploaded development games.
uv run "$EXP/region_stats.py" --positions-tag 5x5-chains --min-undrawn 24 --max-undrawn 30 --num-positions 1400 --save-masks --seed 73
cp "$EXP/data/positions-5x5-chains-24-30.txt" "$TASK/data/probe-masks-24-30.txt"
uv run "$TASK/probe_calibration.py"
uv run "$TASK/probe_slices.py"
for B in boxes-ab-d6 boxes-ab-d4-s28 boxes-classic-fast boxes-classic-tiny; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run "$TASK/run_arena.py" --b "$B" --num-games 200 --workers 1 --seed 0 --output "$TASK/data/$B.csv"
done
uv run "$TASK/report.py"  # requires all four complete 200-game matches
```

`run_arena.py` uses the same `_one_game`/`play_game`, random openings, game seeds,
pairing, and `MatchResult` aggregation as `alpha_go.boxes.arena --opening_moves 4
--num_games 200`. It adds a flushed per-game CSV and safe resume after interruption.
Each pair uses the same four-edge opening with colors exchanged. Timing is total
agent move time divided by that agent's move count, not an unweighted mean of games.
Pair confidence intervals are the repository arena's Wilson intervals (ties count
one half). Raw paired margins are also saved for alternative interval calculations.

An initial four-game smoke match uses seed 42 and is excluded from final 200-game
measurements. It used the original 30-edge/1,024-node setting. That setting's
incomplete 116-game depth-6 match is preserved under `data/preliminary-probe30/`
and excluded from the final matches, which restart with the corrected build.
The GPU machine should use `boxes-classic` for the full-strength
yardstick, with 200 paired games and four random opening moves, then compare the
net with and without its exact solver as separate experiments.
