# CPU tasks for cloud agents

Self-contained tasks for an agent on a CPU-only cloud box (no GPU) with plenty of cores and
time. The GPU machine keeps training; these run beside it. Each task is independent: give an
agent "clone github.com/carrino/autoboxes, read ASTRA_TASKS.md, do Task N" plus the path of
the uploaded data. Tasks are in priority order.

## Setup (every task)

1. Read `CLAUDE.md` (conventions, mergeability rules), `EXPERIMENTS.md` (sections 5 to 10:
   what has been measured and why) and section 8 of `GETTING_STARTED.md`.
2. Clone, branch from `main` as `astra/<task-slug>`. Never push to `main` or to any other
   branch, never merge.
3. `uv sync && bash scripts/build_cpp.sh`, then `uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q`.
   Stop unless it is green.
4. `export GAME_DATA_DIR=$HOME/autoboxes-data/game_data_root` and unpack the uploaded games
   (`tar xzf boxes-games-slim.tgz -C $GAME_DATA_DIR`, made by `slim_games.py`). You should get
   `experiments/2026-10-08_14-42-boxes-3x3-loop/5x5-chains` and `.../5x5-mid`. If not, stop
   and say so. Below, `EXP=experiments/2026-10-08_14-42-boxes-3x3-loop`.
5. Record `nproc` and free memory. Parallel jobs use at most `nproc - 1` processes and
   about 0.5 GB each.

Rules for every task:

- `5x5-mid` games are the clean evaluation set. Measure on them; never label them, train on
  them or tune heuristics against them. Develop on `5x5-chains` positions.
- Code follows `CLAUDE.md`: Go untouched; Boxes code in Boxes files; shared files
  (`bindings.cpp`, `cpp/mcts/*`, `mcts.py`, ...) only get additive, flag-selected edits; new
  behaviour off by default; a new game rule or exact formula gets a pure-Python reference
  and a property test first; ruff and strict mypy clean on new Python; no try/except.
- One experiment folder per task, `experiments/<TZ=America/Los_Angeles date
  +%Y-%m-%d_%H-%M>-boxes-<slug>/`, self-contained: `README.md` (goal, machine, every
  command), the scripts, `report.md` (results, conclusions, what to do next), `data/*.csv`.
- Finish with a pull request against `main`: what changed, the numbers, how it was tested.

## Task 1: extend the exact solver's reach (highest value)

Why: in the 29..34 band the move quality comes from the search reaching exact leaves. Two
more edges of solver reach (28 -> 30) were worth nine points of optimal moves at equal
time, more than doubling the simulations (EXPERIMENTS.md section 8). Self-play now runs the
solver at 30 undrawn edges with a 100k-node budget. The goal is solver 32 at today's cost
of 30, then further.

1. Benchmark set. Dump real positions with
   `uv run $EXP/region_stats.py --positions-tag 5x5-chains --min-undrawn 30 --max-undrawn 34 --num-positions 3000 --save-masks`
   (development) and the same with `--positions-tag 5x5-mid` (final report only). Each line
   is `undrawn mask`; `BoxesSolver.value(mask)` needs only the mask. Copy both files into the
   task folder.
2. Baseline. Write a bench script over a mask file: per undrawn count, median / p90 / max
   nodes and milliseconds with a fresh solver per position, and the share solved within
   100k and 1M nodes (`value_within`). Record every position's exact value in a CSV: from
   then on every candidate must reproduce all of them exactly. Also run
   `region_stats.py --positions-tag 5x5-mid --min-undrawn 30 --max-undrawn 32 --solver-nodes 30 32`
   and answer: at a fixed undrawn count, does the solver's cost fall when the largest
   independent region is small (EXPERIMENTS.md section 10)? Answered on 2026-10-12: yes,
   five to nine times from unsplit to a largest region of 20 edges. The open question is
   what decomposition could still add: for split positions, time the largest region alone
   (draw every edge of the other regions, then solve) against the whole position. The
   ratio bounds decomposition's gain; build it only if that bound is large where the
   budget binds.
3. Make it faster, one change at a time: build, check exactness on the whole benchmark and
   the existing solver tests (`tests/test_boxes_cpp_solver.py`, `tests/test_boxes_solver.py`,
   the Catch2 binaries), measure, keep or revert, log every attempt with its numbers.
   Starting points, not a prescription: profile where the nodes go first; region-aware
   search (by 32 undrawn the board has usually split into one large region plus small
   pieces: region-wise canonical keys, ordering, or exact handling of the small pieces);
   bounds from Nimstring values and controlled value (Berlekamp, *The Dots and Boxes
   Game*; Winning Ways) that settle positions where control decides; extending the exact
   loony-endgame shortcut (today only when every box has at most two open sides);
   transposition-table replacement; move ordering.
4. Constraints: `BoxesSolver`'s Python API unchanged (a drop-in); one thread per solver
   (self-play runs many solvers at once); memory per solver no larger than today's default;
   values identical to the current solver everywhere.
5. Report the before / after table on both mask files and the largest `SOLVER_N` whose
   positions mostly settle within the 100k budget at about today's cost of 30.

## Task 2: exact labels above the solver's reach

Why: the net's moves matter only above the solver's reach, and its policy there improves
with better targets (EXPERIMENTS.md section 9). The existing labelled set mostly covers 26..30,
where the solver already plays. This produces exact best moves and outcomes at 31..36.

1. Pilot: `nproc - 1` processes of
   `uv run $EXP/solver_label.py --positions-tag 5x5-chains --min-undrawn 31 --max-undrawn 36 --num-positions <100 x processes> --node-budget 20000000 --save-name 5x5-oracle-31-36-pilot --shard K --num-shards N --seed 0`,
   one log each. Record wall time, labelled, skipped on the budget.
2. If more than 20% were skipped, double `--node-budget` and repeat the pilot once.
3. Full run sized from the pilot to about 12 hours, at most 60,000 positions, `--seed 1`,
   `--save-name 5x5-oracle-31-36`.
4. Verify: per-file counts add up to the logs' totals, every policy row sums to 1; report
   positions per undrawn count and the distribution of final margins.
5. `tar czf` the output folder and report its path and size (it stays out of git).

If Task 1 lands first, rerun this with its solver and go higher (36..38). This task is pure
compute: on a box with four cores it yields a few thousand labels a day at most, so give
such a box the engineering tasks (1 and 3) and run labelling where the cores are.

## Task 3: a strong classical engine as a yardstick

Why: every baseline is saturated (the nets win 10-0 against greedy and alpha-beta depth 4
and 6), so nothing measures absolute strength, and the goal is to beat real 5x5 programs.
A classical engine built on chain theory is the closest stand-in we can run.

1. A C++ agent in `src/alpha_go/cpp/boxes/`: iterative-deepening alpha-beta under a time
   limit (about 1 s a move on one core), transposition table, forced captures and
   equivalent-edge merging as the solver does, the exact solver below N undrawn edges, and
   an evaluation from chain theory: score margin plus an estimate of who gets control (long
   chain rule parity) and what control is worth (controlled value). Python reference and
   tests for the evaluation first. Bind it additively in `bindings.cpp`; register
   `boxes-classic` (and a few strengths) in `src/alpha_go/boxes/agents.py`.
2. Measure with the paired arena (`uv run -m alpha_go.boxes.arena --opening_moves 4 --num_games 200`)
   against `boxes-ab-d6`, `boxes-ab-d4-s28` and its own weaker settings. Report pair scores
   with intervals and seconds per move.
3. The GPU machine then plays it against the nets; say in the report which registered name
   and settings to use.

## Task 5: exact labels where the games are decided (35..40 undrawn)

Why: replaying 200 games against the classical engine (EXPERIMENTS.md, "Replay from 38
undrawn") showed the games are decided between 38 and 34 undrawn edges, where the net loses
exact margin on two of every three moves. The 31..36 labels (Task 2) cover the lower half of
that zone. This produces labels for its upper half. Use the merged solver (#7, 1.4x faster).

1. Values along game lines, cheap: `uv run $EXP/arena_exact.py` solves every position of a
   game from 38 down at about 2.4 CPU-seconds each with one solver per game. Write a
   `line_label.py` that does the same for stored `5x5-chains` games (never `5x5-mid`), and
   saves, per game, the exact player-1 final margin of every position from --max-undrawn
   down (the `exact_margin_p1` array of `alpha_go/boxes/relabel.py`, which `BoxesDataset`
   reads) into a copy of the game under a new dataset name, e.g. `5x5-chains-exact40`.
   Report positions labelled by undrawn count and the mean |outcome - exact| by band.
2. Policies, expensive: `solver_label.py --min-undrawn 37 --max-undrawn 40` with the
   pilot / sizing procedure of Task 2 (expect several times Task 2's cost per position;
   measure it in the pilot and size the full run to about 24 hours). Both the value-only set
   and the policy set are useful; deliver the value set first.
3. Archive both outside git; report paths, sizes and hashes as in Task 2.

## Task 6: a stronger yardstick

Why: the nets beat `boxes-classic` two games in three. When they pass it clearly, the
measurement saturates like the earlier baselines did.

1. Register `boxes-classic-5s` (five seconds a move) and `boxes-classic-solver30` (exact
   probes at <= 30 undrawn with a 50k-node budget, the nets' own solver setting) in
   `src/alpha_go/boxes/agents.py`; measure both against `boxes-classic` with the paired
   arena (200 games) and report pair scores and seconds per move.
2. Investigate a stdio bridge to an external 5x5 engine that runs on Linux (PLAN.md §3):
   which engines exist, their licences, and a working prototype that plays one game
   through `alpha_go.boxes.arena` if one is feasible. Windows-only programs are out of
   scope here.

## Task 4: tune the search at solver 30 (needs two checkpoints)

Needs `iter15.pt` and `iter54.pt` (about 12 MB each) uploaded to
`$EXP/checkpoints/5x5-chains/`. Everything runs with `--cpu`.

1. Metric: `uv run $EXP/oracle_eval.py --tag 5x5-chains --positions-tag 5x5-mid --min-undrawn 29 --max-undrawn 34 --solver_max_undrawn 30 --solver_node_budget 100000 --num_simulations 400 --prove_terminals 1 --iterations 54 --cpu`
   plus search flags; the searched-move column (about 0.69 on the GPU machine). 300
   positions: differences under 3 points are noise.
2. Knobs: `--c_puct`, `--policy_temperature`, `--leaf_batch_size`,
   `--margin_utility_lambda` / `--margin_utility_k`, `--prove_terminals`. One change at a
   time, keep or discard, every run logged with its flags and result.
3. Known inconsistency to fix first, behind a flag: with the margin utility on, net leaves
   are scored as win probability plus a margin term while solver-settled leaves score
   1 / 0.5 / 0 (EXPERIMENTS.md section 2). Make settled leaves use the same utility from
   their exact margin, then tune lambda.
4. Report the best settings, and confirm them on iteration 15 as well.
