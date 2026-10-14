# Analysis while the 3070 runs

The completed job compares `5x5-exact/iter0.pt` and `iter22.pt` on 100 frozen development
positions from `selfplay-it21`, including both crossed policy/value combinations. Its
startup reported 68 source families. This measures changes over that loop; it does not
isolate the original change from outcome targets to exact targets, since iter0 already
had exact-value training.

This analysis was prepared in a Linux CPU workspace limited to four CPU equivalents
and 16 GiB RAM. No GPU job or new exact-label production run was launched here.

The completed result and paired intervals are in `report.md` and
`data/probe-PCcE6W-{report.json,comparisons.csv,provenance.json}`. The candidate regresses
at 35..36; the 37..38 result is inconclusive. Keep iter0 as the reference.

Fable's CPU policy/value audit has now completed on the uploaded, hash-matched weights.
See `audit-report.md` for interpretation, limitations, sensitivity and the next controlled
target/data experiment. Both policies place about 19.7% probability on optimal moves;
collapsed margin ordering at 35..36 drops from 65.0% to 62.3%. Full summaries and the
downloadable per-position tables are in `data/policy-value-audit-*`. These results do not
identify a causal explanation for MCTS regression or demonstrate a stronger player.

## Completed 3070 job: policy influence on search

The uploaded `astra-policy-temp-XaIrXX` result rejects temperature 2 as a fix on this
development sample. Reference regret rises from 2.633 to 3.551; candidate regret rises
from 3.714 to 3.918. Keep iter0 at temperature 1. The temperature-1 repeat matches every
selected move and exact value from the earlier component probe. Paired results are in
`report.md` and `data/temperature-XaIrXX-{report.json,comparisons.csv,provenance.json}`.

The next task was Fable's CPU-only policy/value audit in PR17, now completed.
Use the frozen `logs/astra-probe-PCcE6W/reference.pt` (iter0) and `candidate.pt` (iter22)
from the original loop experiment; their hashes are recorded with these results. The
audit ran on all 100 roots / 3,640 child labels in the cloud workspace.
See `audit-report.md` for the result. The command below documents the completed
temperature probe; another identical GPU run is unnecessary.

From the repository root on the user's GPU machine:

```bash
bash experiments/2026-10-14_09-58-boxes-probe-analysis/policy_temperature_probe.sh
```

This runs reference/candidate at temperatures 1 and 2, with the same frozen positions,
400 simulations and lambda 0.5. Temperature 2 flattens the prior distribution while
preserving the greedy policy ranking. Temperature 1 repeats the current setting as a
control. Each run has a one-hour wall-time cap and bounded label solves. Existing exact
labels are reused; new chosen children may still need solving. Inputs remain in the
previous log folder; output goes to a fresh folder and its `results.tgz` also preserves
partial artifacts/logs if a run fails. The probe itself already exists on main at 236a4f2;
the shell command can be pasted directly without adopting any training or solver changes.

Compare both runs on their shared exact-known positions. Temperature is recorded in
`summary.arguments`, not case names. Compare each model's T2-T1 change, then the change
in their gap. This is a development diagnostic; verify any candidate on fresh positions
or full games before changing training defaults. Fable's value-ranking task below remains
independent and useful.

The paired comparison for the two temperature outputs is:

```bash
uv run python experiments/2026-10-14_09-58-boxes-probe-analysis/compare_temperatures.py \
  --baseline "$RUN/t1" --candidate "$RUN/t2" --output "$RUN/comparison"
```

It verifies matching frozen positions, checkpoints, code/build/device and all other search
settings. It intersects exact coverage across both runs, then reports each model's T2-T1
effect and the difference between those effects with paired source-family intervals.

## CPU work: exact child diagnostics

The uploaded frozen prefixes allow exact child solves without a checkpoint or GPU.
`child_label_pilot.py` reuses the tested perspective handling and label cache from
`search_probe.py`. It uses one bounded solver per root and preserves partial progress;
only roots with every legal child solved support optimal-move claims. The resulting
positions remain development diagnostics, not a held-out set or a training collection.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 10m \
  uv run python experiments/2026-10-14_09-58-boxes-probe-analysis/child_label_pilot.py \
  --input EXTRACTED/mix --output work/child-pilot \
  --per-band 6 --workers 3 --seconds 360
```

Use `--all-positions --seconds 480` for the whole frozen sample, still under the outer
10-minute cap. `--label-cache PATH` can supply a merged exact-only cache from completed
pilot files; assert overlapping remaining-margin entries agree before merging them.
The deadline is checked between finite-budget solves, so an in-flight solve can finish
after it. Bulk per-root cache files stay outside git.

The completed cloud pass is available in `data/child-diagnostic-positions.jsonl` (replay
prefixes plus original `position` indices), `child-diagnostic-children.csv` (every legal
edge and exact final margin for the root mover), `child-diagnostic-roots.csv` (root values,
coverage and uniform-move baselines), and `child-diagnostic-summary.json` (provenance and
timing). All 3,640 children are exact. Two root values are inferred as the maximum over
their complete exact child sets; `root_source` identifies them. These tables are for the
component diagnostic only, not training. Fable can use them without repeating any solve.

For Fable, extend the value-ranking task below with a policy-distribution read: report
entropy, top-move probability, and probability assigned to the entire exact-optimal move
set, for each checkpoint at temperatures 1 and 2. Positive temperature preserves ranking;
changes in optimal-set mass can reveal how search allocation differs. Report exact-optimal
share of legal moves as a uniform-prior baseline. Also report probability assigned to all
win-preserving moves on exactly winning roots, because winning and maximizing margin are
different objectives. Only compute optimal-set metrics on fully solved roots. Use
inference only in Fable's CPU workspace, leave the running 3070
job alone, and do not change training defaults based on this development diagnostic.

## Analyze the completed probe without using the GPU

```bash
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
RUN="$EXP/logs/astra-probe-PCcE6W"
uv run python src/alpha_go/boxes/probe_report.py \
  --input "$RUN/mix" --output "$RUN/analysis"
```

The input must be complete. The output is a fresh directory outside `mix`. The CLI
prints comparisons and writes `report.json` and `comparisons.csv`; this standalone
invocation needs only NumPy and the standard library. It does not load checkpoints,
solve positions, change input files, or promote a model. For a sweep with
multiple reference cases, specify one with `--reference-case reference:s400:lambda0.5`.

Every case uses the same positions with exact root, selected policy-child, and selected
search-child values. Unknown values are excluded, never treated as zero regret. Report
coverage alongside the result: this common subset can favor easier-to-solve positions.
Results are given overall and separately at 35..36 and 37..38 undrawn edges.

The paired bootstrap resamples source families and retains all positions in a selected
family, preserving position-weighted means. Each replicate uses the same family counts
for every case. The 95% percentile intervals are exploratory and not corrected for
multiple comparisons; they do not establish generalisation from training-derived positions.
Negative margin-regret differences and positive winning-position preservation differences
are improvements. Regret is in final score-margin points, not boxes captured.

Interpret the crossed networks before changing training: a better old-policy/new-value
case supports investigating policy degradation; a better new-policy/old-value case
supports investigating value degradation. Both components can interact, and inconclusive
intervals warrant a larger frozen sample or a repeated search before selecting settings.
Any resulting candidate still needs fresh, paired full-game evaluation with frozen opponents.

## Fable: test value ranking within a position

Prepare an isolated diagnostic on main in a new experiment folder. Start with the same
iter0 and iter22 checkpoints. The question is whether they distinguish legal children,
even when root winner accuracy is near the majority-class baseline. That accuracy alone
does not prove constant outputs or absence of useful rankings.

- Freeze a small sample of development/training positions in the 35..36 and 37..38 bands.
  Keep `5x5-mid` and saved arena games out of fitting, calibration, and new training labels.
- Evaluate legal children with each frozen model. Convert probabilities and margins to
  the root mover's perspective, negating/complementing only when the mover changes.
  Include tiny-board tests for a normal turn, retained capture turn, and a double capture.
  For conclusions about search, match its forced-capture collapse and identify children
  settled by the solver; do not attribute unused raw-child predictions to the search.
- Use bounded exact root/child solves, cache completed values, and report coverage. Compare
  checkpoints on identical known children. Only claim best-of-all-legal regret/ranking
  metrics when the needed child set is completely solved. Report ties/undefined ranks.
- Report within-root spread of P(win) and expected margin, pairwise ordering of children
  with distinct exact values, and regret of the move favored by predicted win probability,
  predicted margin, and the actual shaped utility at lambda 0.5. Separate discrimination
  from calibration; include class counts and a constant baseline for root-level metrics.
- Save raw per-child predictions/labels and checkpoint/source hashes. Use fresh outputs
  and preserve inputs. Prepare on CPU; do not launch another job on the user's machine
  during the active probe. If a separate CPU workspace has the data, time a bounded pilot
  there before scaling. No automatic training/default changes or promotion.

## Reproduce the arena-regret concentration read

This uses the previously solved replay CSV from the user's 300-game A/B arena archive,
not new training data. The historical CSV field `regret_boxes` actually stores score-margin
points; the script records that correction in its output.

```bash
uv run python experiments/2026-10-14_09-58-boxes-probe-analysis/regret_concentration.py \
  --positions /path/to/replayed-arena/positions.csv --output work/regret-concentration
```

Committed `data/concentration.json` records the source hash and measured aggregates.
This is hindsight concentration within the four-edge band, not evidence that a label
curriculum will generalise or that the same moves explain every full-game loss.

## Validation

From a configured checkout with the C++ extension built:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
uv run ruff check src/alpha_go/boxes/probe_report.py tests/test_boxes_probe_report.py \
  experiments/2026-10-14_09-58-boxes-probe-analysis/regret_concentration.py
uv run -m mypy --follow-imports=silent src/alpha_go/boxes/probe_report.py \
  tests/test_boxes_probe_report.py \
  experiments/2026-10-14_09-58-boxes-probe-analysis/regret_concentration.py
```
