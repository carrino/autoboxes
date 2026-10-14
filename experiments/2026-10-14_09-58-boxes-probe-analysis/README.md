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

## Next 3070 job: policy influence on search

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
