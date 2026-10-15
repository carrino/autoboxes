# Matched exact-policy training pilot

The iter0/iter22 audit found weak move priors and no useful improvement in child
ordering after exact-value training. This experiment asks whether exact move targets
can improve the prior on fresh games. It does not restart the league or promote a model.
There are no 5x5 training or strength results yet.

## Overnight run with a fixed iter0 comparison

After the external Windows comparison has finished, update the Windows Git mirror
and pull `astra/policy-target-ab` in WSL. From the repository root:

```bash
ITERATIONS=3 FINAL_GAMES=200 \
  bash experiments/2026-10-14_17-34-boxes-policy-target-ab/run_loop.sh
```

The training recipe is the integrated loop below: frozen iter0 initialization,
three batches of 100 fresh full games at 1600 simulations, exact move supervision
at selected mistakes with 35–36 undrawn edges, successor value targets, natural
loss weights and four training epochs per iteration. Ordinary trajectory examples
remain in each replay window. The first window contains only the first new batch;
older replay archives are not automatically imported. This is a bounded first test
of the new targets, not an established improvement over iter0.

After the three iterations, the script runs **200 actual games of iteration 3
against frozen iter0**: 100 random four-edge openings, each played with sides
swapped, seed 20261016. Both models use 400 simulations, solver reach 30 and the
same search settings. The final checkpoint is fixed before reading these results;
the loop's champion selection does not change this opponent or candidate. Match
output includes game wins, a paired label-swap p-value, and the legacy pair score
based on summed margins. Use the paired game-score test when assessing a win-rate
gain. These tests do not establish empty-board strength or external-bot performance.

The regular per-iteration arena remains a small progress check (60 games against
the current champion, 20 against classic). The final 200-game comparison is the
main overnight read against iter0. It does not promote or replace any existing
run's champion, and all intermediate checkpoints remain available.

Allow several hours. The actual cost of this new 5x5 labelling stage is unmeasured:
each iteration has a two-hour labelling cap, and the final match has a 90-minute
cap. These are stop limits, not predicted durations. A timeout stops the run and
packages diagnostics. Use the printed `resume.sh` to continue: completed training
iterations are reused, and an unfinished final match restarts without retraining.
A finished final match is reused only after checking checkpoint, settings and
result hashes. Its moves, log and checksums are included in the same `results.tgz`.

`FINAL_GAMES` defaults to zero for the original training-only workflow. Optional
`FINAL_SEED`, `FINAL_WORKERS` (default 8) and `FINAL_TIMEOUT` (default 90m) are saved
with the run. Use a new run tag when changing the experiment settings.

## Integrated self-play loop

The same teacher can now run automatically after every fresh self-play batch:

```
latest model -> new games -> exact move/child targets -> replay -> training -> arena
      ^                                                            |
      +------------------------------------------------------------+
```

After pulling `astra/policy-target-ab` through the usual Windows-to-WSL mirror, run
this from the repository root:

```bash
bash experiments/2026-10-14_17-34-boxes-policy-target-ab/run_loop.sh
```

This creates a fresh `5x5-policy-loop-XXXXXX` tag from the surviving iter0 snapshot
and runs **three iterations**, each with 100 new full games at 1600 simulations,
four collection processes, and four CPU labelling workers. The first run labels
35–36 undrawn edges, at most two mistakes per game, then trains for four epochs.
It keeps normal full-game examples in the replay window and adds the collapsed
value-only children. Exact policy targets replace the selected root targets; the raw
copies are not added again. There is no offline pilot or old-game archive prerequisite.

The loop reports training/validation probability mass on exact-optimal moves, top-move
optimality and source-family counts. These are replay diagnostics, not evidence of
playing strength. Arena games still compare with the champion and classic. `GATE=0`
uses the latest trained model for the next batch; the independent champion/promotion
record stays in place. We stop after three iterations to inspect the learning signal
before increasing the budget. BoxesZero matches remain a separate external test.

The wrapper deliberately uses `RELABEL_N=0`: selected roots and children already get
exact values, and this avoids a second blanket value-solving pass. The underlying loop
can combine `RELABEL_N` and `POLICY_LABELS` if requested. Unselected trajectory rows
keep their existing values and search policies. This is a training recipe experiment,
not the policy-only A/B contrast described below.

Each game has bounded root/child solves (20 million nodes per call); each iteration's
labelling stage has a two-hour cap. There is no unbounded solve. If any child remains
unknown, that root receives no replacement policy or child group. Ordinary game rows
remain available. Coverage is printed; a batch with no complete policies does not prove
that policy learning occurred.

The command prints a `resume.sh` and a `results.tgz` path. Send back the results archive.
To resume the same run after interruption, run its printed `resume.sh`; rerunning
`run_loop.sh` intentionally creates a new tag. The resume script skips completed
iterations. During an unfinished iteration, it reuses completed self-play and verified
per-game labels. An interrupted collection is preserved under the run's logs and replaced
with a new complete collection; interrupted labelling only repeats unfinished games.
Changing the checkpoint, collection settings, source arrays, label settings or labeller
code is detected. Use a new tag for a different experiment. Training/arena stages can be
repeated while the arena result is uncommitted. A committed arena result is verified
against the checkpoint hash and reused even if its timing write was interrupted, avoiding
duplicate promotions or candidate-versus-itself matches. Collection process groups are
terminated and reaped on interruption; a run lock prevents two loops writing the same tag.

For an existing compatible run, the underlying switch is `POLICY_LABELS=1` on
`run_iteration_local.sh`. Settings include `POLICY_MIN_UNDRAWN`, `POLICY_MAX_UNDRAWN`,
`POLICY_MAX_PER_GAME`, `POLICY_WORKERS`, `POLICY_NODE_BUDGET`, `POLICY_CHILD_BUDGET`,
`POLICY_TIMEOUT`, and optional `POLICY_EXCLUDE_POSITIONS`. Completed collections from
this mode have receipts; it will not silently adopt an existing unverified raw collection.
The default loop remains unchanged when `POLICY_LABELS=0`.

For a shorter first run, set `ITERATIONS=1` before the wrapper. `SP_GAMES` can be changed
but must be divisible by `SP_PROCS`. No C++ rebuild is needed for these Python changes.

## The contrast

Both arms start from the frozen iter0 checkpoint and see identical positions, exact
values, family splits, update counts, augmentation, optimizer and schedule. Both keep
the ordinary replay window, the fresh full-game rows, and exact value-only child rows.
Only selected root policy targets differ:

| Arm | Move targets at selected roots |
|---|---|
| control | Original MCTS visits |
| exact | Uniform over all exact-optimal legal moves |

Preparation scans fresh games at 35–38 undrawn edges and selects at most two consequential
learner mistakes per game. A mistake must have positive original search visits and lose
at least two final score-margin points. It solves every legal child with a node budget.
If any child is unknown, the entire root policy/child training group is omitted. The
coverage tables retain those failures so hard positions are not silently counted as solved.

Known child values are stored after forced captures have collapsed, matching the states
search evaluates. Their final P1 margin is unchanged by that collapse; the dataset converts
it to the current mover's perspective. Children have no searched policy, so `policy_mask=False`
gives them value loss only. No played-move fallback or fake uniform policy is introduced.
This shared child augmentation is controlled between arms; the experiment does not isolate
its benefit against the previous loop.

Natural value weights, q-mix zero, two epochs, batch size 256, AdamW at 0.001 and the
loop's capped warmup/cosine schedule are fixed. The GPU budget is updates, not elapsed time.
The driver checks the batch/augmentation/LR trace matches across arms. CUDA bitwise
determinism is not promised.

## Run after the 20-game collection pilot finishes

Fetch this branch through the normal Windows-to-WSL mirror workflow. These changes are
Python-only; an already working C++ extension does not need rebuilding. From the repository
root, set `PILOT` to the **actual directory printed by the collection command**:

```bash
PILOT="$GAME_DATA_DIR/experiments/2026-10-08_14-42-boxes-3x3-loop/move-target-pilot-XXXXXX"
bash experiments/2026-10-14_17-34-boxes-policy-target-ab/run_pilot.sh "$PILOT/games"
```

Replace `XXXXXX`; do not use a wildcard that could combine multiple runs. If GAME_DATA_DIR
is not already exported, first run:

```bash
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
```

The script uses the existing `dataset-5x5-exact-it22.txt` replay manifest. An optional second
argument selects a different explicit manifest. It uses the surviving frozen
`logs/astra-probe-PCcE6W/reference.pt`, not the missing original iter0 path. It excludes
families from that probe's `mix/positions.jsonl` from both fresh and replay data. It rejects
known `5x5-mid` paths. Family exclusion includes canonical opening prefixes and exact root
states, and propagates across declared game pairs; it cannot identify every transposition.

The labelling stage uses four CPU workers with 20-million-node budgets per root/child
solve and a two-hour overall cap. Training then runs sequentially on the 3070 with a
30-minute cap. These are limits, not runtime predictions: the pilot measures actual cost
and complete-root coverage. The script packages logs and tables even if a stage fails.
It prints the `results.tgz` to send back; datasets and checkpoints stay local.

For a collection that has not finished, wait for `collection.json` to report `complete`.
Do not point preparation at a directory another process is still writing.

## Reuse prepared labels

The output contains `prepared/` (portable manifests, frozen NPZs, child/root CSVs and
hashes) and `train/` (`reference.pt`, `control.pt`, `exact.pt`, per-arm JSON and `summary.json`).
The input files remain unchanged. A training retry reuses preparation directly:

```bash
uv run python -u -m alpha_go.boxes.policy_target_ab \
  --reference "$RUN/reference.pt" --prepared "$RUN/prepared" \
  --output "$RUN/train-retry" --epochs 2 --batch-size 256 \
  --min-undrawn 24 --seed 20261015 --split-seed 0
```

Set RUN to the printed results directory. Each output must be new. Interrupted preparation
keeps completed batches for diagnosis but is not automatically resumable; review its
coverage and timeout before repeating solver work. A completed preparation is reusable.

## What determines the next step

First read complete roots and distinct families, then exact-optimal prior mass and top-move
hit rate on training and validation roots. Both models are scored against the same exact
targets. Overall value/CE metrics also include outcome-labelled replay rows and are
diagnostic; the optimal-set metrics include only completely labelled roots. Children inherit
their parent's split. Warm-start weights may have seen replay games, so validation is not
an untouched generalisation test.

- Too few complete roots or held-out families: increase diverse collection or reduce solve
  reach based on measured cost. Twenty games are an integration pilot, not a strength test.
- Training optimal mass fails to improve: check fitting/gradient behaviour on this fixed
  data before spending more on collection or architecture.
- Training improves but held-out roots do not: increase source diversity and keep the same
  targets. Do not compensate by heavily weighting a handful of roots.
- Held-out move quality improves: test equal-budget search regret on fresh diagnostic
  positions, then paired full-game matches against iter0, classic and BoxesZero. Report
  equal-time performance too; do not promote from replay metrics alone.

The original plan still has policy-target training, diverse data, controlled child ranking
experiments and external-engine validation available. This pilot addresses the first two.

## Verification

Tests cover exhaustive tiny-board child values across normal turns, retained captures,
double captures and terminal collapses; incomplete-group suppression; lineage/exclusion;
source immutability and portable hashes; policy-mask gradients and legacy compatibility;
and end-to-end tiny training with equal update traces and a common validation target.
Use `--cpu` only for such small smoke runs. Full 5x5 training belongs on the 3070.

Local verification: 620 passed, 29 skipped (`test_gpu_lease.py` excluded as in CI).
The integrated loop is exercised on CPU for two iterations, with a labelling-stage
interruption, source-game reuse, rolling exact/child replay, training metrics, and arena
commit recovery. Separate tests terminate active collectors and exercise the generated
fresh-run/resume wrapper.
The overnight wrapper test also injects a failed final match, verifies resumption
without repeating training, checks fixed candidate/baseline/search arguments,
reuses completed results, and rejects changed result bytes.
Repository CI-scope Ruff passes. Strict mypy with `--follow-imports=silent` passes for
the four changed core modules. The broader CI mypy command reports 112 errors in 12
unchanged imported files in this environment, also reproduced on base commit `236a4f2`.
