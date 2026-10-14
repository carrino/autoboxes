# Implementation and validation

Date: 2026-10-13, America/Los_Angeles. Base: autoboxes main
`bb5653597e4d7e37b05fdcf062fa1cddf901b801`. CPU-only cloud environment, five logical
CPUs, Python 3.12. The existing current C++ extension was reused; this change contains
no C++ modifications. All training defaults remain unchanged.

## Implemented

| Question | Tool/change |
|---|---|
| Did exact training help value but harm policy? | `mixed_evaluator.py` routes outputs from complete networks; `search_probe.py` compares all four old/new combinations. |
| Do better move targets help? | `policy_labels.py` labels consequential mistakes only after every legal child is solved. |
| Does score shaping help win-preserving decisions? | Probe sweeps margin utility, applying it consistently to neural and shaped terminal values. |
| Does more search improve choices? | Probe sweeps simulation budgets on frozen positions, reporting chosen-move regret and time. |
| Does remaining-margin prediction simplify learning? | Optional model target, exact score shift, feasible outcome/parity mask and explicit readout reset. |
| Does opponent diversity expose shared blind spots? | `mixed_collect.py` collects weighted, colour-paired games against self, frozen checkpoints and baselines. |

Two data changes support these experiments. `train_mask` keeps complete mixed-game
trajectories but selects only searched learner rows for training. `source_replay_group`
lets derived exact-policy positions inherit their original source family's stable split.
`exact_fit.py` and exact-target counts respect the selected source indices. Positive
q-mixing is rejected for replay explicitly marked as margin-shaped utility.

## Checks

- Unchanged relevant main code: **438 passed, 24 skipped**.
- Integrated full Python suite: **492 passed, 24 skipped in 61.67 seconds**, excluding
  the repository's known-broken `tests/test_gpu_lease.py`.
- Ruff passed for every changed Python file and all six new test files.
- Strict mypy with `--follow-imports=silent` passed for the seven changed/new core
  modules: model, dataset, validation, mixed evaluator, policy labels, mixed collection
  and search probe.
- All README Bash examples passed `bash -n`; `git diff --check` passed.
- Independent code review and focused re-execution: **54 new tests passed**. Review
  tightened source-directory/replay checks and corrected a sampling fixture.

The full suite command used the installed environment and current solver extension:

```bash
PYTHONPATH=src:/workspace/work/autoboxes-improvement/work/cpp OMP_NUM_THREADS=1 \
  /workspace/autoboxes/.venv/bin/python -m pytest tests/ \
  --ignore=tests/test_gpu_lease.py -q
```

On the GPU machine, the equivalent uses the installed extension:

```bash
uv run python -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
```

Tests include exhaustive tiny-board optimal policies and mover signs, captures that
retain the turn, incomplete-child rejection, masked non-prefix dataset indexing,
source-family inheritance, read-only source/cache inputs, legacy checkpoint loading,
explicit value-target conversion and inherited resume settings. Remaining-margin tests
check mass conservation, odd/even support, gradient equivalence, terminal outcomes,
CPU bf16 loss/backward, and a full-graph `torch.compile` forward using the eager backend.

## CPU smoke experiments

- Targeted policy CLI: four complete 2x2 training games generated through shared
  gameplay; 40 band moves, 22 qualifying mistakes, eight complete policy attempts,
  seven distinct labels saved and one duplicate removed.
- Mixed collector: six complete 2x2 games against self, an old synthetic checkpoint
  and classic; legal replays, correct extra turns, learner-only searched targets,
  paired lineage, and unchanged shaped root Q verified.
- Search probe: actual module CLI completed four policy/value cases on two roots.
  Tests also ran a 16-case objective/search sweep and repeated it with frozen positions
  and the read-only exact cache.

These smoke experiments use small synthetic networks and establish integration only.
No uploaded evaluation games were turned into training examples.

## Interpretation limits and next run

CUDA behavior and real 5x5 playing strength remain unmeasured here. Mixed evaluation
does two forwards and therefore is not an equal-time comparison. Finite node budgets
bound exact search work, not elapsed wall time; incomplete labels remain visible.
The probe's common-known subset must be reported alongside its metrics. C++ search RNG
is independent of the Python seed, and classic's wall-clock search is sensitive to load.

Start with the README's four-way policy/value probe. It uses familiar development
positions, so its job is diagnosis. Select promising changes using these diagnostics,
then confirm on fresh paired complete games with frozen settings. The remaining-margin
A/B must reset both value readouts and is a test of score shifting plus support masking.
No new setting is enabled automatically, and no checkpoint is promoted by these tools.
