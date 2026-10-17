# Solver34 learning-path review

Audited parent `ed3fe2c` (PR21 proof-filtered teaching targets), without changing
production code or the active overnight run. Two bounded CPU experiments supplement
the source review. Neither establishes a new model-strength result.

## Findings

### A latent Q-mixing metadata bug

The ordinary shared gameplay writer does not save `root_value_semantics` for
Boxes self-play. The optional trainer `--q-mix` rejects explicitly tagged
`shaped_utility` replay and rejects Q values outside `[0, 1]`, but shaped utility
can also lie inside that interval. Such ordinary self-play silently passes both
guards and is interpreted as a win probability. The mixed-opponent collector
does write this metadata, so the two collection paths differ.

`reproduce_q_semantics.py` uses actual `play_game`, NPZ serialization and
`BoxesDataset`, with a tiny constant-value network on 2x2. Its neural value
distribution is P(margin=-1)=.25 and P(margin=+3)=.75. Thus P(win)=.75 and expected
margin=2; shaping with lambda=.5 adds `.5*tanh(2/6)` to the evaluator's value.
Both games finish normally; `min_undrawn=12` keeps the opening root for a small
dataset reproduction.

| collection | saved opening Q | saved semantics | both trainer guards |
|---|---:|---|---|
| ordinary, unshaped | 0.583333 | absent | accept |
| ordinary, shaped | 0.636919 | absent | accept |

These are search averages, not the network's raw .75. Their difference is caused
by shaping, while the writer supplies no indication of that change of meaning.
`data/shaped-q-training.log` confirms the actual trainer accepted this replay
with `--q-mix .25`, completed a gradient step and saved its checkpoint.

**Current overnight impact: none from this bug. Its `q_mix=0` never uses Q as
a value-training target.** Before trying Q mixing, tag newly collected Q values
and require known probability semantics whenever `q_mix` is nonzero. Metadata
tagging alone cannot repair legacy untagged data: it needs trustworthy source
configuration/provenance or recollection, rather than inferring semantics from
numeric range. Do not interpret this finding as the explanation of previous value-only or
exact-policy experiments. No production fix is included here.

### Forced captures do invoke probes whose results are discarded

`BoxesMCTSAgent.select_move` constructs a solver-equipped search state before
checking its forced prefix. The constructor collapses captures and probes the
resulting state; when a prefix exists, the agent returns its first forced edge
and clears `last_search_result`, even if the probe solved the state.

`forced_probe.py` scans the completed 200-game solver34/50k versus solver30/50k
arena, collecting forced roots with 24–38 undrawn edges whose forced collapse
does not end the game. There were 971 eligible roots. A deterministic hash
shuffle chooses 24 sequences from 24 distinct opening families. Each sequence
starts with a fresh agent/table; that agent retains its table while playing the
forced prefix. Neural inference is forbidden by the experiment, so every
measured call is the actual forced-capture return path.

| measured work | result |
|---|---:|
| forced moves | 36 |
| bounded solver calls | 34 |
| solver nodes | 364,408 |
| exhausted 50k probes | 6 |
| elapsed wall time | 0.438 s |
| repeated calls after the first forced move | 12 |
| nodes across those repeated calls | 24 |

All 12 repeated calls hit the already warmed table and cost two nodes each;
they consumed about 0.00145 s altogether. All six exhausted probes were first
probes. This is real avoidable immediate work, but this sample does **not** show
repeated expensive solves along the same forced sequence. Some first-probe work
also warms the table for the following decision, so skipping it is not a free
end-to-end saving. It is lower priority than the many expensive probes at quiet
MCTS leaves found in the preceding audit.

Timings are descriptive: the cloud CPU was shared with other agents. This is a
cold-start sample from completed arena games, not profiling of the unavailable
overnight learner. Every sampled prefix and per-step counter is saved in
`data/forced-probes.jsonl`.

## Path checks without a new bug

- `stop_when_solved` keeps the solved root's board, optimal-action policy and
  mover-perspective exact outcome before ending the game. It does not discard
  that teaching endpoint.
- Extra turns use the stored mover, rather than alternating row parity. The
  exact child labels convert to player-one final margin and remain valid after
  forced collapse; exhaustive tiny-board tests include retained and double
  captures and terminal children.
- Live policy preparation overrides the selected original row. It does not
  retain the old conflicting policy as a second copy of that row. Complete child
  groups are required; their collapsed value-only rows mask the policy loss.
- Source/child replay-family identities propagate through the stable split.
- Proof filtering teaches the temperature-one visit support, independent of
  temperature-zero actual move selection. Solved roots already teach a uniform
  distribution over their exact optimal set.
- The timed solver34 loop continues the latest model weights. Optimizer and
  learning-rate schedule still restart per iteration, as already documented.

Focused verification: 81 tests passed covering proof targets, policy A/B and
live preparation, masks, augmentation, the NN agent and remaining-margin/Q-mix
guards. Two additional solved-termination tests passed. Ruff passes for this
experiment. These checks do not replace a matched GPU learning experiment.

The exact focused command (81 passed in 24.31 s) was:

```bash
PYTHONPATH=src:/workspace/work/cpp-current OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/autoboxes/.venv/bin/python -m pytest \
  tests/test_boxes_proof_policy_targets.py tests/test_boxes_policy_ab_data.py \
  tests/test_boxes_policy_loop_data.py tests/test_boxes_policy_mask.py \
  tests/test_boxes_augment.py tests/test_boxes_nn_agent.py \
  tests/test_boxes_remaining_margin.py -q
```

The same environment ran
`tests/test_boxes_gameplay.py::TestSolvedTermination -q` (2 passed in 1.64 s).

## Reproduction

Use the current extension, not an older installed `alpha_go_cpp`. Substitute
your Python/extension paths below. The profiling extension is built with the
preceding audit's `instrument.py`; it changes counters only.

```bash
EXP=experiments/2026-10-17_10-35-boxes-learning-path-audit
PYTHONPATH=src:/workspace/work/cpp-current OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/autoboxes/.venv/bin/python "$EXP/reproduce_q_semantics.py" \
  --output work/q-semantics-fresh

PYTHONPATH=src:/workspace/work/solver-profile-extension OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/autoboxes/.venv/bin/python "$EXP/forced_probe.py" \
  --games /workspace/work/solver34-50k-vs30-UV94Z4/games.jsonl \
  --output work/forced-probes.jsonl
```

To reproduce actual trainer acceptance, put the absolute path to the generated
`shaped` directory in a fresh manifest, then run the existing loop's `train.py`
with `--rows 2 --cols 2 --dataset-txt ABSOLUTE_MANIFEST --tag FRESH_TAG
--iteration 0 --cpu --channels 8 --n-blocks 1 --features basic --q-mix .25
--time-budget 0 --max-epochs 1 --batch-size 1 --min-undrawn 12 --val-fraction 0
--seed 20261030 --split-seed 0`.
