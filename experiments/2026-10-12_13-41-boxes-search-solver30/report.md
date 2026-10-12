# Task 4: consistent margin utility and solver-30 search tuning

The selected change is `--shape_terminals 1 --margin_utility_lambda 0.5`.
Keep k=6, c_puct=1.5, policy temperature=1, leaf batch=16 and proof propagation on.
The terminal fix is off by default. Search budgets and the solver API are unchanged.

## Final clean evaluation

Settings were frozen in `data/frozen.json` before accessing the `5x5-mid` evaluation
positions. Both checkpoints use the same 300 positions at 29–34 undrawn edges, 400
simulations, solver 30, 100,000 solver nodes and a 1,048,576-entry agent table.
The metric is exact margin-optimal searched moves, not game win rate.

| Checkpoint | Original | Selected | Gain | Original seconds | Selected seconds |
|---|---:|---:|---:|---:|---:|
| 54 | 197/300 (65.7%) | 245/300 (81.7%) | +16.0 points | 184.3 | 183.8 |
| 15 | 205/300 (68.3%) | 236/300 (78.7%) | +10.3 points | 177.2 | 170.0 |

Above the root solver threshold:

| Checkpoint | n (31–34) | Original | Selected |
|---|---:|---:|---:|
| 54 | 210 | 51.9% | 73.8% |
| 15 | 210 | 55.7% | 69.5% |

Paired comparisons on the full 300 positions:

- Iteration 54: 48 positions improved, 0 regressed; paired 95% interval for the gain [12.0, 20.3] points.
- Iteration 15: 38 positions improved, 7 regressed; paired 95% interval for the gain [6.0, 14.7] points.

Intervals use 10,000 position-bootstrap resamples (seed 20261012). They are
descriptive: positions may share a source game, and that correlation is not modeled.
`data/paired.json` contains the numerical intervals. The independent held-out results
are the confirmation; no held-out result selected a knob.

## Development trials and decisions

All tuning used iteration 54 on 300 `5x5-chains` positions (seed 0), never on
`5x5-mid`. Require at least 9 additional correct moves (3 percentage points) to
replace the retained configuration. Nonzero lambdas first enable the same exact-leaf
utility; all later candidates change one flag relative to lambda 0.5. Every run's
JSON records the full flags, command, hashes and timing; every CSV records its moves.

| Run / single change | Correct / 300 | Accuracy | 31–34 accuracy | Search seconds | Decision |
|---|---:|---:|---:|---:|---|
| Original task-command defaults | 176 | 58.7% | 44.8% | 336.7 | reference |
| Shape exact leaves, lambda 0.25 | 213 | 71.0% | 60.4% | 349.2 | discard |
| Shape exact leaves, lambda 0.5 | 218 | 72.7% | 62.7% | 361.4 | retain |
| Shape exact leaves, lambda 1 | 214 | 71.3% | 60.8% | 366.9 | discard |
| c_puct 3 | 217 | 72.3% | 61.8% | 388.2 | discard |
| Policy temperature 0.7 | 206 | 68.7% | 57.1% | 319.4 | discard |
| Leaf batch 4 | 216 | 72.0% | 61.8% | 327.2 | discard |
| Disable terminal shaping (ablation) | 176 | 58.7% | 44.8% | 335.8 | diagnostic |
| Utility k=3 | 211 | 70.3% | 59.4% | 361.6 | discard |
| Utility k=12 | 215 | 71.7% | 61.3% | 348.5 | discard |
| Proof propagation off | 212 | 70.7% | 59.9% | 355.5 | discard |
| Leaf batch 8 | 221 | 73.7% | 64.2% | 328.7 | discard |

The original score was 58.7%; consistent utility at lambda 0.5 reached 72.7%, a
14-point gain for about 7% more measured search time. The 31–34 sub-band rose from
44.8% to 62.7%. Leaving exact leaves unshaped with that same lambda returned to 58.7%.
This ablation supports fixing the utility/proof inconsistency before using the margin
term. It does not separate the numerical shaping from the necessary proof correction.

The three nonzero lambdas are within 1.7 points of each other, so the data do not
identify a precise optimal lambda. Batch 8 has the highest observed development
score (73.7%) and takes 328.7 seconds, but its one-point gain over the retained setting
is below the declared threshold; it was not promoted or tested on the clean set.
Sharper priors at 0.7 lost four points here. None of the other knobs justified a change.

## What changed

- `--shape_terminals 1` makes natural terminals, collapsed terminals and solver-settled
  leaves use `win + lambda*tanh(final_margin/k)`, matching the network-leaf scale.
  Direct solver results expose Q on that same scale, while their win probability
  remains available separately. Exact margins include boxes already captured.
- With nonzero shaping, a winning child is not proof of the best margin. A node is
  proven only after all legal children are proven. Unproven roots keep exact losses
  because their margins can differ; proven roots keep only maximum-utility moves,
  including when all utilities are negative. Capture turns still use the correct
  perspective and `1-u` only when the player changes.
- The formula has a pure-Python reference and property checks for monotonicity,
  draw value and perspective symmetry. C++ tests compare natural/solver leaves and
  minimax actions with the Python oracle, including unequal winning and losing
  margins, sequential and batched searches, and Go with the new fields set.

Go ignores the Boxes utility knobs. Lambda zero retains the existing proof rules
and utility values. The exact solver, checkpoints, training data and game rules were
not changed. Implementation file hashes are in `data/implementation.json`.

## Reproduction and validation

See `README.md` for the complete environment, build, tests, sampling, tuning and
validation commands, and `replay.sh` for every individual measured run. Recommended
original evaluator command, from the repository root:

```bash
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
uv run "$EXP/oracle_eval.py" --tag 5x5-chains --positions-tag 5x5-mid \
  --min-undrawn 29 --max-undrawn 34 --solver_max_undrawn 30 \
  --solver_node_budget 100000 --num_simulations 400 --prove_terminals 1 \
  --iterations 54 15 --cpu --shape_terminals 1 --margin_utility_lambda 0.5 \
  --margin_utility_k 6 --c_puct 1.5 --policy_temperature 1 --leaf_batch_size 16
```

The sampler's keys and order were independently checked against
`oracle_eval.late_positions`: 300/300 identical on both datasets. Development had
103,589 eligible positions from 24,474 games; clean evaluation had 7,437 from 1,714.
The agent, seed per position, root temperature and solver-table lifetime match the
original evaluator. Only development reference moves are cached; clean oracle labels
stay in memory and were never written as training targets. All runs use CPU with one
Torch/OMP/MKL thread per worker and at most four workers on five available cores.
Timings exclude exact-reference construction and model loading, and include agent
inference and solver work. The initial development oracle cost about 444–448 seconds;
subsequent candidates reuse its reference cache. This is separate from search cost.

Validation: baseline 351 passed / 24 skipped; final 382 passed / 24 skipped, with the
documented upstream `test_gpu_lease.py` excluded. All three Catch2 binaries passed
(219 assertions in 37 cases). Ruff passed. Strict mypy passed on Boxes and new Python
files with `--follow-imports=silent` to isolate existing errors in unchanged upstream
and generated code. No production code changed during the measurements.

## Limits and next step

This is a bounded CPU sweep, not evidence of a global hyperparameter optimum. The
evaluation measures midgame move quality; it does not measure opening play or whole-game
win rate. GPU inference uses a different precision/batching regime, so GPU throughput
and arena strength need a separate check. Use these frozen flags for that comparison,
especially with iteration 15, without retuning on the clean oracle set.
