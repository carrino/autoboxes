# Completed policy/value audit of the frozen iter0 and iter22 checkpoints

The network distinguishes better from worse continuations, but that discrimination is
weak at the top of its move ranking. Iter22 does not improve probability mass on good
moves and loses some value-ordering accuracy at 35..36 undrawn edges. This supports a
controlled target/data experiment; it does not establish a new architecture requirement
or prove which component caused the MCTS regression. Keep iter0 at temperature 1.

## Inputs and execution

Fable's CPU-only audit from PR17, source commit
`bbbfb67cf49586ae57b53f9b290ab0a547b3dab0`, ran on the complete diagnostic tables from
PR16: 100 roots, 68 source families, and 3,640 exact legal children. Both uploaded
checkpoint hashes match the original GPU probes exactly. There are 73 winning roots:
38/54 at 35..36 and 35/46 at 37..38. This uses all 100 roots, including the two proven
from complete child sets; earlier search summaries used 98 roots and 71 winning roots.

This was CPU float32 inference with four PyTorch threads, no solving or training.
The original audit module was executed directly with the current Boxes modules and C++
extension. The generated summary's `code_commit` names the local base worktree, not the
externally loaded audit module; `data/policy-value-audit-provenance.json` records the
audit source commit, module hash, imported-source hashes and extension hash explicitly.

After checking out the audit code from PR17, reproduce with:

```bash
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
SRC="$EXP/logs/astra-probe-PCcE6W"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -u -m alpha_go.boxes.policy_value_audit \
  --tables experiments/2026-10-14_09-58-boxes-probe-analysis/data \
  --checkpoint "iter0=$SRC/reference.pt" --checkpoint "iter22=$SRC/candidate.pt" \
  --output work/policy-value-audit-fresh
```

The output directory must be new. Temperatures are 1 and 2; margin utility is
P(win) + 0.5 tanh(expected margin / 6); paired intervals use 10,000 source-family
bootstrap draws, seed 20261014. They are exploratory and uncorrected for the many
comparisons. All roots are training-derived development positions, not a final test.

## Policy: little change in good-move probability

| Metric, temperature 1, all 100 roots | iter0 | iter22 | Uniform legal move |
|---|---:|---:|---:|
| Probability on all exact-optimal moves | 19.7% | 19.7% | 13.7% |
| Probability on win-preserving moves, on 73 winning roots | 32.5% | 32.2% | 23.3% |
| Mean highest move probability | 8.8% | 9.6% | varies with legal count |

Iter22 is slightly sharper: entropy changes by -0.033 nats [-0.057, -0.011]. But its
optimal-set mass change is effectively zero, interval [-0.007, 0.009], and win-preserving
mass changes by -0.003 [-0.011, 0.005]. This does not demonstrate destructive policy
overconfidence. It also does not clear the policy: aggregate mass can hide redistribution
among moves and roots, which can affect finite-budget search.

Temperature 2 lowers optimal-set mass to roughly 18.0% and win-preserving mass to 30.0%
in both models. That is consistent with the already completed GPU temperature probe's
failure to improve either model. A broader temperature sweep is not the next priority.

## Value: useful ordering, poor best-move selection

The table below uses expected-margin predictions after the search's forced captures.
Pairwise accuracy compares legal children with different exact margins and gives ties
half credit; random ordering has expected accuracy 50%. Means weight roots equally.

| Metric | iter0 | iter22 |
|---|---:|---:|
| Pairwise accuracy, 35..36 (54 roots) | 65.0% | 62.3% |
| Pairwise accuracy, 37..38 (46 roots) | 67.5% | 67.3% |
| Pairwise accuracy, all roots | 66.2% | 64.6% |
| Exact-optimal move chosen by this one-step score, all roots | 17.0% | 19.0% |
| Mean regret of that chosen move, all roots | 4.34 | 4.44 |

Root final-margin MAE is essentially unchanged: 2.373 versus 2.372. Root winner accuracy
is 75% versus 71%, against a 73% constant-winning baseline. Those root metrics conceal
within-position discrimination: these value predictions are not constant, but selecting
the highest prediction often still chooses a bad move. Root calibration and move ranking
should remain separate diagnostics.

At 35..36, candidate-minus-reference margin pairwise accuracy is -2.76 percentage points
[-5.12, -0.61], and Spearman changes by -0.056 [-0.108, -0.008]. Chosen-move regret
changes by +0.519 [-0.269, 1.321], which is inconclusive. The actual shaped-utility
ranking has a weaker change. At 37..38, iter22's one-step utility regret improves from
4.04 to 3.30, exploratory interval [-1.49, -0.05], while overall utility regret is
inconclusive. Do not reduce this to a universal value regression or claim a causal
explanation of the MCTS result.

## Search matching and sensitivity

The audit's `search` mode means the network score after forced-capture collapse, not
MCTS. It matches the board encoding and mover conversion, including retained captures.
Of 3,640 children, 1,834 collapse, 904 reach a take-all/keep-control decision and 17
enter the solver's <=30-edge zone. None is terminal. The audit network-scores those
17 children; actual search attempts a bounded exact solve there and can exhaust its
budget. The audit help text overstates this as guaranteed exact resolution.

All 17 solver-zone children belong to roots 20, 89 and 96. Removing these entire roots
leaves 51 lower-band roots from 47 families. Margin pairwise accuracy is still 64.46%
versus 61.94%, difference -2.52 points [-4.95, -0.23]; Spearman changes by -0.0531
[-0.1064, -0.0027]. Chosen-move regret and shaped-utility ordering differences remain
inconclusive. The lower-band margin-ordering finding is not driven by solver-zone roots.

CPU float32 reproduces all 100 reference greedy-policy choices and 99/100 candidate
choices from the CUDA fp16 temperature-1 probe. At position 99, the candidate chooses
edge 45 on CPU (regret 2) rather than edge 11 on GPU (regret 4). These inference numbers
must not silently replace the GPU probe's saved move-quality numbers.

## Next controlled experiment

Start both arms from iter0, with the same replay and new labelled training positions,
family split, seeds, batches, update count and natural weighting. Collect the new data
from source families outside this diagnostic collection; do not fit on these 100 roots,
their siblings, `5x5-mid`, or prior evaluation games.

Both arms receive the same exact child-value rows and root value targets. The control
keeps saved root MCTS visit-policy targets; the treatment substitutes uniform probability
over every exactly optimal legal move. Mask policy loss on child rows without a search
policy target in both arms. This tests exact-policy supervision conditional on common
child-value augmentation; it does not separately measure the gain from adding value data.
The existing labeller is a useful starting point, but this matched comparison still needs
a driver and a policy-loss mask; it is not an instruction to restart the current loop.

Choose stopping using a separate development split. Use this audit only as a declared
reused diagnostic, then freeze settings before a fresh position read and paired games
against iter0 and classic. A strength claim against BoxesZero still requires that match.

## Artifacts and validation

`data/policy-value-audit-summary.md` and `.json` contain Fable's full output.
`data/policy-value-audit-tables.tgz` contains the per-child, policy, ranking and root CSVs,
the unmodified generated summaries, run log, provenance, and the independent solver-zone
sensitivity script/results. Checkpoint weights are excluded. Archive and file hashes are
in `data/policy-value-audit-provenance.json`.

Independent review ran all seven focused audit tests successfully. Input checkpoints
and tables were hash-checked; row counts, legal moves, child maxima, model coverage and
paired family identities were checked. No model was trained or promoted by this audit.
