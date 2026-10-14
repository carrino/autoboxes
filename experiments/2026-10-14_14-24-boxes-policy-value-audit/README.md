# Policy mass and value ranking of iter0 / iter22 on the exact child tables

Question (ASTRA PR #16, "Fable: test value ranking within a position"): on the 100 frozen
roots at 35..38 undrawn edges whose every legal child is exactly solved
(`experiments/2026-10-14_09-58-boxes-probe-analysis/data/child-diagnostic-*`), how much
probability does each checkpoint's policy put on the exact-optimal and the win-preserving
moves at temperatures 1 and 2, and does its value head rank the legal children at all,
even though its root winner accuracy sits on the majority baseline?

Inference only, on CPU, with `alpha_go.boxes.policy_value_audit`. No position is solved
and nothing is trained; the roots and child values are read from the tables.

## What is measured

- **Policy** (per root, per temperature): entropy of the search's prior
  (`policy_dict`: softmax over the legal edges at temperature T), top-move probability,
  the mass on the exact-optimal set (children with the root's exact margin), the mass on
  the win-preserving set on winning roots (children with a positive exact margin), the
  uniform-prior baselines (set size over legal moves), and the exact regret of the top
  move. T = 2 keeps the ranking and flattens the distribution.
- **Value ranking** (per root, per predictor, per mode): the children are scored by
  P(win), by the expected final margin and by the search's shaped utility
  P(win) + 0.5 tanh(margin / 6). `search` mode scores each child as the search sees it:
  `BoxesSearchState(child)` plays the opponent's forced captures first and the net scores
  the collapsed position for its side to move; the value is carried to the root mover by
  complementing P(win) and negating the margin only when the mover changed. `raw` mode
  scores the child before the collapse. Metrics: the exact regret of the predictor's
  favourite child, pairwise ordering accuracy over children with distinct exact margins
  (ties count one half), Spearman correlation, and the within-root spread of the scores.
- **Root calibration**: the root's own P(win) against the exact sign and the majority
  baseline, the children's sign accuracy and margin error, and how many children
  collapse, reach a decision point, or fall into the solver's reach (30 undrawn), where
  the real search would use the exact value instead of the net.
- Everything is reported for 35..36 and 37..38 separately and overall, and the
  iter22-minus-iter0 differences carry a source-family bootstrap interval (10,000
  resamples, seed 20261014), as in the probe report.

On these roots no root is at a decision point and every child passes the turn (a quiet
root has no box to complete), so `mover_same` is false throughout and the collapse acts
through the opponent's captures: 1,834 of the 3,640 children collapse, 904 of them to a
take-all / keep-control choice, 17 reach the solver zone, none ends the game.

## Run

```bash
CK=experiments/2026-10-08_14-42-boxes-3x3-loop/checkpoints/5x5-exact
uv run python -m alpha_go.boxes.policy_value_audit \
  --tables experiments/2026-10-14_09-58-boxes-probe-analysis/data \
  --checkpoint iter0=$CK/iter0.pt --checkpoint iter22=$CK/iter22.pt \
  --output experiments/2026-10-14_14-24-boxes-policy-value-audit/data
```

The checkpoints are the probe's: sha256 `46c7264c…` (iter0, the reference) and
`a6eaaeab…` (iter22, the candidate), recorded in full in the probe's
`probe-PCcE6W-provenance.json`; `summary.json` records the hashes of the files actually
used. The run takes about a minute on a CPU (7,280 forwards per checkpoint).

## Outputs (`data/`)

- `children.csv`: every legal child of every root for each checkpoint: exact margin,
  collapse prefix length, mover change, undrawn after the collapse, terminal / decision
  flags, the search-matched and raw P(win) and margin in the root mover's perspective.
- `policy.csv`: one row per root, checkpoint and temperature with the policy metrics.
- `ranking.csv`: one row per root, checkpoint, mode and predictor with the ranking metrics.
- `roots.csv`: root calibration and collapse counts per root and checkpoint.
- `summary.json` / `summary.md`: the aggregates by band, the paired differences, and the
  provenance (checkpoint and table hashes, code commit, arguments).

Tests: `tests/test_boxes_policy_value_audit.py` (2x3 and 2x2 against the exhaustive
oracle: the collapse and the perspective match `BoxesSearchState`, a retained capture and
a double capture keep the turn, metric baselines, the CLI end to end).
