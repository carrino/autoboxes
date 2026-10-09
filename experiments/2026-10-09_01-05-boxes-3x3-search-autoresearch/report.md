# 2026-10-09_01-05-boxes-3x3-search-autoresearch report

Baseline search_optimal 0.8233 -> best 0.9167 (+9.3 points)

Runs: 25; keep 10, discard 15, crash 0; keep rate 0.40

## Kept runs by gain over the previous kept run

| run | gain | metric | description |
|---|---|---|---|
| 19 | +2.0 | 0.8983 | margin_utility_lambda=0.75 (k=6) |
| 8 | +2.0 | 0.8533 | policy_temperature=0.7 (sharper priors) |
| 20 | +1.8 | 0.9167 | margin_utility_lambda=1.0 (k=6) |
| 11 | +1.2 | 0.8717 | margin_utility_lambda=0.5 (k=6) |
| 10 | +0.7 | 0.8600 | margin_utility_lambda=0.25 (k=6), PLAN 6.4 |
| 17 | +0.7 | 0.8783 | sims=200 (double budget; cost in sec_per_move) |
| 6 | +0.5 | 0.8333 | leaf_batch_size=4 |
| 3 | +0.3 | 0.8267 | c_puct=2.5 |
| 5 | +0.2 | 0.8283 | leaf_batch_size=8 (more sequential, less virtual loss) |
| 1 | +0.0 | 0.8233 | baseline: sims=100 c_puct=1.5 leaf=16 T=1 lambda=0 solver=0 merge=off |

## All runs

| run | metric | status | description |
|---|---|---|---|
| 1 | 0.8233 | keep | baseline: sims=100 c_puct=1.5 leaf=16 T=1 lambda=0 solver=0 merge=off |
| 2 | 0.8183 | discard | c_puct=1.0 |
| 3 | 0.8267 | keep | c_puct=2.5 |
| 4 | 0.8200 | discard | c_puct=0.75 |
| 5 | 0.8283 | keep | leaf_batch_size=8 (more sequential, less virtual loss) |
| 6 | 0.8333 | keep | leaf_batch_size=4 |
| 7 | 0.8000 | discard | leaf_batch_size=32 |
| 8 | 0.8533 | keep | policy_temperature=0.7 (sharper priors) |
| 9 | 0.8383 | discard | policy_temperature=1.5 (flatter priors) |
| 10 | 0.8600 | keep | margin_utility_lambda=0.25 (k=6), PLAN 6.4 |
| 11 | 0.8717 | keep | margin_utility_lambda=0.5 (k=6) |
| 12 | 0.8600 | discard | margin_utility_lambda=0.25 k=3 |
| 13 | 0.8600 | discard | merge_equivalent=True (one action per chain/loop) |
| 14 | 0.8650 | discard | solver_max_undrawn=6 (exact leaves below the sampled band) |
| 15 | 0.8667 | discard | solver_max_undrawn=8 |
| 16 | 0.8717 | discard | solver_max_undrawn=10 |
| 17 | 0.8783 | keep | sims=200 (double budget; cost in sec_per_move) |
| 18 | 0.8467 | discard | sims=50 (half budget) |
| 19 | 0.8983 | keep | margin_utility_lambda=0.75 (k=6) |
| 20 | 0.9167 | keep | margin_utility_lambda=1.0 (k=6) |
| 21 | 0.8983 | discard | margin_utility_k=10 at the best lambda |
| 22 | 0.8933 | discard | policy_temperature=0.5 |
| 23 | 0.8817 | discard | solver_max_undrawn=10 on top of the best (tied round 1 at a third of the time) |
| 24 | 0.8783 | discard | leaf_batch_size=16 on top of the best (is leaf=4's gain worth 2x the time?) |
| 25 | 0.9067 | discard | c_puct=1.5 on top of the best (was c_puct=2.5's gain real?) |
