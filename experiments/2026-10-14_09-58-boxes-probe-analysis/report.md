# CPU audit and completed policy/value probe

No new full-game strength result is claimed. The user's 100-position policy/value probe
has completed on the 3070; its paired analysis and the prior CPU audit are recorded here.

## Completed 3070 probe: astra-probe-PCcE6W

The uploaded `results.tgz` was produced on code commit
`236a4f295e88591686c1edb640fb812819dc7ba6`. It compares the frozen `5x5-exact/iter0.pt`
reference and `iter22.pt` candidate, including both policy/value crosses. Both checkpoints
already had exact-value training, so this is loop drift, not outcome-vs-exact treatment.
The input archive hash and checkpoint hashes are in `data/probe-PCcE6W-provenance.json`.

All 100 frozen positions replay correctly. There are 98 common-known positions across
68 source families; the other two root solves exceeded the budget. The sample is from
`selfplay-it21`, a training/development source. Intervals resample whole source families
10,000 times, seed 20261014; they are exploratory, not multiple-comparison corrected, and
do not include variation from repeated search runs or establish held-out strength.

| Policy / value | All 98: margin regret | 35..36 (54) | 37..38 (44) | Winning roots preserved |
|---|---:|---:|---:|---:|
| Original / original | 2.633 | 2.148 | 3.227 | 36/71 |
| New / new | 3.714 | 4.296 | 3.000 | 32/71 |
| Original / new | 3.061 | 2.704 | 3.500 | 32/71 |
| New / original | 3.388 | 3.296 | 3.500 | 32/71 |

Regret is final score-margin points, with lower better. Candidate minus reference is
**+1.082 [0.143, 2.061]** overall and **+2.148 [0.863, 3.564]** at 35..36. At 37..38
it is **-0.227 [-1.455, 0.884]**, inconclusive. In the lower band the candidate improves
9 positions, worsens 26, and ties 19; the upper band is 15 better, 14 worse, 15 tied.

Replacing only the policy outputs gives +0.755 [0.022, 1.520] overall; replacing only
value gives +0.429 [-0.119, 1.082]. This points more strongly toward a policy/search
interaction, without proving overconfidence, identifying defective head weights, or
clearing the value function. Each mixed output retains its own complete network/trunk.
Raw greedy-policy regret barely changes (4.510 to 4.408), and its winning preservation
improves from 20 to 24 of 71; the search-prior distribution deserves examination.

Mixed evaluators take two network forwards, so the 317.6/280.4-second hybrid search totals
are not equal-time strength measurements against 81.5/111.8 seconds for reference/candidate.

Decision: keep iter0 as the reference and do not resume the unchanged loop from iter22.
The next bounded probe compares policy temperatures 1 and 2 for both complete models,
holding positions, simulations, value utility, and solver settings fixed. Temperature 2
flattens priors without changing their ranking. A fresh temperature-1 control checks
reproducibility; the active search disables its usual stochastic branches, although the
C++ generator itself is not seeded by the probe. Do not presume randomness caused this
regression. Existing exact caches avoid most repeat labelling; new chosen children and
previously unknown roots can still need solving.

Across temperature runs, align frozen position and checkpoint hashes and use a common
known subset across all cases/runs. Temperature is in `summary.arguments`, not the case
key. Compare each checkpoint's T2-T1 difference and whether the candidate/reference gap
narrows. Similar gains in both models indicate a general search-setting effect. Any
winner still needs fresh-position/full-game validation before a training-default change.

## Concentration of earlier arena mistakes

The prior exact A/B arena contains 100 games per arm. In its already-solved replay, all
model moves in the 35..38 band had known regret. Selecting each game's worst model move
in hindsight captures about four fifths of that arm's total margin regret in this band.

| Arm | Known model moves | Total margin regret | Share from one worst move per game |
|---|---:|---:|---:|
| Reference | 171 | 538 | 81.4% |
| Outcome control | 184 | 626 | 81.8% |
| Exact-value candidate | 186 | 634 | 79.5% |

At most two positive-regret model moves per game account for all observed regret in this
short band. That is partly a consequence of alternating turns and the four-edge window;
it is not a broad compression claim. The result supports a small consequential-mistake
policy-label pilot on separate training games. These arena records remain evaluation only.
The input hash, counts, and reproducible script accompany the result.

## Verified training mechanics, not identified causes

`train.py` resumes weights and BatchNorm buffers, but creates fresh AdamW and a new
warmup/cosine schedule every invocation. With 131 batches per epoch, four epochs mean
524 updates and 52 warmup steps. Eight/twelve epochs have 1,048/1,572 updates and 100
warmup steps. At scheduler step 524 their learning rates are about 0.000582/0.000809,
while the four-epoch schedule has finished. An epoch sweep changes both exposure and
schedule; an optimizer-continuity test must hold batch order and learning-rate sequence
fixed while changing only whether moments survive a boundary.

Both reported training and validation diagnostics run in eval mode and FP32. The training
metric samples at most 40 shuffled batches; validation covers the whole validation set.
Training uses CUDA bf16, while search/oracle inference uses fp16. No train/eval mode bug
was demonstrated by the audit; small CPU checks found unchanged BatchNorm buffers during
evaluation and only floating-point-scale batch-composition differences.

## Analysis controls

The new report aligns cases by position ID, verifies frozen metadata, excludes the same
unknown positions from every case, and resamples source families together. Root winner
accuracy equal to the majority baseline is insufficient to conclude that predictions
are constant; Fable's sibling-ranking diagnostic is the independent next check. The
probe and its game-family-aware report can identify which component deserves that work,
but cannot alone establish a stronger full-game player.

## Validation

Before the later temperature-comparison addition, the CPU regression run passed:
566 passed, 29 skipped in 100.58 seconds, excluding
the known-broken upstream `test_gpu_lease.py`. All 26 focused report tests passed; they
cover paired alignment, shared family resampling, unknown labels, unavailable intervals,
input validation, and read-only output handling. Ruff and strict mypy passed on all three
new Python files. The CLI also read an existing completed two-position, four-case search
probe and wrote a report without loading a model or rerunning a solve. That smoke run
checks artifact compatibility only, not playing strength.

## CPU preparation while the temperature probe runs

The comparison now handles both temperature directories together, preserving a shared
exact-known denominator across every model/run. It reports each checkpoint's T2-T1
change and the difference between those changes, with shared source-family bootstrap
draws. The comparison/report tests pass together (48 tests), and the existing exact-child
perspective tests against a tiny exhaustive oracle pass (2 tests). Ruff and strict mypy
pass on the new comparison, label-pilot script and comparison tests. A clearly synthetic
duplicate of the prior artifact gives zero temperature effects and zero difference of
changes; this is validation, not evidence from the active GPU run.

A three-worker CPU pilot selected six positions per band without consulting model errors.
All 435 legal children at all 12 roots were exactly solved in 56.36 seconds, reusing the
previous probe's cache. Aggregate time inside solver calls was 98.68 seconds. This small,
cached sample is not a production throughput benchmark. The frozen development prefixes
and exact child tables allow Fable to inspect policy probability mass and value rankings
without repeating the search or fitting on the clean evaluation set.

The full extension solved **all 3,640 legal children at all 100 roots** in 415.03 seconds
on three workers (1,167.11 aggregate solver seconds), under the eight-minute scheduling
limit. Direct root/cache solves knew 98 roots. At positions 6 and 57, every legal child
was exact despite the root solve exceeding its budget; taking the maximum child value,
already expressed in the root mover's perspective, proves the other two root values.
`root_source` records that distinction. The maximum also agrees with all 98 independently
known root labels, and all 800 previously recorded selected-policy/search child values
agree with these tables. Original probe summaries retain their original 98-root scope.

The small diagnostic tables are committed as `data/child-diagnostic-*`; bulk worker
transposition-cache artifacts stay outside git. The tables include every replayable prefix,
source family, legal edge, final margin in the root mover's perspective, coverage and
source/code/build hashes. They contain no checkpoint predictions or new strength result.

| Band | Roots | Exact children | Mean optimal share of legal moves | Uniform-move margin regret |
|---|---:|---:|---:|---:|
| 35..36 | 54 | 1,917 | 0.1414 | 5.3881 |
| 37..38 | 46 | 1,723 | 0.1320 | 4.1850 |

The uniform numbers describe drawing one legal move uniformly at each root; they do not
measure MCTS using uniform priors. Fable can now read policy probability mass on exact
optimal/win-preserving moves and value rankings against a complete child table. Inference
must match mover perspective and, for search claims, forced-capture collapse. No fitting
or training should use this diagnostic collection.
