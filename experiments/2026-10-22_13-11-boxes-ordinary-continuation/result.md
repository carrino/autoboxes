# Ordinary continuation: confirmed starting gain and a stronger final learner

Run `ordinary-next-vm_dguq5`, code `c9af2812c083c6cee8abb6e7ac2a9dcbe2781106`,
completed the confirmation, both training cycles, final match and both probes.
The ordinary fitting endpoint's advantage over iter24 replicated on new openings.
The two-cycle learner then beat that frozen starting endpoint. Carry the final
learner forward as the working baseline, retaining its predecessor for regression
checks. No league state was automatically changed.

## Predeclared full-game matches

| Comparison (candidate first) | Wins–losses | Game score | Paired 95% interval | Mean margin |
|---|---:|---:|---:|---:|
| Ordinary fitting endpoint vs proof34 iter24 | 245–155 | 61.25% | 56.75–65.75% | +1.185 |
| Final two-cycle learner vs ordinary fitting endpoint | 244–156 | 61.0% | 56.25–65.75% | +1.170 |

Each match used 200 new random four-edge openings, played with sides swapped,
and 400 simulations with solver34/50k for both players. Confirmation seed was
20261108; final seed was 20261109. The opening lists are disjoint. Training ran
regardless of confirmation score; intermediate snapshots never selected the
final endpoint. This is the declared fixed comparison, not a best-of-many read.

The two-sided exact opening-pair label-swap p-values are 0.000003303 and
0.000015503. Confirmation had 69 candidate sweeps, 107 splits and 24 opponent
sweeps; the final comparison had 73/98/29. First/second-player win rates were
53%/69.5% in confirmation and 56%/66% in the final match.

The previous ordinary-versus-iter24 match was 222–178. This independent 245–155
confirmation supports the same checkpoint's strength gain. The final learner
did not play iter24 directly, and neither these win rates nor any implied Elo
increments should be added to claim that unplayed result. External strength
against Dabble, qdab or a verified published BoxesZero is still unmeasured here.

Final candidate time per move was 0.6643s versus 0.6734s for its starting model.
Confirmation was 0.6359s versus 0.7960s for iter24. The search settings were
identical; these observations are not an independent equal-time benchmark.

## What was trained

Each cycle collected 200 full-start games using its current learner at 400
simulations and solver34/50k, stopping at an exact endpoint. The first cycle
produced 168 complete mistake policies and 4,875 child-value rows; the second
produced 147 policies and 4,181 child-value rows. All 400 source games were
included, and all 315 selected policy groups completed within their budgets.
There were 36 and 21 unknown adjacent comparisons respectively; these were not
silently treated as correct decisions or exact targets.

Each cycle added its data to all previous replay and continued the latest
weights for 8,192 updates, batch 256, constant LR 0.0001 and fresh AdamW.
Actual exact-root exposure was 33,139 and 32,985 rows out of 2,097,152 samples
per cycle, about 1.58%. The generic training settings also print `root_fraction`
0.25, but the `control` arm does not use that quota; the exposure trace confirms
ordinary sampling. Training took 1,119 and 1,148 seconds, 37.8 minutes total.

| Training data | Cycle 1 | Cycle 2 |
|---|---:|---:|
| Training rows | 55,718 | 64,660 |
| Exact-policy training roots | 881 | 1,017 |
| Held-out exact-policy roots | 98 | 109 |
| Final training optimal-move rate | 36.4% | 37.0% |
| Final held-out optimal-move rate | 30.6% | 26.6% |

Validation pools changed as new families arrived, so these two columns are not
a matched generalization comparison. Within cycle 2, held-out policy
cross-entropy improved 3.318 to 3.267 and margin MAE improved 1.565 to 1.494,
while top-move accuracy fell 28.4% to 26.6%. These mixed raw-head diagnostics
do not explain where the full-game advantage came from. Source-family splits
and D4 input exclusions were retained; replay validation remains diagnostic.
[Per-step, per-band fitting reads](data/fit_metrics.csv) include saved-snapshot
steps 2,048, 4,096 and 6,144. Those snapshots were not arena-tested.

## Position diagnostics: a real caveat at 35–36

| Fixed development read | Starting learner | Final learner |
|---|---:|---:|
| 35–36 search regret, 79 exact-known roots | 0.633 | 0.987 |
| 35–36 winning roots preserved | 42/43 | 41/43 |
| 37 search regret, 27 exact-known roots | 1.407 | 1.111 |
| 37 winning roots preserved | 18/19 | 19/19 |
| 37 raw-policy optimal moves | 6/27 | 9/27 |

Regret is lost final score margin. The 35–36 candidate-minus-start delta is
+0.354, with source-family bootstrap interval [+0.025, +0.734]. Of the 28 total
additional margin points lost, 24 were on already losing roots and four on
winning roots; one previously preserved win became a loss. This does not erase
the full-game gain, but it is a regression to monitor rather than dismiss.

At 37 the regret delta is −0.296 with interval [−0.667, +0.074], so the apparent
improvement is inconclusive. The 19/19 win-preservation result is only 19 roots,
not mastery of 37. These are small reused development sets and exploratory,
uncorrected intervals. One of the 80 sampled 35–36 roots remains unresolved.
The inherited report band name `37..38` contains exactly 37 in this experiment.

Recommendation: retain the ordinary recipe and the final checkpoint; its
continuation passed the fixed full-game comparison. Before widening to 38,
refresh the position benchmark with more independent source families at 35–37
and check the easier-band regression and external opponents. More overnight
continuation is now justified as a bounded experiment, without assuming the
gain repeats or attributing it solely to additional optimizer updates.

## Checkpoints and verification

Final checkpoint on the 3070 machine:
`experiments/2026-10-08_14-42-boxes-3x3-loop/logs/ordinary-next-vm_dguq5/cycle-2-train/attempt-ln176d15/train/control.pt`

- Final SHA256: `674b27ff345c705a6ee144a305f7fcb69d812fbc7393da3133523d7d9da57108`
- Cycle 1 SHA256: `18b0d1b06ec34b73a8defbea5f8b25e6d0253b579f137374009c4b163dbc8c07`
- Starting SHA256: `e913bc0af1b6b0df4ecd44b55f2f41af5d5c7e823af7e0a12f912773408fc153`
- Archive SHA256: `da506557900349d5989dd91798525b9ae5287f3828882af9b97f42d8f56bb68a`

Cloud verification independently replayed all 800 games through Python rules,
regenerated the 400 seeded opening pairs, recomputed paired intervals and exact
p-values, and reconstructed all 214 probe records from board states and exact
caches. Both family-bootstrap reports reproduce. All 3,577 checked included
artifact/input hashes and per-game receipts agree. The copied final label tables
contain 1,127 distinct roots and 40,678 legal child labels, with consistent
minimax values. Checkpoint lineage and ordinary exposure counts match the plan.
Checkpoint and NPZ bytes were intentionally omitted from the archive, so cloud
verification did not rerun inference or independently reconstruct NPZ targets.
