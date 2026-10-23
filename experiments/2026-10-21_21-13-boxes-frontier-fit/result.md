# Saved-label fitting result: ordinary replay wins; 25% exact-root quota regresses

Run `fit37-wau7ku0x`, code `3096d615fa83dad3f2d070ff7dd1aaaa6936fc52`, completed
all ten stages. The ordinary-replay final endpoint is a promising next candidate;
it should receive a fresh-opening confirmation before replacing iter24 as the
benchmark. The exact-heavy endpoint should not be promoted.

## Fixed full-game results

Both final endpoints were predeclared and evaluated, with no selection from
intermediate reads. Each played 400 games against proof34 iter24 on the same
200 new four-edge openings, with sides swapped (seed 20261106). Search was
400 simulations and solver34/50k for every player.

| Endpoint | Wins–losses | Game score | Paired 95% interval | Mean margin |
|---|---:|---:|---:|---:|
| Ordinary replay | 222–178 | 55.5% | 51.0–60.0% | +0.655 |
| 25% exact-root quota | 173–227 | 43.25% | 38.5–48.0% | −0.705 |

The reported two-sided paired p-values are 0.024623 and 0.009335 respectively.
These are modest, single-training-seed development results, not an external
engine ranking. Ordinary-minus-quota game score against the common opponent is
+12.25 percentage points, with shared-opening bootstrap 95% interval
[+5.5,+18.75]. That comparison is not a direct match between the two endpoints.
Ordinary won both games on 55 openings, split 112 and lost both on 33; quota's
counts were 37/99/64. Ordinary's first/second-player wins were 49%/62%; quota's
were 36%/50.5%.

Reported seconds per move were 0.6262 ordinary versus 0.7687 iter24, and 0.6264
quota versus 0.7443 iter24. This is equal simulation budget, not equal time.

## The fitting question is answered for these targets

The diagnostic model fit all 128 selected 37-edge training roots: optimal-move
rate 100%, probability mass on optimal moves 99.95%, exact-margin MAE 0.000020.
Its run took 155 seconds. This establishes capacity to fit this finite target
set; it does not establish strategic generalization or rule out a better
architecture. Its weights were never entered in an arena.

The main arms shared 46,384 training positions, including 731 exact-policy roots,
with the same initialization, targets, family split, augmentation schedule and
8,192 optimizer updates. Clean validation comprised 80 exact roots from 59
families (27 roots at 37), after removing 1,149 training-input overlaps and 91
within-validation duplicates. One exact root was removed for training overlap.

| Metric, all labelled 35–37 roots | Iter24 | Ordinary | 25% quota |
|---|---:|---:|---:|
| Training optimal-move rate | 16.96% | 32.01% | 98.77% |
| Held-out optimal-move rate | 15.0% | 27.5% | 26.25% |
| Held-out policy cross-entropy (lower better) | 3.497 | 3.359 | 4.425 |
| Held-out margin MAE | 1.729 | 1.428 | 1.607 |

The quota endpoint fits its labelled training roots almost perfectly, while its
held-out policy loss deteriorates. That is evidence of overfitting under this
sampling/update recipe. The treatment also changes policy/value exposure and
batch-normalization composition, so these results do not identify which
mechanism causes the search regression.

Uniform replay exposed 33,054 exact-root rows; the quota exposed 524,288—about
15.9 times as many—at the same total 2,097,152 sampled rows per arm. Ordinary
training took 1,251 seconds; quota training took 1,140 seconds. Fresh AdamW with
constant lr 0.0001 and one fixed dataset differ from the continuing loop's
short, repeatedly restarted schedules; the gain cannot be assigned solely to
more epochs.

Intermediate reads show why blindly chasing training accuracy is risky. At
1,024 updates, quota training/validation optimal rates were 42.8%/31.25%; by
8,192 they were 98.8%/26.25%. On the small 37-edge validation subset the rate fell
from 11/27 to 7/27. No earlier checkpoint was saved or arena-tested; this is a
post-hoc diagnostic, not evidence that the 1,024-step model is stronger.
[Full fitting reads](data/fit_metrics.csv) include the separate edge bands.

## Search and the curriculum

| Development read | Iter24 | Ordinary | 25% quota |
|---|---:|---:|---:|
| 35–36 search regret, 79 known roots | 0.962 | 0.633 | 2.101 |
| 35–36 winning roots preserved | 40/43 | 42/43 | 32/43 |
| 37 search regret, 27 known roots | 1.852 | 1.407 | 1.037 |
| 37 winning roots preserved | 18/19 | 18/19 | 18/19 |

Regret units are final score-margin points. One 35–36 root remains unknown.
Ordinary-minus-reference regret intervals are [−0.785,+0.076] at 35–36 and
[−1.185,+0.296] at 37: neither diagnostic establishes a regret improvement.
Quota clearly regresses at 35–36: delta +1.139 [+0.608, +1.722], despite a better
37-edge regret read, delta −0.815 [−1.481, −0.074]. These are exploratory,
uncorrected source-family intervals on small reused development sets.

The shared CSV/report schema calls the 37 read `37..38`; this experiment used
**exactly 37**, all from held-out mistake-selected descendant self-play. No
claim of general 37-edge mastery or readiness to train 38 follows.

Next: confirm the ordinary endpoint on fresh paired openings with unchanged
search. If confirmed, carry it into the next full-game replay cycle, retaining
ordinary sampling and an explicitly bounded increased training budget. Keep
35–36 regression checks and 37 development reads; do not repeat the 25% quota
with 8,192 updates or advance the edge band based on this result alone.

## Reproducibility

Candidate to retain:
`experiments/2026-10-08_14-42-boxes-3x3-loop/logs/fit37-wau7ku0x/train-control/attempt-3vvkv1eq/train/control.pt`

- Ordinary checkpoint SHA256: `e913bc0af1b6b0df4ecd44b55f2f41af5d5c7e823af7e0a12f912773408fc153`
- Quota checkpoint SHA256: `3bb9d2cc388289d7bc2b6dcbd13b70bfd91b37f7387afa6bd3ddfc52ff3c3604`
- Reference SHA256: `73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d`
- Archive SHA256: `deb3399f21e8eaec82757a57c4107f218592f07e75f4f4d425e73ff8155ec502`

Cloud verification independently replayed all 800 full games with Python rules,
regenerated all 200 seeded openings in each arm, checked included receipt/input
hashes, and reconstructed all 428 probe records from Python board states and
saved exact caches. Both arm datasets/splits and augmentation-schedule hashes
match; all four family-bootstrap reports reproduce. The copied label tables
contain 812 distinct roots and 29,293 complete legal children, with consistent
exact minimax values. Checkpoints and NPZs are intentionally absent from the
archive, so inference and underlying training bundles were not rerun here.
