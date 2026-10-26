# Four ordinary cycles: another full-game gain, frontier checks still mixed

`ordinary-four-aqowx5f8` completed all four cycles and the fixed final comparison.
The last learner beat the preceding ordinary cycle-2 winner **271–129 in 400
games (67.75%; paired 95% interval 63.5–72.0%)**. Carry this final checkpoint
forward as the working baseline and retain its predecessor for comparisons.
No league state was automatically changed.

## Fixed final match

Both players used 400 simulations, solver34 with a 50,000-node budget and identical
search parameters. The 200 random four-edge openings (seed 20261110) were played
with sides swapped. They are distinct from both preceding confirmation/final
opening sets. The final checkpoint was fixed in advance, with no intermediate
arena selection or reset to an earlier learner.

| Measure | Result |
|---|---:|
| Final versus frozen start | 271–129 |
| Game win rate | 67.75% |
| Paired game-score 95% interval | 63.5–72.0% |
| Exact paired label-swap p-value | 2.8269e-13 |
| Opening pairs: candidate sweeps / splits / opponent sweeps | 86 / 99 / 15 |
| First / second-player win rate | 64.5% / 71.0% |
| Mean final score margin | +1.520 |
| Candidate / reference seconds per move | 0.6282 / 0.6546 |

This is a strong whole-game improvement under the declared protocol. It does not
locate the source of that improvement, isolate optimizer time from fresh-data
effects, or establish strength against external engines. Successive win rates
against different predecessors must not be added to infer an unplayed matchup.

## Training and elapsed time

Each cycle collected 200 fresh full-start games, then continued the latest
weights for 8,192 ordinary-sampling updates (batch 256, constant LR 0.0001,
fresh AdamW). All prior replay was retained. Total: **800 new games, 32,768
updates, 524 complete exact policies and 15,155 child-value rows**.

| Cycle | Training rows | Exact-policy training roots | New complete policies | Training seconds |
|---|---:|---:|---:|---:|
| 1 | 73,361 | 1,139 | 134 | 1,223 |
| 2 | 81,682 | 1,251 | 128 | 1,101 |
| 3 | 89,957 | 1,366 | 134 | 1,156 |
| 4 | 98,197 | 1,481 | 128 | 1,156 |

All selected policy groups completed. The 23/24/34/28 unknown adjacent
comparisons were retained as unknowns, not treated as correct decisions.
Actual exact-root exposure remained ordinary: 1.55%, 1.53%, 1.52%, 1.51% of
sampled rows, respectively. The generic printed `root_fraction=0.25` setting
does not apply to this `control` arm.

Measured wall time from original archive timestamps was **6h11m39s**: 87.05
minutes collection, 59.12 labelling, 1.75 preparation, 80.88 training stages,
119.90 final match and 22.88 position checks. Reported optimization time inside
training was **77.27 minutes**. The run used the intended fixed workload.

Within-cycle held-out root reads remain mixed. For example, cycle 4 policy
cross-entropy improved 3.373 to 3.356, but top-move optimality fell 23.1% to 20.7%
and margin MAE increased 1.440 to 1.507. These replay diagnostics do not explain
the large full-game advantage. Their pools change between cycles, so compare
endpoints within a cycle rather than treating columns as one fixed test set.
The diagnostic trace is in [data/fit_metrics.csv](data/fit_metrics.csv).

## Reused development positions

| Search measure | Frozen starting winner | Final learner |
|---|---:|---:|
| 35–36 mean margin regret, 79 exact-known roots | 0.987 | 1.215 |
| 35–36 winning positions preserved | 41/43 | 39/43 |
| Exactly 37 mean margin regret, 27 roots | 1.111 | 1.259 |
| Exactly 37 winning positions preserved | 19/19 | 18/19 |

The regret changes are +0.228 with paired family-bootstrap interval
[-0.177, +0.633], and +0.148 [-0.298, +0.667]. Both include zero. At 35–36,
the additional 18 total margin points lost comprise +26 on winning roots and
-8 on losing roots; two previously preserved wins were lost. At 37, one
previously preserved win was lost. These are small reused development sets,
but their direction remains a concern rather than evidence of mastery.

Keep the final whole-game winner. Before widening the exact-label band to 38,
compare it with its frozen predecessor on the larger, newly frozen cloud
benchmark and an external engine. Retain the ordinary training recipe as the
supported baseline; do not change the sampling quota based on these small reads.

The independent cloud job completed all 8,640 child solves for 240 fresh
development roots (80 each at 35, 36, 37) in 696.7 seconds. Nine direct root
solves were unknown; complete child sets permit their root values to be derived
by minimax. Preserve that distinction in the cache provenance. The positions
come from the preceding full-game arena, exclude old monitor opening families,
and were selected before solving. Full historical NPZ input overlap still needs
checking before calling them an untouched benchmark. They are evaluation data,
not additional training examples.

## Provenance and verification

Final checkpoint on WSL:
`experiments/2026-10-08_14-42-boxes-3x3-loop/logs/ordinary-four-aqowx5f8/cycle-4-train/attempt-1u32cfzo/train/control.pt`

- Final SHA256: `c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`.
- Frozen start SHA256: `674b27ff345c705a6ee144a305f7fcb69d812fbc7393da3133523d7d9da57108`.
- Uploaded archive SHA256: `d6514340c10e268b4164728600055ad19eb0ff1313a00957fffa3b115b5e9378`.

Cloud verification independently replayed all 400 games, regenerated all 200
paired openings, recomputed paired intervals and the exact p-value, reconstructed
all 214 saved probe records, and reproduced both family-bootstrap reports.
All 9,386 included artifact/input hashes checked agree, and the four-cycle
checkpoint lineage and collection settings match the frozen protocol. Final
copied label tables contain 1,651 distinct roots and 59,663 legal child labels.
Checkpoint and NPZ bytes were intentionally omitted, so cloud verification did
not rerun inference or independently reconstruct the NPZ targets.
