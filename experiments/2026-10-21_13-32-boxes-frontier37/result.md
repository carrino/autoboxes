# Frontier37 result: no established full-game improvement

Run `5x5-frontier37-ezpgdpxy`, code `75ee82274083bf00e8e9a884ca9f1f4f048000cd`,
completed ten continuing iterations and both development diagnostics. The final
learner won **98–102 against frozen proof34 iter24**: game score 49.0%, paired
opening-bootstrap 95% interval **[42.0%, 55.5%]**, paired p=0.887725, mean final
margin −0.27. Iter24 remains the incumbent. This sample does not establish an
improvement or a regression; do not promote iter10 or widen to 38 on its basis.

Both players used 400 simulations and solver34/50k. There were 100 fresh
four-edge openings, each played with sides swapped (seed 20261103). Candidate
won both games on 24 openings, split 50, and lost both on 26. Candidate played
first at 36% wins and second at 62%. Reported time per move was 0.6644 versus
0.7666 seconds; this equal-simulation comparison does not establish an
independent equal-time strength gain.

## Fixed development probes

All comparisons below use the same exact-known roots for both checkpoints.
Regret is final score-margin lost by the selected move, not boxes captured.
These retained diagnostic sets are small, reused development samples.

| Read | Frozen iter24 | Final iter10 |
|---|---:|---:|
| 35–36: search regret, 78 roots | 0.923 | 0.923 |
| 35–36: search preserves winning root | 39/42 | 40/42 |
| 35–36: raw-policy optimal move | 22/78 | 26/78 |
| 37: search regret, 38 roots | 1.526 | 1.316 |
| 37: search preserves winning root | 21/26 | 23/26 |
| 37: raw-policy optimal move | 10/38 | 6/38 |
| 37: raw-policy regret | 2.526 | 3.158 |

Two sampled roots per band were excluded because the common exact labels were
incomplete. For 37-edge search regret, candidate minus reference is −0.211,
with source-family-bootstrap 95% interval [−0.632, +0.105]. The apparent gain is
inconclusive. The raw policy moved in the wrong direction on this sample. This
is not evidence that the network mastered 37, even though search saved two
additional winning roots. At 35–36 search regret is unchanged, with a delta
interval [−0.333, +0.359]. The shared report schema calls the second band
`37..38`, but this run contains **exactly 37**, with 40 sampled roots.

## Teaching and fitting

The run generated 1,000 full-start self-play games, with 812 complete mistake
policies: 206 at 35, 339 at 36, and **267 at 37**. Of the 37-edge labels, 41
corrected winning-to-losing decisions. All attempted policy groups completed;
180 of 3,000 adjacent root/move comparisons were unknown and could not be
classified for mistake selection. Counts describe labelled rows, not a claim
of globally distinct states. This remains a mistake-selected dataset.

The ten iterations took 15,792.6 active seconds (4.39 hours), including the
iteration that finished after the four-hour budget. Training logs report
**3,744 optimizer steps and 486 seconds (8.1 minutes)** of training. These timings
exclude training-data loading and final diagnostic evaluation; they are not a
GPU-utilization measurement.

The last replay window contained 37,208 training positions, including 593
exact-policy roots across 424 families. On those labelled training roots, the
final model put 0.150 probability on the exact-optimal set and selected an
optimal move 22.1% of the time. Validation replay had 70 exact roots, mass 0.166
and optimal rate 18.6%. These aggregate reads span 35–37, not 37 alone. Replay
metrics are diagnostic, with a changing window, and do not establish a causal
undertraining diagnosis. Nevertheless the low training-set fit is a concrete
reason to test optimization before collecting another large batch.

Next recommendation: keep iter24 and run a bounded fit-versus-generalization
study on the saved exact labels. First determine whether training can learn
the exact-optimal moves on its own training roots; then measure disjoint-family
37-edge roots, 35–36 regression and search regret. Reuse the collected data and
hold targets, split and initialization fixed. Do not assume that more epochs,
more exact-label weight, or a move to 38 will improve full-game play: previous
runs already show that better raw-policy metrics may not improve search.

## Verification and provenance

The cloud read independently replayed all 200 full games with Python rules,
regenerated every paired opening from seed, checked frozen input/final hashes
and diagnostic receipts, and reconstructed all 240 probe records from the
Python board state plus saved exact caches. It reproduced both paired
source-family reports. No checkpoint inference or additional solving was
needed. The archive does not include the checkpoints or training bundles, so
training-label coverage is taken from the archived preparation reports; it
was not independently regenerated from NPZs.

- Uploaded archive SHA256: `03a8f9ed560894ea83f18561997fed1c6ff7fda677cb2112ece5004ab205bcae`
- Reference: `73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d`
- Candidate iter10: `6c85c44a6eb22c4f41e77cbebc98bf5358e004de192e18e9b80d23aa01038439`
- Final games SHA256: `3310cabbdf71e260a12e5da474de1b4216e7ddac64110039b9a9c2e9fcefb26e`
