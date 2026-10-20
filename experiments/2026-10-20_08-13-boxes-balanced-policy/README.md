# Balanced exact-root sampling on fresh games

The completed `fresh35-36-anmyR2` collection has 2,000 games, 3,364 complete exact
move policies, and 94,902 child value rows. Both arms start from proof34 iter24 and
use the same exact-labelled dataset. The comparison changes how positions are
sampled, not their labels.

The uploaded archive audit found 2,000 distinct eight-move families and 1,404
canonical four-move openings. All 3,364 eligible policy solves completed; another
636 band positions had no search policy and were excluded. The predeclared split
contains 3,017 training roots and 347 validation roots before the full input check.
Reconstruction from the supplied tables found no validation-root D4 overlaps with
training roots, available original prefixes or child rows at 35+ edges. The runner
still checks the actual NPZ arrays because the archive omitted them.

```bash
bash experiments/2026-10-20_08-13-boxes-balanced-policy/run_balanced.sh
```

The default input is
`experiments/2026-10-08_14-42-boxes-3x3-loop/logs/fresh35-36-anmyR2`.
Use `--source-run PATH` if the complete original run was moved. It must include
the checkpoint and prepared NPZs; the exported diagnostics archive is insufficient.
The runner checks those files against their saved hashes and requires the original
native extension and search source. No rebuild or recollection is needed.

## Fixed comparison

- Ordinary arm: shuffled sampling of all training rows.
- Balanced arm: each batch reserves 25% of rows for completely solved decision
  roots and samples the other 75% from the remaining training rows. Child rows
  continue to receive value loss only.
- Both: 2,048 updates, batch size 256, constant learning rate 0.0001, fresh AdamW,
  the same checkpoint, D4 augmentation and random seed.
- The final update is the primary endpoint. Earlier reads are diagnostics, not
  opportunities to choose a favourable checkpoint.
- The family split was fixed before collection: 10% validation, split seed 0.
  Validation inputs identical or symmetry-equivalent to training inputs are
  removed, and validation inputs are deduplicated. Counts are reported.
- Raw evaluation separates 35 and 36 undrawn edges and reports their base rates.
  Two matched 400-simulation probes use the retained fresh validation roots:
  ordinary vs balanced, then iter24 vs balanced, all with solver34/50k.
  Source-family bootstrap intervals accompany the comparisons.

This changes the frequency of policy and value examples, and the batch
composition. It is not a pure policy-loss weighting ablation. The fresh roots
come from iter24 self-play, so they test that distribution rather than stronger
external-engine games.

## Runtime and recovery

Training and each of the two probes have separate one-hour caps: at most three
hours of stage execution, plus verification, shutdown and reporting. These are
caps, not a GPU runtime forecast. Existing exact root and child labels are
exported for the probes, so offline target generation is not repeated. Search
still invokes its normal budgeted solver while choosing moves.

The runner prints a unique directory, a `resume.sh` command and a `results.tgz`
path. Completed stages are hash-verified and reused after interruption.
An interrupted stage restarts in a new attempt directory. Source data, iter24
and league state are unchanged. The results archive excludes weights and NPZs;
they remain in the run directory.

Do not change checkout or rebuild during a run. A favourable result here is a
reason to run a fresh full-game confirmation match, not an automatic promotion.

## Recovering an evaluation timeout

The first 347-root GPU comparison exceeded the original one-hour cap after
training completed. Extend evaluation time without editing the frozen config:

```bash
uv run python -u experiments/2026-10-20_08-13-boxes-balanced-policy/resume_evaluation.py \
  --run experiments/2026-10-08_14-42-boxes-3x3-loop/logs/balanced-policy-k3s3hb6v \
  --probe-minutes 180
```

This verifies and reuses completed training and comparisons. An interrupted
comparison restarts in a new attempt directory; old partial results stay intact.
The two evaluation stages each have a three-hour cap, for up to six hours of
remaining stage execution plus verification/reporting. This is a safety limit,
not a forecast. Position-level partial search work is not resumed.

The recovery writes a separate `evaluation-extension-180m.json` provenance record
and a `resume-evaluation.sh` command. It preserves the original configuration,
checkpoint hashes, data, training budget, positions, search settings and seed.
It refuses to start if training is incomplete or any recorded artifact changed.
Use the new resume command for later interruptions; the original `resume.sh`
still has its original one-hour limits. Send the printed `results.tgz` afterward.
