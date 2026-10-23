# Give the value head explicit structure

The user is right that we can compute useful chain information instead of asking
the network to reconstruct it. Autoboxes already receives coarse chain planes;
the next useful change is **richer descriptors fed directly to the value head**.
This is a small, testable hypothesis, not evidence that a CNN cannot learn Boxes.

Status: isolated feature/residual-head prototypes and the bounded offline
preparation/fitting pipeline are implemented in this folder. They are **not
connected to production checkpoint loading or arena inference**. The current
collection and checkpoint are unchanged. No strength improvement has been measured.

## What the present model has

`src/alpha_go/boxes/encode.py:15–30` defines ten extra planes alongside the eleven
board/score planes:

- Per-box chain membership in sizes 1, 2, or 3+; loop membership; opened-component
  membership.
- Safe undrawn edges.
- Global long-chain count, loop count, safe-edge count and long-chain parity.

The 128-channel residual trunk processes these planes. The value head receives
the spatially averaged trunk activations and the current score margin
(`model.py:110–125`). It does not directly receive the exact lengths of all chains
and loops, their endpoint types, or their complete length histograms. Those facts
remain recoverable from the board, but the trunk must learn to extract them.

Qdab's much smaller value network directly consumes structural statistics,
including link-length and loop-length histograms and both scores; see
[QDAB_REVIEW.md](QDAB_REVIEW.md). Its strength also depends on its rollout and
search, so copying descriptors alone is not a guarantee.

## Descriptor v1: 123 scalars on 5×5

[`structural_features.py`](structural_features.py) implements
`structural-value-v1`, using the existing Python `chains.components` and `degrees`
definitions. It computes observed graph properties only, without a solver.

| Group | Scalars on 5×5 | Meaning |
|---|---:|---|
| Ground-to-ground quiet chain lengths | 25 | Count of each box length 1 through 25 |
| Quiet paths attached to junctions | 25 | Full length histogram; a junction has at least three undrawn sides |
| Opened path lengths | 25 | Full lengths of paths with a capturable endpoint |
| Closed loop lengths | 25 | Full length histogram, without collapsing all loops into one count |
| Box degree histogram | 5 | Boxes with 0 through 4 undrawn sides |
| Edge statistics | 4 | Safe, immediately capturing, double-capturing, and all undrawn edges |
| Scores | 4 | Mover score, opponent score, unclaimed boxes and signed score margin |
| Path endpoint categories | 6 | Ground/ground, ground/junction, ground/open, junction/junction, junction/open, open/open |
| Additional structural facts | 4 | Paths returning to the same junction; long-chain parity; presence of junctions; quiet closed-endgame indicator |

Histogram/degree/endpoint counts divide by the number of boxes, edge counts by
the number of edges, and scores by the number of boxes. The last three indicators
are unscaled. The schema and exact feature order must be stored in checkpoints
and caches. Features are invariant under board rotations/reflections. Player
exchange swaps the two score inputs and negates the margin; structural counts
stay unchanged.

The endpoint categories distinguish a closed independent chain from a path that
still connects to the unsettled middlegame. The same-junction count exposes a
specific topology; it is not claimed to exactly reproduce Qdab's named motifs.
The quiet closed-endgame indicator describes a position class, not its winner.
No feature claims to know who controls the game, and no closed-chain scoring
formula is applied to a position containing junctions or opened paths.

## Direct value attachment that preserves the existing model

The isolated [`structural_value.py`](structural_value.py) contains:

```
structural descriptors (123) → Linear(64) → ReLU → Linear(51)
                                                   ↓
existing final-margin logits ───────────────────── add
```

There are 11,251 additional parameters on 5×5, about 0.38% of the current
2,990,452-parameter model. The final residual projection is initialized to zero.
Attaching it therefore initially preserves every existing value logit and policy
output. It leaves existing layer shapes intact rather than widening a trained
layer and discarding its weights.

For the first diagnostic, freeze the entire existing network, including BatchNorm
statistics, and train only this residual. The wrapper enforces evaluation mode on
the base even when the correction is in training mode. This gives a particularly
clear test: can explicit structure correct the existing value function while
keeping the policy exactly fixed?

The prototype also exposes the small structural head alone, for testing whether
the descriptors carry enough signal without the CNN. It currently accepts
final-margin base models only, matching the uploaded checkpoint. Remaining-margin
models need their own explicit integration decision; do not silently reinterpret
an existing head.

Production integration, after the diagnostic passes, is additive:

1. Add an opt-in structural-head schema/config field with disabled default, so
   existing checkpoints and callers retain their present shapes and behavior.
2. Compute descriptors from the **actual state being evaluated**, including the
   updated scores/mover after forced-capture collapse. Cache training descriptors.
3. Extend the Boxes inference and training paths to pass the vector separately
   to the value head. Do not force it through the shared spatial trunk or add it
   to policy logits.
4. Before arena use, verify Python/native descriptor parity and zero-residual
   checkpoint round trips. Preserve the frozen policy/trunk until separate evidence
   justifies joint fine-tuning.

The existing model loader and these production paths have not been changed.

## Matched fitting experiment

Use the exact root/child bundle described in [LABEL_PLAN.md](LABEL_PLAN.md),
with the collector's source-family split preserved. The frozen current network
is the reference read. Fit three inexpensive candidates with identical examples,
labels, family weighting, update budget, optimizer and predeclared selection rule:

| Candidate | Trained parameters | Purpose |
|---|---|---|
| Existing-input calibration control | A new 64-hidden zero-initialized residual over the existing pooled trunk features and score | Measures how much gain comes merely from fitting another readout to these labels |
| Structural-only readout | The 123→64→51 head, without base value logits | Tests how predictive the explicit structure is by itself |
| Structural residual | The implemented 123→64→51 correction added to frozen base logits | Tests whether direct structure adds information beyond the present value estimate |

The calibration control has nearly the same size as the structural residual and
also starts at the frozen model's predictions. Comparing only a newly trained
structural head against an untouched old head would confound feature quality with
additional fitting. The structural-only arm is diagnostic; it need not become the
deployed model.

For the first fit, use final-margin cross entropy consistently across all arms.
Do not simultaneously change to winner-only targets, add a ranking loss, change
search utility, or upweight exact rows ×8. Child ordering is an evaluation metric
from the outset. If a later ranking-loss experiment is needed, change that one
factor separately for the same arms.

Choose source families and root groups uniformly, then a fixed-size sample of
their known states. Do not let a root with 37 solved children implicitly receive
37 times the weight of a root-only label. Keep unknown children unknown. A
position's final game loss is not an exact label for all earlier states.

The network is frozen, so base logits/pooled activations and structural features
can be cached once. Fitting these small readouts then avoids repeated CNN forward
or backward passes. Use a short fixed update budget with a fixed development
read; one predeclared extension is reasonable only if training and held-out
ordering are still improving. There is no reason to begin with another 120-epoch
memorization run.

## Data and success criteria

Primary fitting/evaluation states are the exact root and child states in the
35–38 band from fresh strong-opponent games. Include the capture-collapsed child
states that search actually evaluates, with score and perspective recomputed by
replay. Raw children are a separate diagnostic slice; they are not extra independent
evidence merely because collapse changes their board.

All siblings, captures, symmetries and branched descendants inherit their source
family's split. Deduplicate state transpositions across train and validation,
including score/mover. The user's GPU benchmark and the cloud pilot remain
excluded. The first 60-family collection may yield only a handful of validation
families under its frozen assignment: report that sample size and do not call a
small, correlated validation improvement a reliable generalization result. Scale
to a separately frozen monitor with at least 50 families before making a strong
claim about the value head.

Report per band, root group and source family:

- Final-margin error, probability calibration and sign accuracy beside its
  constant-predictor base rate.
- Exact regret of the value head's preferred child; winning moves preserved;
  pairwise ordering accuracy excluding exact ties; and rank correlation.
- Raw versus capture-collapsed reads, with mover conversion based on whether the
  mover actually changed. Final-margin child values already include captured
  score; do not add a captured box twice during conversion.
- Fixed-search regret and win preservation with **the same frozen policy**,
  400 simulations, solver34/50k and current utility. Bootstrap changes by source
  family. One selected error can be a regression example, never the test set.

A candidate proceeds only if it improves held-out child choice/ordering and that
gain survives fixed-search evaluation without a clear calibration regression.
Then run a fresh full-game comparison against pinned Qdab. A lower training loss,
root sign improvement alone, or a fitting win on old self-play does not establish
that this makes the bot stronger.

## Cost and validation so far

### Executable offline pipeline

`prepare_structural.py` writes replayable root/child JSON and exact-query receipts.
Unknown values remain null. Complete-child decisions are evaluated only when
every legal child is exact; partial groups can still supply individual value
targets. Semantic state keys include D4 geometry and mover-relative scores, so
transpositions and physical-color swaps cannot cross the training/validation
boundary. Known diagnostic sources are checked now; actual GPU benchmark histories
remain an explicitly pending exclusion input.

`fit_structural.py` caches the frozen model's logits and actual value-layer input,
then trains the three readouts with 1,024 updates of 256 examples, fresh AdamW at
0.001 and one common saved sample-index matrix. Only these roughly 11,000-parameter
heads learn. It saves final heads, per-root child reads, state calibration reads,
family-weighted comparisons and family-bootstrap intervals. It reports the number
of complete validation families; zero such families means no held-out ranking
evidence. It neither selects a best epoch nor promotes a candidate.

`queue_structural.py` waits for all planned source games and a successful collection
supervisor receipt before starting any heavy work. It holds the Qdab runtime lock
while solving/fitting, freezes source code/native/checkpoint hashes, gives labeling
45 minutes and fitting 30 minutes, and preserves stage receipts for resume. The
45-minute label cap can yield a partial bundle. Pending GPU exclusions are accepted
only through the explicit provisional-development option; these outputs cannot
be used to claim a fresh benchmark improvement.

To queue this cloud-only experiment against a collection with a supervisor:

```bash
EXP=experiments/2026-10-23_11-07-boxes-qdab-learning
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -u \
  "$EXP/queue_structural.py" --collection /path/to/qdab-training-run \
  --supervisor /path/to/collection-supervisor/job.json
```

The printed queue directory contains `resume.sh`, `state.json` and per-stage logs.
At completion or interruption, `results.tgz` contains receipts, reports and small
head checkpoints; the large frozen-feature/prediction caches remain on disk.
Those heads are intentionally not loadable as ordinary Boxes checkpoints. Search
integration and fresh opponent matches follow only if the offline result supports
them. Existing benchmark weights remain frozen throughout.

### Runtime and semantic checks

Feature extraction is linear in this small graph using existing chain analysis.
The cached float32 vector is 492 bytes per 5×5 state: about 49.2 MB for 100,000
states. The readout is roughly 11,000 multiply-add terms per state, negligible in
parameter count compared with the CNN. Actual end-to-end latency still needs
measurement, especially Python extraction; production should reuse the structural
analysis already computed for chain planes rather than duplicate it.

The prototype tests cover full-length chains, loops versus opened loops, capture
and double-capture counts, score perspective, D4 invariance, invalid score rejection,
exact preservation at zero residual, finite trainable gradients, and an unchanged
base policy/parameters/BatchNorm buffers after a correction update. These are
semantic/isolation checks, not learning or playing-strength results.

Run the small checks with:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest -q \
  experiments/2026-10-23_11-07-boxes-qdab-learning/test_structural_features.py \
  experiments/2026-10-23_11-07-boxes-qdab-learning/test_structural_value.py
```
