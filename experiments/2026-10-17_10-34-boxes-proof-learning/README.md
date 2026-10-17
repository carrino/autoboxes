# Proof-filtered policy targets: a matched 2×3 learning check

The corrected targets teach a substantially better policy on this small-board
test, **while making the shared network's value head worse**. This establishes
the proposed policy-learning mechanism near an exact solver frontier. It does
not establish stronger 5×5 play, better MCTS, or the cause of the overnight
training plateau.

## Fixed endpoint results

Three matched initialization/batch seeds, 800 updates per arm, averaged over
the same 123 untouched canonical board families:

| Heldout metric | Raw exploration visits | Proof-filtered visits |
|---|---:|---:|
| Probability mass on exact-optimal moves | 0.597 | **0.834** |
| Top move is exact-optimal | 75.1% | **87.8%** |
| Exact margin lost by top move | 0.846 | **0.455** |
| Expected margin lost under the policy | 1.634 | **0.648** |
| Value margin MAE | **1.260** | 1.838 |

The policy improves in all three seeds. Top-move regret (raw → filtered) is
0.992 → 0.455, 0.780 → 0.520, and 0.764 → 0.390 boxes. Averaging each root's
paired difference across the three fitted seeds, then bootstrapping the 123
roots, gives a regret reduction of 0.390 boxes, 95% interval [0.168, 0.612].
The analogous value-MAE increase is 0.578 [0.392, 0.764]. These intervals are
conditional on these three trained models; there are **123 independent roots,
not 369**. They do not measure variability over new training datasets or large
board distributions.

Training reads show the same tradeoff. Optimal policy mass rises from 0.572 to
0.943 and top-move regret falls from 0.828 to 0.008; value MAE rises from 1.207
to 1.980. Identical value targets do not imply identical value learning:
the changed policy gradients also change the shared trunk. No extra tuning
was performed to eliminate this result.

## Why the target correction matters here

Every collected root was proved and search selected an exact-optimal move.
However, 411 of 512 training roots retained visits on inferior actions in the
old target, averaging **40.1% target mass** across all training roots. The
corrected target assigns zero mass to those inferior actions. The untouched
heldout roots show 37.4% inferior mass in the raw targets and zero after
filtering. This is a direct teaching-target mismatch, not imperfect oracle
labels or a stronger policy choosing a different data distribution.

This regime deliberately exposes the mechanism: quiet 2×3 roots have nine
undrawn edges and the exact solver starts at eight. All roots become proved.
The earlier real 5×5 audit found changed targets in only 2/48 sampled roots;
the effect here must not be extrapolated to the whole production replay
window.

## Protocol

- **Collection:** shared `alpha_go.gameplay.play_game` with uniform random
  legal play for eight opening edges, then one shared-loop MCTS decision.
  Roots with an automatic forced-capture prefix are rejected. The dataset
  keeps one root per canonical edge mask under the four rectangular board
  symmetries, regardless of ownership; no canonical mask appears in both
  training and heldout data. No continuation positions are collected.
- **Search:** uniform evaluator (equivalent to an untrained BoxesNet's zero
  readouts), 160 simulations, leaf batch eight, PUCT 1.5, no noise, temperature
  zero, forced collapse, shaped terminal proofs, utility lambda 0.5 / k=6.
  Solver frontier eight, 50,000 nodes per call, 16,384 table entries. Raw and
  filtered targets come from the **same search and GameRecord**.
- **Oracle:** independent Python exhaustive 2×3 minimax supplies final-margin
  value targets and every legal child's exact remaining margin. Regret is
  root-optimal remaining margin minus that child's margin. Root score cancels.
  Truncated collection outcomes are never used for training.
- **Split:** collect 640 canonical families with seed sequence beginning
  20261031, then sort by a fixed SHA256 family hash; the first 512 train and
  the last 128 are held out. A separate feasibility pilot used 64 families
  generated from seed 70101. A protocol audit found five of the 128 heldout
  masks overlapped that pilot; those five are excluded from the primary read,
  leaving 123. No roots were replaced and training was not changed. Their IDs
  and masks are in `data/summary.json`. Sixteen training families overlapped
  the feasibility pilot, which is allowed. The original 128-root results
  remain available in `data/metrics.json` and the per-root files.
- **Feasibility:** without a solver, both 160- and 1,600-simulation pilots
  produced zero completed shaped proofs and zero changed targets. With the
  solver at eight, all 64 roots were proved and 51 targets changed. This
  selected the mechanism-exposing regime before training any model. The
  pilot family list and diagnostics are in `data/feasibility.json`.
- **Models:** BoxesNet, 24 channels, two residual blocks, 32 value-hidden
  units, group normalization, chain features. One identical initialization
  per seed (101, 202, 303) is copied into both arms.
- **Training:** 800 updates, batch 64, AdamW learning rate 0.001 / weight
  decay 0.0001; ordinary policy cross-entropy plus exact final-margin
  cross-entropy with equal loss weights. All four rectangular augmentations
  are precomputed; both arms receive exactly the same sampled augmented
  batches. The sole changed training input is the policy target. One CPU and
  one Torch thread, deterministic algorithms, no GPU. Six fits took roughly
  one minute in total. Evaluation occurs only at the fixed final endpoint;
  no best-epoch selection or parameter sweep uses these results.

## Reproduction and validation

Base code: `ed3fe2c` (PR #21). The data config records the run script and C++
extension SHA256 hashes. The main executable is unchanged after collection
and training; `analyze.py` applies the documented pilot-family exclusion and
paired root bootstrap.

From a checkout with the current extension built:

```bash
EXP=experiments/2026-10-17_10-34-boxes-proof-learning
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python "$EXP/run.py" \
  --output work/proof-learning-reproduction
uv run python "$EXP/analyze.py" --data work/proof-learning-reproduction
```

The committed `data/roots.jsonl` freezes every opening, both target arrays,
chosen edge, proof status and exact labels. To retrain on precisely those
artifacts, copy `roots.jsonl` into the new output directory first. To rescore
the committed endpoint without training, run `analyze.py` with no arguments.

Collection asserts legality, nonempty targets, exact optimality of every
retained move in a proved root, and disjoint canonical train/heldout masks.
An independent post-run audit replayed every prefix, recomputed all exact
child labels, checked all selected moves, target normalization and symmetry
family separation, and verified both arms' sample counts. Ruff passes. No
production files were changed.

The next practical test remains a matched 5×5 training comparison after the
current overnight results arrive, measuring both policy quality and value
ordering before a full paired arena. This small-board result is sufficient
to motivate that comparison, not to promise its outcome.
