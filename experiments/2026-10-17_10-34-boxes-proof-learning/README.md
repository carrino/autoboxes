# Proof-filtered policy targets: a matched 2×3 learning check

The corrected targets teach a substantially better policy on this small-board
test, **while making the shared network's value head worse**. This establishes
the proposed policy-learning mechanism near an exact solver frontier. It does
not establish stronger 5×5 play, better MCTS, or the cause of the overnight
training plateau.

## Fixed endpoint results

Three matched initialization/batch seeds, 800 updates per arm, averaged over
the same 128 untouched canonical board families with disjoint move-RNG seeds:

| Heldout metric | Raw exploration visits | Proof-filtered visits |
|---|---:|---:|
| Probability mass on exact-optimal moves | 0.578 | **0.823** |
| Top move is exact-optimal | 69.8% | **86.5%** |
| Exact margin lost by top move | 1.151 | **0.578** |
| Expected margin lost under the policy | 1.625 | **0.739** |
| Value margin MAE | **1.160** | 1.672 |

The policy improves in all three seeds. Regret and MAE are measured in
**score-margin points**: transferring one box from one player to the other
changes that margin by two points. Averaging each root's paired difference
across the three fitted seeds, then bootstrapping the 128 roots, gives a regret
reduction of 0.573 points, 95% interval [0.297, 0.870]. The analogous value-MAE
increase is 0.512 [0.369, 0.654]. These are descriptive root-bootstrap intervals
conditional on these three trained models, treating the 128 distinct families
as sampling units; there are **128 roots, not 384**. They do not measure
variability over new training datasets or large board distributions.

Training reads show the same tradeoff. Optimal policy mass rises from 0.556 to
0.940 and top-move regret falls from 0.992 to 0.026; value MAE rises from 1.080
to 1.734. Identical value targets do not imply identical value learning:
the changed policy gradients also change the shared trunk. No extra tuning
was performed to eliminate this result.

## Why the target correction matters here

Every collected root was proved and search selected an exact-optimal move.
However, 421 of 512 training roots retained visits on inferior actions in the
old target, averaging **41.3% target mass** across all training roots. The
corrected target assigns zero mass to those inferior actions. The untouched
heldout roots show 36.5% inferior mass in the raw targets and zero after
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
- **Split and random streams:** collect 640 canonical families with game seed
  `20261107 * 1_000_000 + attempt * 64`, then sort by the same fixed SHA256
  family hash; the first 512 train and the last 128 are held out. Shared
  `play_game` adds move index to the game seed. A stride of 64 exceeds the
  entire 17-edge game, so no games reuse those move RNG seeds. All 683 family
  masks previously seen in the original collection or feasibility pilot are
  rejected **before** the new collection and split. Consequently all 640 new
  roots are fresh, the 128 heldout roots have no training or earlier evaluation
  symmetry equivalents, and no after-training exclusions are needed. The
  fixed exclusion list is committed and hashed in `data/config.json`.
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

Base code: `ed3fe2c` (PR #21), experiment revision based on `6dd61dc`. The data
config records the run script and C++
extension SHA256 hashes. The main executable is unchanged after collection
and training; `analyze.py` computes the fixed endpoint read and paired root
bootstrap. Sample counts are derived from the roots, so nondefault
`--train` / `--heldout` settings are reported correctly.

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
child labels, checked all selected moves, target normalization, symmetry
family separation and absence of previously seen masks, and verified both
arms' sample counts and disjoint move-seed windows. Ruff passes. No production
files were changed.

## Superseded collection and reason for rerun

The original collection used consecutive game seeds. Independent review found
that `play_game(seed + move_index)` therefore reused a move RNG seed in nearby
games. Although canonical masks were distinct and the split had no exact
symmetry leakage, the root-level uncertainty calculation could understate
correlation. That methodological defect triggered this complete fresh-data
rerun; it was not a search for a more favorable result. Architecture, losses,
optimizer, 800-step endpoint, model seeds and search budgets remain unchanged.

`superseded-correlated-seeds/` retains every original result, root, configuration
and the old five-pilot-mask exclusion for audit. The new exclusion list is the
union of all 640 original roots and all 64 pilot roots (683 distinct masks).
Both runs show the same qualitative finding: policy improves, value degrades.
Only the corrected collection above supports the primary numerical report.

The next practical test remains a matched 5×5 training comparison after the
current overnight results arrive, measuring both policy quality and value
ordering before a full paired arena. This small-board result is sufficient
to motivate that comparison, not to promise its outcome.
