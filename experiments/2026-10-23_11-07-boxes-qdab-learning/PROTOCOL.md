# Next learning pilot: approximate Qdab moves against matched search targets

Status: collection and the matched exporter are implemented. The bundled payload
contains 60 independently replayed teaching games and 240 approximate actions.
`run_teacher_ab.sh` prepares local replay, fits both endpoints and completes the
fixed 200-game-per-arm overnight comparison; see [OVERNIGHT.md](OVERNIGHT.md).
This is a controlled experiment, not a promise that imitating Qdab will beat it. The separate [mechanism review](QDAB_REVIEW.md)
describes search improvements that should keep their own experimental comparison.

## Question and frozen starting point

Can teaching earlier decisions from a strong opponent improve full games more
than another equal amount of ordinary training on our own searched policies?
Previous Qdab curricula supplied later positions for exact labelling. They did
not teach Qdab's early decisions. The new intervention changes those policy
targets while retaining the same positions, outcomes, replay and update count.

Both arms start from the latest ordinary learner, SHA256
`c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`
(`work/qdab-latest-upload/control.pt` in this cloud workspace). Keep the existing
chain features and model. The cloud diagnostic lost both sides of two opening
pairs by 38 undrawn edges; improving only the last decisions above solver 34
cannot repair those earlier losses. Child value ordering also fails on a selected
37-edge mistake, so policy supervision is a hypothesis, not a complete diagnosis.

## Collect and prepare

1. Collect 40–60 complete learner-versus-Qdab games within the bounded three-hour
   session. The learner uses the current 400-simulation search, solver 34/50,000;
   Qdab receives five seconds per requested decision and the existing external
   timeout. These are specified settings, not equal-time settings. Preserve every
   legal move, true final score, learner search target and raw Qdab response.
2. Query Qdab on at most four searched learner decision states per game, one from
   each of 48–50, 45–47, 42–44 and 38–41 undrawn edges. Preserve the learner's
   original policy and Qdab's separate recommendation. Forced-capture-only moves
   do not count as strategic teaching examples. A recommendation is one
   approximate action, never an exact optimal set or an exact value.
3. Freeze opening families and the 80/20 training/holdout assignment before game
   outcomes. The collector canonicalizes the four-edge opening mask under D4;
   opposite sides of an opening stay together. Exclude the existing external
   benchmark opening banks, including seeds 20261112 and 20261116, plus fixed
   diagnostic families. Preserve explicit family/state exclusions, not just
   different seed numbers. Interrupted or failed games cannot supply invented
   outcomes; retain completed games and record failures without counting them as
   wins, losses or replacement successes.
4. Build two portable bundles containing the same existing full-trajectory replay
   and exact root/child anchors, plus the same eligible new game rows. Keep
   ordinary sampling and `min_undrawn=24`. Do not introduce exact-root quotas,
   teacher oversampling, or extra value weighting in this first comparison.

At four queries per game, this is only 160–240 recommendations before holdout and
eligibility filtering. It tests whether useful supervision transfers; it is not
enough data to expect dominance over Qdab.

### Required matching and truthful target semantics

The preparer must verify that both bundles have identical ordered positions,
movers, moves, outcomes, value targets, family identifiers and eligibility masks.
Only the policy vectors at an explicit `approximate_teacher_mask` may differ:

| Row | Control bundle | Qdab bundle |
|---|---|---|
| Queried learner state | Original normalized 400-simulation search policy | One-hot legal Qdab recommendation |
| Other learner state | Original searched policy | Identical searched policy |
| Existing replay/exact anchor | Existing target | Identical target |
| Qdab turn without matched learner search | Excluded from fitting | Excluded from fitting |

Use the first legal edge in an independently replay-verified Qdab reply as its
root action; preserve the entire capture bundle as provenance. Keep the raw
learner visits, teacher response, source family, root prefix, mover, scores and
checkpoint/config hashes in the export. Unknown/budget-failed queries do not
become targets. Real mixed-opponent game outcomes remain observed outcomes.

The ordinary dataset already accepts any positive `mcts_visits` vector as a
normalized policy target. A one-hot approximate action can therefore use this
route without changing the model or pretending it is exact. Teacher rows must
not acquire `exact_margin_p1`, `exact_policy_root_mask`, or
`policy_ab_root_mask` from the teacher. Retain real exact labels only where an
independent solver supplied them. Store approximate provenance separately.

Exclude unmatched opponent turns with `train_mask=False`, leaving `policy_mask`
true/default. In the current loader, `policy_mask=False` requires a known exact
value even for a row excluded by the training mask; it is not an appropriate way
to discard an unlabeled opponent move.

Exclude every collector-heldout family from **both** fitting bundles. The
trainer's `split_stable(.1, 0)` adds its own replay validation split and does not
honor the collector's arbitrary 80/20 metadata. Report eligible training counts
after that additional split; use the untouched collector families separately.
Remove D4-equivalent input leakage into all diagnostic sets and propagate family
exclusions. Do not silently equate this collector's four-edge mask family with
the older replay preparer's ordered eight-edge opening key.

## Matched GPU fit

Use two calls to the existing `alpha_go.boxes.frontier_fit` with **`--arm control`
in both calls**, each reading its corresponding bundle. Here `control` selects
ordinary sampling; it does not disable the real exact labels already in replay.
Both bundles must retain the verified exact CSV anchors required by this tool,
including clean validation roots at 37 edges.

Predeclare 2,048 updates, batch 256, learning rate `1e-4`, seed `20261120`, and
`min_undrawn=24`. Existing training uses fresh AdamW, weight decay `1e-4`, gradient
clipping at 1, constant learning rate, and D4 augmentation. At roughly 100,000
eligible replay positions, this is about five ordinary passes. It is a modest
extension of the successful ordinary sampler, avoiding the previous heavy
weighting and 120-epoch memorization experiments. Exact exposure depends on the
actual bundle size and must be reported.

The exporter produces verified bundles; the existing fit interface
for each role is:

```bash
uv run python -m alpha_go.boxes.frontier_fit \
  --reference "$REFERENCE" --prepared "$BUNDLES/$ROLE" \
  --output "$RUN/$ROLE" --arm control \
  --steps 2048 --batch-size 256 --lr 0.0001 --seed 20261120 \
  --min-undrawn 24 --snapshot-every 1024
```

`ROLE` is `learner-targets` or `qdab-targets`; these are the roles created by the local preparation stage. Each call produces an internally named `control.pt`;
the outer manifest must distinguish the two roles. Only the predeclared final
2,048-update checkpoint is the candidate. The intermediate snapshot is diagnostic.

Verify identical index and augmentation traces between arms. Count actual
approximate-teacher exposures using their explicit mask and the saved sample
indices; the trainer's exact-root exposure counter does not measure them.
Report training and heldout Qdab-action probability/agreement separately from
exact optimal-move probability. Qdab agreement is not proof of correctness.

## Evaluation and continuation decision

First check heldout teacher agreement and the existing exact 35–37-edge search
diagnostics, including regret and wins preserved. Also check child value ordering:
both arms train value identically, but the shared trunk may still move. Failure
to fit the supplied training targets calls for a target-consumption/schema check;
fitting them without transfer calls for more distinct families, not automatically
more epochs or higher weight.

Then play both final checkpoints against pinned Qdab on the same **fresh** bank
of 100 four-edge opening pairs: 200 full games per arm, 400 games total. Freeze the
bank and seed `20261121` before either arm plays; exclude training and diagnostic
families. Keep ours at 400 simulations, solver 34/50,000 and the same shaping;
keep Qdab at the same requested five seconds and external timeout. Verify every
full game by independent replay. Log failures, stop for technical repair, and
resume the same slot; never discard a failed slot based on who appeared ahead.

The primary quantity is Qdab-arm game score minus control-arm game score, with
opening-family paired uncertainty. Each cluster contains both sides and both
models' games. Report each arm's raw wins/losses, first/second rates, game-score
interval and elapsed time. The 200-game endpoint per arm is the overnight comparison, not an automatic
promotion. The sample was increased before either fit or match at the user's
request for an unattended overnight run; smaller explicit --games runs are
screening experiments only. Do not stop early or choose the better-looking snapshot.
Use the exact same settings to distinguish learned improvement from more search.

A promising screen earns an independent, predeclared larger confirmation; it
does not earn an overnight production rollout based only on head metrics.
Beating Qdab requires a fresh full-game score whose uncertainty supports more
than 50%, not merely beating the old checkpoint. Ultimately measure dominance
with hundreds of fresh games and separately report empty-board matches. Relative
Elo against this pinned engine/configuration is not its published absolute Elo.

## Implemented interfaces

`prepare_teacher.py export` verifies and packages the completed cloud collection.
`prepare_teacher.py augment` builds both portable replay bundles, checks actual
loaded-dataset matching, excludes heldout/evaluation overlap, and verifies exact
anchors survive. `run_teacher_ab.sh` verifies the packaged payload, freezes the
full experiment and runs the existing trainer and Qdab gameplay interfaces.

The runner checks sample and augmentation traces, reports approximate-teacher
exposure and heldout agreement, then compares both fixed final checkpoints using
complete opening-pair clusters. It does not change the production loop or
represent approximate actions as exact policies. See [OVERNIGHT.md](OVERNIGHT.md).

If this small pilot transfers, obtain more approximate targets cheaply from
Qdab's **own already logged decision states**. Generate matched learner search
policies on the GPU for those states before adding them to either arm. This adds
strong-engine decisions without repeating its CPU queries. Increase distinct
families before heavily replaying the same few recommendations.

In parallel, a frozen 400/1,600-simulation read can test search headroom. The
selected cloud mistake was rescued at 1,600, but that was one cold-search example
with a different evaluator wrapper, not a replication of its played action or a
general strength result. Keep that experiment separate from the learning A/B.

## Separate hypothesis: give the value head richer chain structure

Our model already receives chain/loop information among its 21 input planes;
the value head receives the pooled shared trunk and known score directly. The
question is whether richer structural scalars supplied directly to that head
improve its ordering of children. This is a representation experiment, separate
from changing policy labels; see the companion `STRUCTURAL_VALUE.md` design.

Compare the unchanged model with a compatible residual structural-value branch
using identical ordinary/exact data, family splits and update counts. Keep the
policy-target source identical between these two arms. Prefer a zero-initialized
residual so the starting predictions match; report any unavoidable initialization
or optimization differences. Read heldout exact child ordering, chosen-child
regret and searched wins preserved, then full games. Root sign accuracy alone
cannot establish useful search guidance. Do not combine this architecture change
with the approximate-Qdab policy intervention until their separate effects are
measured, and do not edit the model during the current frozen collection.
