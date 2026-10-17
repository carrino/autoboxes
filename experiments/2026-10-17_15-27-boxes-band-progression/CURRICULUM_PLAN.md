# Bootstrap the value frontier: 36, then 38, then 40

Agreed direction, 2026-10-17. This is the next experiment plan; the staged
controller and frontier teacher described below are not implemented yet.
The active eight-hour proof-target run continues unchanged.

## Objective and starting evidence

Build one model whose useful policy and value estimates extend progressively
earlier into realistic games. A temporary frozen teacher checkpoint can guide
the next stage; deployment still uses one network and search.

The verified solver34 iter15 won 126/200 games against its starting checkpoint
with identical search. The active run starts from that iter15 and finishes with
a fixed 200-game match against it. We have evidence of overall learning, but no
demonstration of mastery through 36 or 38.

The current recipe uses full-start self-play, solver34/50k, exact move corrections
at 35–36, and exact values for the selected roots and collapsed children. Training
continues the latest weights. The 240 frozen development roots and small bounded
label-cost pilot are recorded in [README.md](README.md). At 20M nodes, the pilot
solved 12/12 roots at 35–36, 6/12 at 37–38, and 0/12 at 39–40. Full policy labels
require additional child solves, so their yield remains to be measured.

“Strong value at 36” means evaluations that help choose good continuations.
Root winner-sign accuracy and replay loss alone do not establish that. Exact
supervision can also teach earlier positions directly; this curriculum is a
practical way to make better use of limited exact labels, not a mathematical
requirement that every later position must be mastered first.

## Stage 0: establish the incumbent and locate its weaknesses

When the active run finishes:

1. Verify the final match, checkpoint ancestry, target audit, hashes and replay.
   Compare the declared endpoint against the frozen old iter15. Use the current
   run's predeclared game count and seed; do not choose the luckiest intermediate
   arena result or extend the same match until it becomes significant.
2. Score both checkpoints on the same frozen decision roots, separately at
   35–36, 37–38 and 39–40. Use the same solver, simulations and utility settings.
   Fill exact labels with explicit budgets and preserve the cache. Unknown
   labels remain unknown; report common coverage for every comparison.
3. Start with root and selected-child solves for move regret; these are cheaper
   than a complete child ranking. At 35–36, build a manageable exact-labelled
   child set to read value ordering,
   including the child states after the forced-capture collapse used in search.
   Retain both raw and collapsed diagnostics so a representation mismatch is
   visible. Exclude tied exact child values from pairwise ordering accuracy.

Report mean selected-move regret, winning-position preservation, raw policy
regret, and value ordering. Include raw-value MAE and sign/base-rate numbers as
secondary diagnostics. Use whole source/opening families for intervals.
Report shallow/solver-fallback states as well if errors cluster near the boundary:
solver34 attempts are bounded, not a guarantee of exact play below 34.

If the full-game result is inconclusive, keep old iter15 as the proven incumbent
and preserve the new learner for diagnostics. A small arena must never silently
reset the learner. Curriculum readiness and full-game champion promotion are
different decisions.

## Stage 1: strengthen the 35–36 foundation

Collect diverse, replayable prefixes from recent strong self-play. Add verified
Qdab-generated histories as a separate source where its collector is available.
Use legal histories with their true scores, owners and mover; avoid relying on
greedy-only openings or arbitrary edge masks. Record opponent and source-family
provenance. Keep development families out of training and reserve fresh families
for confirmation before inspecting candidate outputs.

Keep joint policy/value training and natural-weight exact targets. Do not repeat
the failed heavy reweighting or 120-epoch memorization approach. First measure
whether new examples improve fit and held-out decisions at modest update counts.

The first curriculum change to test is a **25% branch-start / 75% full-start game
mix**, with branches at 35–36. This is a proposed starting allocation, not an
established optimum. Measure the resulting position distribution: game fractions
are not training-position fractions. Within the branch pool, sample source
families broadly rather than repeating a handful of difficult roots.

Compare against the current full-start recipe from the same frozen checkpoint.
Keep architecture, optimizer settings, search settings and exact-label weighting
fixed. Predeclare matched optimizer-update and collection-work budgets; report
actual wall time and target counts. If the first signal is promising, confirm on
another training seed before spending a full overnight budget on the changed mix.

Readiness primarily requires lower searched regret on separate source families,
with no clear loss of win preservation. Value ordering on completed child sets
helps diagnose whether the value head supplies the gain.
Inspect absolute residual mistakes too: a tiny improvement over a poor baseline
does not establish a reliable teacher. The next stage's teacher-search pilot is
the functional test of whether this foundation is useful.

If exact training targets still cannot be fitted, inspect representation, loss,
perspective and capacity. If they fit but held-out decisions worsen, increase
data diversity or reduce reuse. If values improve but search does not, inspect
policy coverage and search consumption before adding more epochs.

## Stage 2: make the 36 value function improve decisions at 37–38

Freeze the best supported lower-band checkpoint as a temporary teacher. The
student starts from it and remains the single model we train and deploy.

Before producing a large training set, prototype a bounded **search to the
trusted frontier** on 37–38 roots:

- Explore continuations until they reach at most 36 undrawn edges, or terminate
  earlier. For the initial one/two-edge gap, benchmark a batched shallow
  full-width implementation against ordinary MCTS before choosing the teacher.
  Explicitly limit wall time, leaf states and memory; a state-count limit must
  record incomplete coverage rather than silently claim a complete search.
- Apply the production forced-capture collapse. Use exact solver results when
  available; otherwise evaluate the frontier with the frozen teacher. Record
  actual leaf edge counts and the fraction settled exactly.
- Back up values using the actual player to move: captures can retain the turn.
  Preserve the existing final-margin/value-distribution and utility contracts;
  do not blindly negate on every edge or add the already-known score twice.
- Compare its chosen moves with ordinary MCTS on identical, exactly labelled
  development roots. Match wall-time budgets as well as recording node/leaf
  counts. Reject a teacher that is worse or buys too little for its extra cost.
  Also compare the same frontier procedure using the previous value head, to
  distinguish better learned values from a gain caused by different search.

This is more specific than raising `POLICY_MAX_UNDRAWN`: it tests whether later
value estimates can actually supply better supervision for earlier moves.

Keep teacher-derived targets explicitly approximate, with their teacher hash and
search settings. Never write them into the exact-label fields. On a bounded
sample of fresh training mistakes, also attempt complete exact policies at
37–38. Start with the measured 20M per-solve limits; record completed policies
per CPU-hour and incomplete-child rates before increasing budgets. A one-hour
cloud cost pilot is sufficient to decide whether to fund a larger label batch;
it is not a model-strength experiment.

For the first learning comparison, change the 37–38 **policy-target source**
only; keep the established value-target recipe. Continue ordinary joint learning
so value can improve from the better resulting trajectories and available exact
anchors. Test bootstrapped value targets as a separate later change if needed.
This avoids attributing a result to several new losses or data sources at once.

Retain full games and 35–36 examples. Evaluate the candidate against its frozen
start on all bands, then use fresh full games for promotion. A stronger teacher
search without a stronger learned student is a search result, not learning.

## Stage 3: repeat from 38 toward 39–40

Advance when the 37–38 teacher/learning experiment demonstrates useful gains and
the previous band is retained. Freeze the improved model, move the evaluation
frontier to 38, and repeat the short teacher-quality and training pilots at
39–40. Keep a bounded exact audit even if exhaustive labels there remain costly.

Do not make solver40 a prerequisite or claim that four edges always have the
same difficulty. Connected-region structure and pending captures affect cost.
The point of bootstrapping is to use reliable learned evaluations where exact
search becomes too expensive, while continuing to test their errors.

## Decision rules and run schedule

- The existing 80 roots per band are development diagnostics, not final tests.
  For a stage decision, seek a paired improvement in searched regret whose
  family-bootstrap interval excludes zero, with value ordering as a supporting
  diagnostic and no clear deterioration in win preservation or retained
  bands. Show effect sizes and remaining errors. A wide interval means
  inconclusive, not permission to relabel the same result as success.
- Before a confirmation, fix fresh families, search settings, training endpoint,
  budgets and primary metric. If the pilot is too small, plan a larger independent
  confirmation once; do not repeatedly sample until a favorable result appears.
  Unknown-label coverage and its selection bias remain part of the report.
- Full-game champion promotion uses a predeclared fresh paired match (initially
  200 games) and its game-score interval. A clear improvement requires its lower
  bound above 0.5. A band improvement alone can justify another bounded experiment
  without crowning a new overall champion.
- Start changed recipes with a resumable **1–2 hour work-budget pilot**. An
  all-child labelling pilot may consume that allowance before a useful training
  comparison; report that outcome. Only promote a promising recipe to an
  eight-hour work budget, followed by the separately budgeted final match.
- Continue latest learner weights within a declared run. Save every completed
  stage. Keep separate learner, frozen teacher, and proven-incumbent identities;
  record why each pointer changes. Do not auto-advance bands merely because an
  iteration counter or timer expired.

Once a new checkpoint improves against the incumbent, measure it against Qdab
and the verified Dabble/PRsBoxes adapters. Keep frozen-position tests, full-game
matches, and equal-time comparisons separate. BoxesZero remains a source of
ideas until its usable checkpoint provenance is established.

## Work split and implementation order

| Work | Owner / resource | Deliverable |
| --- | --- | --- |
| Finish current declared run | User's 3070 | Results archive and checkpoint identities |
| Complete lower-band labels and family-aware reports | Codex cloud CPU | Cached labels, coverage, regret/order diagnostics; no monitor data in training |
| Audit stage-1 branching and ancestry | Codex; independent review can be delegated | Small-board/replay tests and a resumable short-pilot launcher |
| Train and score the matched stage-1 pilot | User's 3070, one supplied command | Fixed-budget control/candidate results and retained-band checks |
| Prototype frontier teacher and verify backup semantics | Codex CPU tests; 3070 for batched timing | Exact small-board agreement, capture/turn tests, bounded real-board teacher comparison |
| Run stage-2 short comparison, then overnight if justified | User's 3070 | Separate target provenance, latest-weight continuity, fixed final evaluation |

The next engineering deliverable is the **band report plus short-pilot launcher**,
not an automatic 40-edge training run. In parallel, a collaborator can review
value-target perspective/collapse handling or measure exact-policy label cost on
disjoint training families. GPU runs remain sequential. No current-run code,
weights, labels or settings need changing to prepare these pieces.
