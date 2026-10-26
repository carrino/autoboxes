# Learn from a stronger opponent's decisions

The latest learner won 17/100 paired games against pinned Qdab at 400 simulations
and 24/100 at 1,600, using the same weights. The matched improvement interval
includes zero. See the [completed comparison](../2026-10-23_10-19-boxes-qdab-benchmark/result-1600.md).
The first teaching comparison completed at 28/200 control wins versus 27/200
with Qdab targets. Teaching occupied only 0.293% of policy samples and did not
demonstrate better heldout choices; see [the result](TEACHER_RESULT.md).
The next command is the [short teaching screen](SCREEN.md), using 3,072 teacher
positions and an explicit 8/256 quota. It performs two fixed fits and diagnostic
reads, with no arena. The [old overnight recipe](OVERNIGHT.md) remains documented
for reproduction and is not the next recommended run.

This experiment collects fresh full games against Qdab and asks Qdab what it
would do in selected positions actually reached by our learner. The first batch
is 60 distinct opening families, with a three-hour collection budget and at most
four additional teacher queries per game. Complete games and individual teacher
receipts survive interruption. This is data collection; no weights change.

Qdab's recommendations are approximate expert actions. They are stored separately
from the learner's real search visit distributions and from any later exact
solver labels. The comparison will change the earlier policy targets while
holding positions, value targets, replay, updates and initial weights fixed.
See [PROTOCOL.md](PROTOCOL.md) for the matched training experiment and
[LABEL_PLAN.md](LABEL_PLAN.md) for bounded exact labeling after collection.

A separate [structural value experiment](STRUCTURAL_VALUE.md) supplies full
chain/loop length histograms and graph counts directly to a small value readout.
The isolated prototype leaves the existing policy and base weights frozen and
starts with a zero value correction. The initial matched fits completed without a heldout move-selection improvement;
none is a candidate for a playing-strength promotion. The current model already receives ten coarse
chain/loop planes; this experiment tests richer information and a direct path
to the value head, not the first addition of chain awareness.

The bounded offline pipeline is implemented in `prepare_structural.py`,
`fit_structural.py` and `queue_structural.py`. The queue waits for every frozen
collection game and a successful supervisor receipt, then labels roots/children
for at most 45 minutes and fits three small readouts on cached frozen-network
features. All arms use the same family-balanced sample indices and 1,024 updates.
This is a development diagnostic. Independent audits of both actual GPU benchmark
archives found no overlap with its frozen states or opening families. It does not alter the running GPU job,
production weights or arena model loading. See [STRUCTURAL_VALUE.md](STRUCTURAL_VALUE.md)
for commands, interpretation and limits.

## Evidence motivating the experiment

The cloud diagnostic used checkpoint
`c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`,
400 simulations, solver34/50k and Qdab's requested five-second setting.
Independent Python replay verified all 480 played edges.

* At 38 undrawn edges, all eight roots were solved exactly: two were winning
  for autoboxes and six losing. Two side-swapped opening pairs were losing in
  both colors. Consequently, at least one originally winning side in each of
  those pairs had already been lost before that cut. A single losing entry
  alone would not establish an earlier mistake.
* Earlier exact coverage was 4/8 at 39, 1/8 at 40, and 0/8 at 41 or 42 under
  the 20-million-node cap. Unknown results do not locate a first error.
* One autoboxes move at 37 edges changed a forced +1 win into a -1 loss.
  All 37 legal children were solved: edge49 was the only winning continuation.
  The policy assigned it 3.8% probability, rank13; the played losing edge32
  received 25.8%. Several losing children also ranked above the winner in the
  network's value estimates, before and after forced-capture collapse.
* A cold-search diagnostic on that selected position found a losing move with
  400 simulations and the winning edge49 with 1,600. The winner received
  35 versus 908 root visits. It was already explored at 400; this is evidence
  of additional search resolving this position, not proof of root starvation.
  CPU times were 7.07 and 14.35 seconds. The direct evaluator and cold solver
  differ from the original game's batched evaluator/cache history; the 400
  simulation move differed. This is neither an independent strength test nor
  a GPU timing prediction.

Raw audit artifacts remain in
`work/qdab-cloud-pilot-w895zcit/{frontier-diagnostics,earlier-frontier}/` and
`work/qdab-cloud-pilot-w895zcit/search-headroom.json`. The diagnostic games are
excluded from training collection. One selected mistake cannot establish the
sole cause of the overall strength gap.

## What the cloud and GPU each do

The cloud collects fresh strong-opponent trajectories, expert corrections and
bounded exact labels. It can also fit these small readouts after caching frozen
CNN features; the GPU remains useful for full-network training and larger fixed
full-game comparisons. Timed Qdab collection runs alone; CPU-heavy solving and
fitting wait until it finishes. Four agents can research in parallel without
pretending four shared CPU cores are four independent machines.

Collection excludes canonical four-edge families from both known benchmarks and
assigns training/validation families before results exist. Later-position
transpositions must also be checked against the completed GPU benchmark before
training. Reusing benchmark games would require explicitly retiring that test;
the current experiment does not do that.

The first batch is a transfer pilot, not a promise that a few hundred corrections
will beat Qdab. A useful next scale-up is to reuse Qdab's own saved decision
states and compute matched learner policies on the GPU. That adds supervision
without paying for the same expert query again. Success is measured on fresh
paired games against Qdab, with search settings and actual time reported.
