# Qdab teaching result: no demonstrated strength gain

Completed run `qdab-teacher-ab-lb2xd5_9`: 200 full games per endpoint on the same
100 side-swapped opening families. Ordinary learner targets won **28/200 (14%)**;
Qdab targets won **27/200 (13.5%)**. The matched score difference is **−0.5
percentage points**, with a 100-family bootstrap interval **[−7,+6] points**.
This does not establish either improvement or equivalence. Keep the starting
`c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`
checkpoint as incumbent; neither endpoint is promoted.

The original 17/100 benchmark used different openings. Comparing it to 28/200
does not by itself establish regression from training. Qdab used about 2.1
seconds per edge versus our 0.52–0.54 seconds, so this is configured strength,
not equal-time evaluation. The two complete matches took 8.18 hours.

## Teaching uptake

There were only **164 eligible teacher rows among 98,747 training rows**. Both
arms sampled the same 524,288 examples with the same augmentation schedule;
teacher rows appeared 869 times, or **0.166% of all draws / 0.293% of policy
draws**, averaging 5.30 exposures per teacher example.

| Read | Ordinary targets | Qdab targets |
|---|---:|---:|
| Teacher probability, collector-training families | 3.534% | 3.681% |
| Teacher top-move agreement, collector-training families | 7/180 | 11/180 |
| Teacher probability, heldout families | 3.561% | 3.577% |
| Teacher top-move agreement, heldout families | 3/60 | 3/60 |
| Exact-optimal raw-policy choices, 37 edges | 14/59 | 14/59 |
| Wins preserved by raw policy, 37 edges | 26/40 | 26/40 |

The collector-training read includes 16 examples outside the trainer's actual
fitting split. Teacher agreement is approximate supervision, not exact optimality.
Heldout teacher NLL improves slightly (3.77079 to 3.74985); a post-hoc 15-family
interval for the difference is [−0.03786,−0.00243] nats. The mean teacher-action
probability difference is +0.0159 percentage points [−0.0503,+0.0803]. These
small distribution changes did not demonstrate better heldout choices or wins.

Dilution is a supported concern, not proof that it is the sole bottleneck or
that a particular stronger weighting will work. Before another long arena,
use a larger teacher bank and a predeclared short matched fit with explicit
teacher exposure, preserving ordinary replay. Check target uptake, heldout
move quality and exact-position regressions first. Avoid another long run
memorizing the same small set. No new sampler or fitted candidate is claimed here.

## Verification and limitations

An independent board replay verified all 400 complete games / 24,000 edges,
9,652 Qdab RPC states and 11,138 Qdab edges, including retained capture turns.
Both arms share all 200 slots, runtime, search/evaluator settings and 71 runtime
source hashes. The original outer configuration is unchanged; the only accepted
runner change is the recorded duplicate-lock correction. One failed first-game
attempt is retained; all 400 scored games completed normally.

Control has 138 one-box losses; the teacher endpoint has 144. Mean winning
margins are about +14.4, leaving positive overall mean margins (+0.62/+0.68)
despite poor win rates. Margin alone is not the success measure, and narrow
losses do not identify when the winning opportunity was lost.

Endpoint weights, raw sample indices and prepared replay were omitted from the
upload. Their hashes match across available receipts, but omitted bytes cannot
be independently checked. Compact original result and training receipts are in
`teacher-result.json`; full local audits are in `work/qdab-teacher-result-audit`
and `work/qdab-teacher-learning-review`.

## Data reuse and next experiments

The user authorized recycling completed evaluation games after freezing their
result. Preserve this report and original game receipts. Derived Qdab move
targets must remain approximate, with capture bundles and mover state verified;
observed outcomes must not become exact value labels. Retire all used opening
families from future evaluation and keep related games in the same split.
Future tests must use fresh families; the archived result is not rewritten.

The completed local exporter produced **2,493 unique approximate Qdab teaching
positions** at 38–50 edges remaining from these 400 games / 100 families.
It verified all 9,652 RPCs, skipped 10 roots with forced captures pending, and
found no family or relevant semantic-state overlaps with the 35 heldout families
from the two cloud collections. Source receipts and original outcomes are retained;
no learner search visits were invented. `work/qdab-retired-match-teachers/bank-final`
contains the candidate JSONL bank and retired-family ledger. This is a prepared
data artifact, not an NPZ training bundle or a fitted model.

Independent cloud collection has completed 120 additional games, split into
100 training and 20 validation families. Exact labeling finished for 471/480
complete root/child groups at 35–38 undrawn edges (23,271 known state records).
These are correlated states from 120 games, not 23,271 independent games.
Known benchmark exclusions were supplied; later-state overlap with this newly
uploaded match must also be accounted for before combining fitting datasets.

Next search diagnostic: unchanged starting weights, matched 400 simulations and
solver34/50k, comparing current margin shaping against win-only utility on a
small predeclared training-family set. Keep its result separate from learned
improvement, and leave fresh validation families unused for tuning.
The bounded cloud probe has now launched on 32 roots from 32 distinct training
families, eight roots per edge count, with alternating condition order and fresh
solver/agent state. Its results are pending; it does not run full Qdab games.
