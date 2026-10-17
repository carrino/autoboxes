# Where to extend the learning curriculum

The agreed next steps, stage checks and cloud/3070 work split are in
[CURRICULUM_PLAN.md](CURRICULUM_PLAN.md). The plan is preparation, not a change to
the active run or evidence that a band has been mastered.

The completed solver34 run beat its frozen iter0 126–74 with matched search. That
establishes stronger overall play, not mastery through 38 undrawn edges. Leave the
active proof-target run unchanged. The next candidate experiment should retain
35–36 corrections and pilot additional corrections at 37–38, rather than move the
entire exact-label budget to 39–40.

## What the active run teaches

The recipe in `experiments/2026-10-17_14-14-boxes-proof-overnight/overnight.py` uses
the timed solver34 controller's recipe:

| Range | Supervision/search |
| --- | --- |
| Early game, including 39–40 and 37–38 | Ordinary full-start self-play targets; MCTS policy on searched decisions and eventual game or solver-adjudicated outcome |
| 35–36 | The same, with exact policy/root-value replacements at up to two consequential searched mistakes per game |
| Children of completed exact-policy roots | Additional exact value-only targets after forced-capture collapse |
| At most 34 | Bounded exact probes during search: 50,000 nodes per probe, not guaranteed exact play throughout this region |
| Below 24 | Excluded by the training cutoff |

`START_FRACTION=0` and `START_FROM=""`: the configured `START_UNDRAWN="36 40"`
does not start games in that band. The current approach is full-board training
with targeted corrections, rather than a staged backward curriculum. The new
run also corrects proof-filtered search policy targets; that change has not yet
been evaluated in its final match.

## Frozen development monitor and bounded cost pilot

Source: all 200 complete games from `5x5-solver34-8pt6ai3d`'s final iter15-vs-iter0
match. Both checkpoints used solver34/50k and 400 simulations. Source provenance
and the verified final result are in the adjacent experiment
`2026-10-17_15-12-boxes-solver34-learning-result`.

We replayed every edge using both Python and native board implementations,
checking legality, board contents, player to move, and final recorded margin.
We froze **240 decision roots: 80 per band, 40 per exact undrawn count**. Each
band has 80 distinct symmetry-canonical four-edge opening families. Both games
of an arena opening pair share a family, even if their fifth moves differ.
The original eight-edge training-family identity is also saved.

Selection preceded solving. There is no forced-capture prefix at these roots;
one of the 240 is a take-all/keep-control decision, so “quiet” in the original
snapshot's metadata means no forced prefix, not absence of every capture choice.
The same 36-position subset (six per exact undrawn count) was tested twice.

Each root receives a fresh 2^22-entry transposition table. Its actually played
child reuses that table. Each solve has its own finite node budget. This is a
**root/selected-child cost pilot**, not a full optimal-policy labelling pass.

| Undrawn | Root labels at 2M | Root labels at 20M | Root + played child at 20M | Aggregate solve seconds at 20M |
| --- | ---: | ---: | ---: | ---: |
| 35–36 | 8/12 | 12/12 | 12/12 | 34.8 |
| 37–38 | 0/12 | 6/12 | 6/12 | 197.1 |
| 39–40 | 0/12 | 0/12 | 0/12 | 361.7 |

Total wall time was 28.0 seconds at 2M and 172.7 seconds at 20M with four workers.
Aggregate solve seconds sum workers' elapsed solve time, not wall time or
individually measured CPU time. These times are for this cloud machine, not a
prediction for the user's desktop.

This sample shows a sharp label-coverage drop above 36 at the current training
label budget. It does not show that every 40-edge position is intrinsically hard;
topology matters, and these roots come from two particular policies. Unknown
labels are neither correct moves nor mistakes. The original summary represents
zero observed mistakes when zero labels are known; that is a zero denominator,
not a zero error rate.

The known played moves include mistakes at both 35–36 and 37–38, but those moves
belong to two different models on different positions. Do not use their averages
to compare checkpoints or infer that the learner mastered a band. Full optimal
policy targets additionally require every legal child to be solved; root coverage
alone does not establish their cost or yield.

## How to choose the next experiment

1. Keep these development roots out of training. They are from the match already
   used to assess iter15, so they are not an untouched final test set. The
   four-edge opening families define evaluation clusters; retain the eight-edge
   identities and prefixes for training exclusions too.
2. After the active run, evaluate the frozen incumbent and new candidate on these
   same roots at matched solver34/50k, 400 simulations, and search parameters.
   The existing `alpha_go.boxes.search_probe` accepts the saved `positions-*.jsonl`
   and exact-only `labels.json` cache. Compare raw-policy and searched move regret
   and preservation of winning positions. Report coverage and whole-family
   bootstrap intervals using only the common known positions.
3. Before expanding training, pilot **all-child** policy labels at consequential
   mistakes in fresh training games at 37–38. Record completed policies per
   CPU-hour, unknown-root rate, incomplete-child rate and training throughput.
   Do not turn a partially solved child set into a claimed exact optimal set.
4. If useful labels are affordable, compare a limited 37–38 allocation against
   keeping the current allocation. Retain 35–36 examples and check that the new
   learner reduces development regret without sacrificing earlier-band play.
   Confirm the selected recipe with a fresh full-game match.
5. Consider 39–40 once the above demonstrates useful yield and learning, or after
   a solver improvement makes those labels affordable. No arbitrary percentage
   accuracy gate or aggregate win rate establishes “mastered through 38.”

The archive supplies game traces and hashes, not iter15's checkpoint bytes.
Its checkpoint SHA256 is
`3f17b19f3caa66905be1f33a960c038fec01fa499cda483ed00367b99e164dac`.
Cloud neural comparisons must wait for that checkpoint or run on the user's
machine later; the old cloud `candidate.pt` is a different, regressed iter22.

## Reproduction and validation

```bash
EXP=experiments/2026-10-17_15-27-boxes-band-progression
tar -xzf "$EXP/data/pilot-results.tgz" -C "$EXP/data"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python "$EXP/prepare.py" \
  --games "$EXP/data/pilot-20m/final-games.jsonl" \
  --output work/band-pilot-reproduction --workers 4 \
  --per-edge 40 --pilot-per-edge 6 --node-budget 20000000 --seed 20261017
```

Use a fresh output directory. Each run preserves its script snapshot, source-game
bytes, hashes, frozen sample, per-solve values/nodes/times, exact-only reusable
cache, and coverage summary. The recorded snapshots precede a documentation/type
cleanup in the current script; frozen prefixes and label computations are
unchanged. The current script uses `null` for mistake counts with zero known
labels rather than the original snapshot's zero.

The complete 153,733-byte evidence archive has SHA256
`896b200391fc157f3941aebb1f01f5bbb945ab0f0acb04139b0bae1a388943ec`.
Summaries, frozen positions and the 20M label cache are also stored as plain files
for inspection and later checkpoint probes.

Validation: all 12,000 source moves replayed with Python/native parity; generated
position files passed `search_probe.load_positions`; frozen sets agree between
budgets; exact root/child perspectives independently recomputed with Python rules;
60 relevant rules, symmetry, validation and search-probe tests passed; Ruff and
focused mypy passed. No active training or search implementation was changed.
