# Solver cost and proof-filtered teaching targets

The completed solver34/50k match won **163–37** against the same frozen iter0
with solver30/50k, over 100 side-swapped openings. Independent replay verified
all 200 games. The opening-pair bootstrap game-score interval was [0.755, 0.87],
mean margin +4.11, and recorded mean move times 1.7564s versus 0.3968s (4.43x).
The match took 3,211s. This establishes stronger search at higher cost; it does
not establish learned improvement or equal-time superiority.

While the new overnight training run continued on its original code, this audit
used the completed match to check solver work and the policy target path.

## A real target mismatch

`BoxesMCTSTree.get_action_probabilities()` applies the existing proof filter
before selecting a move. Shared gameplay previously saved raw child visit
counts into `mcts_visits`. Those counts include exploration of actions that
search subsequently proved inferior, so the network could be taught a policy
that search itself would never play.

A 1x2 regression case has legal edges 5 and 6, with exact child win values 1
and 0. Batched search gives both edges eight raw visits, then proves the root
and selects edge 5 with probability one. The old training target is 50/50.
The corrected target retains edge 5's eight visits and zeros edge 6.

The fix exposes `training_visit_counts()` on Boxes results and applies it only
to Boxes metrics in shared gameplay. It uses the existing temperature-one
search policy as the support mask, preserving relative visit weights and the
search's fallback rules. A temperature-zero played move does not make the
training target one-hot. Exactly solved roots retain uniform-optimal targets.
Raw counts remain available from the tree and in a new `mcts_raw_visits` NPZ
array; `policy_target_semantics=proof_filtered_visits_v1` identifies new
collections. Existing data and search decisions are unchanged. Later exact
policy relabelling can still replace the teaching counts as before.

In the 48 sampled quiet roots, solver34 produced two affected targets (IDs 1
and 10, at 32 and 34 undrawn): **73.61% and 91.67%** of raw visit mass was on
actions excluded from selection. Both roots were proven during MCTS. Six other
roots were solved before MCTS and already had uniform-optimal targets. The
remaining 40 searched roots, including every 35–40 root here, were unaffected.
Solver30 produced no affected targets in this sample.

These are equal-count sampled bands, not the traffic distribution of self-play.
They cannot estimate the bug's contribution to past failed learning runs. The
35–36 mistake-only exact-policy stage does not repair these 32/34 examples;
more generally a correctly selected move can still have a misleading raw
target. Only newly collected data receives the fix. A matched learning test is
still required before claiming a stronger model.

## Where solver time goes

Eight quiet roots per undrawn count were selected deterministically from the
completed arena. Opening families are distinct within each band, but can recur
across bands. These are decision-cost diagnostics, not a strength test. Both
agents use the same checkpoint, 400 simulations, leaf batch 16, c_puct 1.5,
no noise, temperature zero, terminal proofs and shaping (lambda 0.5, k 6), and
a 50,000-node **per-call** solver budget. Every root gets a fresh agent/table.
Inference is CPU float32 with one Torch thread.

| Solver reach | Bounded calls | Nodes across those calls | Aborted calls | Bounded solver wall time | Neural wall time |
|---|---:|---:|---:|---:|---:|
| 30 | 4,486 | 38,334,285 | 337 | 43.20s | 74.58s |
| 34 | 7,377 | 191,639,804 | 2,707 | 222.02s | 40.88s |

Solver34's total sampled search time was 263.44s, with about 84% inside bounded
solver calls. Unbounded calls used to enumerate exact root choices took only
0.179s (200 calls, 126,907 nodes). They are not the main cost in this sample.
At root 28 (36 undrawn), solver34 made 366 bounded calls, visited 14,409,679
nodes and exhausted the budget on 243 calls. A 50k budget therefore does not
mean 50k nodes per played move.

The 30/34 jobs ran concurrently and other checks overlapped part of the run;
timings are descriptive wall times under that load, not a controlled speed
comparison or a prediction of GPU throughput. Node/call counts show the work
independently of that timing limitation. The scratch instrumentation records
only entry/exit counters. The first eight solver34 roots were rerun with the
uninstrumented extension: visits, chosen moves, selection distributions and
root proof status matched exactly.

The next performance experiment should control repeated expensive leaf probes:
for example a conservative structural trigger or allocation of a total solver
budget per move. That changes which proofs search receives, so it must retain
the current solver34 baseline and compare strength as well as cost. No solver
budget change is included in this fix.

## Reproduce

`data/provenance.json` records source, checkpoint and extension hashes. The
original match and frozen roots are included; weights are not. From the repo:

```bash
EXP=experiments/2026-10-17_10-17-boxes-search-target-audit
uv run python "$EXP/summarize.py" "$EXP/data"

# Build scratch instrumentation; use a new output directory on each rebuild.
uv run python "$EXP/instrument.py" --output work/solver-profile-extension
PYTHONPATH=src:work/solver-profile-extension uv run python "$EXP/bench.py" \
  --games "$EXP/data/source-match/games.jsonl" --positions work/frozen-roots.jsonl

# REFERENCE must be the checkpoint whose hash is in provenance.json.
for REACH in 30 34; do
  PYTHONPATH=src:work/solver-profile-extension OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -u "$EXP/bench.py" --positions "$EXP/data/positions.jsonl" \
    --checkpoint "$REFERENCE" --solver "$REACH" --output "work/solver$REACH.jsonl"
done
```

Validation: 784 tests passed, 29 skipped, using an uninstrumented extension
built from the current sources; the known upstream GPU-lease test is excluded.
The eight added cases cover batched/unbatched proofs with/without shaping,
unproven targets with/without proofs, non-one-hot teaching despite deterministic
selection, and actual `play_game` -> NPZ -> dataset policy against the exhaustive
1x2 oracle. The running overnight branch is unchanged; this fix is separate.
Ruff passed for the changed Boxes code, tests and experiment scripts. Targeted
type checking passed with imported modules skipped and missing-extension/unused-
ignore diagnostics suppressed. The unrestricted CI-style mypy command reports
112 errors in 12 imported files on both this branch and its unmodified base;
the diagnostics are identical after ignoring shifted line numbers.
