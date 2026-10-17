# Overnight learning with proof-filtered targets

This continues learning after `5x5-solver34-8pt6ai3d`. Its predeclared final
checkpoint, **iter15, beat original iter0 126–74**, with a paired 95% game-score
interval of 56–69.5% under identical solver34/50k and 400 simulations. The archive,
independent replay and ancestry analysis are retained in
`experiments/2026-10-17_15-12-boxes-solver34-learning-result/`.

The new run starts **fresh replay and a new league from verified iter15**, with
corrected policy targets and the exact-equivalent faster solver. Original iter0
and the previous experiment remain preserved. The new run's local `iter0.pt`
is a copy of the old run's **iter15**; startup and final summaries identify it
as `solver34 iter15` and retain its hash to avoid confusing the two baselines.

The question is whether the corrected self-play targets teach a stronger 5x5
network. The small-board mechanism test improved policy decisions but worsened
value predictions, so a learning gain is not assumed. The final match compares
the last learner with frozen **iter15** under identical search. This is an exploratory loop,
not a matched-data A/B attribution experiment: faster collection can also change
the number of iterations completed in eight hours.

## Start the continuation

Update the Windows GitHub mirror, then in WSL:

```bash
git switch astra/proof-target-overnight
git pull
bash experiments/2026-10-17_14-14-boxes-proof-overnight/run_overnight.sh
```

The fresh-run launcher **rebuilds the C++ extension automatically** in the active
uv environment, verifies that the loaded binary matches the new build, and runs
the proof-policy and solver-canonicalization checks. Build/check logs are printed
under `work/proof-preflight-*`; this takes a few minutes before collection starts.
The venv, current checkout and imported modules are pinned for every subprocess.
The usual C++ build prerequisites from `GETTING_STARTED.md` must be installed.

The default starting checkpoint is
`experiments/2026-10-08_14-42-boxes-3x3-loop/checkpoints/5x5-solver34-8pt6ai3d/iter15.pt`.
Its SHA256 must equal
`3f17b19f3caa66905be1f33a960c038fec01fa499cda483ed00367b99e164dac`.
`--reference /path/to/iter15.pt` handles another location. A missing or changed
default fails rather than silently reverting to older weights.

An explicit reference may also select the original iter0 for a separate control
experiment, with SHA256
`46c7264c4a0e9c7efa1cfccbb1b511c1c5faa3824fb69a62705d8cbfb1f953f0`.
Other checkpoints are rejected. Whichever verified reference is selected is
both the learner's initialization and its frozen final opponent.

## Fixed recipe and evaluation

- Eight active work hours, finishing the current iteration, then 200 paired
  full games of the last learner versus frozen iter15. Build/check time and final arena
  are additional. `--hours 10` changes a fresh run's work budget.
- Both final players use solver34/50k, 400 simulations, identical utility,
  four-edge paired openings and fresh final seed **20261029**. The previous run
  used 20261028. This final seed was fixed before viewing the previous run's result.
- Each iteration collects 100 full-start self-play games with the latest weights,
  four processes/eight workers and the same search settings as the previous run.
- Before any relabelling or training, every collected game must carry
  `proof_filtered_visits_v1`, raw diagnostics and valid masked visit counts.
  `targets-itN.json` reports changed positions and removed raw mass by undrawn
  band. This validates saved targets; it is not a new exact proof audit.
- The existing natural-weight exact mistake labels at 35..36, successor value
  labels, four training epochs, learning rate 1e-4, eight-iteration window,
  chain features, D4 augmentation and q_mix=0 remain in place.
- All weights continue from the latest iteration. Small development arenas do
  not reset the learner. Optimizer state is still initialized each iteration.
- No previous replay or Qdab data is imported. Archives retain all checkpoints'
  hashes; the checkpoints themselves remain on WSL.

The label stage still has a two-hour timeout. A timeout or invalid collection
stops visibly and preserves completed work; incomplete labels are not silently
used. This is not a guarantee of eight uninterrupted hours on pathological games.

## Resume and return results

The run prints its saved `resume.sh` and `results.tgz`. Resume validates source,
build inputs, currently imported binary, interpreter and checkpoint identities;
it does **not** rebuild. Use that saved resume command after an interruption.
Do not switch code or rebuild while a run is active or needs resuming.

The archive includes preflight/build logs, recipe, target-accounting reports,
all stage logs, timings, checkpoint hashes and independently replayed final games.
Send that archive for analysis. The corrected-target continuation has not run yet.

## Validation

The actual fresh-run preflight rebuilt the optimized extension and passed its
17 proof-policy/canonicalization checks. The continuation's Python suite passed **817
tests, 29 skipped**, excluding the known upstream GPU-lease test. The controller
integration runs real collection, audit, bounded exact labelling, training, paired
final games, independent replay, completed-run resume and interrupted-state
recovery, with and without the new audit. Nine wrapper tests cover stale runtime,
source and baseline identities, including rejection of an accidental reset to
older weights or a different frozen opponent on resume. All three native C++ suites passed (143 Go,
42 Boxes and 34 MCTS assertions). Ruff, focused strict type checks and shell syntax
checks passed. The continuation's GPU throughput and further learned improvement
remain unmeasured; the previous run's gain predates both new fixes.
