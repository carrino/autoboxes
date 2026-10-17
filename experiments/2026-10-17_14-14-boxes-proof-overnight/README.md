# Overnight learning with proof-filtered targets

This is the next prepared run after `5x5-solver34-8pt6ai3d`; that run's results
are still pending. It starts a **new replay and league from verified frozen
iter0**, with the corrected policy targets and the exact-equivalent faster
solver. It does not resume or alter the previous experiment.

The question is whether the corrected self-play targets teach a stronger 5x5
network. The small-board mechanism test improved policy decisions but worsened
value predictions, so a learning gain is not assumed. The final match compares
the last learner with iter0 under identical search. This is an exploratory loop,
not a matched-data A/B attribution experiment: faster collection can also change
the number of iterations completed in eight hours.

## Start after the previous results have been collected

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

The known iter0 is found in the prior solver arena, Qdab run, probe snapshot or
pending solver34 run's iter0. `--reference /path/to/iter0.pt` handles another
location; its SHA256 must still equal
`46c7264c4a0e9c7efa1cfccbb1b511c1c5faa3824fb69a62705d8cbfb1f953f0`.
Starting from a different candidate is a separate decision after the pending
results arrive.

## Fixed recipe and evaluation

- Eight active work hours, finishing the current iteration, then 200 paired
  full games of the last learner versus iter0. Build/check time and final arena
  are additional. `--hours 10` changes a fresh run's work budget.
- Both final players use solver34/50k, 400 simulations, identical utility,
  four-edge paired openings and fresh final seed **20261029**. The previous run
  used 20261028. Seeds are fixed before viewing either final result.
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
Send that archive for analysis. No training result is recorded yet.

## Validation

The actual fresh-run preflight rebuilt the optimized extension and passed its
17 proof-policy/canonicalization checks. The full Python suite passed **814
tests, 29 skipped**, excluding the known upstream GPU-lease test. The controller
integration runs real collection, audit, bounded exact labelling, training, paired
final games, independent replay, completed-run resume and interrupted-state
recovery, with and without the new audit. Six wrapper tests reject stale runtime,
source and baseline identities. All three native C++ suites passed (143 Go,
42 Boxes and 34 MCTS assertions). Ruff, focused strict type checks and shell syntax
checks passed. GPU throughput and learned 5x5 strength remain unmeasured.
