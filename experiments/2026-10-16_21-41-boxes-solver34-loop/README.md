# Continuing self-play with solver 34

The same frozen iter0 won 175–25 against solver30/50k when given solver34/200k,
at roughly nine times the recorded per-move time. The follow-up solver34/50k
match is still running: 62–18 after 80 games, about half the previous match's
elapsed time at that point. These are search improvements, not learned gains.

This exploratory training run starts again from frozen iter0. Each iteration
plays 100 fresh full-start self-play games at 400 simulations with solver34/50k,
stopping each game when its final outcome is proven. The teacher is the latest
network with this stronger search. Exact policy labels replace consequential
mistakes at 35–36 undrawn edges (at most two per game); successor values are
also labelled. Training uses four epochs over an eight-iteration replay window,
natural label weight, chain features, D4 augmentation, and learning rate 1e-4.
This conservative warm-start rate differs from the old online loop's 1e-3.

The learner always continues from the previous iteration's weights. The small
20-game development arenas cannot reset it. AdamW is still initialized anew for
each iteration by the existing trainer; this is weight continuity, not optimizer
state continuity. All checkpoints are retained. No old Qdab-labelled replay is
added in this experiment: it asks whether newly generated stronger-search targets
teach useful play. It does not isolate every difference from earlier runs.

Run from the repository in WSL after the current arena finishes:

```bash
bash experiments/2026-10-16_21-41-boxes-solver34-loop/run_overnight.sh
```

The default is **eight hours of work**, including collection, labelling, training
and development matches. It finishes the current iteration after that budget,
then plays **200 fresh paired full games of the last learner against frozen
iter0**. Both sides use solver34/50k and 400 simulations, so the comparison
measures learning on top of the stronger search. Total elapsed time exceeds eight
hours by the remainder of an iteration plus the final match. There is no fixed
three-iteration stop. `--hours 10` changes the work budget for a fresh run.

The launcher prints `resume.sh` and `results.tgz` paths. Resume uses saved settings,
counts previous active work, reuses completed collections/labels, and validates
checkpoint, source and extension hashes. Completed runs are verified and reported
without retraining. Failures stop visibly and preserve evidence; they do not silently
substitute incomplete labels. The label stage retains its two-hour safety timeout.

The standard reference is automatically found in the recent solver-arena folder,
the previous Qdab run, or the original probe snapshot, and its known SHA256 is
verified. `--reference /path/to/model.pt` explicitly selects a different checkpoint.
Existing checkpoints, game data and leagues are preserved under a fresh tag.

The output archive includes the recipe, code hashes, all stage logs, training
manifests, league, timings, checkpoint hashes and independently replayed final
games. Checkpoints and NPZ data stay on WSL. A stronger network is a hypothesis;
the final matched arena is the readout, not the small development promotions.

Validation: 776 tests passed and 29 skipped (the known upstream GPU-lease test
excluded), using a freshly compiled C++ extension. The new real CPU integration
covers collection, preparation, training, final paired games, independent replay,
completed-run resume, recovery of a committed iteration, and rejection of modified
results. Ruff, launcher type checking and shell syntax checks passed. GPU timing
and learning improvement remain to be measured by the overnight run.
