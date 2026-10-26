# One-edge curriculum pilot from proof34 iter24

The balanced 35–36 fine-tune did not beat iter24 (103–97). The following cloud
trajectory audit found specific winning-to-losing decisions at 37, while search
already converted most known winning 36-edge positions. This experiment returns
to the continuing self-play loop that produced the previous measured gains,
extending exact mistake supervision from 35–36 to **35–37**.

```bash
bash experiments/2026-10-21_13-32-boxes-frontier37/run_pilot.sh
```

Pull `astra/all-policy-pilot` into the Windows mirror first, then pull that branch
in WSL. No C++ rebuild is needed. The launcher verifies the imported extension and
search sources against the completed collection's saved identity. It reads
`balanced-policy-k3s3hb6v/inputs/iter24.pt` and `collection-config.json` under the
old experiment's logs directory. `--reference PATH` permits another copy of the
same verified iter24 weights; it does not permit the unpromoted balanced weights.

The default budget is **four active work hours**, including collection, exact
labels, training and development arenas. The current iteration finishes before
the final match starts. Then the **last** checkpoint plays 200 complete games
against frozen iter24, with identical 400-simulation search and solver34/50k.
The 100 four-edge openings are side-swapped, using fresh final seed 20261103.
Every saved game is independently replayed. Two development probes follow:
35–36 as a regression check and exactly 37 as the new band. Each probe has a
two-hour wall cap; final match and training survive a diagnostic timeout.

Expect roughly **six to eight hours including evaluation**, with substantial
uncertainty from 37-edge solve cost. Four hours is a work budget, not a total
completion deadline: finishing an iteration and evaluation add time. To set a
different work budget explicitly, pass `--hours N` on the initial launch.

## Fixed recipe

- Fresh 100-game full-start self-play batches, 400 simulations, four processes.
- Latest learner generates games and continues training; `GATE=0`.
- Exact mistakes at 35–37, up to three positions per game, 20M-node root and
  child budgets, four label workers. Win-to-loss mistakes rank first. Three
  slots avoid excluding 37 merely because 35/36 also contain mistakes.
- Only completely solved child groups become exact policy targets. Unknowns
  remain unknown. Complete groups also provide successor value targets.
- Four epochs, lr 1e-4, eight-iteration replay, chain features, minimum 24 edges.
  Full trajectories remain in replay, including opening and 35–36 positions.
- Natural sampling/weight: this is not another 25%-quota balanced fine-tune.
- Solver34/50k, search utility and simulation budgets remain unchanged.

This is an exploratory continuation, not a matched ablation isolating the new
37-edge labels from additional self-play. The final match evaluates the complete
recipe; no automatic promotion or claim of 36-edge mastery is made. Small league
matches never reset the learner or select the final checkpoint. The older loop's
iteration-based collection seeds are retained; a fresh tag does not imply an
independent RNG schedule.

## Recovery and evidence

The launcher prints a unique run directory, `resume.sh`, and `results.tgz`.
Run its printed resume command after interruption. Completed collection and
per-game labels are reused; source data and older runs are untouched. A two-hour
label-stage timeout stops visibly and requires that resume command. Even if the
work budget elapsed during interruption, this pilot finishes the started
iteration before selecting its last checkpoint. Old launchers retain their
previous behavior unless they opt into this recovery rule.

Previous policy manifests and their bundles are verified before each iteration,
preventing a missing manifest from falling back to excluded raw replay. Frozen
monitor families, symmetry-equivalent first-eight-move openings and exact root
states are excluded from entire source games. This does not exclude every
possible transposition or information already seen by the starting weights;
the probes are development diagnostics. Both monitor files are frozen before
training. The 38-edge monitor is excluded too, although it is not evaluated.

Keep the checkout and native build unchanged while running. `label-coverage.json`
reports attempted/complete policies and win-to-loss targets by edge band, so a
lack of usable 37-edge labels is visible. Send the final `results.tgz`; large
checkpoints and training NPZs stay on the WSL machine.
