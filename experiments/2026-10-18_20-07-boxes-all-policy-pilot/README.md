# Complete-policy supervision at 35–36

This isolated pilot compares broader policy supervision with mistake-only
policy targets at 35–36 undrawn edges. Both arms start from the current verified
proof34 iter24 checkpoint. It does not change the ongoing training loop.
Run from the repository root after pulling `astra/all-policy-pilot`:

```bash
bash experiments/2026-10-18_20-07-boxes-all-policy-pilot/run_pilot.sh
```

The default is 200 fresh full-start games collected once from the verified
proof34 iter24 with 400 simulations, solver34/50k and four processes. Games stop
when the solver settles the result, as in the successful overnight run. The
launcher verifies the original run configuration, unchanged search code and
the same native extension; no rebuild is required.

Both arms start from that same checkpoint and use identical full-game positions,
exact root values, complete-group collapsed child values, family splits, batch
order, augmentation, learning-rate schedule and optimizer updates. Only policy
targets differ:

- **Control:** complete exact policies where the played move lost at least two
  final score-margin points (the mistake-only selection).
- **Treatment:** complete exact policies at every eligible searched 35–36 root,
  including roots where the played move was already optimal.

Both arms share the extra value supervision. Thus this isolates policy coverage;
the control is not an exact replay of the entire old loop. Incomplete child
groups never become policy targets. No loss upweighting or branch starts are
introduced. Full-game trajectories retain opening supervision. Known diagnostic
families, including symmetry-equivalent first-four-edge prefixes, are excluded
before training preparation. There is no claim that earlier warm-start weights
have never seen any related positions.

Training uses four epochs per arm, batch 256, learning rate 0.0001, natural target
weight and the existing matched trainer. The actual batch/augmentation/LR traces
must match. Read the changed target count and training/validation optimal mass:
a short null is not evidence that complete-policy supervision cannot work.

Evaluation uses the frozen 80 roots per band at 35–36 and 37–38, 400 simulations and
solver34/50k. The recent `proof24-bands-me7lvz_n` exact caches are reused when
present; the committed finite-budget cache is the fallback. No new labels at 40
are requested. There are two paired comparisons:

1. **Primary: treatment versus control.** Read selected-move exact regret,
   preserved wins and source-family intervals on common known roots, with policy
   optimality as a secondary diagnostic.
2. **Retention: treatment versus frozen iter24.** Check for regression under the
   same search settings. Do not subtract gains/intervals across comparisons when
   their common-known populations differ.

This is a directional development pilot. 80 roots per band cannot establish a
small improvement reliably; incomplete labels can bias coverage toward easier
roots. It does not promote a checkpoint or claim full-game strength. A promising
result should be followed by a fresh paired 200-game match before replacing the
incumbent or scheduling a longer run.

Stage limits are 60 minutes collection, 90 minutes labelling, 15 minutes training and
60 minutes for each of two probes: 4h45m total, plus shutdown/reporting. These are
caps, not runtime predictions. Progress appears every 30 seconds. On timeout or
interruption, completed stages and completed game labels remain reusable. An
incomplete collection, training stage or probe restarts in a fresh attempt;
successful exact probe labels are retained. The printed `resume.sh` resumes the
same frozen run. Source checkpoints, completed experiments and league state are
untouched.

Send the printed `results.tgz`. It includes configuration, hashes, coverage,
training metrics, probe rows and paired reports. Checkpoints and NPZs remain
local. New code is covered by small-board exact-label/matched-training tests and
launcher interruption/resume tests; the current 5x5 weights require the user's
3070 machine.
