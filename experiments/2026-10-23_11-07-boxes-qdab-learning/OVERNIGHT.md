# Overnight Qdab policy transfer

Train two models from the same current checkpoint, then finish **200 full games
per model against Qdab**. Expected duration on the user's RTX 3070/WSL setup is
approximately **8–12 hours**, dominated by 400 Qdab games. This is a fixed
experiment with saved progress, not a time-based training loop or automatic
champion promotion.

Pull the branch into Windows first, as usual. In the WSL checkout:

```bash
(
set -euo pipefail
git switch astra/all-policy-pilot
git pull
bash experiments/2026-10-23_11-07-boxes-qdab-learning/run_teacher_ab.sh
)
```

No C++ rebuild or separate data download is required for this change. The
hash-checked teaching payload is included on the branch. Leave the terminal
running and prevent the host from sleeping. The command prints the run directory,
its `resume.sh`, and the `results.tgz` to return when finished. After interruption,
run the printed resume command rather than starting a second experiment.

## What runs

1. Verify and freeze the current checkpoint, existing local replay, code and
   teaching payload. Choose 100 fresh opening families, paired across both seats,
   excluding recoverable replay families, all teaching families and old benchmarks.
2. Prepare two bundles with identical positions, outcomes, real exact labels,
   masks and family splits. Only explicit approximate-Qdab policy targets differ.
   The control retains the learner's real search target at those same positions.
3. Fit both models for **2,048 updates**, batch 256, learning rate `1e-4`, seed
   `20261120`, ordinary sampling and the existing chain features. Neither model
   resumes from the other. The final endpoint is fixed before results are read.
4. Check identical sample/augmentation traces and count how often the approximate
   targets actually appeared. Report teacher agreement separately on training and
   heldout families; agreement is not exact correctness.
5. Play **200 games per endpoint** against pinned Qdab on the same frozen openings:
   ours at 400 simulations, solver34/50k, unchanged utility; Qdab at requested
   five seconds per decision. All games use shared full gameplay and independent
   replay verification. The primary result is the difference in game score with
   opening-family uncertainty, not the highest intermediate score.

Both completed fits and each completed game survive resume. An interrupted fit
restarts that incomplete fit from the frozen reference; it does not modify or
repeat completed stages. Failed attempts and diagnostics are retained. Preparation
has a 30-minute technical cap and each fit a 90-minute cap; the final matches have
no whole-run time cutoff. Qdab retains its 30-second per-decision external timeout.
Keep the checkout, Python environment, GPU and native build unchanged during the run.

## Inputs and scope

Default reference is
`experiments/2026-10-08_14-42-boxes-3x3-loop/logs/qdab-match-1yafbn9k/inputs/checkpoint.pt`,
SHA256 `c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`.
Default replay is the existing ordinary-four cycle-4 preparation:
`logs/ordinary-four-aqowx5f8/cycle-4-prepare/attempt-tbyadiie/prepared` under the same
loop folder. Explicit `--checkpoint` and `--replay` paths are available if those
files were moved; checkpoint identity must still match.

The cloud collected **60 new games**, with 45 training and 15 heldout families,
1,051 learner-searched positions and 240 approximate teacher recommendations.
The payload contains the original complete-game and recommendation receipts.
It is 733,841 compressed bytes; SHA256 is
`375869d1835178f1749d425ca3d1c6ef8b3c4e12301039b4cad90464e85a703b`.
The exporter includes exclusions from both completed 120-game Qdab benchmarks,
their 35–38-edge raw/collapsed children and the collector's heldout families.
It retains source provenance and never represents Qdab's advice as an exact label.

The trainer applies its own stable replay split after the collector's heldout
families are excluded. Eligible teacher counts and actual exposures are therefore
reported from the loaded fitting dataset. This is deliberately a small transfer
experiment; 240 recommendations do not establish enough data to dominate Qdab.
The untrained starting model agrees with Qdab on only 5% of the 60 heldout queried
actions, but multiple good moves may exist, so disagreement alone is not a blunder.

An optional `--prepare-only` checks and constructs the bundles without training or
playing. An explicit `--games 40` makes a shorter screening experiment; the
overnight default remains 200 per arm. Do not change the sample after seeing scores.

## Cloud work alongside the GPU

The cloud handles opponent-game collection, provenance checks and exact labeling.
The first structural-feature and child-ranking readout tests completed without
better heldout move selection. Ranking improved ordering among losing moves more
than separating wins from losses; none of those heads is being integrated into
this GPU run. Further fresh teaching games are kept separate from this frozen
experiment and may supply a later training round.
