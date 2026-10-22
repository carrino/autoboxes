# Ordinary replay: confirmation and two fresh-data cycles

The saved fitting endpoint won 222–178 against proof34 iter24 (55.5%, paired
95% interval 51–60%). This run confirms that fixed checkpoint on new openings,
then asks whether the ordinary-training recipe improves further with new games.
The 25% exact-root quota is not used: its endpoint regressed in the fitting study.

Pull `astra/all-policy-pilot` into the Windows mirror, then in WSL:

```bash
git switch astra/all-policy-pilot
git pull
bash experiments/2026-10-22_13-11-boxes-ordinary-continuation/run_overnight.sh
```

No native rebuild is required. The launcher verifies the existing native binary,
source code, saved replay receipts, and both starting checkpoint hashes before
work. It prints an isolated `ordinary-next-*` results directory and `resume.sh`.
Leave the checkout and native build unchanged while it runs.

## Fixed protocol

1. **Confirmation:** 400 games of the frozen ordinary fitting endpoint against
   proof34 iter24. Use 200 fresh four-edge openings with sides swapped, seed
   20261108. Both players use 400 simulations and solver34/50k, with the same
   search settings as the fitting study. The score does not select, promote,
   reset or gate training; this continuation is declared in advance.
2. **Cycle 1:** collect 200 full-start self-play games using the ordinary endpoint
   at 400 simulations, solver34/50k and stop-when-solved. Retain proof-filtered
   policies. Label consequential mistakes at 35–37, at most three per game,
   using the saved label settings and four CPU workers. Exclude frozen monitor
   families/positions. Combine these games with all ten existing preparations
   (1,000 source games), preserving exact child values and source-family splits.
   Train 8,192 ordinary updates, batch 256, constant LR 0.0001, fresh AdamW.
3. **Cycle 2:** repeat from cycle 1's final weights, retaining both new collections
   and the original replay. Another 200 games and 8,192 ordinary updates. The
   learner always continues; there is no small per-cycle promotion match.
4. **Final comparison:** 400 games of the LAST learner against the frozen
   **ordinary starting endpoint**, not iter24, using new paired openings, seed
   20261109. This directly measures the continuation's gain. A successful
   confirmation plus this match is not a direct final-vs-iter24 measurement and
   does not establish an external-engine ranking.
5. **Development probes:** final versus start on the fixed 80 roots at 35–36 and
   the fitting run's 27 held-out decision roots at exactly 37. Reuse exact caches;
   report regret and winning positions preserved. These are small, reused
   development sets, not a fresh generalization test or a promotion gate.

Intermediate weights and fitting reads are saved every 2,048 updates without
resetting the optimizer or schedule. These are diagnostic snapshots; only the
predeclared final endpoint enters the final arena. The family split stays fixed;
validation removes D4-equivalent training inputs and duplicates. Because the
starting checkpoint already saw the old training replay, replay fitting metrics
remain diagnostic. Preserve old position coverage while testing the larger
training budget; do not widen to 38 based on training accuracy.

## Timing and recovery

Expected total is roughly **6–9 hours**, not a hard total deadline. Per-stage caps
are four hours per arena, two hours per collection, two hours per labelling pass,
30 minutes per replay copy, 90 minutes per training cycle, and two hours per
development probe. They limit pathological stages; the run ends after its fixed
two cycles and evaluations even if some of the night remains.

Completed stages verify hashes and are reused. An interrupted labelling pass
reuses completed per-game bundles. An interrupted collection or training stage
restarts that stage in a fresh attempt; intermediate snapshots are not optimizer
resume points. Confirmation/probe timeouts preserve completed work and do not
block independent stages. A failed collection, label pass or training cycle
stops dependent work instead of silently omitting new data. A changed checkpoint,
code file, source game, receipt or native runtime is rejected.

Return the printed `results.tgz`; it excludes checkpoints and NPZs, which remain
available locally. This run does not alter existing league state, source games
or checkpoints. No outcome automatically promotes a model.

## Starting provenance

- Source run: `fit37-wau7ku0x`
- Source config SHA256: `19109763407d8edce1169acf0b03e899c542ca85263be78332718f4829527da6`
- Ordinary endpoint: `e913bc0af1b6b0df4ecd44b55f2f41af5d5c7e823af7e0a12f912773408fc153`
- Iter24: `73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d`
