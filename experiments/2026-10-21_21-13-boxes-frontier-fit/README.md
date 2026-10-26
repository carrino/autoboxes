# Overnight fitting test on saved frontier37 labels

The frontier37 continuation won 98–102 against iter24. Its final net chose an
exact-optimal move on 22% of its labelled training roots (35–37 pooled). This
experiment asks whether increased exposure can fit those targets and whether
any fitting gain transfers to search and complete games. It reuses all ten
completed policy preparations from `5x5-frontier37-ezpgdpxy`: no collection or
relabelling stage is run.

After pulling `astra/all-policy-pilot` into the Windows mirror and then WSL:

```bash
bash experiments/2026-10-21_21-13-boxes-frontier-fit/run_overnight.sh
```

No native rebuild is needed. The launcher verifies the original source config,
checkpoint and binary, original game files, per-game receipts and prepared NPZs.
Leave the checkout/build unchanged until the run finishes. It creates an isolated
`fit37-*` directory and prints a resume command and final `results.tgz` to return.
Completed stages survive interruption; an incomplete stage restarts in a new
attempt. Original games, previous results and incumbent weights remain intact.

## Fixed protocol

All training begins from verified **proof34 iter24**, not iter10, which already
trained on the saved games. Use the existing 10% opening-family validation split
(seed 0) and remove D4-equivalent training inputs from validation. Family and
child lineage remain intact. This is development validation, not an untouched
final dataset; the games come from descendants of iter24.

1. **Diagnostic fitting check:** select at most 128 unique-input 37-edge training
   roots. Train 2,048 updates, batch 64, lr 0.001, using only those roots and their
   exact policy/value labels. Read baseline and final training/validation by
   band. These weights never enter an arena or checkpoint-selection decision.
2. **Ordinary replay:** 8,192 updates, batch 256, lr 0.0001, uniform saved replay.
3. **Exact-root quota:** same 8,192 updates, initialization, targets and augmentation
   schedule, but reserve 25% of each batch for exact roots at 35–37. Retain ordinary
   trajectories and value-only children in the remaining pool. This changes both
   policy/value exposure and batch-normalization composition. It is not a pure
   policy-loss-weight ablation. Both arms use fresh AdamW and constant LR.
4. **Two fixed final arenas:** 400 games of EACH final arm vs frozen iter24, using
   the same 200 new four-edge openings with sides swapped (seed 20261106). Both
   players receive 400 simulations and solver34/50k. Saved games are independently
   replayed. The small development reads never choose an arm or checkpoint.
5. **Search diagnostics:** both final arms on clean held-out 37 roots and the frozen
   35–36 regression set. Exact raw-child caches avoid new solving on the freshly
   exported validation set. The older regression set may require bounded solves.

Expected total roughly **4–7 hours**, depending on search and diagnostic costs.
The protocol finishes when these fixed tasks finish; it does not collect games
just to occupy the rest of the night. Caps are 30 minutes for preparation/canary,
90 minutes per main training arm, three hours per arena and two hours per probe.
Independent arms/arenas continue after a timed-out diagnostic. Every failed stage
is visible in `summary.json`; use the printed resume command to finish it.

The held-out 37 set may contain only a few dozen mistake-selected positions.
Training-set fitting gains establish optimization capacity for those targets,
not game strength. Fitting failure does not establish an architectural limit:
learning rate, joint losses, augmentation and normalization remain possible
causes. Exact-child value rows do not directly dilute policy loss, which is
normalized by supervised policy weights. Promotion still requires the full-game
evidence; both comparisons are exploratory, with no automatic promotion.
