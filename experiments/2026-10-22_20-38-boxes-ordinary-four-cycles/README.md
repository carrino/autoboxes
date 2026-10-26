# Four ordinary training cycles from the verified winner

Start from `ordinary-next-vm_dguq5` cycle 2, which beat its frozen starting model
244–156 under identical search settings. This run tests four more cycles of the
same ordinary-replay recipe. The prior 35–36 margin-regret regression remains a
development check; neither that band nor 37 is treated as mastered.

Pull `astra/all-policy-pilot` into the Windows mirror, then run in WSL:

```bash
git switch astra/all-policy-pilot
git pull
bash experiments/2026-10-22_20-38-boxes-ordinary-four-cycles/run_overnight.sh
```

The launcher verifies the completed parent run, final checkpoint, native runtime
and all saved replay receipts. No C++ rebuild is needed. It creates an isolated
`ordinary-four-*` directory, prints a resume command and packages `results.tgz`.

## Fixed workload

- **Four cycles**, each with 200 fresh full-start self-play games and 8,192
  ordinary-sampling training updates. Each cycle continues the latest weights.
- Retain all 1,400 prior source games and append each new collection: 800 new
  games total and up to 2,200 source games in the final replay pool. Source-family
  validation and D4 input exclusion stay in place.
- Exact consequential-mistake policies at **35–37**, at most three per game,
  with the same bounded root/child solving and value-only child rows. Reuse old
  labels. Solver remains **34/50k**, self-play and arena use 400 simulations.
- Ordinary training uses batch 256, constant LR 0.0001 and fresh AdamW per cycle.
  Save diagnostic snapshots every 2,048 updates; evaluate only the fixed LAST
  checkpoint in the final arena. There are no per-cycle promotion gates.
- **400 final games against the frozen cycle-2 starting winner**, on 200 fresh
  four-edge openings with sides swapped (seed 20261110). No opening confirmation
  match is repeated. Training seed 20261023 defines distinct collection seeds.
- Then compare final versus start on the same fixed 35–36 and exactly-37
  development positions. These reused probes check regression; they are not a
  new independent benchmark or proof of readiness to advance to 38. A fresh
  position benchmark and external-engine comparisons remain separate work.

Expected total is approximately **6–9 hours**. It finishes its four cycles and
evaluations, rather than running until a wall-clock deadline. Stage failure caps
are inherited from the tested controller: two hours per collection/label pass,
30 minutes per replay copy, 90 minutes per training cycle, four hours for the
final match and two hours per probe. These caps can extend the worst-case total.

Completed stages are verified and reused on resume; completed per-game labels
survive an interrupted solve pass. Interrupted collection/training stages restart
in a fresh attempt. Snapshots are diagnostic weights, not optimizer resume points.
Keep the checkout and native build unchanged until completion. Original games,
checkpoints and league states are not overwritten, and no result auto-promotes a
model. The output archive omits large checkpoints/NPZs, which remain local.

## Provenance

- Parent: `ordinary-next-vm_dguq5`, completed two-cycle run.
- Parent config SHA256: `c7b69dcbf3476720a5220dab3a1cc25dff981fa0a16b226d4069c57bd273f152`.
- Starting checkpoint SHA256: `674b27ff345c705a6ee144a305f7fcb69d812fbc7393da3133523d7d9da57108`.
- Starting checkpoint location: `experiments/2026-10-08_14-42-boxes-3x3-loop/logs/ordinary-next-vm_dguq5/cycle-2-train/attempt-ln176d15/train/control.pt`.

The previous controller's default two-cycle/confirmation behavior is preserved.
The new launcher explicitly disables that confirmation and pins the new starting
hash in its own verifier. Tests cover four-cycle weight lineage, fixed final
opponent, no repeated confirmation, preservation of parent replay and files,
resume reuse, and rejection of incomplete or changed parent checkpoints.
