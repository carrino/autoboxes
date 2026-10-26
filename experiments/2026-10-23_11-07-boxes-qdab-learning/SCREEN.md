# Short Qdab teaching screen

Run two fixed fits and inspect whether the additional Qdab supervision changes
heldout choices. No arena, Qdab executable, solver labeling or automatic
promotion is part of this command. Expected RTX 3070 duration is about
15–30 minutes, depending on replay preparation and inference speed.

Pull into Windows first as usual, then in WSL:

```bash
(
set -euo pipefail
git switch astra/all-policy-pilot
git pull
bash experiments/2026-10-23_11-07-boxes-qdab-learning/run_teacher_screen.sh
)
```

No C++ rebuild or separate data download is required. Return the printed
`results.tgz`. The command also prints a `resume.sh`: use it for the same run
after interruption. Completed arms are hash-checked and reused; an incomplete
arm restarts from the frozen reference and keeps its previous attempt. A
90-minute technical cap prevents a stuck fit from occupying the machine all
night. Keep the checkout, build and environment unchanged until completion.

## Why this run

The previous 400-game comparison was 28 wins versus 27. Only 164 fitting rows
carried Qdab actions, producing 869 teacher exposures across 524,288 sampled
positions. Both endpoints still agreed with Qdab on 3/60 heldout decisions.
See [the completed result](TEACHER_RESULT.md).

This payload contains **3,072 distinct approximate teaching positions from 245
opening families**: 2,493 from the now-retired completed comparison and 579 from
the two fresh cloud collections. All are at 38–50 undrawn edges. A separately
frozen set of **80 recommendations from 20 fresh families** is the primary
development read. The original 60 recommendations from 15 previously examined
families are reported separately as diagnostic data.

The two fits each use **2,048 updates**, batch 256, learning rate `1e-4`, seed
`20261128`, fresh AdamW and D4 augmentation. Every batch has **248 ordinary
replay rows plus 8 teacher rows**. This guarantees **16,384 teacher exposures**,
about 19 times the old exposure, spread across about 19 times as many distinct
teacher positions. Teacher and ordinary sample indices and augmentation agree
between the two arms.

- `reference-policy`: on teacher positions, retain the frozen starting model's
  raw legal policy as a soft target. This is a distillation control, not an
  invented MCTS visit distribution or the earlier control recipe.
- `qdab-policy`: on those identical teacher positions, use Qdab's verified
  approximate action targets.

Both arms otherwise use the same ordinary replay and its existing policy/value
targets. **Teacher rows contribute policy loss only**. Their observed game
outcomes are not used as exact values, and no learner search visits are invented.
Value loss is normalized over ordinary replay rows only. The joint model is
trained, so changing policy supervision can still affect its shared features.

## Reads and interpretation

The fixed final endpoint is primary. Intermediate teacher reads at steps 256
and 1,024 are diagnostic and never select a checkpoint. Reports include:

- Actual teacher exposure and distinct examples sampled.
- Teacher probability, cross-entropy and action agreement on fitting, fresh
  heldout and old diagnostic families, with paired family intervals.
- Exact-root raw-policy move quality on the existing replay train/validation
  split, before and after each fit.
- Input, source, checkpoint, sample-index and augmentation hashes.

Teacher agreement is not exact correctness. Existing replay validation is a
diagnostic because the starting model may have seen those older games. The
screen does not prove playing-strength improvement. Review both transfer to
fresh families and exact-position regressions before selecting a candidate for
searched-move checks or a later full-game test. There is no automatic promotion.

## Frozen inputs and data separation

Default reference remains the original ordinary-four cycle-4 checkpoint:
`logs/qdab-match-1yafbn9k/inputs/checkpoint.pt` under the loop experiment,
SHA256 `c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`.
Default ordinary replay is
`logs/ordinary-four-aqowx5f8/cycle-4-prepare/attempt-tbyadiie/prepared`.
The two new endpoints live in their own unique run directory.

The portable payload records 35 heldout families, 50 retained historical
benchmark families and 66,594 semantic state exclusions. Replay fitting removes
those families/states; replay validation additionally removes new teacher-training
families/states, then removes remaining D4 input overlap. The original replay
split is preserved. The 100 recycled match families are explicitly retired from
future evaluation. Original match results and receipts remain unchanged.

Bundled payload SHA256:
`d46826494b829c6797cd43577df9c18465327f14cb03bb4447199adbe4f5c762`.
The archive is 751,711 bytes. Its train/validation/diagnostic records are verified
again after safe extraction. CPU overrides exist for small implementation tests;
the default command requires CUDA.

## Implementation checks

Eighteen focused tests cover replay/heldout separation, teacher action transforms,
policy/value gradient separation, matched sample/augmentation traces, interrupted
and completed-arm resume, source tampering and archive safety. Ruff, strict mypy
and shell checks pass. A full-size original-c24 CPU smoke used the actual bundled
3,072-position payload and older cloud replay for two updates per arm. Both arms
finished, sample/augmentation hashes matched, teacher value exposure was zero,
and the shell resume reused completed fits. This validates execution and recovery;
the default 2,048-update CUDA experiment remains to be run on the 3070.
Receipts are in `screen-validation.json`.
