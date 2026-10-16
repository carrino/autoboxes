# Qdab-sourced backward curriculum

The preceding 24-iteration policy loop finished 87–113 against frozen iter0 in
200 side-swapped games (paired p=0.10479). It did not demonstrate improvement.
Its final replay contained 686 exact training roots among 41,131 positions.
This experiment starts again from iter0 and makes exact decision roots the main
training population. It continuously adds searched Qdab histories and solver
labels instead of extending that unsuccessful run.

## Run on the RTX 3070 in WSL

Update the Windows Git mirror first, then in the WSL checkout:

```bash
git switch astra/qdab-curriculum
git pull
bash experiments/2026-10-15_20-57-boxes-qdab-curriculum/run_overnight.sh
```

This change is Python only and needs no C++ rebuild if the extension used by the
previous policy run is still installed. It uses the existing `uv` environment.
CUDA is checked before teacher collection. Qdab installs its pinned Linux x86_64
engine and ANN files automatically under `work/qdab-linux`; no Windows UI is used.

Default work budget: **eight hours, then finish the current cycle and run a
200-game comparison of the last candidate against frozen iter0**. This is a soft
work deadline, not a process kill: a slow cycle and the final arena can extend it.
The run does not stop after three iterations. To request a different budget:

```bash
HOURS=10 bash experiments/2026-10-15_20-57-boxes-qdab-curriculum/run_overnight.sh
```

The first phase collects 16 fresh Qdab histories for a fixed development monitor.
At the observed cloud speed of roughly 2½ minutes per history, that is about
40 minutes, plus solving. Local timing can differ. This phase uses CPU; an idle
GPU is expected. Progress and the current log path are printed throughout.

The launcher prints a unique run directory and an exact resume command:

```bash
bash experiments/2026-10-15_20-57-boxes-qdab-curriculum/logs/<run>/resume.sh
```

Use that command after an interruption. Re-running `run_overnight.sh` starts a
new experiment. Completed histories, source labels and training stages are reused;
an interrupted training stage restarts from its frozen parent and retains the
partial attempt for inspection. Do not switch branches or edit the implementation
during a run: resume verifies code, settings, checkpoints and data hashes.
Three bounded attempts are allowed for a timed-out Qdab game; other engine errors
stop with saved diagnostics rather than producing substitute labels.

At completion, send the printed `results.tgz`. It includes configuration, logs,
exact root tables, per-position monitor reads and the final full-game moves.
Weights and NPZ data remain on the WSL filesystem. Existing checkpoints and
league state are never overwritten or automatically promoted.

## What changes

1. Generate genuine searched opening histories with Qdab at a requested five
   seconds per decision, through shared `play_game`. Stop after 30 edges have
   been drawn. These histories supply positions only: their truncated scores
   are **never** used as outcome/value targets and their chosen moves are not
   assumed exact.
2. Freeze the first 16 fresh histories as the development monitor. Exclude all
   their source families, symmetry-canonical eight-edge openings and known
   matching root states from training and opening rehearsal. Failed solves do
   not make a monitor source eligible for training.
3. Each cycle adds four fresh Qdab histories and up to 64 previously unselected
   searched games from retained replay. The included cloud starter histories
   enter the first training batch only. Retained replay is shuffled once and
   frozen; it is exhausted without recycling files. Fresh teacher collection
   continues afterward.
4. Solve **every eligible decision root at 31–36 undrawn edges**, using finite
   20-million-node root and 5-million-node per-child budgets. A policy label is
   admitted only when every legal child is solved. Incomplete roots are counted
   and skipped. Labels remain cached in each completed source shard.
5. Initially train only roots at **31–34 undrawn**. Each distinct exact root is
   visited once per epoch. Add at most 25% distinct searched opening rehearsal
   rows from the iter0 collector, at 39+ undrawn, to limit opening forgetting.
   No mass of value-only child examples dilutes these policy targets. This is a
   deliberate change of training population, not an exact-value ×8 multiplier.
6. Warm-start from the currently accepted development checkpoint and train
   12 epochs at learning rate `1e-4`, batch 128, with existing D4 augmentation
   and chain features. Report training fit and fixed-monitor optimal-set mass,
   optimal-move rate, exact score-margin regret and wins preserved.
7. Keep a new development parent only if its current-band mean regret improves
   by at least 0.05 margin points without reducing the number of wins preserved.
   Broaden training to 31–36 only after an accepted candidate has at least 32
   monitor roots from eight families, eight winning roots, 90% win preservation
   and mean regret at most one margin point. Otherwise stay at 31–34 and collect
   more examples. Earlier roots remain in replay after the boundary moves.
8. At the work deadline, compare the **last trained candidate**, including an
   unaccepted one, against frozen iter0 on 200 games from fresh paired openings.
   This endpoint is chosen in advance; it does not cherry-pick the best arena
   result. The saved games are independently replayed and an opening-pair
   bootstrap win interval is added to `final.json`.

Development acceptance is an operational data-selection rule, not statistical
promotion. Repeatedly consulting this monitor can overfit it. The independent
final arena and later strong-opponent tests remain the strength criteria.

## Inputs and overrides

Default iter0 lookup, in order:

- `experiments/2026-10-08_14-42-boxes-3x3-loop/checkpoints/5x5-policy-loop-cS6u3n/iter0.pt`
- `experiments/2026-10-08_14-42-boxes-3x3-loop/logs/astra-probe-PCcE6W/reference.pt`

Retained replay is read beneath `$GAME_DATA_DIR` (default
`$HOME/autoboxes-data/game_data_root`) from `5x5-policy-loop-cS6u3n`, with
`5x5-exact` and `5x5-s1600` as fallbacks. The selected run must have searched
`selfplay-it0` games for opening rehearsal. Override locations explicitly:

```bash
bash experiments/2026-10-15_20-57-boxes-qdab-curriculum/run_overnight.sh \
  --reference /path/to/iter0.pt \
  --replay-dir /path/to/searched/selfplay \
  --rehearsal-dir /path/to/iter0/selfplay
```

Other bounded controls: `--monitor-games`, `--teacher-games`, `--retained-games`,
`--workers`, `--epochs`, `--final-games`, `--final-workers`. The fixed-port Qdab
adapter runs serially and refuses an occupied port. Run only one collector per
machine. External decision timeout defaults to 20 seconds; Qdab's own exact
presearch can overrun its requested five-second search time.

## Qdab provenance

Repository: https://github.com/yimmon/dots-and-boxes

Pinned commit: `57c6a5fd849022bf3754c4aa530a2ab2ee7f300c`.
Release: `release/Qdab_0.1_lnx_amd64.tar.bz2`.
Archive SHA256: `cddc6a6c533acd5c47c99ea3c708b342b78ef31a2e260b3f85e65d1ad041dafa`.
Individual executable/library/model hashes are checked by `qdab.py` and written
into collection metadata. No upstream executable or model weights are committed.
The engine uses a wall-clock RNG, so gameplay seeds do not make it bit-reproducible.

Prior development-position check: Qdab preserved 72/73 known wins on 100 frozen
35–38 roots, with 62% margin-optimal moves and 1.18 mean margin regret. This was
not a full-game opening-strength benchmark or an equal-time match. The included
starter histories demonstrate collection validity, not model improvement.

This first curriculum experiment uses external searched histories plus retained
agent replay. It does not yet add every visited internal MCTS node, generate new
learner self-play each cycle, or reproduce every BoxesZero training technique.
