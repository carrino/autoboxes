# Exact labels above the solver's reach

Generated **2,891 exact positions** with 31–36 undrawn edges from 5x5-chains; 56 of 2,947 attempted positions were skipped (1.9%) at the node limit.

Only development games were replayed. No 5x5-mid positions were labeled, trained on, or used for sizing or tuning.

## Pilot and sizing

| Stage | Processes | Node budget | Attempted | Labeled | Skipped | Wall minutes |
|---|---:|---:|---:|---:|---:|---:|
| 5x5-oracle-31-36-pilot | 4 | 20,000,000 | 400 | 388 | 12 | 48.86 |

The production request was 2,947 positions, seed 1, 20,000,000 nodes per solver call, 2 processes. Measured pilot throughput was scaled to 12 hours and capped at 60,000 positions. Two production processes left CPU capacity for tasks 1 and 3.

Actual production wall time: **8.05 hours**.

| Production shard | Labeled | Skipped | Solve seconds | Files |
|---:|---:|---:|---:|---:|
| 0 | 1,448 | 26 | 26,970 | 16 |
| 1 | 1,443 | 30 | 28,938 | 16 |

## Distribution

| Undrawn edges | Positions |
|---:|---:|
| 31 | 540 |
| 32 | 440 |
| 33 | 542 |
| 34 | 502 |
| 35 | 445 |
| 36 | 422 |

Final margins are from player 1's perspective. The stored `to_play` field supplies the training reader's mover perspective.

| Final margin | Positions |
|---:|---:|
| -15 | 2 |
| -13 | 8 |
| -11 | 28 |
| -9 | 89 |
| -7 | 176 |
| -5 | 398 |
| -3 | 576 |
| -1 | 609 |
| +1 | 415 |
| +3 | 239 |
| +5 | 165 |
| +7 | 106 |
| +9 | 45 |
| +11 | 20 |
| +13 | 12 |
| +15 | 3 |

## Verification and artifact

Validated all 32 files: counts equal the per-shard logs; every policy has at least one optimal legal edge and sums to 1 after normalization; stored moves belong to that set. Board shapes, player fields, undrawn counts, ordering, final margins, and exact root outcomes also pass. The CSVs contain per-file counts, per-file SHA256 digests, and both distributions.

Archive: `/workspace/work/5x5-oracle-31-36.tgz` (78,053 bytes).

SHA256: `5bbde9a869aa67c122f4da752935eb1ad81ade53017baa3a3f902c1ae7ab9807`.

The configured chat delivery directory was absent and could not be created in this environment. After generation finished, `archive_completed.py` reverified every file and packaged the archive at the workspace path above. The original packaging failure log is retained; no solves were repeated.

The NPZ data and tar archive stay out of Git. Extract the folder under the GPU machine's game-data experiment directory, then add `experiments/2026-10-08_14-42-boxes-3x3-loop/5x5-oracle-31-36` to `EXTRA_DATA`.

## Interpretation

These are exact outcomes and uniform-optimal move targets, not self-play outcomes. Budget skips select harder positions out of the dataset, so its accuracy as a training target does not make it an unbiased evaluation set. The solver is the pinned main version, independent of task 1's candidate changes. The clean evaluation games remain available for measuring any downstream net improvement.
