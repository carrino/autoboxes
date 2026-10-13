# A classical 5x5 yardstick

Use **`boxes-classic`**: iterative-deepening alpha-beta with a one-second move deadline, chain/control evaluation, transposition table, exact forced-capture and equivalent-edge reductions, and budgeted exact-solver probes at ≤26 undrawn edges. `boxes-classic-fast` uses 100 ms and `boxes-classic-tiny` uses 20 ms.

## Paired arena

Each comparison has 200 games / 100 pairs, seed 0, four uniformly random opening edges shared within each color-swapped pair. Pair wins are decided by summed margin; ties count one half. Intervals are the repository arena's 95% Wilson intervals on paired scores. Seconds per move are weighted by actual move counts, including forced captures and fast exact finishes.

| Opponent | Pair W/D/L | Pair score [95% CI] | Mean margin | Classic s/move | Opponent s/move |
|---|---|---|---:|---:|---:|
| boxes-ab-d6 | 91/5/4 | 0.935 [0.869, 0.969] | +6.17 | 0.3605 | 1.1636 |
| boxes-ab-d4-s28 | 93/2/5 | 0.940 [0.875, 0.972] | +7.28 | 0.3564 | 0.0471 |
| boxes-classic-fast | 88/8/4 | 0.920 [0.850, 0.959] | +5.35 | 0.3732 | 0.0512 |
| boxes-classic-tiny | 94/3/3 | 0.955 [0.895, 0.981] | +7.55 | 0.3714 | 0.0128 |

The initial 30-edge/1,024-node build's seed-42 smoke match finished 4–0 against depth 6, mean margin +7.5. Its incomplete 116-game match is preserved under `data/preliminary-probe30/`. Both are excluded from the final matches, which restarted with the corrected build. Raw per-game timings, paired margins, settings, and summaries are retained in `data/`.

## Probe calibration

The original 1,024-node cap at 30 edges often left a heuristic leaf. Simply reducing the threshold to 26 did not resolve the low completion rate. The corrected engine uses a cumulative 16,384 nodes at ≤26 edges, retaining its solver table across 1,024-node slices and checking the clock between slices.

Calibration sampled 1,400 development positions at 24–30 edges, seed 73. Every successful probe was checked against a fresh unrestricted exact answer. Selection preferred threshold 26 and the smallest tested cumulative budget (4,096 / 16,384 / 65,536) with ≥90% completion at each sampled count at or below the threshold. No held-out games or match outcomes selected this setting.

| Undrawn | Positions | Original 1,024 solved | Sliced 16,384 solved | Sliced probe median / p90 ms |
|---|---:|---:|---:|---:|
| 24 | 103 | 62.1% | 99.0% | 3.14 / 19.53 |
| 25 | 88 | 48.9% | 98.9% | 7.06 / 32.73 |
| 26 | 109 | 45.0% | 98.2% | 8.17 / 43.82 |
| 27 | 132 | 27.3% | 86.4% | 12.89 / 73.29 |
| 28 | 304 | 17.8% | 72.7% | 25.19 / 85.79 |
| 29 | 327 | 9.2% | 45.9% | 71.80 / 96.98 |
| 30 | 337 | 5.3% | 40.7% | 74.87 / 98.53 |

These are fresh-table probe completion rates on sampled game positions, not the engine's in-search hit rates. Warm tables and the remaining move deadline affect actual leaves. A timed-out or budget-exhausted probe remains heuristic; the engine never treats an unfinished solve as proof.

## Evaluation and search details

The score is the board's current margin plus an estimate of its remaining margin. The Python reference and C++ agree on randomly generated masks and symmetries. Simple chain/loop endings use the existing exact recurrence. For unresolved junctions, long-chain parity estimates who will control the ending, controlled value estimates the reward for that control, and the junction count reduces confidence. This is a heuristic, never an exact bound.

The search uses safe/history ordering with a remembered best move, bounded transposition storage, and a steady-clock deadline. Each exact-solver probe is at most 16,384 cumulative nodes, with deadline checks on both sides of each 1,024-node slice. An unfinished probe falls back to search/evaluation. On timeout, the agent returns the last completed iteration's move. Forced captures can return immediately.

The engine is opt-in by registered name. Each agent is single-threaded and allocates a 4 MiB search table plus the default 16 MiB solver table. A new game starts fresh tables, and bindings release the GIL for independent games.

## Validation and next measurement

Python reference first: 8 tests, including controlled-value theorem checks against the exact recurrence and exhaustive small endgames. Reference plus C++: 29 tests, including exact tiny-board minimax with tiny collision-prone tables, move optimality, perspective, timeouts, configuration checks, and regression positions whose one-slice heuristic leaves now resolve exactly. Full suite: **380 passed, 24 skipped**. New Python is ruff/strict-mypy clean. Build SHA256: `f4e170317b6d77a1d5a9e66675d6694a376066982fb53242b781665e3c45c441`.

On the GPU machine, use `--a ckpt:<checkpoint> --b boxes-classic --opening_moves 4 --num_games 200` with the usual net search settings. Record the net's solver reach and budget explicitly. This CPU arena does not establish strength against the nets or external Windows engines; that is the next comparison.
