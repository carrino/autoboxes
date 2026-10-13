# Exact solver reach: results

Final implementation: **candidate2**. All 6,000 benchmark values match the baseline; both candidates were checked on both complete mask files.

The Python API and one-thread-per-solver model are unchanged. The default table remains 1,048,576 entries / 16 MiB. Candidate 1 removes temporary heap allocations; candidate 2 uses existing entry padding to retain both alpha-beta bounds.

## Attempts

1. Profile: five development positions, about 1.27 million search nodes. Component walking, temporary vectors, and repeated chain analysis were prominent hotspots. The initial 100-position instrumented run was stopped; it supplied no reported timing.
2. Allocation-free chain walking: same traversal order and values. The initial 20-position development probe preserved every node count and had a provisional median paired speedup of 1.29×. Full results below supersede that small probe.
3. Two table bounds in 16 bytes: kept. Development geometric-mean speedup versus candidate 1: 1.107×; total search nodes 2,753,168,161 → 2,332,104,979. The predeclared rule required ≥2% geometric-mean speedup and no increase in total nodes. Held-out timings did not enter this decision.

## Before and after

Times exclude solver construction. Nodes and milliseconds are shown as median / p90 / maximum. Every unrestricted and budgeted call starts with a fresh table. Raw per-position CSVs include all candidates, masks, exact values, and timings.

| Dataset | Attempt | Geometric mean paired speedup | Total solve-time speedup | Total nodes |
|---|---|---:|---:|---:|
| 5x5-chains | candidate1 | 1.290× | 1.291× | 2,753,168,161 |
| 5x5-chains | candidate2 | 1.428× | 1.538× | 2,332,104,979 |
| 5x5-mid | candidate1 | 1.613× | 1.309× | 913,260,852 |
| 5x5-mid | candidate2 | 1.409× | 1.408× | 756,033,397 |

Candidate 1 preserves every per-position node count as well as every value. Timing measurements share the machine with other single-threaded CPU tasks; node counts and budget success rates do not depend on scheduling.

The second change is not a uniform timing improvement over candidate 1: on held-out positions its geometric-mean paired speedup is 0.874×, while its total solve-time speedup is 1.075×. The development-only selection rule was retained; the held-out result was not used to retune or change the candidate.

### 5x5-chains

| Version | Undrawn | Positions | Nodes med / p90 / max | ms med / p90 / max | Solved ≤100k | Solved ≤1M |
|---|---:|---:|---|---|---:|---:|
| baseline | 30 | 627 | 25,881 / 214,413 / 2,410,411 | 92.44 / 762.12 / 9,769.48 | 78.0% | 99.2% |
| baseline | 31 | 662 | 77,668 / 472,625 / 6,451,297 | 289.03 / 1,846.29 / 34,247.55 | 55.4% | 97.3% |
| baseline | 32 | 552 | 127,900 / 1,045,587 / 5,453,844 | 466.08 / 4,332.80 / 20,740.09 | 42.8% | 88.4% |
| baseline | 33 | 640 | 406,860 / 2,824,646 / 37,640,843 | 1,486.42 / 10,275.24 / 113,845.12 | 20.0% | 71.9% |
| baseline | 34 | 519 | 1,032,611 / 5,532,680 / 190,849,217 | 3,129.66 / 22,196.21 / 635,091.41 | 13.7% | 49.3% |
| candidate2 | 30 | 627 | 24,597 / 198,742 / 2,117,267 | 64.35 / 596.78 / 5,832.07 | 79.3% | 99.4% |
| candidate2 | 31 | 662 | 73,558 / 434,549 / 4,956,250 | 226.60 / 1,197.00 / 20,880.78 | 56.9% | 97.9% |
| candidate2 | 32 | 552 | 122,147 / 962,363 / 5,075,796 | 341.16 / 2,853.84 / 20,926.87 | 44.2% | 90.9% |
| candidate2 | 33 | 640 | 370,319 / 2,364,828 / 30,037,472 | 1,083.35 / 6,116.91 / 95,768.61 | 21.1% | 74.4% |
| candidate2 | 34 | 519 | 921,726 / 4,825,243 / 155,285,130 | 2,198.13 / 14,201.16 / 501,982.84 | 15.2% | 53.0% |

### 5x5-mid

| Version | Undrawn | Positions | Nodes med / p90 / max | ms med / p90 / max | Solved ≤100k | Solved ≤1M |
|---|---:|---:|---|---|---:|---:|
| baseline | 30 | 558 | 6,400 / 61,811 / 1,783,782 | 25.72 / 245.67 / 4,443.84 | 93.5% | 99.6% |
| baseline | 31 | 576 | 12,954 / 121,381 / 1,914,667 | 51.80 / 468.81 / 7,935.28 | 87.8% | 99.3% |
| baseline | 32 | 595 | 21,554 / 228,824 / 2,827,972 | 91.84 / 778.53 / 8,212.55 | 81.3% | 98.5% |
| baseline | 33 | 646 | 51,343 / 488,312 / 82,397,497 | 192.71 / 1,871.43 / 230,442.91 | 71.7% | 95.5% |
| baseline | 34 | 625 | 111,546 / 695,419 / 235,104,447 | 477.12 / 2,910.01 / 823,300.60 | 43.5% | 93.0% |
| candidate2 | 30 | 558 | 6,018 / 59,442 / 1,514,263 | 16.97 / 165.55 / 3,292.13 | 93.7% | 99.6% |
| candidate2 | 31 | 576 | 12,242 / 113,970 / 1,783,358 | 36.84 / 389.71 / 3,831.50 | 88.9% | 99.3% |
| candidate2 | 32 | 595 | 19,671 / 208,113 / 2,418,633 | 64.81 / 725.68 / 6,599.15 | 82.2% | 98.8% |
| candidate2 | 33 | 646 | 46,840 / 423,644 / 60,364,968 | 146.65 / 1,239.82 / 150,407.70 | 73.2% | 96.1% |
| candidate2 | 34 | 625 | 103,602 / 615,789 / 185,632,607 | 341.51 / 2,132.80 / 529,584.19 | 48.0% | 93.6% |

## Practical reach

For this report, ‘mostly’ was predeclared as at least 80% solved within 100k nodes, and ‘about today's cost of 30’ as median time no more than 110% of that dataset's baseline at 30. Both datasets must meet the condition.

Largest qualifying SOLVER_N: **none in the tested 30–34 band**. Development qualifying counts: []; held-out: [30].

The requested 32-at-30-cost target was not established. Do not raise the self-play threshold to 32 on these results alone. The tables show the speed gain and the budget failures separately; faster nodes need not make more positions fit a fixed node budget.

## Independent regions at fixed undrawn count

This comparison uses exact baseline costs on the held-out mask file, comparing largest region ≤24 edges with an unsplit board at the same undrawn count.

| Undrawn | Small-region n | Median nodes | Unsplit n | Median nodes | Ratio |
|---|---:|---:|---:|---:|---:|
| 30 | 307 | 2,386 | 134 | 23,931 | 10.03× |
| 32 | 239 | 8,912 | 163 | 74,235 | 8.33× |

Yes, smaller largest regions correlate with lower cost at both counts.
This is an observational comparison, not a claim that the regions can be solved independently and their margins summed. The required separate `region_stats --solver-nodes 30 32` run requested 4,000 positions and sampled 3,630 distinct positions from the available games. It is recorded in `logs/region-cost.log`; its 2M-node cap censors the hardest tails.

## Validation and next work

See README for commands, machine, setup, and the baseline checks. Exactness uses the existing Python oracle, exhaustive tiny boards, full-window/MTD(f) comparisons, new tiny-table budget-restart tests, all Catch2 binaries, and the full 6,000-position corpus. Candidate CSVs are retained even for a reverted change.

Final solver suite: **357 passed, 24 skipped**; all three Catch2 binaries pass. Logs are in `data/final-tests.txt` and `data/final-catch2.txt`.

Combined with the corrected classical engine: **386 passed, 24 skipped**, plus all three Catch2 binaries. This separate integration build did not merge either branch. See `data/integration-validation.txt`.

Next: attack the remaining tree size with a proved region-aware reduction. Keep using development positions for design and reserve 5x5-mid for final reports.
