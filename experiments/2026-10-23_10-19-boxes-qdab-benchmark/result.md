# First completed Qdab benchmark

The frozen latest learner won **17 of 100 paired-opening games** against pinned
Qdab `quctann`: **17%, opening-pair bootstrap 95% interval 9–26%**. This corresponds
to **−275 relative Elo**, interval **−402 to −182**, assigning this Qdab setup zero.
The separate empty-board panel was **1 win, 19 losses**; those repeated starts
are descriptive and excluded from the rating.

| Panel | Wins / games | First / second wins | Mean margin | Our seconds / edge | Qdab seconds / edge |
|---|---:|---:|---:|---:|---:|
| 50 side-swapped opening pairs | 17 / 100 | 9 / 50; 8 / 50 | +0.76 | 0.516 | 2.126 |
| Empty board | 1 / 20 | 1 / 10; 0 / 10 | −0.40 | 0.498 | 2.246 |

This is a clear configured-strength gap, **not equal-time Elo**. Qdab used about
4.1 times as much average decision time per played edge in the paired panel.
That ratio includes cheap late-game moves and different trajectories; four times
the neural simulations need not take exactly the same time or close the gap.

## Why positive margin does not mean competitive play

Of the 83 paired losses, **65 were by one box**, 14 by three, and four by five
or more. Every one of the 17 wins was by at least seven boxes. Wins averaged
**+12.53**, losses **−1.65**. Large wins lift average margin above zero while Qdab
wins more than four games in five.

Qdab's source optimizes winning probability and treats reaching 13 boxes as a
win. The pattern is consistent with preserving small wins and conceding boxes
in already-lost positions. It does not prove our margin shaping caused the losses,
or that turning a one-box loss into a win requires only a small fix. Prioritize
game wins and exact win preservation; mean margin is a secondary diagnostic.

Pair outcomes were **36 lose-both, 11 split, 3 win-both**. Every fixed 5×5 opening
has a perfect-play winning side because 25 boxes cannot be tied. Since autoboxes
occupied both sides of each opening, at least 36 of these games failed to convert
a theoretically winning post-opening state. Seat imbalance or unlucky openings
alone cannot explain the result. This does not locate the errors.

## Frozen configuration

- Run: `qdab-match-1yafbn9k`; seed `20261112`.
- Checkpoint: `c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`,
  the ordinary-four cycle-4 endpoint.
- Autoboxes: 400 simulations, solver34/50k, policy temperature 1, proof and terminal
  shaping enabled, margin utility lambda 0.5 and k 6.
- Qdab: pinned `57c6a5fd849022bf3754c4aa530a2ab2ee7f300c`, `quctann`, five seconds
  requested per decision, 30-second external deadline. Its seed is not exposed.
- Recorded user runtime: RTX 3070, WSL2, Torch 2.11.0+cu130.
- All 120 first attempts completed. Their elapsed times total 9,034.8 seconds,
  about 2.51 hours. No retries or technical forfeits.

## Verification

Independent lattice/box replay reconstructed all **7,200 edges**, checked final
owners/scores/extra turns, and verified all **2,930 Qdab RPC request states** and
returned move bundles, comprising 3,393 Qdab edges. The audit regenerated the
opening schedule, checked every game's configuration hash and all 120 successful
attempt receipts. All 70 frozen Python source hashes match the cloud checkout;
six local pinned Qdab artifacts and the separately uploaded checkpoint match the
recorded hashes.

It reproduced the paired bootstrap interval and relative Elo, plus the exact
whole-pair permutation p-value `3.608874976634979e-08`. See
[`data/review.json`](data/review.json), [`data/verified-games.csv`](data/verified-games.csv)
and [`data/paired-openings.csv`](data/paired-openings.csv). Remote GPU execution and
its native binary were not rerun in the cloud. The archive omits checkpoint bytes;
the earlier separate upload supplies the independently hashed weights.

[`verify_benchmark.py`](verify_benchmark.py) preserves the independent audit and
was also run successfully as a standalone script. It accepts the extracted run,
output directory, checkpoint and pinned Qdab release paths; the matching frozen
source checkout is required for source-hash verification. It performs no inference
or solving.

## Training separation and next experiment

The cloud separately completed **60 fresh Qdab teaching games**, with 45 training
and 15 validation families, 1,051 learner search roots and 240 approximate teacher
recommendations. Those are the training source; these 120 benchmark games remain
evaluation only.

An independent overlap audit replayed all 180 games through Python/native rules
and compared all **11,694 frozen structural states** against benchmark trajectories
and every legal raw/capture-collapsed child at 35–38 edges. It checked 29,232 unique
D4/mover-score semantic keys and found **zero state overlap and zero opening-family
overlap**. The external receipt is
[`data/structural-benchmark-overlap.json`](data/structural-benchmark-overlap.json).
The already-running structural queue retains its original provisional metadata;
this companion receipt supplies the actual benchmark overlap check without
rewriting frozen inputs. It is not a playing-strength claim.

The immediate experiment is the queued matched structural-value fit on the fresh
games. The next useful GPU match is the **same frozen checkpoint at 1,600
simulations**, holding the solver, utility, opponent settings and fixed opening
sample constant. That isolates additional runtime search. Report actual timing
rather than calling it equal-time automatically. A utility-lambda change is a
separate experiment; changing both would obscure the cause of any gain.
