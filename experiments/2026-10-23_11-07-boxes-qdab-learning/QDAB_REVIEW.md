# Why Qdab can see farther, and what to test next

Read-only source audit, 2026-10-23. No production search or training settings changed.
Upstream source is pinned to
[`57c6a5fd849022bf3754c4aa530a2ab2ee7f300c`](https://github.com/yimmon/dots-and-boxes/tree/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c).
The associated release executable and three ANN files are already hash-pinned by
`src/alpha_go/boxes/qdab.py`. Source statements below describe that commit; this
audit did not rebuild the historical release to establish binary reproducibility.

## Main finding

Qdab's advantage is plausibly a combination of **much cheaper, deeper simulation,
structural move selection, and winner-first exact search**. It does not have a
large learned opening-policy network. Simply copying its final game outcomes
would miss much of the mechanism that makes its decisions strong.

The cloud pilot's eight games establish a useful failure example, not a rating:
our current weights won one and lost seven. Six games were already exactly losing
at 38 undrawn edges; two side-swapped opening pairs were losing in both games at
that cut. Those pairs prove that some initially winning sides were lost earlier.
Unknown earlier solves do not locate those first mistakes precisely.

## What the code actually does

| Mechanism | Pinned Qdab | Current autoboxes benchmark |
|---|---|---|
| Selection | Variance-aware UCB; no learned policy prior | PUCT driven by the learned policy |
| Unvisited actions | First-play estimate 0.61, increased to 0.76 for the root mover | Initial action Q is 0, plus its PUCT exploration bonus |
| Simulation | Structural rollout after expansion toward later game stages | Batched network evaluation at the new leaf, with forced captures collapsed and bounded solver calls |
| Value network | 25 scalar structural features → 7 hidden neurons → 1 output; three stage-specific models | 21 spatial planes → 128 channels × 10 residual blocks, joint policy and final-margin heads |
| Search objective | Probability of winning; a player with 13 boxes has won | `P(win) + 0.5*tanh(E[margin]/6)` |
| Exact proof | Winner search can stop after finding a winning continuation | Margin-shaped proof requires all actions settled before the exact node utility is proved |
| Search budget | Requested 5 seconds, with separate exact presearch and external 30-second deadline | 400 simulations, solver threshold 34 and 50,000 nodes per call |

Qdab source references:

- [`quctann.go:39–93`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/quctann/quctann.go#L39): stage-dependent UCB, first-play estimates, rollout/exact thresholds.
- [`quctann.go:231–260`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/quctann/quctann.go#L231): tree descent, expansion, rollout, and backup.
- [`quctann.go:330–407`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/quctann/quctann.go#L330): structural rollout choices, later ANN-guided choices, exact winner search, and further rollout when the ANN abstains.
- [`quctann.go:426–565`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/quctann/quctann.go#L426): root/inner move filtering, expansion ordering, and UCB selection.
- [`qeval.go:114–130`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/ann/qeval/qeval.go#L114): degree counts, link-length histogram, loop histogram, half-heart/four-link counts, and both scores.
- [`qeval.go:144–186`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/ann/qeval/qeval.go#L144): cached recursive winner search returns as soon as a winning move is found.

Autoboxes references: `nn_agent.py:53–60,354–415`; `encode.py:15–30,74–94`;
`model.py:48–55,110–125`; `cpp/mcts/mcts.cpp:35–65,88–108,440–461,571–589`;
`cpp/boxes/boxes_search.cpp:60–86`; `utility.py:7–15`.

### The early-game mechanism is not simple greedy capture

Qdab represents strings-and-coins links explicitly. It prefers particular
structural move classes, filters some edges connecting two degree-4 nodes early,
sometimes samples only 18 or 22 candidates, and later admits sacrifice moves or
the shortest candidate link. Those are heuristic restrictions, not a proof that
every omitted move is inferior. They concentrate simulation effort. See the
`AddChild` and `Sim` references above and
[`qboard.go:191–234`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/qboard/qboard.go#L191).

Autoboxes already has chain/loop planes and forced-capture collapse. Qdab does
more than supply similar input features: it uses structure to choose rollout
moves and reduce the search branching factor. Its network sees detailed global
link-length/loop-length statistics; ours explicitly encodes chain sizes 1, 2,
and 3+, loop membership and global counts, with the spatial board available for
learning the rest. That is a representation difference, not proof that our
network cannot learn it.

Qdab `Turn` counts **noncapturing turn changes**, not all edges played
([`qboard.go:1062–1069`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/qboard/qboard.go#L1062)).
Do not translate model names `24/28/32.ann` directly into 36/32/28 undrawn edges.
Its exact-search threshold uses the number of compressed candidate moves, also
not the physical undrawn-edge count.

### How much search is it buying?

Existing cloud-pilot engine logs, without running another timed match:

| Qdab noncapture turns | Logged decisions | Median simulations | Median root children | Median tree turn depth |
|---|---:|---:|---:|---:|
| 0–14 | 44 | 4,241.5 | 18 | 4 |
| 15–23 | 36 | 3,861.5 | 28 | 5 |

These are logged searched decisions; forced captures and successful exact
presearch returns are omitted. Tree depth excludes the following rollout, so a
depth of four does **not** mean Qdab only considers four more edges. Simulation
counts are not equivalent work across engines. They nevertheless show why
“both use MCTS” does not make 400 of our simulations comparable to Qdab at five
seconds. Qdab also uses multiple CPU workers (`quctann.go:594–598`).

Qdab's requested timeout begins after an exact presearch at turn 24 or later
(`quctann.go:166–217`); the adapter's external deadline bounds this. Near-instant
late decisions can be forced capture, an already secured 13-box win, or early
success in winner search. Timing alone cannot distinguish them.

### Training origins: what is and is not known

The checked-in trainer builds a 25→7→1 network and uses RPROP, balanced records,
and ±0.9 winner targets
([`tr.go:33–58,74–128`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/ann/tr/tr.go#L33)).
The released ANN headers agree with that architecture (their stored layer sizes
include bias neurons). The data generator samples legal structurally guided
histories and labels through winner search
([`ds.go:138–165`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/ann/ds/ds.go#L138),
[`board.go:828–852`](https://github.com/yimmon/dots-and-boxes/blob/57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/src/algorithm/board/board.go#L828)).
The ANN cutoff in the training-side winner routine is commented out at this
commit. We therefore should not describe these checked-in labels as bootstrapped
from the later ANN. The release does not provide a reproducible training ledger
with dataset count, training duration, and model lineage. No claim that these
particular weights trained in a few hours follows from the source.

## A concrete failure and what it rules out

In cloud game 0 at 37 edges, edge 49 is the sole exact winning move (+1).
The network assigns it 3.8% prior, rank 13, versus 25.8% for played losing edge
32 (−1). Several losing children also receive higher value estimates than 49.

Root's follow-up cold-search diagnostic selected a losing edge at 400
simulations but selected 49 at 1,600. Crucially, **49 already received 35 visits
at 400**, then 908 at 1,600. This is not a case where the winning root action was
never discovered. It suggests that further resolution of its subtree can rescue
the decision. Only 14/37 root actions had visits at 400, versus 17/37 at 1,600.

This is one deliberately selected mistake, with cold solver state and a
different evaluator wrapper from the original game. The 400-simulation selected
edge differs from the game. It demonstrates possible search headroom, not a
general strength gain or a causal explanation of the entire match deficit.

## Three bounded experiments, in order

1. **Search-budget and coverage read, with weights frozen.** On a predeclared
   fresh set of exact 35–38 roots from strong-opponent games, compare 400 and
   1,600 simulations. Record exact win preservation, regret, elapsed time,
   root-action coverage, winning-action visits and proven children. Include the
   selected game-0 mistake only as a separate regression example. If 1,600 helps
   consistently, distinguish a stronger search setting from improved learning;
   measure an equal-time setting before choosing the production budget.

2. **First-play and proof semantics, as isolated search variants.** Current
   unvisited Q=0 is deliberately pessimistic on a [0,1] win scale, especially
   against confident positive leaf values. A Boxes-only candidate can compare
   neutral/parent-based first-play values while keeping priors and 400 simulations
   fixed. Measure discovery and resolution separately; simply flattening priors
   already disappointed in earlier experiments. Independently compare lambda
   0.5 with lambda 0 on the same small fixed sample. Lambda 0 enables a proven
   win to settle the node without proving the best margin and drops proven
   losing root choices. **Do not switch production to lambda 0 blindly:** margin
   shaping previously helped our search. If a win-proof certificate is useful,
   it should eventually be represented separately from exact-margin proof, not
   achieved by incorrectly treating a found win as the optimal margin.

3. **Teach the missing decisions, not just their final loss.** Use the fresh
   collection's Qdab recommendations at learner-visited early states as
   approximate policy supervision, with our existing searched targets as a
   matched control. Retain exact root/child value anchors and ordinary replay.
   This directly tests whether the learner can absorb the stronger engine's
   strategic choices. If policy improves but child ordering remains poor, a
   small separate structural-feature value baseline is a useful next diagnostic:
   can explicit link/loop length counts predict exact child order on held-out
   source families better than the current head? Do not start by replacing the
   whole network or adding another long memorization run.

The first two tests help determine how much the learner needs better evidence
versus better use of existing evidence. The third is the learning experiment.
Fresh full games against pinned Qdab, with swapped sides and timing reported,
remain the confirmation test.

## Correctness scope and artifacts

No new perspective/retained-turn error was identified by this read. The current
backup explicitly flips only on mover changes, legal-policy softmax normalizes
over the actual search state's actions, and capture collapse preserves the
take-all/keep-control decision. Existing exhaustive tests cover these mechanisms.
The FPU and shaped-proof behaviors above are visible algorithm choices, not
evidence of a bookkeeping bug.

Local reproducibility artifacts (development evidence, not training data):

- `work/qdab-source-review/`: pinned extracted `.go` sources and
  `pilot-search-stats.json`; source archive SHA256
  `2f91308d556db5b30735ce4c2d4b88ddabbed30f924a2e3009b49f7812f6e363`.
- `work/qdab-cloud-pilot-w895zcit/earlier-frontier/critical-child-audit.json`.
- `work/qdab-cloud-pilot-w895zcit/search-headroom.json`.
- `work/qdab-cloud-pilot-w895zcit/qdab-match-2g8p825p/attempts/*/qdab.log`.

The benchmark games stay quarantined. The cloud pilot is a development diagnostic
and is not pooled into the user's 120-game GPU benchmark.
