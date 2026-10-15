# iter0 vs iter22 on 100 exactly solved roots: where the prior puts its mass, and whether the value head ranks the children

Inputs: the probe analysis' `child-diagnostic-*` tables (100 frozen roots from
`5x5-exact/selfplay-it21` at 35..38 undrawn edges, 54 at 35..36 and 46 at 37..38, 73 of them
won for the mover, every one of their 3,640 legal children exactly solved, 68 source
families) and the frozen checkpoints `5x5-exact/iter0.pt` (sha256 `46c7264c…`, the probe's
reference) and `iter22.pt` (`a6eaaeab…`, the candidate). CPU inference only, 61 s;
`data/` holds every per-child and per-root number and `summary.json` the provenance.
Regret is in final score-margin points (one box moved between the players is 2).
Intervals are iter22 minus iter0 with a source-family bootstrap, 10,000 resamples.

Caveat on the sample: these roots are iteration 21's own self-play positions, which sit in
iteration 22's training window with their exact values; iteration 0 never saw them. The
comparison is therefore tilted toward iter22, and it still does not favour it.

## 1. The prior is nearly flat, and the loop did not move where it points

| checkpoint, T | entropy (max 3.59) | top move | mass on exact-optimal (uniform 0.137) | mass on win-preserving (uniform 0.233) | top-move regret | best optimal move's rank, median / mean |
|---|---|---|---|---|---|---|
| iter0, T=1 | 3.16 | 0.088 | 0.197 | 0.325 | 4.52 | 4 / 5.4 |
| iter22, T=1 | 3.13 | 0.096 | 0.197 | 0.322 | 4.36 | 4 / 6.0 |
| iter0, T=2 | 3.41 | 0.057 | 0.179 | 0.300 | 4.52 | 4 / 5.4 |
| iter22, T=2 | 3.39 | 0.059 | 0.180 | 0.299 | 4.36 | 4 / 6.0 |

- The prior spreads over about 37 legal moves with 0.088 on its top move and 1.4 times
  the uniform mass on the exact-optimal set. The median root gets 0.13 on that set, the
  uniform share; 15 roots get less than uniform; the 90th percentile is 0.48. The
  best-ranked optimal move sits at median rank 4; it is the top move in 23% of roots and
  within the top five in 65% (a uniform ordering: 14% and 43%).
- iter22 is the same prior, slightly sharper: entropy -0.033 [-0.057, -0.011], top move
  +0.008 [0.001, 0.016], optimal mass +0.000 [-0.007, +0.009], win-preserving mass
  -0.003 [-0.011, +0.005]. The best optimal move's mean rank went from 5.4 to 6.0 (median
  unchanged), the top-move regret from 4.52 to 4.36 [-0.89, +0.52]. Twenty-two iterations
  with exact value targets left the policy's targeting where it was.
- Temperature 2 preserves the ordering (same top move, same regret) and gives back a
  third of the prior's excess over uniform on the optimal set (0.197 -> 0.179). With a
  weak value ordering (section 2) the search then spends more of its simulations on
  non-optimal moves; the pending T=2 GPU run should not improve on T=1 by this read.

## 2. The value head orders the children a little, and knows their sign not at all

Children scored as the search sees them (`search`: after the opponent's forced captures,
flipped to the root mover) and before the collapse (`raw`).

| checkpoint, mode, score | favourite child's regret | favourite optimal | pairwise ordering accuracy | Spearman | P(win) range within a root | best optimal move's rank, median / mean |
|---|---|---|---|---|---|---|
| a random legal move | 4.83 | 0.137 | 0.500 | 0 | | 7 / 10.9 |
| iter0, search, margin | 4.34 | 0.17 | 0.662 | 0.33 | 0.48 | 5 / 8.3 |
| iter0, search, P(win) | 4.80 | 0.15 | 0.635 | 0.28 | 0.48 | 5 / 8.3 |
| iter0, search, utility | 4.72 | 0.15 | 0.646 | 0.30 | | 5 / 8.4 |
| iter0, raw, margin | 4.94 | 0.15 | 0.617 | 0.25 | 0.65 | 7 / 10.6 |
| iter22, search, margin | 4.44 | 0.19 | 0.646 | 0.30 | 0.53 | 6 / 9.2 |
| iter22, search, P(win) | 4.76 | 0.17 | 0.623 | 0.26 | 0.53 | 6 / 8.5 |

- The outputs are not constant: within a root P(win) spans 0.48 and the expected margin
  5.1 points across the children. They are ordered above chance (pairwise 0.66, Spearman
  0.33 for the margin head), but the favourite child loses 4.3 points, half a point less
  than a random legal move, and is optimal in 17% of roots against 14% for a random move.
- Sign skill is absent at every level. Only 17% of the children are won for the root
  mover, so "this move loses" scores 0.83; the net's child sign accuracy is 0.81 (iter0)
  and 0.80 (iter22). At the root the mover wins 73% of the time; the net's root sign
  accuracy is 0.75 and 0.71. What the search gets from the value head is its ordering.
- The collapse matters: scoring the child after the opponent's forced captures beats
  scoring it before (pairwise 0.662 against 0.617, favourite's regret 4.34 against 4.94,
  best optimal rank 8.3 against 10.6). Half the children collapse (18 of 36 per root),
  a quarter end at a take-all / keep-control choice, 0.2 per root reach the solver zone,
  none ends the game. Positions with captures pending are the ones the net reads worst.
- The margin head orders better than P(win) (0.662 against 0.635) and the shaped utility
  sits between them, so the margin term in the search utility is doing useful work.

## 3. iter22 against iter0

| band | metric (search, margin head) | iter22 - iter0 [95% interval] |
|---|---|---|
| 35..36 | pairwise ordering accuracy | -0.028 [-0.051, -0.006] |
| 35..36 | Spearman | -0.056 [-0.108, -0.008] |
| 35..36 | favourite child's regret | +0.52 [-0.27, +1.32] |
| 37..38 | pairwise ordering accuracy | -0.002 [-0.023, +0.020] |
| 37..38 | favourite child's regret (utility head) | -0.74 [-1.49, -0.05] |
| all | pairwise ordering accuracy | -0.016 [-0.032, -0.001] |

At 35..36 iter22's value ordering is worse on every score, by a small but consistent
margin; at 37..38 it is unchanged, with a slightly better favourite. The prior's
best-optimal rank at 35..36 went from 5.9 to 6.8. This is the shape of Astra's search
probe (candidate regret +2.1 [0.9, 3.6] at 35..36, inconclusive at 37..38): a slightly
sharper prior on the same moves, and a slightly weaker value ordering, in the band where
the search regressed.

The two heads rarely agree: the prior's top move is the value head's favourite in 17% of
roots (iter22: 12%); one of the two is optimal in 31%, both in 9%.

## 4. What this says

1. **Root winner accuracy at the base rate did not mean constant outputs.** The value
   head varies by half a win probability within a root and orders siblings at 0.66. Its
   sign, at the root and for the children, carries nothing over a constant guess, so
   exact value labels that improved the sign calibration (the 35..36 reads) were never
   going to change what the search consumes; the ordering is what it consumes, and the
   ordering did not improve over 22 iterations (it slipped at 35..36).
2. **The prior is the better locator of optimal moves** (best optimal move within the top
   five in 65% of roots, against 54% by the value ordering and 43% by chance), yet it
   carries 1.4 times uniform mass on them and its top move is barely better than random.
   The search's 2.6 regret (iter0, Astra's probe at 400 simulations) against the heads'
   4.3..4.5 is the search recovering two points over either head alone. Exact policy
   targets (`policy_labels`) address this quantity directly; exact value targets do not.
3. **Flattening the prior (T=2) removes a third of its excess mass on optimal moves and
   changes nothing else**, so the T=2 run is expected to be neutral or worse unless the
   value ordering is stronger inside the search than it is here.
4. **Positions with captures pending are read worst** (raw against search mode). The
   training set holds such positions with a one-hot policy of the played capture and no
   search; their value targets are the outcome or the exact value, as for any position.
   A net that scores them badly is not a problem for the search, which collapses them,
   but it is a sign that the chain planes are not giving the head what a capture
   sequence means.

Not established here: anything about full-game strength, the search's behaviour at T=2
(the GPU run), or generalisation beyond training-derived positions. These are 100
development positions from one iteration's self-play, 68 families.
