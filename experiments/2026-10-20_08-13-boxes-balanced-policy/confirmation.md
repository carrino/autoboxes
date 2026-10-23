# Balanced replay confirmation: no demonstrated playing-strength gain

The fixed 200-game confirmation finished **103–97** for balanced against proof34
iter24: **51.5%**, with an opening-pair bootstrap 95% interval of **[45.0%, 58.5%]**.
Mean score margin is **+0.01** (paired interval **[-0.65, +0.67]**). Keep iter24 as
the incumbent. This result does not establish equivalence, but it does not
support promoting balanced or spending another night on this unchanged recipe.

## Protocol and verification

Run `balanced-policy-k3s3hb6v/arena-vs-iter24-5Qa3Gk` used the predeclared seed
20261102, 100 distinct random four-edge openings, each played with sides swapped.
Both agents used 400 simulations, solver34 with a 50,000-node budget, policy
temperature 1, c_puct 1.5, leaf batch 16, terminal proving/shaping enabled, and
margin utility lambda 0.5 / k 6. Two arena workers ran for 4,116 seconds.

- Balanced won both games in 25 opening pairs; 53 pairs split; iter24 swept 22.
- The two-sided paired label-swap p-value is 0.770867.
- Balanced won 41/100 in the first seat and 62/100 in the second. Side swapping
  prevents that seat difference from becoming an overall advantage.
- Recorded time per move was 0.7064s versus 0.7584s. This is an observation from
  the match, not a controlled inference-speed benchmark.

All 200 games were independently replayed through the Python rules. Every edge
was legal, all games ended after 60 moves, and every final score matched. All
100 opening pairs match the saved seed, and the paired p-value was recomputed.
The interval resamples whole opening pairs, not individual games (10,000
replicates, seed 20261018).

The uploaded checkpoint manifest matches the predeclared candidate and opponent:

| Checkpoint | SHA256 |
|---|---|
| Balanced | `0cbf8ca0ddd2216a308125394c625b76d21c73ae2321442a0c132ac7b34f69d3` |
| Iter24 | `73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d` |

The launcher checked these files before the match. Checkpoint bytes were not
included in this upload, so the cloud audit verifies the manifest rather than
rehashing those weights. Code commit: `1db4b5d9783b281cd04cb3282ea45f214e2ff1e3`.
Source archive SHA256:
`2dda916c3e95b5f21feef40be416a17e731700a6f2090258dfd4b1f6f1c447f9`.

## What transferred from training

The preceding fixed development read had 347 reserved roots, including 167
positions where the mover could win with perfect play. Balanced training did
improve raw policy quality, but search recovered much of the reference's deficit:

| Metric on the same roots | Iter24 | Ordinary replay | Balanced replay |
|---|---:|---:|---:|
| Raw optimal move | 33.4% | 36.6% | 42.4% |
| Raw mean exact margin regret | 2.571 | 2.473 | 2.069 |
| Raw wins preserved | 88/167 | 87/167 | 93/167 |
| Search mean exact margin regret | 0.870 | 0.767 | 0.755 |
| Search wins preserved | 158/167 | 162/167 | 162/167 |

Balanced improved the raw selected move's exact value on 97 roots relative to
iter24, but on 65 of those roots both searches selected moves with the same exact
value. Relative to ordinary replay, those counts are 76 and 60. These are
descriptive decompositions of the existing development sample, not new tests.
The improvement in raw regret was credible; neither search-regret comparison
established a gain, and the fresh full-game result is consistent with that.

This does not show that the model cannot learn Boxes. It shows that better raw
move predictions in the targeted band have not yet produced a demonstrably
stronger complete player. Nor does it establish mastery through 36 edges or
strength against Dabble, PRsBoxes or BoxesZero.

## Cloud trajectory diagnosis

A separate post-hoc audit solved the actual game prefixes at 36, 35 and 34 edges
remaining. Four cloud CPU workers used one million nodes per state, then retried
unknown states with five million nodes. Exact completed labels were retained;
unknowns were not converted into estimates. The two passes took about 100 and
142 seconds. Each worker retained its transposition table, so bounded coverage
can depend on scheduling even though returned exact values do not.

| Boundary | Exact roots / games | Exact winning side differs from final winning side |
|---|---:|---:|
| 36 edges | 169/200 | 5/169 |
| 35 edges | 196/200 | 2/196 |
| 34 edges | 200/200 | 0/200 |

There are 31 unresolved roots at 36 and four at 35. Harder positions are less
likely to be solved, so this table cannot establish a population error rate or
mastery of the band. Matching winners at two boundaries also does not rule out
intermediate mistakes that cancel each other.

Where both adjacent values were exact, the move at 36 turned a win into a loss
twice for balanced and once for iter24 (165/200 moves labelled). At 35 there was
one such mistake per agent (196/200 labelled). These are different positions for
each player, so the counts are not a matched comparison of their policies.

The follow-up selected 25 opening pairs (50 games) by the 25 lowest hashes of
`arena-frontier-v1:<pair index>`, before solving their 37/38-edge prefixes. It
reused exact 36-edge labels and used a five-million-node budget per new state,
four workers, and a ten-minute wall cap; it finished in 134 seconds.

Only 18/50 roots at 37 and 3/50 at 38 were solved, so this is evidence of specific
mistakes rather than a credible error-rate estimate. Of the 17 labelled adjacent
37-to-36 transitions, four turned a win into a loss: balanced in games 121/199,
iter24 in games 34/170 (zero-based game indices). One of the three labelled
38-to-37 transitions did so too: balanced in game 34. Do not treat these five
moves as five independent games, compare the agents on different states, or
infer good play from the unresolved cases.

The next useful training experiment is a small targeted pilot at **37 edges**,
with 35–36 retained as a regression check. Concentrate exact supervision on
decisions that lose winning positions; retain ordinary replay so opening play
and the already useful solver-supported endgame are rehearsed. Keep solver34
and search settings fixed, use fresh training families, and evaluate both exact
win preservation on reserved roots and a fresh full-game match before promotion.
These arena games are now development evidence, not a fresh final test.

This is a recommendation, not a completed training result. There is no claim
that the raw network has mastered 36, no solver-threshold change, and no
promotion. The unknown 36-edge roots still need attention; advancing one narrow
training band is an experiment, not a graduation based on this diagnostic.
