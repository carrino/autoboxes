# Experiment log

Everything tried so far, with the numbers, next to `PLAN.md` (the plan) and
`ARCHITECTURE.md` (how the code fits). Newest at the bottom; each entry says what was run,
what came out, and what it changed in the plan. Commands and knobs are in
`experiments/2026-10-08_14-42-boxes-3x3-loop/README.md`; the exact-oracle evaluation that
scores most of this is `oracle_eval.py` in that folder.

How to read the oracle numbers: every sampled position is solved exactly by the C++ endgame
solver. `policy optimal` is the share of positions where the raw net's top move is an optimal
edge, `search optimal` the same for the agent's searched move, `value sign` whether the net's
P(win) agrees with the exact outcome, `margin MAE` the error of its expected margin. 300
positions give a standard error of about 2.9 points, so differences under 5 points are noise.
"Clean positions" means positions sampled from another run's games, which the scored
checkpoints never trained on; a checkpoint scored on its own run's games is 5 to 10 points
flattered by memorisation (measured below).

## 1. 3x3 loops (2026-10-08)

Pipeline proof on the small board. GPU run (4 iterations, `TAG=3x3`): win rate against
alpha-beta depth 4 went 0.38 -> 0.88, oracle on 6..14 undrawn edges: searched moves
0.80 -> 0.92 optimal, value sign 0.60 -> 0.79 (`data/oracle_eval-3x3.csv`). The same loop
on this container's CPU (`TAG=3x3-cpu`, 5 iterations) reached a champion at iteration 5,
0.95 to 1.00 against alpha-beta depth 4, about 15 minutes per iteration.
Changed: nothing in the plan; it showed the loop learns when the game is small enough that
the net sees the whole fight.

## 2. Search-knob autoresearch on 3x3 (2026-10-09, CPU)

`experiments/2026-10-09_01-05-boxes-3x3-search-autoresearch/`: 25 keep / discard runs of
the MCTS agent's search knobs with a frozen checkpoint against the oracle on 600 positions.
0.823 -> 0.917 optimal searched moves. The margin-utility term in the leaf value (lambda 1.0)
was the dominant knob, sharper priors (policy temperature 0.7) second, leaf batch 4 third;
c_puct and the equivalent-edge merge were noise. The small-N solver hurt once lambda > 0
because solved positions back up 1 / 0.5 / 0 on a different scale from the shaped leaves
(open follow-up: shape terminal outcomes with the margin).
Changed: the knobs became CLI flags on every entry point (`SEARCH_ARGS` in the loop);
policy temperature 0.7 is used in every 5x5 run since; the margin utility waits for the
terminal-outcome fix because every 5x5 run uses the solver.

## 3. The overnight 5x5 solver run (2026-10-09, `TAG=5x5-solver`)

31 iterations, solver at 28 undrawn edges, 400 games x 300 simulations per iteration,
replay window 4 iterations, 300 s of training (about 8 passes), arena 40 games.

- Arena: 1.00 against greedy from the start, 1.00 against alpha-beta depth 4 from
  iteration 5 with margins of plus 5 to plus 10; 16 promotions, all but two with intervals
  including 0.5; four straight non-promotions after iteration 27. Both baselines were
  saturated by iteration 5 and measured nothing after.
- Training curve (`analyze.py`): held-out loss 3.83 at iteration 1, 3.43 at 12, then flat
  between 3.3 and 3.9 while the train loss fell 2.26 -> 1.96; held-out policy accuracy
  peaked at 0.125 (iteration 14) and slid to 0.08. Memorisation of the replay window:
  about 8 passes per iteration, three quarters of each window already trained on.
- Oracle on 8..24 undrawn (inside the solver zone, own positions): searched moves 0.59
  (bootstrap) -> 0.71 at iteration 1, then 0.72-0.78 through 31; alpha-beta depth 4 0.81.
  Value sign 0.86 -> 0.92. The net never learned moves there because solver-played
  positions gave it a one-hot of an arbitrary optimal edge as the policy target.
- Oracle on 26..32 undrawn (just above the solver zone, own positions): searched moves
  0.40-0.49 with no trend, value sign 0.55-0.65, margin error 4.9. Alpha-beta depth 4
  0.53. The net learned nothing in the only band where its judgement decides moves in
  play; the games were won by the solver.
- Self-play took 700 s at iteration 1 and 970 s at iteration 30: better play leaves
  harder endgames for the solver.

Changed: uniform-optimal policy targets for solver-played moves; `WINDOW`, `TRAIN_EPOCHS`
and `TRAIN_MIN_UNDRAWN` to stop memorising and to train above the solver zone; the
warmup capped at a tenth of the steps; `oracle_eval.py --positions-tag` for clean
positions; depth 6 and solver-backed baselines; the interpreter-lock release in the C++
bindings (depth-6 arena games 8 min -> 30 s per 10, self-play threads overlap).

## 4. Repaired pipeline, basic planes (`TAG=5x5-mid`, 4 iterations)

Warm start from the solver run's champion; train only above 24 undrawn edges, window 8, two
passes, uniform targets, policy temperature 0.7, 100-game arena.

- On its own positions (26..32): value sign 0.52 -> 0.66 over three iterations.
- On clean positions from the solver run's games: value sign 0.65 (iteration 0, which had
  trained on those games) -> 0.57, 0.57, 0.54; policy optimal 0.37 -> 0.45; searched moves
  0.42-0.44. The climb on its own positions was memorisation; the policy gain is real.
- Arena: 0.55 / 0.48 / 0.53 against the previous champion with margins within a box of
  zero (ties).

Changed: the representation hypothesis moved to the front (chain planes); the arena was
recognised as blind to the net while the solver plays for both sides.

## 5. Chain / loop planes (`TAG=5x5-chains`, from scratch, 21 iterations)

`FEATURES=chains` (10 planes: chain lengths, loops, opened components, safe edges, chain
count, loop count, safe-edge share, long-chain parity), train above 24 undrawn, window 8,
two passes, solver budget 50k from iteration 4, the solver-labelled set of about 15k
midgame positions in every training set from iteration 5.

- Clean positions, 26..32, net alone: policy optimal 0.43 at iteration 0, 0.47 at 1 and 2
  (the basic net needed three iterations to reach 0.45); searched moves 0.47-0.49; value
  sign 0.53-0.55.
- Arena against iteration 0 with the solver on for both: 0.42, 0.48, 0.53, 0.53, 0.48 with
  margins within two boxes of zero. Ties, as expected (see 6).
- Clean positions, 29..34, agent with its solver at 28 (the band where nothing is solved
  at the root and the search has to reach exact leaves one to six plies down; random
  0.15, greedy 0.34, alpha-beta depth 4 0.39):

  | iteration | policy optimal | searched move optimal | value sign |
  |---|---|---|---|
  | 0 | 0.323 | 0.430 | 0.483 |
  | 4 | 0.367 | 0.483 | 0.563 |
  | 7 | 0.377 | 0.473 | 0.610 |
  | 8 | 0.387 | 0.470 | 0.620 |

  The policy creeps up, the value head rose to the level of a depth-6 hand-written search
  once the labelled set joined, and play quality plateaued at about 0.47 after iteration 4.
- Iteration 15 (arena with the solver at 28 for both sides, 60 games, four opening moves at
  temperature 1): beat the champion, iteration 13, 60-0 with a mean margin of +6.0 (std 3.0);
  9-1 against alpha-beta depth 4 with the solver and 9-1 against depth 6; held-out policy
  accuracy 0.46, value accuracy 0.61 on 122k training positions. The first decisive
  promotion of any 5x5 run; every earlier arena between two nets was a tie. A margin of
  +6 in every game is what winning the chain fight (control) every game looks like, but
  until the ladder below is played it is not known whether iteration 15 learned it or
  iteration 13 is an unusually weak champion.
- The ladder and the clean read, same day. Iteration 15 against iteration 8 stood at
  exactly half the games won at every progress line (3 of 6, 6 of 12, ... 18 of 36), which
  a coin flip does once in a thousand matches: the side moving first (or second) won every
  game, so the two nets are equal and colour decides. Clean positions, 29..34, agent with
  its solver at 28, iterations 4 / 8 / 11 / 15: searched move optimal 0.483 / 0.470 / 0.507
  / 0.503, policy 0.367 / 0.387 / 0.390 / 0.377, value sign 0.563 / 0.620 / 0.700 / 0.687,
  margin MAE 4.08 / 4.15 / 3.70 / 3.66. No step at 15. The value head's climb to 0.70,
  above the depth-6 hand-written value, is the one trend, and it has not moved the
  searched move at 100 simulations. Caveat: the labelled set and these "clean" positions
  were both sampled from the `5x5-solver` games by the same function, so about a fifth of
  the scored positions carry exact labels the net trained on; the climb has to be
  re-measured on games the labelled set never touched (`--positions-tag 5x5-mid`) before
  it counts. So the 60-0 says iteration 13 was a weak champion, let
  in by a 60-game arena with a 0.55 threshold (which promotes an equal net one time in
  four). Self-play always used the latest checkpoint, so the weak champion never
  generated data.

- The run ended at iteration 21 (old arena throughout). Iterations 16 to 21 each scored
  exactly 30-30 against iteration 15 with mean margin 0.0, every six-game block 3-3: colour
  decided every game. Read with the earlier 0.98 / 1.00 / 1.00 promotions (11, 13, 15),
  four temperature-sampled opening moves from a sharpened policy give a handful of lines
  that 60 games replay, so the champion column measured nothing after iteration 7. Both
  baselines 10-0 at 21 (saturated). Held-out accuracy in the tail: value 0.61 (15) ->
  0.67 (16) -> 0.74 (21), policy 0.46 -> 0.49, training loss flat at 4.52 to 4.55; real or
  memorised is for the clean read on `5x5-mid` positions with 21 included.

- The clean read, on `5x5-mid` games that the labelled set never touched (29..34 undrawn,
  agent with its solver at 28, 100 simulations; random 0.15, greedy 0.32, alpha-beta depth 4
  0.37):

  | iteration | policy optimal | searched move optimal | value sign | margin MAE |
  |---|---|---|---|---|
  | 4 | 0.370 | 0.497 | 0.600 | 3.80 |
  | 8 | 0.397 | 0.483 | 0.553 | 3.76 |
  | 11 | 0.380 | 0.510 | 0.637 | 3.47 |
  | 15 | 0.390 | 0.510 | 0.637 | 3.49 |
  | 21 | 0.383 | 0.510 | 0.653 | 3.46 |

  Half of the value climb measured on the solver run's positions was memorisation of the
  labelled set: the real gain is about five points, to the level of depth-6 alpha-beta.
  The searched move has been at 0.50 since iteration 4, the policy at 0.37 to 0.40 even
  with 15k exact uniform-optimal targets in every training set. Seventeen iterations of
  self-play bought nothing in the band that decides games.

Changed: the arena plays colour-swapped pairs from the same random opening and scores each
pair on its summed margin, so colour cancels inside the pair; promotion needs the pair
score's 95% interval to exclude a tie; the per-colour win rates are reported. Every
future read of the chains checkpoints uses `--positions-tag 5x5-mid`.

Changed: nothing about the planes (they help the policy and cost nothing); the labelled
set stays; the plateau pointed at the search budget, measured next.

## 6. Where 5x5 strength comes from (diagnostics on the chains net, iteration 2)

Clean positions, 26..32 undrawn:

| evaluator | value sign | searched move optimal |
|---|---|---|
| alpha-beta depth 2, own value | 0.550 | |
| alpha-beta depth 4, own value | 0.557 | 0.530 |
| alpha-beta depth 6, own value | 0.607 | |
| chains net, value head | 0.550 | |
| net's search, 300 simulations, no solver | | 0.543 |
| net's search, 300 simulations, solver at 28 | | 0.707 |
| net's search, 1000 simulations, solver at 28 | | 0.740 |

No static evaluation reads this band, learned or hand-written to six plies. The search with
the solver two plies below is worth 16 points; tripling the simulations is worth 3. The
midgame is a lookahead problem; the net's useful output there is the policy that aims the
search, and the exact horizon is the lever.

Changed: `ARENA_SOLVER_N` so promotion can compare nets with the solver off; self-play can
branch from stored midgame positions and stop at the solver's exact outcome
(`START_FROM`, `STOP_WHEN_SOLVED`), so the search budget goes to the deciding band; the
solver's reach became the main engineering target.

## 7. Solver cost and the ordering work

- Probe games (weak play, 2026-10-09 morning): 28 undrawn solves in 0.6 ms median, 32 in
  16 ms with 94% within 20k nodes.
- The solver run's own games (strong play): 29 undrawn settles 85% of positions within
  20k nodes (median 12 ms, p99 500 ms), 30 64%, 31 52%, 32 33% (median 94 ms, p99 6.8 s,
  worst 3 million nodes). About 2x the nodes per extra undrawn edge.
- Local hard set (depth-6 games, 60 positions), old solver -> table-move and history
  ordering with MTD(f): 32 undrawn 7,500 -> 6,066 nodes median (p99 27.4k -> 22.0k),
  34 undrawn 62.9k -> 49.1k (p99 235k -> 179k). Values unchanged. 32 -> 34 costs 8x.

Changed: the solver stays at 28 with a 50k budget for now (29 or 30 cost two to three
times the self-play for one or two edges); the next exact gain is structural (Nimstring
values of independent regions), not search tuning.

## 8. Proof propagation in the search (MCTS-Solver), 2026-10-10

The "high/low versus average" question: with the solver two plies below the midgame band,
the search reaches exact leaves, but it backs them up by averaging, so one exact loss among
many 0.5 estimates barely moves a child's value. `--prove_terminals 1` (shared
`MCTSConfig.prove_terminals`, off by default, Go byte-identical with it off) makes
terminal and solver-settled leaves proven, proves a node once a child is a proven win for
its mover or every child is proven (minimax over them), hands proven values up unchanged in
place of the simulation result, and keeps only optimal moves at a proven root, which also
makes the policy targets exact there. Python reference and C++ agree on tiny boards against
the exact oracle (`tests/test_boxes_mcts_sign.py`, `tests/test_boxes_cpp_mcts.py`).

To measure (iteration 15 of the chains run, clean positions, the solver at 28), the flag off
against on at the oracle's 100 simulations and at the self-play budget:

```bash
for P in 0 1; do for S in 100 400; do
  uv run $EXP/oracle_eval.py --tag 5x5-chains --positions-tag 5x5-mid --min-undrawn 29 --max-undrawn 34 \
      --solver_max_undrawn 28 --solver_node_budget 50000 --num_simulations $S --prove_terminals $P --iterations 15
done; done
```

Result on 5x5, iteration 21 of the chains run, clean `5x5-mid` positions (29..34), solver
at 28 with a 50k budget (the 100-simulation row had the 20k default):

| solver at | simulations | proof flag | searched move optimal | seconds for 300 |
|---|---|---|---|---|
| 28 | 100 | off | 0.510 | |
| 28 | 400 | off | 0.550 | 105 |
| 28 | 1000 | off | 0.600 | 194 |
| 28 | 400 | on | 0.587 | 109 |
| 28 | 1000 | on | 0.607 | 244 |
| 30 (100k budget) | 400 | on | 0.693 | 244 |

The simulation budget moves the band at three to four points per doubling, not flat by
1000: the first lever since iteration 4 that moved the searched move. The flag at 400 is
worth about a doubling, free; at 1000 it adds nothing measurable, so the two do not stack.
Two more edges of solver reach are worth nine points over 1000 simulations at the same
wall time (both 244 s, measured while self-play ran). Part of that is the positions at
29..30 undrawn, now solved at the root; scoring 31..34 alone would say how much is
lookahead. On the same positions the net's value head (0.653) beats depth-2 (0.543) and
depth-4 (0.550) alpha-beta's own values by ten points.

A 16x larger solver table (16.8M entries, 268 MB per agent) gives the same 0.693, in 163 s
on an idle machine; the 244 s above ran beside self-play, so the timing does not compare.

Changed: the restart runs the solver at 30 with a 100k budget, 400 simulations and the
flag on, with branched self-play; the solver's reach is confirmed as the main lever.

CPU probe first (`proof_probe.py` in the loop folder: 3x3, a uniform evaluator with no net,
solver at 8, 300 random positions at 11..14 undrawn edges, so the exact leaves sit 3 to 6
plies down as they do in the 5x5 band; a random move is margin-optimal 0.27 of the time):

| simulations | flag | margin-optimal move | keeps the exact outcome | roots proven |
|---|---|---|---|---|
| 100 | off | 0.350 | 0.817 | 0.00 |
| 100 | on | 0.377 | 0.837 | 0.17 |
| 400 | off | 0.453 | 0.917 | 0.00 |
| 400 | on | 0.453 | 0.927 | 0.51 |

Two to three points at 100 simulations and none at 400, inside the noise of 300 positions,
while half the roots are proven at 400. So the averaging backup was not what held the
searched move at 0.50: once the tree reaches the exact leaves the average already orders
the moves right, and the loss is in which children get explored at all, which is the
policy prior's job. The flag is free (proven roots also make the policy targets exact there)
and stays available; the 5x5 read above says whether a real net changes the picture.

## 9. Branched restart of the chains run (iterations 21 to 41)

Resumed from iteration 21 with branched self-play (75% of games start from the
`5x5-solver` games at 36..40 undrawn), games stopped at the solver's exact outcome, 1000
simulations, the proof flag, the solver at 28 (50k budget), the labelled set, and the
paired arena (solver at 28 for both sides, 400 simulations).

- Training sets shrink to about 70k positions over eight iterations, nearly all in the
  band: branched games record about ten positions each.
- Iteration 40 against the champion, iteration 15: pair score 0.38 [0.23, 0.56], mean
  margin -0.7, not promoted. The candidate won 7% of its games moving first and 87% moving
  second: from random four-edge openings, between nets this close, the second player wins
  almost every game and the pairing is what makes the match readable at all. Against
  greedy 10-0, against alpha-beta depth 4 8-2 (both pairs won).
- Clean read, same positions and search settings as section 5 (100 simulations, solver at
  28):

  | iteration | policy optimal | searched move optimal | value sign | margin MAE |
  |---|---|---|---|---|
  | 21 | 0.383 | 0.510 | 0.653 | 3.46 |
  | 25 | 0.410 | 0.530 | 0.640 | 3.46 |
  | 30 | 0.413 | 0.557 | 0.637 | 3.53 |
  | 35 | 0.423 | 0.523 | 0.640 | 3.53 |
  | 41 | 0.413 | 0.500 | 0.623 | 3.54 |

  Twenty iterations of branched self-play at 1000 simulations moved the policy up about
  three points and the searched move not at all; the value sign drifted down within
  noise. The paired arena agrees: every candidate from 22 to 41 scored 0.38 to 0.58
  against iteration 15, none significant. The held-out value accuracy fell from 0.74 to
  0.62 as the window filled with band positions, which are harder; the clean read did not
  move. Collect time rose from 870 s to about 1500 s per iteration over 28..41, and the
  newer nets took about twice as long per move as iteration 15 in the same arena games.

Changed: the search budget is spent on reach instead of simulations. Resumed from
iteration 41 with the solver at 30 (100k budget), 400 simulations, everything else the
same, four self-play processes. The question for it: do search targets that are 69%
right instead of 60% lift the policy, where 60% ones stopped at 0.42? Read at 45 and 49
with the command above.

### The solver-30 run (from iteration 41)

- Iteration 54 against iteration 15, paired arena with the solver at 30 for both: pair
  score 0.25 [0.13, 0.43], mean margin -2.8. The first significant result between two nets,
  and it is a loss: iteration 15, trained before any branching, plays whole games better.
  The branched runs were already drifting the same way (0.38 at 40, 0.43 at 41). Iteration
  54 also spent 1.73 s per move against 0.49 s for iteration 15 in the same games (0.79 /
  0.40 at 40, 1.19 / 0.65 at 41).
- Leading hypothesis: the opening regressed. Branched self-play gives the opening and early
  middle game (60 down to about 40 undrawn) only the quarter of games that start from the
  empty board, a quarter of the data it had before, and the band oracle cannot see it.
  The test: the same paired match started from clean midgame positions instead of random
  openings (`arena.py --start_positions ... --start_undrawn 36 40`, `5x5-mid` games). If
  iteration 54 holds its own from 36..40 undrawn and loses from the opening, the opening is
  what regressed.
- Result. From clean midgame positions (`5x5-mid` games cut at 36..40 undrawn, 50 pairs,
  solver at 30): pair score 0.51 [0.38, 0.64], mean margin +0.3. A tie, where the same two
  nets from random openings scored 0.25. Iteration 54 also spent 1.8x iteration 15's time
  per move from the midgame, against 3.5x from the opening. The opening regressed; the
  middle game did not.
- Clean band read (100 simulations, solver at 28):

  | iteration | policy optimal | searched move optimal | value sign |
  |---|---|---|---|
  | 15 | 0.390 | 0.510 | 0.637 |
  | 41 | 0.413 | 0.500 | 0.623 |
  | 45 | 0.423 | 0.493 | 0.607 |
  | 49 | 0.430 | 0.503 | 0.627 |
  | 54 | 0.447 | 0.520 | 0.613 |

  The policy climbs steadily under the solver-30 targets (0.41 -> 0.45 in thirteen
  iterations, against 0.38 -> 0.41 in twenty under solver-28 targets): better search
  targets do teach the policy. The searched move at 100 simulations has not followed yet.

Changed: the next run restores the opening's data (`START_FRACTION=0.25`) and gates
self-play on the arena champion (`GATE=1`), resuming training from iteration 54.

### The gated restart (`TAG=5x5-gate`, 2026-10-12)

Iterations 55..59 of the chains run under gating (self-play from iteration 15, three
quarters of games from the empty board) still lost to 15 from random openings (pair score
0.40..0.45) while their training window was the branched data. Rather than wait eight
iterations for the window to turn over from weights that carry the drifted opening, the
run restarts from iteration 15's weights as `5x5-gate/iter0.pt`, with:

- gating (`GATE=1`), a quarter of games branched (`START_FRACTION=0.25`), solver at 30;
- the search from PR #9: `--shape_terminals 1 --margin_utility_lambda 0.5`, which put the
  band's searched move at 0.82 for iteration 54 and 0.79 for iteration 15 (from 0.66 /
  0.68), plus proof propagation and policy temperature 1.0;
- 200 games per iteration and a 50k solver budget, after timing probes of eight games on
  an idle machine: the new search costs 1.3 / 1.4 s per move at temperature 1.0 with a
  100k budget, 1.0 / 0.9 at temperature 0.7 with 50k, and single hard games near the
  solver boundary run to 2..4 s per move. A reused solver does not slow down across games
  (its median nodes fall 1154 -> 483 over 900 positions as the table fills), so the cost is
  the number of boundary solves per move, not accumulated state. Temperature stays at 1.0
  for the diversity of the policy targets; the budget carries the saving.

Iteration 15 was never especially strong: it tied iterations 8 and 16..21 colour for
colour. It is the last net trained only on full games, and everything after it lost the
opening. The question for this run: does a net trained from it, under search targets that
are about 80% right in the band, beat it in the paired arena from random openings?

Answer after 12 iterations: no. Nothing was promoted, so all self-play came from
iteration 0 and the run distilled it.
- **Arena.** Pooled over the 12 arenas (different openings each), the challengers scored
  171.5 of 360 pairs (0.48, about [0.42, 0.53]), with a mean margin of -0.1.
- **Clean band read** (`5x5-mid`, 29..34, solver 28, default search). The policy rose from
  0.390 to 0.41..0.43 by iteration 5 and then stayed flat. The searched move went from
  0.51 to 0.51..0.53, value sign stayed at 0.62..0.65 (0.637 at iteration 0), and margin
  MAE went from 3.50 to 3.43.

The weights carry over between iterations (training resumes from the latest), but the
data never advanced. The ungated chains run (16..55), whose data did advance, stalled at
the same level. The striking number is the gap between student and teacher: self-play
targets are about 80% optimal in the band, yet the net's policy is at 42%. Each iteration
trains for 2 epochs (about 400 steps, 30..70 s, from a fresh optimizer and cosine
schedule), so the first thing to rule out is underfitting: train iteration 12 on its own
window to convergence and band-read it.

Underfitting ruled out (`5x5-fit/iter1`: iteration 12 trained for 20 epochs, 4,080 steps,
on its own window).
- **Policy.** The policy loss stayed at 2.55..2.65 nats the whole time, and policy accuracy
  stayed at 0.357 (train) / 0.366 (val). The policy already matches its diffuse targets
  (temperature-1 visits plus 0.1 label smoothing) as closely as this net can.
- **Value.** The value fit its own games much better: value accuracy went 0.65 -> 0.76
  (train) and 0.63 -> 0.74 (val, held-out games from the same window). On the clean band,
  though, value sign fell 0.643 -> 0.610, margin MAE rose 3.43 -> 3.52, and the policy
  stayed at 0.427.

More training fits the outcomes and the position distribution of iteration 0's own games,
not the true values. The game-split validation set shares that distribution (one champion,
branched games with shared prefixes), so it overstates generalisation. The bottleneck is
the targets, not the amount of training.

### First absolute yardstick: the classical engine (PR #6)

`boxes-classic` is one-second iterative-deepening alpha-beta with a chain/control evaluation
and exact probes at 26 undrawn or fewer. In 200-game paired matches it scores 0.935 against
`boxes-ab-d6`. Against it, `5x5-gate/iter12` (400 simulations, solver 30 with a 50k budget,
the #9 search) played 200 games:

- **Games:** 135-65, a win rate of 0.675 [0.61, 0.74]. It won 0.69 as first player and 0.66
  as second.
- **Pairs and margin:** pair score 0.585 [0.49, 0.68], mean margin +0.26 ± 6.6.
- **Time:** 0.16 s per move, against classic's 0.42.

The net wins two games in three, but the margin is even: its wins are narrow and its losses
are large. Two causes look likely. The margin utility makes the net value winning over
margin. And a lost control fight costs most of the long chains.

`arena --save_games` with `arena_exact.py` replays the games against the solver from 34
undrawn edges. It reports whether each loss was already decided when the game came within
reach, or was given away inside it. The 31..36 labels can only fix the second kind.

**Replay of a rerun from the same 200 openings.** The net won 146-54 (0.73 [0.67, 0.79]);
the self-play loop shared the CPU part of the time, which slows the timed engine. Games were
replayed against the solver from 34 undrawn edges (13 s for all 200):

| net's result | games | net winning at 34 | mean exact margin at 34 | net error boxes | classic error boxes | final margin |
|---|---|---|---|---|---|---|
| won | 146 | 0.98 | +4.92 | 1.14 | 0.75 | +4.53 |
| lost | 54 | 0.11 | -4.67 | 3.19 | 0.33 | -7.52 |

- **Where the games are decided: above 34 undrawn edges.** 48 of the 54 losses were already
  lost at 34, and only 6 were thrown away inside the reach. Likewise, 143 of the 146 wins
  were already won at 34. Both sides' results come from the midgame control fight, and the
  net wins it about three times in four.
- **Inside the reach, classic plays better than the net.** The net gives away 1.7 boxes a
  game against classic's 0.6. Its error moves fall at 34 (43), 33 (23) and 32 (7), none
  lower: below 32 its search reaches the solver at 30. Classic's full-width alpha-beta is
  more accurate four edges above the net's solver than 400 simulations of the net are.
- **The net loses big because it plays lost positions badly** (3.2 error boxes in lost
  games, 1.1 in won ones). With the win term dominating the utility, there is little
  incentive to fight for margin once the game is lost.

What follows: the 31..36 labels and more solver reach can repair the 32..34 errors (margin,
plus about 6 of 54 losses). Most of the losses happen above 34. Replaying from 38 undrawn
shows how much higher they happen.

**Replay from 38 undrawn** (2,196 s on two workers, 100M-node budget, nothing skipped):

| net's result | games | net winning at 38 | mean exact margin at 38 | net error boxes | classic error boxes | final margin |
|---|---|---|---|---|---|---|
| won | 146 | 0.58 | +0.48 | 5.85 | 9.90 | +4.53 |
| lost | 54 | 0.30 | -2.33 | 10.96 | 5.78 | -7.52 |
| all | 200 | 0.50 | -0.28 | 7.23 | 8.79 | +1.28 |

- **The games are decided between 38 and 34 undrawn edges.** At 38 the net is winning in
  half the games (mean exact margin -0.3); at 34 it is winning in three quarters. 62 of the
  146 wins were lost positions at 38, and 16 of the 54 losses were won ones.
- **Both players play that zone badly, the net less badly.** From 38 to 35 the net gives
  away 5.5 boxes a game and classic 8.2. The net's error moves by undrawn count: 38: 74,
  37: 65, 36: 65, 35: 33, 34: 43, 33: 23, 32: 7. About 100 of the 200 games have the net
  to move at a given count, so at 36..38 roughly two thirds of its moves lose exact margin,
  one third at 34..35, and almost none from 32 down, where the search reaches the solver.
- **So the net's edge over classic is the 35..38 zone,** and its remaining losses are
  mostly decided there too: 38 of 54 were already lost at 38, and those positions reached
  38 through the same zone of a previous capture fight.

The 31..36 labels cover the lower half of the zone. The upper half, 36..40, is where exact
labels would matter most; the replay solved every position from 38 along a game line at
about 2.4 CPU-seconds each with the table carried along the line, far cheaper than the
labeller's root-plus-children solves (22 CPU-seconds per position at 31..36).

### Why the loop is stalled, and the fix being built (2026-10-13)

Thirty-odd nets trained over two days (gate 1..12, s1600 1..20) tie iteration 0 in the
arena (pooled pair score 0.48; a promotion needs about 0.68 over 30 pairs, which equal
nets clear 3..5% of the time, so zero promotions is what equal nets produce). The classic
column of s1600 is flat at about 0.75 and the band reads are flat. Three measurements say
why:

1. games are decided at 35..38 undrawn, where the net loses exact margin on two of every
   three moves (the replay from 38);
2. there the search cannot reach the solver, so its leaves are scored by the value head,
   which gets the sign right 64% of the time;
3. the value head's targets are the outcomes of games both sides misplay in that zone, and
   training harder on them fits the net's own games and worsens the clean read (the
   fitting probe).

That is a fixed point: bad play at 35..38 -> noisy outcome targets -> bad value head -> bad
search at 35..38. More simulations did not break it (s1600). AlphaZero escapes it with
millions of games; this loop has 100 per iteration, and the solver has only been used at
<= 30 in play and <= 34 in labels, never in the decisive zone.

**Exact value relabelling** (`alpha_go/boxes/relabel.py`, `relabel_exact.py`, loop knob
`RELABEL_N`): after each self-play batch, every game is replayed against the solver from
`RELABEL_N` undrawn edges down, one solver per game with the table carried along the line
(the replay did 200 games from 38 in 37 CPU-minutes), and every position within reach gets
its exact final margin as the value target (`exact_margin_p1` in the NPZ, which
`BoxesDataset` prefers over the outcome). The script also prints how far the outcomes were
from the exact values by undrawn band: the noise it removes. Tests against the brute-force
oracle on 2x2 and 2x3, branched games included, in `tests/test_boxes_relabel.py`.

Plan: relabel the s1600 window offline, train iteration 20 on it once, band-read and play
200 games against classic (the fitting-probe pattern). Pass = value sign on the clean band
from 0.64 to >= 0.70 or classic clearly above 0.75; then the next run is the s1600 recipe +
`RELABEL_N=38` + the 7,445 labels from #4 + the #7 solver. Fail = the net cannot represent
the value function, and the next step is architecture.

**How wrong the value targets were** (`relabel_exact.py` over s1600's iteration 14, 100 games,
902 positions within 38 undrawn, 9 over a 100M-node budget, 19 minutes on 4 workers beside
the running loop):

| undrawn | positions | mean \|outcome - exact\| | outcome sign wrong |
|---|---|---|---|
| 25..30 | 127 | 0.00 | 0.00 |
| 31..34 | 400 | 0.56 | 0.01 |
| 35..38 | 375 | 2.28 | 0.24 |

The other iterations of the window (15..19) read the same: 17..25% wrong-signed at 35..38
with a 2.1..2.8 box error, 0..1% at 31..34, 7..20 positions per 100 games over the budget.
At 35..38, about one value target in five has the wrong sign and the mean error is 2.3 boxes; at
31..34 the targets were already clean (the solver at 30 and STOP_WHEN_SOLVED make the tail
of every game optimal). So the clean band read at 29..34 (value sign 0.64) measures the
head on positions whose training targets were right, which bounds what exact targets can
give there: the gain to expect is at 35..38, and the probe must read that band (33..36 is
affordable to solve: 300 positions in minutes; 35..38 costs hours).

**The A/B (2026-10-13, 33..36 band on `5x5-mid`, 300 positions).** Both arms from
`5x5-s1600/iter20` on its window (`dataset-5x5-s1600-it21.txt`: 845 games, 34k positions,
6,155 of them with exact targets in the exact arm), same seed and stable split, 4 and 12
epochs:

| | policy optimal | search optimal | value sign | margin MAE | margin lost policy / search |
|---|---|---|---|---|---|
| start, iter20 | 0.300 | 0.343 | 0.700 | 2.63 | 3.78 / 3.34 |
| exact, 4 epochs | 0.320 | 0.367 | 0.743 | 2.50 | 3.81 / 3.22 |
| control, 4 epochs | 0.320 | 0.340 | 0.720 | 2.56 | 3.69 / 3.45 |
| exact, 12 epochs | 0.307 | 0.323 | 0.720 | 2.49 | 4.03 / 3.60 |
| control, 12 epochs | 0.300 | 0.340 | 0.713 | 2.56 | 3.71 / 3.37 |

No dose response: the 4-epoch gap of 2.3 points on value sign (one standard error) vanished
at 12 epochs, and the training-set value accuracy was identical in both arms (0.654). The
relabel changes only the targets that were wrong, about 20% of the 2,600 positions at
35..38, so some 550 flipped signs among 34,000 positions that also carry the policy loss;
and half of the 33..36 read (33..34) had correct outcome targets to begin with. At this
density exact values do not register, and `RELABEL_N` in the loop, with the same density
per window, would not either.

Two incidental measurements of the band: the search adds almost nothing at 33..36 (policy
0.30 -> searched 0.34, because it cannot reach the solver) and the searched move gives
away 3.3 boxes on average, no better than depth-4 alpha-beta (3.25).

Next: make the exact signal dominant and read the band where every exact label differs
from the outcome. `train.py --exact-weight W` counts an exact-labelled position's value
loss W times (W = 8 makes the 6,155 exact positions outweigh the 28,000 outcome ones), and
the read moves to 35..36 with 200 positions. If that moves value sign by several points,
exact values work and the loop needs many more of them (line labels at 35..40, the #4
set); if the net fits the weighted labels on the training side and the band still does not
move, that is the capacity evidence for an architecture change.

**Does the net fit exact labels at all?** (`exact_fit.py`: value sign / margin MAE on the
6,155 *training* positions with exact targets, by undrawn band; the `--exact-weight 8` run
is `quick-exact/iter3`, 12 epochs from the same start.)

| checkpoint | 25..30 (n=997) | 31..34 (n=2680) | 35..38 (n=2476) | all |
|---|---|---|---|---|
| s1600 iter20, the start | 0.728 / 2.55 | 0.608 / 2.79 | 0.566 / 2.41 | 0.611 / 2.60 |
| control, 12 epochs on outcomes | 0.763 / 2.34 | 0.646 / 2.65 | 0.588 / 2.39 | 0.642 / 2.49 |
| exact x1, 12 epochs | 0.751 / 2.38 | 0.663 / 2.68 | 0.703 / 2.11 | 0.693 / 2.40 |
| exact x8, 12 epochs | 0.836 / 1.80 | 0.731 / 2.30 | 0.746 / 1.88 | 0.754 / 2.05 |

- The starting net barely knows the value of its own training positions at 35..38 (sign
  0.57), and twelve more epochs on outcomes cannot teach it (0.59): a fifth of those
  targets are wrong.
- Exact labels teach it, and the net can learn them: 0.70 at natural weight, 0.75 at x8,
  MAE 2.4 -> 1.9. Representation is not the first-order problem.
- The x8 gains at 25..34, where the targets are identical in both arms, show the value
  head has been under-trained relative to the policy all along (equal loss weights, two
  epochs per iteration).

**The 35..36 read** (`5x5-mid`, 200 positions, raw net, value sign / margin MAE):

| checkpoint | value sign | MAE |
|---|---|---|
| s1600 iter20, the start | 0.670 | 2.44 |
| control, 12 epochs on outcomes | 0.720 | 2.35 |
| exact x1, 12 epochs | 0.740 | 2.25 |
| exact x8, 12 epochs | 0.705 | 2.29 |

Natural-weight exact labels help a little (+2 points, 0.1 box), within noise at 200
positions and consistent with Astra's +1 at 33..36 and +9 at 37..38. The x8 run memorises:
best fit on the training labels, worst generalisation. And the base rate matters: the
alpha-beta rows (0.345 / 0.355) predict "not winning" on nearly every position here, so a
constant "mover wins" guess scores about 0.65 on this band; iter20's 0.67 is nearly nothing
over it and the best net's 0.74 is +9. (Corrected 2026-10-14: `oracle_eval` now prints the
base rate from the labels, 0.735 on these 200 positions, so 0.74 is the constant guess and
iter20's 0.67 is below it; see the exact-value loop section.) The value head at 35..38 is barely better than the
base rate, which matches the 0.49 at 37..38.

So exact values are the right target and the net can learn them, but 2,500 positions at
35..38 from one net's own games are too few and too alike to teach a value function that
generalises. The lever is label quantity and diversity (line labels at 35..40 from many
games, Astra's Task 5), with the loop relabelling at natural weight (`RELABEL_N=38`,
`5x5-exact` run) to stop training the head on wrong targets meanwhile. (One position at 36
undrawn took 58 of the read's 64 minutes: the labelling solve was unbounded;
`--label-budget` now drops such positions.)

**Astra's `exact_ab.py` run** (`logs/exact-ab-20261013-140131`, natural weight, 4 epochs,
raw net, 100 positions per band on `5x5-mid`; value sign / margin MAE):

| band | reference (iter20) | control | exact |
|---|---|---|---|
| 33..36 | 0.68 / 2.71 | 0.66 / 2.72 | 0.69 / 2.63 |
| 37..38 | 0.49 / 2.38 | 0.50 / 2.40 | 0.59 / 2.28 |

At 37..38 the current value head is a coin flip (Brier 0.245: it outputs about 0.5 for
everything), which is the decisive zone of the replay. Exact labels are the first thing to
move it: +9 points on the same 98 positions, about 1.8 standard errors, with MAE and Brier
improving in step. At 33..36 the arms tie, as in the quick A/B, because few labels changed
there.

A second lead to check first (`target_entropy.py`): iteration 19's candidate took 1.10 s a
move against the champion's 0.49 at equal simulations and solver, which is what a flatter
prior does (more distinct leaves, more solver calls), and s1600's policy cross-entropy is
2.85 against the gate run's 2.6. If the visit-target entropy climbs with iteration, the
targets are flattening themselves (Dirichlet noise plus temperature-1 visits at a few
hundred simulations), and the fix is sharper targets and a lower `c_puct`.

Measured on s1600 (entropy in nats / top-move share, by undrawn band):

| iteration | 31..34 | 35..40 | 41..60 |
|---|---|---|---|
| 0 | 1.91 / 0.44 | 2.05 / 0.42 | 2.42 / 0.21 |
| 5 | 1.74 / 0.49 | 2.20 / 0.38 | 2.93 / 0.14 |
| 10 | 1.81 / 0.47 | 2.19 / 0.38 | 3.01 / 0.13 |
| 20 | 1.78 / 0.49 | 2.19 / 0.38 | 3.12 / 0.12 |

The flattening is real but confined to the opening (41+ undrawn, where most moves are
exactly equal and the arena randomises the first four edges); the decisive 35..40 zone is
stable, undecided (top move 0.38 of the visits among about 37 legal moves) and not drifting.
The gate run, where one net plays every iteration, is the control: nothing moves in any
band (2.15 / 2.53 / 2.10 at iterations 0 and 10), so target entropy is a property of the
playing net. The chains run went the other way in the opening (2.76 -> 1.84 from iteration
0 to 20, top-move share 0.11 -> 0.30) while it was still improving, with 35..40 at 2.6..2.8.
In the decisive zone the 1600-simulation search is the most decided of the three runs
(top-move share 0.38 against 0.23 and 0.14..0.17) and its nets are no stronger for it.
So the policy targets are not the lever; the value head is. (Iteration 0 holds 139 games:
self-play files carry a timestamp prefix, so a restart adds games beside those of a killed
attempt instead of replacing them.)

### The exact-value loop (`TAG=5x5-exact`, 2026-10-14): a regression

The first loop with exact value targets: the s1600 recipe plus `RELABEL_N=38` at natural
weight, `TRAIN_EPOCHS=4`, `WINDOW=8`, `GATE=0`, `EXTRA_DATA` = the 26..34 and 31..36 label
sets, from `quick-exact/iter2` (s1600's iteration 20 plus four epochs on exact labels) as
iteration 0. One iteration took 40..55 minutes: self-play of 100 games at 1600
simulations 20..33 minutes, relabelling 9..14 minutes on six workers (about 920 positions
within 38 undrawn per iteration, 3..5 of them over the 100M-node budget, one game holding
the last four minutes), training 50..90 s, the two arenas 6..7 minutes. Stopped after 22
iterations (`report-5x5-exact.md`).

| | iterations 1..8 | iterations 15..22 |
|---|---|---|
| vs iteration 0, pair score (60 games each) | 0.55 | 0.48 |
| vs classic, win rate / margin (60 games each) | 0.55 / -1.8 | 0.45 / -2.8 |
| relabel of the games just played, 35..38: mean \|outcome - exact\| / sign wrong | 2.51 / 0.24 (iteration 0's games) | 2.83 / 0.29 (iteration 21's) |

- **Against classic the run started at 0.55 and slid to 0.45**, with the worst iterations
  at 0.32..0.38 (9, 14, 19, 21). Iteration 1 ties iteration 0, so iteration 0 itself
  (the exact-trained `quick-exact/iter2`) was at about 0.55. Astra's `exact_ab` arena,
  the one same-session comparison, read the same thing after a single training step:
  the s1600 iteration-20 reference 57/100 against classic, the outcome-trained control
  54, the exact-trained candidate 44. One step of exact value training costs about 13
  points of classic win rate while improving the clean value read.
- **The yardstick is not comparable across sessions.** s1600's 0.75 against classic came
  from another session: classic is wall-clock limited (one second a move), so a faster
  solver (#7, 1.4x), a rebuilt extension, different search arguments or CPU load move it
  by tens of points. The same reference net scored 0.57 in Astra's arena. Only pairs
  measured in one session count, which is an argument for the fixed-simulation BoxesZero
  opponent (#13) as a second yardstick.
- **The window is memorised.** Once the window filled (iteration 8, 34k positions), the
  train loss fell every iteration, 4.35 -> 4.07, while the held-out loss rose, 5.12 ->
  5.53, and held-out policy accuracy fell, 0.21 -> 0.18 (`analyze.py`'s second table,
  the signature its docstring names). Four epochs over an eight-iteration window gives
  every game 32 passes, twice the s1600 recipe.
- **The value head at 35..36 has no sign skill, on either position set, and did not
  change.** On the clean `5x5-mid` positions (200, the mover winning 0.735 of them, so a
  constant "mover wins" guess scores 0.735) iterations 0, 3, 7, 21 and 22 read 0.740 /
  2.25, 0.745 / 2.20, 0.740 / 2.19, 0.730 / 2.25 and 0.730 / 2.23. On 200 held-out
  positions from the run's own games (mover winning 0.645) iteration 0 reads 0.620 /
  2.64 and iteration 21 0.645 / 2.55, depth-4 alpha-beta 0.400. The MAE moved by a tenth
  of a box; the sign sits on the constant guess throughout. This corrects the 35..36
  read above: the base rate there was estimated from the alpha-beta rows as 0.65, and
  `oracle_eval` now prints it from the labels, 0.735, so the exact-trained net's 0.740
  is the base rate, not nine points over it. What one step of exact training did to the
  value head was move it from below the base rate (s1600 iteration 20 at 0.670, which
  under wrong-signed targets predicted "not winning" too often) up to it: it learned the
  prior of the zone, not the positions. The net fits its exact training labels
  (`exact_fit`, 0.70 at 35..38) and does not generalise them.

So exact value targets at this density do not lift play, and the exact-trained value
head makes play worse at once, before any memorisation, while its clean read sits on
the base rate in both arms. The hypothesis to test first: a head that has learned the
zone's prior answers "mover wins about 0.74, margin near zero" for every leaf at 35..38,
whatever the position, so the search's Q differences between siblings vanish and the
move choice falls back on the policy prior. The outcome-trained head, noisier and below
the base rate in sign, may still have ranked siblings, since its errors came from the
targets' noise rather than from a constant output. Astra's four-way probe (reference policy
with candidate value, candidate policy with reference value, exact regret on frozen
positions) measures exactly that, and its lambda / simulation sweep says whether the
margin term or more search recovers it. The training-side fixes are independent of the
answer: two epochs, not four, and a gate on the classic column, since `GATE=0` let the
slide compound for twenty iterations with nothing in the script to stop it. The entropy
diagnostic above stands in its measurement (the visit targets are not flattening) but
not in its last clause: the policy targets are still the net's own moves, which the
replay from 38 found wrong two times in three.

**Merged from Astra (PRs #13 and #14, 2026-10-14)**, no loop default changed:

- `alpha_go.boxes.search_probe`: the four-way policy/value cross of two checkpoints on
  frozen positions, plus sweeps of margin utility and simulations, scored by exact
  regret on the common solved subset (roots and chosen children only, bounded solves).
- `alpha_go.boxes.policy_labels`: exact policy targets at consequential mistakes in
  training games (winning-to-losing first, then regret), saved only when every legal
  child is solved; the output is an `EXTRA_DATA` collection that inherits its source
  game's validation family.
- `alpha_go.boxes.mixed_collect`: colour-paired games against self, older checkpoints
  and classic with a `train_mask` that keeps only the learner's searched turns;
  `BoxesDataset` honours the mask.
- `BoxesNet(value_target="remaining")` with `train.py --value-target remaining
  --reset-value-head`: the head predicts the unclaimed boxes' margin, forward adds the
  known score and masks infeasible outcomes; legacy checkpoints stay final-margin.
- `boxeszero:<config.json>` arena opponent (`alpha_go.boxes.boxeszero_install` fetches
  the pinned author snapshot and checkpoint): the published neural engine as a yardstick
  at a fixed simulation count, scored by game wins (it stops at 13 boxes and the adapter
  completes the game).

Next, in this order, one GPU job at a time: the four-way probe on the `exact_ab` pair
(reference and exact candidate, one training step apart, 57 and 44 against classic) at
35..38, then its lambda / simulation sweep on the same positions; the BoxesZero smoke
and 20-game pilots with the reference; `policy_labels` over the 5x5-exact window, timed
on its first games, and one A/B from the reference (its window with and without the
labels, two epochs, same seed and split) read on the clean bands and in 100 games
against classic in one session. Only a gain in that A/B puts policy labels into the
loop, and the loop gets a classic-column gate and two epochs before it runs again.

### Solver PR #7 (reviewed 2026-10-13)

Two changes to `boxes_solver.cpp` / `boxes_chains.cpp`: chain walking without temporary
vectors, and transposition entries that keep both proven bounds (the entry stays 16 bytes;
`lower`/`upper` replace the single value for cutoffs, `value` stays the MTD(f) guess).
Cutoffs read `lower >= beta` / `upper <= alpha`, bounds only tighten, and an exact entry
sets `lower == upper`. Every one of 6,000 benchmark values is unchanged; the new
`test_boxes_solver_bounds.py` checks tiny tables (2 entries, so constant collisions),
budget restarts and symmetric images against the oracle. Merged into this branch on
2026-10-13 together with the label reports (#4, #8): builds, 417 tests pass. Gain: 1.4x at
equal nodes (15..17% fewer nodes), memory unchanged. Solver 32 still costs 2.5..3.7x solver
30, so `SOLVER_N` stays at 30. The 7,445 labels from #4 (`5x5-oracle-31-36`) are the set to
train on; #8's 2,891 were sampled from the same games with the same seed and are a prefix
of them.

## 10. Does the board split into independent regions? (`region_stats.py`)

4000 positions from the `5x5-mid` games. A region is a set of open boxes joined through
undrawn edges; regions interact only through the turn, which is what Nimstring
decomposition needs.

| undrawn | split | regions | largest region, median edges | largest <= 24 | largest <= 28 |
|---|---|---|---|---|---|
| 29 | 0.84 | 2.77 | 20 | 0.69 | 0.84 |
| 30 | 0.80 | 2.61 | 23 | 0.55 | 0.80 |
| 31 | 0.76 | 2.43 | 25 | 0.47 | 0.74 |
| 32 | 0.73 | 2.39 | 27 | 0.40 | 0.64 |
| 33 | 0.70 | 2.21 | 29 | 0.27 | 0.44 |
| 34 | 0.64 | 2.05 | 30 | 0.19 | 0.39 |
| 35 | 0.61 | 1.92 | 31 | 0.10 | 0.27 |
| 36 | 0.59 | 1.81 | 33 | 0.05 | 0.14 |

The board has usually split by the band, but mostly by peeling off small pieces: the
largest region is three to nine edges smaller than the whole board, median about five at
31..33. If the solver's cost followed the largest region, decomposition would be worth
about four to six edges of reach in the band, more than the two edges that took the band
from 0.61 to 0.69. Whether the solver already gets that on its own (it merges the moves of
independent chains) is the next measurement: nodes by largest-region size at a fixed
undrawn count (`--solver-nodes`).

A first look on local games between depth-4 alpha-beta agents suggested the solver gets
little of the split's benefit. That was wrong: those positions are fragmented and easy. The
real read, on `5x5-mid` positions (cloud agent, `experiments/2026-10-12_10-11-boxes-cpu-solver-work/`,
fresh table per position):

| undrawn | largest region | median nodes | p90 nodes |
|---|---|---|---|
| 30 | 10 to 14 | 1,000 to 2,600 | 2,000 to 5,900 |
| 30 | 20 | 4,166 | 7,312 |
| 30 | 30 (unsplit) | 20,481 | 167,777 |
| 32 | 12 to 16 | 4,000 to 7,900 | 5,600 to 23,000 |
| 32 | 20 | 10,205 | 19,000 |
| 32 | 32 (unsplit) | 88,469 | 573,373 |

At a fixed undrawn count the solver already runs five to nine times cheaper when the board
has split, so it captures much of the split's benefit by itself. The expensive positions,
the ones that set the node budget, are the unsplit or nearly unsplit ones, which
decomposition cannot help. Self-play positions (warm table across positions, solver_bench):
median 4.5k / 22k / 113k nodes at 30 / 32 / 34 undrawn, 97% / 86% / 71% settled within 200k
nodes, about five times the nodes per two edges.

Changed: the solver work aims at single-region search first (ordering, Nimstring and
controlled-value bounds, the table), and measures what decomposition could still add by
timing the largest region alone against the whole position before building it.

## What we believe now

1. The endgame (from 28 undrawn) is exact and cheap. The midgame fight (29 to about 36)
   is decided by lookahead to exact leaves; no static value reads it. The opening is where
   a learned policy and value can still beat a hand-written engine, and it has no
   measurement yet.
2. The net's contribution at 300 simulations with the solver at 28 is a few points of
   move quality from its policy. On clean positions its value head reads the 29..34 band at
   0.65 (exact labels gave five points of that), the level of depth-6 alpha-beta, and the
   searched move has sat at 0.51 since iteration 4 of the chains run: self-play at this
   budget has a ceiling in the band, and exact labels lift the value head without lifting
   play. The search budget does lift it: 0.51 / 0.55 / 0.60 at 100 / 400 / 1000
   simulations on clean positions, and proof propagation adds about a doubling's worth for
   free at 400. The solver's reach lifts it most: at 30 undrawn with 400 simulations the
   band reads 0.69, nine points above 1000 simulations at 28 for the same time. So the
   restart puts the budget where it counts (branched self-play in the band, solver at 30,
   flag on), and the next engineering is more reach: Nimstring decomposition towards
   32..34 undrawn.
3. To be stronger than the 5x5 engines: match their exact midgame (Nimstring
   decomposition), measure against them (engine bridge), then beat them in the opening.

## Experiments worth running in parallel

CPU-only work for cloud agents (solver reach, exact labels, a classical yardstick engine,
search tuning) is written up as self-contained tasks in `ASTRA_TASKS.md`. The arms below
need the GPU.

Each is one GPU box with the loop script and a different environment; all score on the same
clean positions (29..34 with the solver at 28, positions from the `5x5-solver` games) so the
arms compare. Each box needs the evaluation data once: the `5x5-solver` games directory
(the positions source and the branch source) and the labelled set `5x5-oracle-26-34`.

| arm | change from the current run | question |
|---|---|---|
| A, control | branched self-play + arena blind to the solver (the current resume command) | the new baseline |
| B | `SOLVER_N=30 SOLVER_BUDGET=100000` | is two more edges of reach worth 2-3x the self-play time? |
| C | `SP_SIMS=800` | does the search budget move the 29..34 plateau at all? |
| D | `TRAIN_ARGS="--q-mix 1.0"` | are search values better value targets than outcomes above the solver? |
| E | no `EXTRA_DATA` | how much of the value-head rise is the labelled set? |
| F | `WINDOW=16 TRAIN_EPOCHS=4` | more passes over more data, with the memorisation table watched |
| G | `FEATURES=basic` with everything else | are the chain planes still worth their cost under branched self-play? |
| H | `TRAIN_ARGS="--channels 192 --n-blocks 12"` (from iteration 0: the size lives in the checkpoint) | capacity, once the data is right |

Measure every arm after 8 iterations with:

```bash
uv run $EXP/oracle_eval.py --tag <tag> --positions-tag 5x5-solver --min-undrawn 29 --max-undrawn 34 --solver_max_undrawn 28 --value-depths --iterations 0 4 8
```
