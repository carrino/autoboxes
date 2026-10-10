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

## 5. Chain / loop planes (`TAG=5x5-chains`, from scratch, 9+ iterations)

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
  above the depth-6 hand-written value, is the one real trend, and it has not moved the
  searched move at 100 simulations. So the 60-0 says iteration 13 was a weak champion, let
  in by a 60-game arena with a 0.55 threshold (which promotes an equal net one time in
  four). Self-play always used the latest checkpoint, so the weak champion never
  generated data.

Changed: the arena plays colour-swapped pairs from the same random opening and scores each
pair on its summed margin, so colour cancels inside the pair; promotion needs the pair
score's 95% interval to exclude a tie; the per-colour win rates are reported.

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

## What we believe now

1. The endgame (from 28 undrawn) is exact and cheap. The midgame fight (29 to about 36)
   is decided by lookahead to exact leaves; no static value reads it. The opening is where
   a learned policy and value can still beat a hand-written engine, and it has no
   measurement yet.
2. The net's contribution at 300 simulations with the solver at 28 is a few points of
   move quality from its policy. Its value head now reads the 29..34 band at 0.70 (exact
   labels did that), above any hand-written static value, yet the searched move has not
   moved: the next tests are whether minimax backup of proven leaves and more simulations
   convert the better value into moves. Exact labels and branched self-play are the levers
   on the learning side, the solver's reach on the engine side.
3. To be stronger than the 5x5 engines: match their exact midgame (Nimstring
   decomposition), measure against them (engine bridge), then beat them in the opening.

## Experiments worth running in parallel

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
