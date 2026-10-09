# boxes-3x3-loop

Single-machine AlphaZero loop for Dots and Boxes: bootstrap games -> train iter0 ->
[self-play with MCTS -> train -> arena promotion] x N. Written for 3x3 (24 edges, 9 boxes,
no ties; minutes per iteration) and run unchanged on 5x5 (60 edges, 25 boxes; overnight) by
setting `ROWS=5`. Everything runs with plain `uv run`; no cluster, SSH or /nfs. The search
uses the forced-move collapse (`BoxesSearchState`), so captures and chain take-outs are not
searched edge by edge.

Every output is tagged (`TAG`, default `<rows>x<cols>`): game data under
`$GAME_DATA_DIR/experiments/<this folder>/$TAG/`, checkpoints under `checkpoints/$TAG/`,
logs under `logs/$TAG/`, timing under `timing/$TAG/`, manifests `dataset-$TAG-itN.txt`,
league state `league_state-$TAG.json`, report `report-$TAG.md`. 3x3 and 5x5 runs never collide,
and `TAG=5x5-probe` keeps a timing probe apart from the real 5x5 run.

## Run

```bash
export GAME_DATA_DIR=$HOME/autoboxes-data/game_data_root   # WSL filesystem, never /mnt/c
EXP=experiments/<this folder>
bash $EXP/run_iteration_local.sh 0 5              # 3x3, iterations 0..5 on the GPU
ROWS=5 bash $EXP/run_iteration_local.sh 0 20      # 5x5 overnight (128ch x 10 blocks)
bash $EXP/run_iteration_local.sh 0 1 --cpu        # tiny CPU smoke run (smoke budgets)
ROWS=5 TAG=5x5-probe BOOT_GAMES=50 SP_GAMES=16 SP_SIMS=100 TRAIN_BUDGET=60 ARENA_GAMES=8 BASE_GAMES=2 \
    bash $EXP/run_iteration_local.sh 0 0          # 5x5 GPU timing probe, one iteration, kept apart by TAG
uv run $EXP/analyze.py 3x3                        # report-3x3.md from league_state-3x3.json + timing/3x3/
uv run $EXP/oracle_eval.py --tag 3x3              # every checkpoint vs the exact oracle on late positions
```

Every budget (`BOOT_GAMES SP_GAMES SP_SIMS SP_WORKERS TRAIN_BUDGET ARENA_GAMES BASE_GAMES
ARENA_SIMS`) can be overridden from the environment; the script prints the budgets it uses.
`SP_PROCS=4` splits each self-play phase over four processes (disjoint game indices, own
seeds, threads and GPU engines), which is the remedy when `nvidia-smi` shows the GPU idle
while the game threads sit on the interpreter lock. `SOLVER_N=24` turns the exact endgame
solver on in self-play and the arena at 24 undrawn edges (see `solver_bench.py`);
`SOLVER_BUDGET=50000` lets it spend more nodes per position, which is what a larger N needs.
`ARENA_SOLVER_N=0` keeps the solver out of the arena only: with it on, two nets of any quality
tie (the search reaches the solver's exact leaves whichever net steers it), so promotion
cannot see the net; with it off the nets' own midgame and endgame knowledge decides.
`MERGE_EQ=1` gives the search one action per independent chain or loop in quiet positions
(BoxesZero's equivalent edges; value-preserving, tested against the oracle). `BASELINES`
picks the arena's absolute-scale opponents (default `boxes-greedy,boxes-ab-d4`, both C++).
`SEARCH_ARGS="--policy_temperature 0.7 --margin_utility_lambda 0.5"` passes extra search
flags (`nn_agent.add_search_flags`) to self-play and the arena. `WINDOW=8` widens the replay
window (self-play iterations per training set, default 4) and `TRAIN_EPOCHS=2` caps the
passes over it per iteration: the 5x5 solver run at the defaults made about eight passes
per iteration over a window it had mostly trained on already, and its held-out loss stopped
falling at iteration 12 while the train loss kept falling (`analyze.py` shows both).

`FEATURES=chains` trains the net on the 21-plane input (the 11 basic planes plus the chain /
loop structure of `encode.py`: chain lengths, loops, opened components, safe edges and the
long-chain count and parity); the feature set is stored in the checkpoint, so self-play and
the arena pick it up on their own, and a chains run starts from iteration 0.
`EXTRA_DATA="experiments/<this folder>/5x5-oracle-26-34"` adds solver-labelled positions from
`solver_label.py` to every iteration's training set: exact final margins and uniform optimal
policies for midgame positions the solver can settle offline, a supervised foothold for the
value head where self-play outcomes alone taught it nothing.
`START_FROM=experiments/<this folder>/5x5-solver START_UNDRAWN="36 40" STOP_WHEN_SOLVED=1`
branches self-play games from stored positions and ends each one with the exact outcome once
the solver settles it: a full game spends two thirds of its search on the opening and yields
about ten positions in the band that decides the result, so a branched game delivers roughly
three times the decisive-band data per hour, with more variety. `START_FRACTION` (default
0.75) keeps a share of games opening from the empty board so the opening still gets data; the
NPZ records the branch prefix in `start_moves` and `oracle_eval.game_moves` replays it.
Solver-played moves (positions at or below `SOLVER_N` undrawn edges) record one visit per
optimal edge, so their policy target is uniform over the exact optimal set rather than a
one-hot of an arbitrary optimal edge, and their root value is the exact outcome.
Resume by passing the last trained iteration as `<start>`; the script refuses to start from
a missing checkpoint. For 5x5 read `timing/5x5/it1.json` after the first iteration and scale
`SP_GAMES` / `SP_SIMS` / `ARENA_GAMES` so one iteration fits your night.

## Pieces

- `pre_collect.py` — bootstrap games without search (greedy vs random, random vs random)
  into `<tag>/bootstrap-it0/`; iter0 trains on their played moves (label-smoothed) and outcomes.
- `run_games.py` — self-play with `BoxesMCTSAgent` from a checkpoint on both sides,
  `--collect-metrics` so visit counts become policy targets; a shared
  `PlaneBatchedEngine` batches leaves across game threads on the GPU.
- `train.py` — `BoxesDataset` -> `BoxesNet` (size picked from `MODELS` by rows), AdamW +
  cosine, all lattice symmetries as augmentation, policy CE against visit distributions + CE
  over final margins (`--q-mix 0.25` adds BoxesZero's P(win) target mixing the outcome with
  the root Q), a held-out split by game (`--val-fraction`, default 0.1) whose loss and
  accuracies sit next to the training ones in the `===RESULT===` JSON line, time budget,
  checkpoint `checkpoints/<tag>/iter{N}.pt`.
- `arena_promote.py` — candidate vs champion (alternating first player, first 4 moves
  sampled at temperature 1 so games differ, Wilson CI), promote at >= 55%
  (`--threshold`); also reports candidate vs `boxes-greedy` and
  `boxes-ab-d4`; appends to `league_state-<tag>.json`.
- `run_iteration_local.sh <start> <end> [--cpu]` — the loop; `ROWS`/`COLS` pick the board,
  `TAG` the output name, `--cpu` switches to smoke budgets so the pipeline runs anywhere.
- `analyze.py <tag>` — tabulates `league_state-<tag>.json` and `timing/<tag>/` into `report-<tag>.md`.
- `solver_bench.py --tag <tag>` — solve-time distribution (p50/p99/max ms and nodes, share
  solved within each node budget) of the exact endgame solver on positions with N undrawn
  edges taken from the run's own games; sets `SOLVER_N` for a solver arm
  (`SOLVER_N=28 TAG=5x5-solver ROWS=5 bash run_iteration_local.sh 0 20`).
- `solver_label.py --positions-tag 5x5-solver --min-undrawn 26 --max-undrawn 34 --num-positions 20000 --save-name 5x5-oracle-26-34`
  labels midgame positions from a run's games exactly (value and optimal-edge set) and writes
  them as training data for `EXTRA_DATA`.
- `oracle_eval.py --tag <tag>` — the exact check that training works: samples late positions
  (6..14 undrawn edges by default) from the run's own games, solves them with the oracle, and
  reports per checkpoint the share of oracle-optimal moves for the raw policy and for the
  searched agent, value-sign agreement and margin error, next to greedy and alpha-beta on the
  same positions; the chance level (a random legal move) is printed too. Writes
  `data/oracle_eval-<tag>.csv`. Both move rates must rise with the iteration.

## Budgets (GPU defaults)

| step       | 3x3                                              | 5x5                                   |
|------------|--------------------------------------------------|---------------------------------------|
| bootstrap  | 400 games per matchup (3 matchups)               | 600 per matchup                       |
| self-play  | 200 games/iter, 200 sims/move, 8 workers         | 400 games/iter, 400 sims/move         |
| train      | 64ch x 6 blocks, batch 256, lr 1e-3, 3 min/iter  | 128ch x 10 blocks, 5 min/iter         |
| arena      | 100 games vs champion, 40 vs each baseline       | 60 vs champion, 10 vs each baseline   |

Both profiles use leaf batch 8 per game, T=1 for the first 8 moves then 0, and fp16 autocast
for inference / bf16 for training on CUDA. `boxes-ab-d4` is pure Python and needs about
2 s/move on 5x5 (0.3 s on 3x3), which is why the 5x5 baseline matches are few; the
`vs champion` column is the promotion signal, the baseline columns are an absolute scale.
