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
