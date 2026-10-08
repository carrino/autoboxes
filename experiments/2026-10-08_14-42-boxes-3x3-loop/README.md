# boxes-3x3-loop

Single-machine AlphaZero loop for 3x3 Dots and Boxes (24 edges, 9 boxes, no ties):
bootstrap games -> train iter0 -> [self-play with MCTS -> train -> arena promotion] x N.
Everything runs with plain `uv run`; no cluster, SSH or /nfs. Data lives under
`$GAME_DATA_DIR/experiments/<this folder>/`, checkpoints under `checkpoints/` here.

## Run

```bash
export GAME_DATA_DIR=$HOME/autoboxes-data/game_data_root   # WSL filesystem, never /mnt/c
EXP=experiments/<this folder>
bash $EXP/run_iteration_local.sh 0 5          # iterations 0..5 on the GPU
bash $EXP/run_iteration_local.sh 0 1 --cpu    # tiny CPU smoke run (see SMOKE_* in the script)
uv run $EXP/analyze.py                        # report.md from league_state.json + timing/
```

## Pieces

- `pre_collect.py` — bootstrap games without search (greedy vs random, random vs random)
  into `bootstrap-it0/`; iter0 trains on their played moves (label-smoothed) and outcomes.
- `run_games.py` — self-play with `BoxesMCTSAgent` from a checkpoint on both sides,
  `--collect-metrics` so visit counts become policy targets; a shared
  `PlaneBatchedEngine` batches leaves across game threads on the GPU.
- `train.py` — `BoxesDataset` -> `BoxesNet`, AdamW + cosine, all 8 lattice symmetries as
  augmentation, policy CE against visit distributions + CE over final margins, time budget,
  `===RESULT===` JSON line, checkpoint `checkpoints/iter{N}.pt`.
- `arena_promote.py` — candidate vs champion (alternating first player, Wilson CI),
  promote at >= 55% (`--threshold`); also reports candidate vs `boxes-greedy` and
  `boxes-ab-d4`; appends to `league_state.json`.
- `run_iteration_local.sh <start> <end> [--cpu]` — the loop; `--cpu` switches to the
  SMOKE_* budgets so the pipeline can be exercised anywhere.
- `analyze.py` — tabulates `league_state.json` and `timing/` into `report.md`.

## Budgets (GPU defaults)

| step            | setting                                              |
|-----------------|------------------------------------------------------|
| bootstrap       | 400 greedy-vs-random + 400 random-vs-random games     |
| self-play       | 200 games/iter, 200 sims/move, leaf batch 8, T=1 for the first 8 moves then 0 |
| train           | 64ch x 6 blocks, batch 256, lr 1e-3, 3 min/iter      |
| arena           | 100 games vs champion, 40 vs each baseline            |
