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
flags (`nn_agent.add_search_flags`) to self-play and the arena; `--prove_terminals 1` among
them is MCTS-Solver proof propagation: terminal and solver-settled leaves are proven, their
exact values back up by minimax instead of averaging, and a proven root keeps only its
optimal moves (so the policy targets there are exact). `WINDOW=8` widens the replay
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
- `arena_promote.py` — candidate vs champion in colour-swapped pairs from the same random
  4-edge opening (`--opening_moves`), each pair scored on its summed margin so the
  side-to-move advantage cancels; promote when the pair score is >= 55% (`--threshold`)
  and its 95% interval excludes a tie (60 games with a plain 0.55 threshold promoted an
  equal net one time in four); also reports candidate vs `boxes-greedy` and
  `boxes-ab-d4` with the per-colour win rates; appends to `league_state-<tag>.json`.
  Self-play uses the latest checkpoint unless `GATE=1`, which makes it use the arena champion
  (AlphaGo Zero's gating) while training still continues from the latest: a candidate that
  loses to the champion then stops generating the data.
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

## Exact-value A/B probe (after PR #10)

From the repository root, with CUDA available and the stalled run's local checkpoints,
manifests and games present:

```bash
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
uv run "$EXP/exact_ab.py" --source-tag 5x5-s1600 --source-iteration 20 \
    --reach 38 --workers 4 --epochs 4 --arena-games 100
```

Pause concurrent self-play, training and labelling before running this command. Timed
classic opponents lose strength under CPU contention. Keep the WSL terminal running
(or run in an existing tmux session). This is a screening experiment; runtime depends
mostly on CPU solving and the three arena matches, and is not capped at two hours.

The script reads `checkpoints/5x5-s1600/iter20.pt` and the actual
`dataset-5x5-s1600-it20.txt` manifest, including any extra datasets that were in that
training window. Use `--manifest /absolute/path/to/manifest.txt` if it was archived elsewhere;
it deliberately fails rather than inventing a different window. Evaluation games must
exist under `$GAME_DATA_DIR/experiments/<this-folder>/5x5-mid/`.

If `relabel_exact.py` already processed some or all of the replay window, wait for any
active labelling to finish, then reuse those labels:

```bash
uv run "$EXP/exact_ab.py" --reuse-labels --workers 4 \
    --bands 33:36 37:38 --positions 100 --arena-games 100
```

`--reuse-labels` and `--reuse-exact` are equivalent. This still snapshots both arms and
strips exact arrays only from the control. It reuses existing arrays and solves only games
without an `exact_margin_p1` array, in the copied exact arm. Before solving, it prints
the reuse/solve counts for the full manifest and lists every directory needing work.
Progress includes completed reuse/solve counts and a 30-second heartbeat. An entirely
labelled window needs no relabelling solver; malformed arrays still fail before training.
Existing unknown labels stay unknown and use the outcome fallback. Labels above `--reach`
are ignored. Stored arrays do not record the attempted solver reach or budget, so inspect
coverage: `--reach 38` does not extend a previously shallower relabelling run. Clean
evaluation positions still require solving. This smaller read uses 100 roots in each band;
the three arena matches remain 100 games each. `--arena-games 0` skips them for a value-only
probe. This option starts a fresh A/B run; it does not resume an interrupted run.

New labels are saved in the run's exact copy; original replay files stay untouched.
To reuse that work in another run, including labels saved before an interruption, pass
`--manifest "$EXP/logs/<previous-tag>/exact-manifest.txt" --reuse-labels`. Both copy
manifests are written before solving begins. Use a fresh tag and the same reach; existing
unknown entries are not retried or extended to a larger reach.

It performs these steps sequentially:

1. Copy the checkpoint and every NPZ in the manifest into a fresh timestamped run. Record
   source hashes and settings; leave the original run and league untouched. Allow disk
   space for two copies of the replay data. Existing tags are never overwritten.
2. Keep original outcome targets in the control copy; relabel the exact copy from 38
   undrawn edges down with a 100M-node per-position budget. Print progress and report
   known/unknown labels at every undrawn count. Existing solver-labelled collections
   stay identical in both arms and are never mistaken for sequential games.
3. Train both arms from the same checkpoint for four complete epochs, using the same
   seed, sample ordering, augmentations, split and batch size. Time limits are disabled
   so both complete the same number of optimizer steps. `q-mix` stays zero. The checkpoint's
   complete architecture is preserved on resume.
4. Compare the original, control and exact checkpoints on identical clean positions in
   31..34 and 35..38. Solve roots only, avoiding the cost of enumerating optimal policies.
   Report value sign, margin MAE, Brier score and label coverage; all models see the same
   solved subset. These are value diagnostics, not searched-move optimality measurements.
5. Play all three checkpoints against `boxes-classic` from the same colour-swapped random
   openings, at 400 simulations, solver 30 / 50k nodes, proof propagation and shaped
   terminals with lambda 0.5. This evaluation budget is fixed even when the source was
   trained with 1600 simulations. Each model gets 100 games with the command above
   (the script's default is 200). One arena worker is the default.

The script prints its run tag and log directory. Read `logs/<tag>/summary.json` first;
`completed: true` indicates every requested stage finished. `settings.json` records source
hashes and the code revision. `train-*.log`, `value-*.csv`, `arena-*.log` and `arena-*.jsonl`
contain the detailed evidence. Candidate checkpoints are `checkpoints/<tag>-control/iter1.pt`
and `checkpoints/<tag>-exact/iter1.pt`. It never promotes either candidate automatically.
With insufficient VRAM, choose a fresh tag and `--batch-size 128` for both arms.

Interpret the result by comparing exact against both control and the unchanged reference.
A value gain should lower clean-band error and improve calibration in 35..38; a replay
validation gain alone is insufficient. A 100-game match is a screen, not proof of a small
strength improvement. Confirm a promising recipe using a fresh pre-sized opening suite
and another training seed before launching a long run. Root-label budget failures must
remain visible because skipping the hard cases can flatter results.

Training now uses stable, symmetry-canonical opening-family hashes rather than reshuffling
validation games each iteration. Keep `--split-seed` and `--val-fraction` fixed. Branches
after the first eight moves share their parent's assignment; an explicit NPZ `split_group`
can describe other lineage. Hash splits need not yield exactly the requested fraction.
This does **not** remove historical leakage from warm-start weights, all transpositions,
or overlapping solver-labelled collections: replay validation remains diagnostic, and
`5x5-mid` must stay out of all training manifests. A fresh final test set is still needed.

Promotion now uses game score >= 0.55 and a two-sided paired label-swap p-value < 0.05,
swapping complete opening pairs. Box-margin pair scores and legacy Wilson columns remain
for diagnosis and comparison; the latter are not used to promote. League entries record
`promotion_metric=game_score_paired_swap`. The p-value is for a fixed match, not repeated
peeking, and does not correct for selecting among many candidates. The existing 60-game
loop arena is still a small screen; use a larger, fresh confirmation match for modest gains.
