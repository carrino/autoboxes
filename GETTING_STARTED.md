# Getting started: Dots and Boxes training on one WSL2 machine

Everything below was exercised on Ubuntu 24.04 under WSL2 with one RTX 3070 (8 GB). It takes
you from a bare WSL shell to the first training loop. `README.md` (upstream Go) and `PLAN.md`
(the Boxes plan) say what the pieces are; this file is only the how.

## 0. Before you start

- A Windows NVIDIA driver from the 580 series or newer: the default torch wheel bundles CUDA 13.
  WSL uses the Windows driver, so never install a Linux NVIDIA driver inside WSL. `nvidia-smi`
  must work in the WSL shell.
- Keep the repository and all data on the WSL filesystem (`~/...`), never under `/mnt/c`. Git
  checkouts, the virtualenv and NPZ game data are all far slower on the Windows mount.
- Roughly 10 GB free on the WSL disk: torch and its CUDA libraries are about 3 GB, the rest is data.

## 1. One-time machine setup

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake python3-dev git curl
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv: Python and dependency manager
source ~/.local/bin/env                              # or open a new shell
git config --global core.autocrlf false              # LF line endings (the repo forces them too)
nvidia-smi                                           # must list the GPU
```

`python3-dev` provides `Python.h` and `libpython` for the C++ extension; without it the build
stops inside `find_package(Python3)`.

## 2. Clone, install, build

```bash
git clone https://github.com/carrino/autoboxes.git ~/autoboxes && cd ~/autoboxes
git checkout claude/awesome-fermi-ygaeuj      # the Boxes branch, until PR #1 merges into main
uv sync                                        # creates .venv, downloads torch (~2.5 GB, once)
bash scripts/build_cpp.sh                      # C++ board + MCTS -> .venv (module alpha_go_cpp)
```

Check the two things that matter:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -c "import alpha_go_cpp; print(alpha_go_cpp.BoxesBoard(3, 3).num_edges())"
```

Expect `2.x.x+cu130 True NVIDIA GeForce RTX 3070` and `24`. Every training and collection
script asserts that CUDA is available and exits otherwise; `--cpu` exists for smoke runs only.

## 3. Verify correctness

```bash
uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
./src/alpha_go/cpp/build/go_game_test && ./src/alpha_go/cpp/build/boxes_game_test && ./src/alpha_go/cpp/build/mcts_test
```

Expect `258 passed, 24 skipped` (the skips are upstream Go tests that need tools or data not
present here) and three `All tests passed` lines from Catch2. `tests/test_gpu_lease.py` is
broken upstream and is always excluded.

For anyone changing code, these must stay clean:

```bash
uv run ruff check src/alpha_go/boxes tests/test_boxes_*.py
uv run -m mypy src/alpha_go/boxes src/alpha_go/games
```

## 4. Where data goes

```bash
echo 'export GAME_DATA_DIR=$HOME/autoboxes-data/game_data_root' >> ~/.bashrc && source ~/.bashrc
```

Game records (one NPZ per game) land under `$GAME_DATA_DIR/<save-name>/`. Checkpoints, logs,
timing and league state stay inside the experiment folder and are gitignored. The upstream
default is an NFS path that does not exist here, so set this before any self-play.

## 5. First games

```bash
# two baselines on 5x5; prints a ===RESULT=== JSON line: win rate, Wilson CI, mean margin, s/move
uv run -m alpha_go.boxes.arena --rows 5 --a boxes-ab-d4 --b boxes-greedy --num_games 4
# an untrained net searching on the GPU beats random on search alone
uv run -m alpha_go.boxes.arena --rows 3 --a boxes-mcts-untrained-3x3 --b boxes-random --num_games 10
# self-play that writes NPZ games under $GAME_DATA_DIR/dev/
uv run -m alpha_go.self_play --game boxes --board_size 3 --black boxes-greedy --white boxes-random --num_games 20 --save-name dev
```

Registered Boxes agents: `boxes-random`, `boxes-greedy`, `boxes-ab-d2`, `boxes-ab-d4`,
`boxes-ab-d6` (alpha-beta at that depth) and `boxes-mcts-untrained-3x3`. The alpha-beta agents
are pure Python: about 0.3 s/move on 3x3 and 2 s/move on 5x5 at depth 4.

## 6. The training loop

The loop lives in `experiments/2026-10-08_14-42-boxes-3x3-loop/` (its README has the details):
bootstrap games, train iter0, then [self-play with MCTS, train, arena promotion] repeated.

```bash
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
bash $EXP/run_iteration_local.sh 0 2           # 3x3, iterations 0..2, minutes per iteration
cat $EXP/report-3x3.md                         # promotion table + seconds per phase
grep -h "model:\|peak_vram" $EXP/logs/3x3/train-it*.log
```

The report's `vs greedy` and `vs ab-d4` columns are the absolute scale; `vs champion` decides
promotion. Then a short 5x5 probe to measure your iteration time before committing a night:

```bash
ROWS=5 TAG=5x5-probe BOOT_GAMES=50 SP_GAMES=16 SP_SIMS=100 TRAIN_BUDGET=60 ARENA_GAMES=8 BASE_GAMES=2 \
    bash $EXP/run_iteration_local.sh 0 0
cat $EXP/timing/5x5-probe/it1.json
```

Collection time scales with `SP_GAMES x SP_SIMS`; the overnight default (400 x 400) is 100
times this probe. Pick values that put one iteration at about 45 minutes, then:

```bash
ROWS=5 SP_GAMES=<n> SP_SIMS=<n> bash $EXP/run_iteration_local.sh 0 20    # overnight
uv run $EXP/analyze.py 5x5
```

Every budget can be set from the environment and the script prints what it uses. `TAG`
(default `<rows>x<cols>`) names every output, so probes never touch a real run. Resume by
passing the last trained iteration as `<start>`; the script refuses to start from a missing
checkpoint. Two more knobs matter on 5x5:

- `SP_PROCS=4` splits each self-play phase over four processes with their own GPU engines.
  Use it when `nvidia-smi` shows the GPU mostly idle during self-play: the game threads of
  one process share the Python interpreter lock.
- `SOLVER_N=24` turns the exact endgame solver on in self-play and the arena once 24 or
  fewer edges are undrawn (`TAG=5x5-solver` keeps that run apart from the baseline). Pick N
  from `uv run $EXP/solver_bench.py --tag 5x5-probe --undrawn 20 24 28 32 --selfplay-only`:
  the largest N whose `solved_within_20000` column is close to 1.0.
- `SEARCH_ARGS="--policy_temperature 0.7 --margin_utility_lambda 0.5"` passes extra search
  flags to self-play and the arena (`c_puct`, `leaf_batch_size`, `policy_temperature`,
  `margin_utility_lambda`, `margin_utility_k`; the same flags work on `oracle_eval.py` and
  `alpha_go.boxes.arena`). The values come from the autoresearch folder's report
  (`experiments/2026-10-09_01-05-boxes-3x3-search-autoresearch/report.md`).
- `WINDOW=8 TRAIN_EPOCHS=2` widens the replay window and caps the passes over it per
  iteration. Watch the train / held-out loss columns of `analyze.py`: when the held-out loss
  stops falling while the train loss keeps falling, the net is memorising the window.
- `TRAIN_MIN_UNDRAWN=24` trains only on positions above the solver zone, and
  `FEATURES=chains` gives the net the chain / loop structure as input planes (new run from
  iteration 0; the feature set lives in the checkpoint).
- `EXTRA_DATA="experiments/2026-10-08_14-42-boxes-3x3-loop/5x5-oracle-26-34"` adds
  solver-labelled midgame positions (`solver_label.py`, exact values and optimal edges) to
  every training set.

After any run, `uv run $EXP/oracle_eval.py --tag <tag>` scores every checkpoint against the
exact oracle on late positions from the run's own games; the raw-policy and searched-move
columns must rise with the iteration, which is the real check that training learns.

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `$'\r': command not found`, `set: - invalid option` | CRLF checkout | `git config --global core.autocrlf false`, then `git rm -r --cached -q . && git reset --hard` |
| `uv: command not found` right after installing | PATH entry not loaded | `source ~/.local/bin/env` or open a new shell |
| cmake: `Cannot find the library libpython3.12.so`, no `patchlevel.h` | no Python headers | `sudo apt-get install -y python3-dev`, rebuild |
| `torch.cuda.is_available()` is `False` | Windows driver too old for the CUDA 13 wheel | update the Windows NVIDIA driver (580+); never install one inside WSL |
| `AssertionError: CUDA not available; pass --cpu` | by design | fix CUDA; `--cpu` is for smoke runs only |
| pytest `PermissionError ... 'gnugo'` at collection | something non-executable named `gnugo` on PATH (WSL appends the Windows PATH) | fixed on this branch; `type -a gnugo` shows the culprit |
| `Unknown agent: ...` | name typo | the error lists every registered name; Boxes agents start with `boxes-` |
| CUDA out of memory while training | another program holds the GPU | the 5x5 net at batch 256 fits an 8 GB card comfortably; close the other program or lower `BATCH_SIZE` in `train.py` |
| self-play is slow and `nvidia-smi` shows low utilisation | the Python search driver is bound by the GIL across threads | known limit; lower `SP_SIMS` per iteration for now |
| `No space left on device` | WSL virtual disk full | delete old runs under `$GAME_DATA_DIR`; data never belongs on `/mnt/c` |

Memory in WSL2 is capped by `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=16GB`, for
example). Every entry point prints the byte footprint of its buffers and tables at startup so
they can be sized against that cap.

## 8. What next

`PLAN.md` §2 has the Phase 1 status and §4 the Phase 2 work (exact endgame solver, chain
features). `ARCHITECTURE.md` explains how the upstream Go code works and where the Boxes code
plugs in.
