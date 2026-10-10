# Workflow

```
# user plays an interactive Go game against a random policy
uv run -m alpha_go.play

# Run self-play between two agents
uv run -m alpha_go.self_play --black random --white random --num_games 10 --board_size 9 --save-name dev

# One iteration of the selfplay-only loop (random-init iter0 + collect + train).
EXP=experiments/2026-04-26_22-32-train-fromscratch
bash $EXP/run_iteration.sh 0 5

# Typecheck
uv run -m mypy src/

# Run all tests
uv run -m pytest tests/
```

# Libraries used:
- uv / pyproject.toml - dependency management
- pytest / pytest-asyncio - Testing framework with async support
- pytest-cov - Coverage reporting
- mypy - Static type checking (strict mode)
- ruff - Linting and formatting
- pandas - for manipulating research results and data
- rich - for interactive printing when running the training loop
- torch - deep learning. No other torch wrapper frameworks!


# Code style guidelines

- prioritize simplicity and compactness of code over generality, do not add try-except clauses
- Always be trying to reduce complexity of the codebase, minimize if statements / branching.
- Design the code to run on single GPU
- Code like how an effortlessly smart Anthropic engineer would do it
- tensors should have shape suffixes:

```
Dimension key:

B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)

def attention(input_BLD, params):
   input_BLD = layer_norm(input_BLD, params.layernorm_params)
   query_BLHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params.w_q_DHK)
   key_BMHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params.w_k_DHK)
   value_BMHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params.w_k_DHK)
   logits_BHLM = torch.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)
   B, L, H, K = query_BLHK.shape()
   logits_BHLM /= K ** 0.5
   masked_out_LM = torch.arange(L).unsqueeze(1) < torch.arange(L).unsqueeze(0)
   logits_BHLM += torch.where(masked_out_LM, -inf, 0)
   weights_BHLM = torch.softmax(logits_BHLM)
   wtd_values_BLHK = torch.einsum('BMHK,BHLM->BLHK', value_BMHK, logits_BHLM)
   out_BLD = torch.einsum('BLHK,HKD->BLD', wtd_values_BLHK, params.w_o_HKD)
   return out_BLD
```

- experimental scripts and result files should be generated in a way that is self-contained, reproducible, and easy for an AI such as Claude to interpret results.

All experiments should use `alpha_go.gameplay.play_game` or `uv run alpha_go.self_play` to evaluate and generate data. If these functions are not able to serve the experiment, suggest changes to these functions.

# Codebase structure

- `src/alpha_go` - Main package
    - `go.py` - implements the Go game logic
    - `analysis` - implement methods for analyzing experimental data
    - `agents/` - implements various agents
    - `model.py` - torch network definitions
    - `cpp/` - C++ Go board + MCTS, exposed via pybind11 as `alpha_go_cpp`
- `infra/` - cluster bringup, SSH dispatch, GPU leases
- `experiments/<datetime>-<slug>/` - Self-contained experiment folders
    - `*.py` - experiment scripts
    - `report.md` / `README.md` - findings
    - `data/*.csv` - result data files
    - `figures/*.png` - generated figures
    - `checkpoints/*.pt` - model checkpoints (if any)
- `game_data/9x9/<dataset-name>/*.npz` - training and validation datasets

If you ever need to generate a datetime string, format it in PST time zone, i.e. `TZ="America/Los_Angeles" date +"%Y-%m-%d_%H-%M"`

---

# autoboxes fork — Dots and Boxes conventions (appended; everything above is upstream)

This repository is a fork of `ericjang/autogo` that adds Dots and Boxes ("Boxes") as a
second game. Read `ARCHITECTURE.md` (how upstream works, where Go leaks in), `PLAN.md`
(phased plan, shared-file list) and `EXPERIMENTS.md` (every run so far, its numbers and what
it changed) before changing anything. `GETTING_STARTED.md` takes a bare
WSL2 shell to the first training run, with the failures seen on the way and their fixes.

## Mergeability rule (non-negotiable)

- Never delete, rename or re-flow the Go implementation. Go stays the default game.
- Boxes code goes in new files: `src/alpha_go/boxes/`, `src/alpha_go/cpp/boxes/`,
  `tests/test_boxes_*.py`, `experiments/<date>-boxes-*/`.
- Shared files (`mcts.py`, `cpp/mcts/*`, `gameplay.py`, `self_play.py`, `agents/base.py`,
  `bindings.cpp`, `CMakeLists.txt`, infra) get minimal, additive edits selected by a flag
  (`--game go|boxes`, `play_game(game=...)`). Prefix those commit messages with `shared:`.
- Upstream Go tests must keep passing after every commit.
- `upstream` remote = `https://github.com/ericjang/autogo`. Sync with
  `git fetch upstream && git merge upstream/main`.

## Environment (WSL2 on Windows, single RTX 3070 8 GB)

- Treat it as Linux. Run directly with `uv` in the WSL shell; Docker/devcontainer is
  optional, the SSH cluster path (`infra/`, `cluster.toml`) is kept but never required.
- Verify CUDA early: training/collection scripts assert `torch.cuda.is_available()` and
  exit loudly otherwise (`--cpu` for tests and tiny boards).
- All generated data stays on the WSL filesystem. Set
  `export GAME_DATA_DIR=$HOME/autoboxes-data/game_data_root` (upstream default is
  `/nfs/game_data_root`). Never default output paths to `/mnt/c`. Checkpoints go under
  `experiments/<exp>/checkpoints/` (gitignored).
- Memory is capped by `.wslconfig`: replay-buffer size, transposition-table entries and
  inference batch size are CLI flags, and every entry point prints their byte footprint
  at startup.
- Precision: fp16 autocast for GPU inference, bf16 autocast + GradScaler for training.
- Dabble and PRsBoxes are Windows programs; an optional stdio/TCP bridge (see `PLAN.md` §3)
  may wrap them as arena opponents only, never inside the training path.

## Build / test commands

```bash
uv sync                                   # Python env (first time: downloads torch)
bash scripts/build_cpp.sh                 # C++ extension -> .venv site-packages (alpha_go_cpp)
uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py   # test_gpu_lease is broken upstream
uv run -m pytest tests/test_boxes_*.py    # Boxes only
uv run -m mypy src/alpha_go/boxes src/alpha_go/game.py     # keep NEW code strict-clean (upstream has ~157 errors)
uv run ruff check src/alpha_go/boxes tests/test_boxes_*.py
./src/alpha_go/cpp/build/go_game_test && ./src/alpha_go/cpp/build/boxes_game_test && ./src/alpha_go/cpp/build/mcts_test   # Catch2

# Boxes self-play / arena (Phase 1 targets; see PLAN.md for the exact flags)
uv run -m alpha_go.self_play --game boxes --board_size 3 --black boxes-random --white boxes-greedy --num_games 100
uv run -m alpha_go.boxes.arena --rows 5 --cols 5 --a boxes-ab-d6 --b boxes-greedy --num_games 200
uv run -m alpha_go.boxes.arena --rows 5 --a ckpt:$EXP/checkpoints/5x5-solver/iter12.pt --b boxes-ab-d6 --sims 400 --solver 28 --num_games 100 --num_workers 8   # a checkpoint vs anything (or two ckpt: agents)
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop && bash $EXP/run_iteration_local.sh 0 5   # 3x3; add --cpu for a smoke run
ROWS=5 bash $EXP/run_iteration_local.sh 0 20   # 5x5 overnight; outputs tagged 5x5 (checkpoints/5x5, league_state-5x5.json)
SOLVER_N=24 SP_PROCS=4 TAG=5x5-solver ROWS=5 bash $EXP/run_iteration_local.sh 0 20   # solver arm, 4 self-play processes
SEARCH_ARGS="--policy_temperature 0.7 --margin_utility_lambda 0.5" TAG=5x5-tuned ROWS=5 bash $EXP/run_iteration_local.sh 0 20   # tuned search knobs
SEARCH_ARGS="--policy_temperature 0.7 --prove_terminals 1" TAG=5x5-proof ROWS=5 bash $EXP/run_iteration_local.sh 0 20   # MCTS-Solver: exact leaves back up by minimax
FEATURES=chains TRAIN_MIN_UNDRAWN=24 WINDOW=8 TRAIN_EPOCHS=2 SOLVER_N=28 TAG=5x5-chains ROWS=5 bash $EXP/run_iteration_local.sh 0 20   # chain planes, train above the solver zone
uv run $EXP/analyze.py 5x5                     # report-5x5.md
uv run $EXP/oracle_eval.py --tag 3x3           # checkpoints vs the exact oracle on late positions
uv run $EXP/solver_bench.py --tag 5x5 --undrawn 20 24 28 32 --selfplay-only   # picks SOLVER_N
uv run $EXP/solver_label.py --positions-tag 5x5-solver --min-undrawn 26 --max-undrawn 34 --num-positions 20000 --save-name 5x5-oracle-26-34   # exact midgame labels for EXTRA_DATA
```

`scripts/build_cpp.sh` reads the venv Python's libpython from `sysconfig`, so any
interpreter uv picks works (the upstream script hard-coded 3.10).

## Style rules for new code (in addition to upstream's list above)

- Follow the surrounding file exactly: 4-space indent, docstrings on public functions,
  `from __future__ import annotations`, dimension-suffixed tensor names
  (`planes_BKHW`, `policy_BE`, `value_BM` for margins), braces/`.cpp` layout like
  `go_game.cpp` (namespace `alpha_go`, `snake_case` methods, flat `std::vector` state).
- Pure-Python reference first, property tests, then C++ that must match it under random play.
- No try/except for control flow; minimise branching; single-GPU design.
- Tests: pytest classes grouped by topic like `tests/test_cpp_go.py`; parity tests seed
  `random`/`numpy` explicitly; tiny boards (1×1, 1×2, 2×2, 2×3, 3×3) for exhaustive checks.
- Experiment folders are self-contained and reproducible from the folder alone
  (`README.md`, scripts, `report.md`, `data/*.csv`, `figures/*.png`).

## Boxes rules summary (what the code implements)

- Board: `R × C` boxes = `(R+1) × (C+1)` dots. Default 5×5 (60 edges, 25 boxes, no ties);
  any `R, C` supported (3×3 for fast tests; 1×1, 1×2, 2×2, 2×3 for the exhaustive oracle).
- Edges: `(R+1)·C` horizontal then `R·(C+1)` vertical; action = edge index; legal move =
  any undrawn edge; no passing. Lattice coordinates `(2R+1) × (2C+1)` interleave dots,
  edges and boxes (edge `h(r,c)` at `(2r, 2c+1)`, `v(r,c)` at `(2r+1, 2c)`, box at `(2r+1, 2c+1)`).
- Completing the fourth side of a box captures it for the mover, and the mover **must move
  again** (one edge can complete two boxes → +2 and one extra move). Game ends when every
  edge is drawn. Score = boxes captured; value = margin for the side to move.
- Impartial: both players have identical moves, so there is no colour channel. State that
  matters = (edge bitmask, score margin for the side to move); 64-bit mask when `E ≤ 64`.
- Symmetries: 8 for square boards, 4 for rectangular; actions are permuted by the same
  lattice transform as the board (derived from the transform applied to an index grid).
- MCTS-Solver proof propagation (`MCTSConfig.prove_terminals`, `--prove_terminals 1`):
  terminal and solver-settled leaves are proven, a node with a proven winning child for its
  mover (or only proven children) takes the exact minimax value, proven values replace
  averages on the way up, and a proven root keeps only its optimal moves. Off by default;
  with it off the Go search is byte-identical. Tests on tiny boards against the exact oracle
  in `tests/test_boxes_mcts_sign.py` (Python) and `tests/test_boxes_cpp_mcts.py` (C++).
- MCTS backup: negate the value **only when the mover changes**. After a capture the same
  player moves again → no sign flip. This is the most likely bug; it has explicit tests in
  both the Python and the C++ search and must never be "simplified" away.
