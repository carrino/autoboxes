# ARCHITECTURE.md — how upstream `autogo` is built, and where Go leaks in

Phase 0 reconnaissance for the `autoboxes` fork. Written against upstream
`ericjang/autogo` at commit `54ac8d4` ("bugfixes"); the fork's `main` is
byte-identical to `upstream/main` (`git diff main upstream/main` is empty).

Sections 1–4 describe upstream as it is. Section 5 lists every place a Go
assumption leaks into code that reads as generic. Section 6 proposes the
minimal seam for a `Game` interface. Section 7 lists pre-existing defects
found while reading.

---

## 1. Component map

```
src/alpha_go/
  go.py                 Python Go rules (FastGoBoard) + MCTS adapter (GoState)
  cpp/go/go_game.*      C++ Go rules (GoBoard) — the production rules engine
  cpp/mcts/mcts.*       C++ MCTS over GoBoard — the production search
  cpp/bindings/         pybind11 module `alpha_go_cpp` (GoBoard, MCTSConfig, MCTSTree, run_mcts)
  mcts.py               Python MCTS over a GameState Protocol (tests + NNMCTSAgent only)
  model.py              torch nets: GoTransformer, MuPGoResNet, SizeInvariantGoResNet
  agents/base.py        Agent ABC + name registry (PASS=(-1,-1), RESIGN=(-2,-2))
  agents/random.py      random legal move
  agents/nn_agent.py    raw policy sampling (local / gRPC / shared-engine)
  agents/nn_mcts.py     evaluators (Local, LeafBatched, BatchedLocal, RPC) + CppMCTSAgent + NNMCTSAgent
  inference/batched_engine.py  LocalBatchedInferenceEngine (cross-game GPU batching, padding+mask)
  gameplay.py           play_game() loop, GameRecord/MoveMetric, save_game_data() NPZ writer
  self_play.py          CLI: N games between two registered agents, threads, stats, NPZ output
  dataset.py            GoDataset: NPZ dirs -> (board, move, winner, mcts_policy, is_teacher)
  engine.py             GTP wrapper for GNU Go (web UI + tests only)
  play.py               FastAPI web UI (human vs agent, NPZ replay)
  analysis/plotting.py  board rendering, loss/FLOPs plots
  proto/                gRPC inference service schema (+ generated stubs)
infra/                  SSH + docker-run dispatcher, GPU leases, cluster bringup
experiments/<date>-<slug>/  train.py, run_games.py, collect_driver.py, update_league.py, run_iteration.sh
tests/                  pytest; C++ Catch2 tests under cpp/ (go_game_test wired, mcts_test NOT wired)
```

### 1.1 Rules

Two implementations of the same rules, kept in parity by `tests/test_cpp_go.py`:

* `src/alpha_go/go.py::FastGoBoard` — numpy int8 board, values `EMPTY=0, BLACK=1, WHITE=2`,
  flood-fill liberties, simple ko only, all suicide illegal, area scoring. **Hard-codes komi 6.5**
  (`score()`, `go.py:289`) while the C++ board defaults to 7.5. No superko.
  `GoState` wraps it to satisfy the Python MCTS `GameState` Protocol:
  `get_legal_actions() -> list[(r,c)|None]`, `apply_action`, `is_terminal` (two passes),
  `get_reward(player) -> {0, 0.5, 1}`, `current_player() -> 0|1`, `clone`.
* `src/alpha_go/cpp/go/go_game.h/.cpp::GoBoard` — flat `std::vector<int8_t>`, precomputed
  neighbour tables, Zobrist hash + positional-superko `seen_hashes_`, simple ko point,
  all suicide illegal (KataGo-aligned), Tromp-Taylor area scoring with komi,
  `is_game_over() == consecutive_passes_ >= 2`. Actions are flat ints `row*size+col`;
  `pass()` is a separate method. `set_from_array()` wipes superko history.
  Public surface used by the rest of the code: `play/play_flat/pass/is_legal/is_legal_flat/
  get_legal_moves_flat/is_game_over/score/get_winner/size/to_play/move_count/komi/at/row_col/
  flat_index/set_from_array`, plus `to_numpy/set_from_numpy/copy` in the binding.

### 1.2 MCTS

* **C++ (production)** `cpp/mcts/mcts.h/.cpp::MCTSTree`. Nodes live in a flat
  `std::vector<MCTSNode>` (index links, never pointers — the vector reallocates).
  Each node stores `N`, `N_virt`, `Q`, `first_eval_value`, `player_at_parent`, `depth`,
  `children: unordered_map<int action, int node_idx>`, `logP_A: unordered_map<int, float>`,
  and a full `GoBoard state` copy.
  * `Q` is a **win probability in [0,1] from the perspective of `player_at_parent`**
    (the player who moved into the node). The root's `player_at_parent` is set to the
    opponent of the side to move, so `get_root_q_value()` is the *opponent's* win prob and
    callers flip it (`gameplay.py:280`, `nn_mcts.py:946`).
  * Evaluator contract: `EvaluatorFn(const GoBoard&) -> (unordered_map<int action, float prob>, float v)`
    with `v` the side-to-move win probability. The policy dict defines the legal action set
    for the node (MCTS never calls `get_legal_moves_flat` itself).
  * `run_simulations(n, eval)`: recursive `perform_playout` — select by PUCT
    (`Q + c_puct * P * sqrt(ΣN+1)/(1+N)`), expand, evaluate, back up with
    `Q += (U - Q)/N`. Optional AlphaGo-style rollout mixing (`lambda_`, `max_depth`).
  * `run_simulations_batched(n, leaf_batch_size, batched_eval)`: leaf-parallel search with
    virtual loss (`N_virt` added to visit counts during selection, `q_eff = q*N/(N+N_virt)`),
    collects up to `leaf_batch_size` leaves, one batched evaluator call, then backs up each
    path. Terminal leaves bypass the evaluator.
  * Playout-cap randomisation (`pcr_sims/pcr_probs`), Dirichlet root noise, temperature
    action selection `N^(1/τ)`.
  * Accessors used by data collection: `get_child_visit_counts`, `get_child_q_values`,
    `get_root_policy_priors`, `get_child_first_eval_values`, `get_child_max_subtree_depths`.
* **Python** `mcts.py` — same algorithm in dataclasses, generic over a `GameState[Action]`
  Protocol. Only used by `NNMCTSAgent` and `tests/test_mcts.py` (toy games). Useful as the
  reference implementation for new games because it is game-agnostic *except for one line*
  (see §5.1).

### 1.3 Network

All nets take `board_BHW` with values `{0: empty, 1: side-to-move, 2: opponent}` (callers
swap colours before inference), one-hot it into 3 planes, and return
`(policy_BC, value_B)` with `C = H*W + 1` (last logit = pass) and `value_B` a single
win-probability logit trained with BCE.

* `GoTransformer` — legacy, tests only.
* `MuPGoResNet` + `create_mup_model(config="3M"|"18M")` — muP ResNet with `set_base_shapes`;
  fixed board size (policy FC over `policy_channels*H*W`). Used by older checkpoints/agents.
* `SizeInvariantGoResNet` — **the production model** (`train.py` default: 128ch × 10 blocks,
  value_hidden 64, ~3M params). Fully convolutional: `MaskedBatchNorm2d`/`MaskedGroupNorm2d`,
  `MaskedSEBlock`, `MaskedResBlock`; policy = 1×1 conv → per-cell logit (+ pass logit from
  masked-avg-pooled features); value = pooled features → FC → FC. Accepts a `mask_BHW` so a
  9×9 board can be padded inside a 19×19 canvas. `compute_dense_loss(board, mask, π*, winner,
  is_teacher)` = per-sample CE against MCTS visit distribution + BCE.
  The masked building blocks are reusable for a non-Go lattice unchanged.

### 1.4 Agents and evaluators

`Agent.select_move(board: alpha_go_cpp.GoBoard, seed) -> (row, col) | PASS | RESIGN`, plus
`start_game/notify_move/end_game` hooks. `@register_agent("name")` populates a global
registry; `get_agent(name)` instantiates with no args, so every registered agent hard-codes
its checkpoint path (all under `/nfs/checkpoints/...`, see §7).

* `LocalNNEvaluator` — one board per forward.
* `LeafBatchedNNEvaluator` — `batch_evaluate(list[GoBoard])`, bf16 autocast, auto-detects
  `SizeInvariantGoResNet` vs `MuPGoResNet` from the checkpoint. **This is what `run_games.py`
  uses** (one process, N game threads, each thread's MCTS batches its own leaves).
* `BatchedLocalNNEvaluator` — submits to a shared `LocalBatchedInferenceEngine` so many game
  threads share one GPU batch (`self_play.py --batched-inference`).
* `RPCEvaluator` — gRPC to `alpha_go.inference_server` (module not in the repo).
* `CppMCTSAgent(evaluator, num_simulations, c_puct, temperature, add_noise, lambda_,
  max_depth, resign_threshold, resign_consec_turns, min_turns_before_resign, pcr_*,
  leaf_batch_size)` — builds `alpha_go_cpp.MCTSTree` per move; resigns when
  `1 - root.Q < resign_threshold` for `resign_consec_turns` consecutive turns; stores
  `last_search_result` for metric collection.

### 1.5 Inference engine

`LocalBatchedInferenceEngine(model, device, batch_size, batch_timeout_ms, board_size)`:
one `queue.Queue` of `(board_np, native_size, Future)`, a worker thread that drains up to
`batch_size` requests within `batch_timeout_ms`, zero-pads every board to
`(max, max)` with a 0/1 mask, runs the model once, slices each row's logits back to
`native*native+1`, resolves the futures with `(policy_logits, sigmoid(value), entropy)`.
Requires square `(H, W)` boards with `{0,1,2}` cell values (`submit`, line 126).

### 1.6 Self-play and data

`gameplay.play_game(black_agent, white_agent, board_size, max_moves, seed, collect_boards,
collect_metrics, komi, black_is_teacher, white_is_teacher) -> GameRecord`:

1. `board = alpha_go_cpp.GoBoard(board_size, komi)`; `agents = {BLACK: ..., WHITE: ...}`.
2. Loop until `board.is_game_over()` or `max_moves`: snapshot `board.to_numpy()`, ask
   `agents[board.to_play()]`, optionally harvest MCTS stats from `agent.last_search_result`
   into a `MoveMetric` (dense arrays of length `size*size+1`, pass at the end), apply
   `PASS`/move/`RESIGN`, call `notify_move` on both agents.
3. Score: `board.score()` sign → winner, `result` string `"B+7.5"`/`"W+3.0"`/`"Draw"`,
   `termination ∈ {double_pass, max_moves, resign}`.

`save_game_data()` writes one `.npz` per game:
`boards (n,H,W) int8, moves (n,2) int16, winner {0,1,2}, result, board_size, black_agent,
white_agent, *_checkpoint_path, num_moves, komi, termination, code_version`, and when MCTS
stats exist: `mcts_visits (n,A) int16, mcts_q_values, mcts_policy_priors, mcts_temperatures,
mcts_root_values, is_teacher, mcts_first_eval_values, mcts_max_subtree_depths`.
**There is no per-position side-to-move field**; consumers infer it from position parity.

`self_play.py` CLI: `--black/--white <agent name>`, `--num_games`, `--board_size {9,13,19}`,
`--max-moves` (default `2*size²`), `--seed`, `--komi 7.5`, `--save-name` (dir under
`$GAME_DATA_DIR`, default `/nfs/game_data_root`), `--num_workers` (threads; agents cached
per thread), `--batched-inference/--batch-size/--batch-timeout-ms`, `--collect-metrics`,
`--black-is-teacher/--white-is-teacher`, `--profile-memory`. Prints a rich table + JSON.

`GoDataset(dirs, load_mcts_policy, load_is_teacher, in_memory)`: builds `index.json`
(file → num_moves) per dir, global cumulative index, `__getitem__` returns
`board` (colours swapped so side-to-move = 1), `move`, `winner` (1 iff side-to-move won),
`mcts_policy` = `visits^(1/τ)` normalised (label-smoothed one-hot fallback), `is_teacher`.
Side to move = `local_idx % 2` (`dataset.py:170`).

### 1.7 Training loop, arena, promotion (all in `experiments/`, nothing in `src/`)

Reference: `experiments/2026-04-26_22-32-train-fromscratch/`.

* `pre_collect_random.py` — 5000 random-vs-random games → `random-it0/` (bootstrap).
* `train.py` — one iteration: load every NPZ in `dataset-it{N}.txt` into RAM, `SizeInvariantGoResNet`,
  AdamW lr 1e-3 wd 5e-3, cosine w/ warmup 200, batch 128, bf16 autocast + GradScaler,
  grad-clip 1.0, random D4 augmentation of board and spatial policy (pass logit kept aside),
  policy CE masked by `is_teacher` (**overridden to `winner` — see §7**), value BCE,
  time budget 15 min or train policy acc ≥ 95%, saves `/nfs/checkpoints/<EXP>/iter{N}_best.pt`,
  prints a `===RESULT===` JSON line.
* `run_games.py` — registers two `CppMCTSAgent`s (`LeafBatchedNNEvaluator`, 1024 sims,
  PCR 95/5 1024/2048, c_puct 5.0, T 0.3, resign 0.05×5, leaf batch 8, no noise) and calls
  `self_play.main()` with `--collect-metrics` + both teacher flags.
* `collect_driver.py` — the **arena/gauntlet**: iter N plays the last K=3 iterations on both
  colours (`as-black-it{N}/vs-it{M}`, `as-white-it{N}/vs-it{M}`) plus `selfplay-it{N}`,
  50 games each, sharded into ~12 jobs and dispatched with `infra.remote_exec.run_pool`.
* `update_league.py` — **promotion rule**: aggregate iter N's win rate per colour across the
  gauntlet; it becomes `best_black`/`best_white` in `league_state.json` only if its rate
  strictly exceeds the recorded rate of the current champion.
* `run_iteration.sh <start> <end>` — bootstrap → loop {collect, league, build
  `dataset-it{N+1}.txt` (carry forward + append this iter's dirs), train iter N+1}.
  Training is dispatched remotely too (`python -m infra.remote_exec --role train`).

Lessons recorded upstream (fastlearn report, tutorial): policy must train on *every* MCTS
position not just the winner's (DAgger-style target π*), Dirichlet noise helps one step but
compounds badly across iterations, game variety beats per-game sim count, leaf batching +
cross-game batching gave ~19× sims/s over Python MCTS, co-training 19×19+9×9 beat 9×9-only.

### 1.8 Config

There is no config file or config object. Hyperparameters are module constants in experiment
scripts; agents are registered classes with baked-in parameters; `self_play.py` is argparse;
`self_play._AGENT_MODEL_CONFIGS` maps agent names to model configs for `--batched-inference`;
`cluster.toml` describes workers. A `game = "go" | "boxes"` flag has to be introduced
(§6) — the natural places are a `--game` CLI flag on `self_play.py` and a `game` field on
`GameRecord`/NPZ.

### 1.9 Infra

`infra/remote_exec.py` — `Worker`, `Job(push_files, pull_dirs, gpu-type affinity)`,
`build_ssh_argv` (ssh → `docker run --rm --gpus all -v /data/eric:/nfs ...`), `run_one`
(retries, rsync push/pull, periodic partial pulls), `run_pool` (per-GPU worker threads,
FIFO with affinity, dead-host rescheduling, cluster.toml hot reload, NFS flock leases via
`infra/gpu_lease.py`). `infra/cluster.py` — `add/ping/build/pull/status`. Hard-coded paths:
`/workspace/cluster.toml`, `/nfs`, `/data/eric`, `~/.ssh/id_ed25519`, image
`ghcr.io/ericjang/alphago-worker`. None of this is needed on one machine; `run_games.py`
and `train.py` run fine with plain `uv run` once `GAME_DATA_DIR` and the checkpoint dir are
redirected.

### 1.10 Tests (baseline)

| file | what | status on a fresh checkout |
|---|---|---|
| `tests/test_go.py` | GNU Go GTP wrapper | skipped unless `gnugo` installed |
| `tests/test_cpp_go.py` | C++/Python Go parity, captures, suicide, numpy I/O | runs (needs `alpha_go_cpp`) |
| `tests/test_cpp_mcts.py` | C++ MCTS basics; NN parity classes | basics run; NN classes skipped (`alpha_go.inference_server` missing, checkpoint missing) |
| `tests/test_cpp_mcts_batched.py` | leaf-parallel vs sequential visit distribution (TV distance) | runs |
| `tests/test_mcts.py` | Python MCTS on toy games incl. perspective tests | runs |
| `tests/test_mcts_data.py` | NPZ round trip, visit→policy parity, GoDataset | runs |
| `tests/test_model.py` | GoTransformer / MuPGoResNet shapes, loss, grads | runs (dataset tests skipped) |
| `tests/test_gpu_lease.py` | flock leases | **ImportError** — imports `_read_priorities` which does not exist in `infra/remote_exec.py` |
| `cpp/go/go_game_test.cpp` | Catch2 rules tests | built by CMake (`go_game_test`), not run by pytest |
| `cpp/mcts/mcts_test.cpp` | Catch2 MCTS tests | **not in CMakeLists.txt** |

---

## 2. Data flow of one self-play iteration

```
checkpoint iterN.pt ─┐
                     ▼
run_games.py ──► self_play.main ──► ThreadPool(num_workers) ──► play_game()
                                                                  │ per move:
                                                                  │  CppMCTSAgent.select_move
                                                                  │    MCTSTree.run_simulations_batched
                                                                  │      ◄── LeafBatchedNNEvaluator.batch_evaluate (GPU, bf16)
                                                                  │  MoveMetric(visits, Q, priors, root_value, temperature)
                                                                  ▼
                                               save_game_data → $GAME_DATA_DIR/<save-name>/*.npz
                                                                  ▼
train.py: GoDataset(load_mcts_policy, load_is_teacher, in_memory) → D4 augment → CE(π*)·is_teacher + BCE(z)
                                                                  ▼
                                               /nfs/checkpoints/<EXP>/iter{N+1}_best.pt
                                                                  ▼
collect_driver (gauntlet vs last K) → update_league (promotion) → next iteration
```

---

## 3. Perspective and sign conventions (the part Boxes must get right)

* Evaluator `v` = win prob for **side to move** at the evaluated state.
* Node `Q`/`U` = win prob for **`player_at_parent`** = the player who *made the move into*
  the node. At a leaf, `U = v` if `to_play == player_at_parent` else `1 - v`
  (`mcts.cpp:112-115`, `mcts.cpp:505-506`, `mcts.py:284-287`, `:305-308`). This part is already
  mover-aware.
* On the way back up, every level does `U = 1 - U` unconditionally
  (`mcts.cpp:171`, `mcts.cpp:520`, `mcts.py:360`). That is only correct when the mover
  changes at every ply. **For Boxes this must become
  `U = (child.player_at_parent == node.player_at_parent) ? U : 1 - U`.** For Go the two
  are identical (passes also switch the player), so upstream behaviour and tests are
  unchanged by the fix.
* Terminal value: `get_winner()` → `1.0 / 0.5 / 0.0` for `player_at_parent`
  (`mcts.cpp:88-99`, `mcts.py:276`).
* Root: `player_at_parent = opponent of to_play` (`mcts.cpp:16`, `mcts.py:451`), hence the
  `1 - root.Q` flips in `gameplay.py:280` and `nn_mcts.py:946`.
* Training label: `winner == side_to_move` with side-to-move from **position parity**.

---

## 4. Where the board size and action layout are assumed

Action index = `row * size + col`, pass = `size*size` (dense arrays) or `-1`
(`alpha_go_cpp.PASS_ACTION`, sparse dicts). Every consumer re-derives `n_actions =
size*size + 1`. Square boards everywhere (`LocalBatchedInferenceEngine.submit` rejects
non-square input; `SizeInvariantGoResNet` pads to `(max, max)`; `self_play --board_size`
is restricted to `{9, 13, 19}`; `GoDataset` optionally appends `"{n}x{n}"` to paths).

---

## 5. Go-specific assumptions that leak into "generic" code

Grouped by theme; `file:line` refers to upstream `54ac8d4`.

### 5.1 Player alternation (highest risk for Boxes)
| where | assumption |
|---|---|
| `mcts.py:360` `U = 1.0 - child_value` | mover changes every ply |
| `mcts.py:451` root `player_at_parent = 1 - current_player()` | two players, alternate |
| `cpp/mcts/mcts.cpp:171`, `:520` | same unconditional flip, sequential + batched paths |
| `cpp/mcts/mcts.cpp:16,112,159,454,505` `to_play()==BLACK ? 0 : 1` | player index derived from stone colour |
| `gameplay.py:217` `agents = {BLACK: black, WHITE: white}`; `:230` | agent chosen by colour; colour ≡ player |
| `dataset.py:170` `is_white = local_idx % 2 == 1` | side to move = parity of move index (no `to_play` stored in NPZ) |
| `dataset.py:177` colour swap so side-to-move = 1 | colours exist and are asymmetric |
| `tests/test_mcts.py` toy games | all alternate |

### 5.2 Pass moves and game end
| where | assumption |
|---|---|
| `agents/base.py:9` `PASS=(-1,-1)`, `:14` `get_pass_index = size²` | pass exists, last action |
| `cpp/mcts/mcts.h:16` `PASS_ACTION=-1`; `mcts.cpp:152,245,447` `new_state.pass()` | pass is a distinct branch of `apply` |
| `cpp/mcts/mcts.cpp:187,240` rollout falls back to pass when policy empty | pass always legal |
| `model.py:125,333,589,645` `n_actions = H*W + 1`, `pass_head/pass_fc` | one extra logit |
| every evaluator (`nn_mcts.py:150,259,356,438,544`, `nn_agent.py`) adds pass prob | |
| `gameplay.py:305-313` `PASS` → `board.pass_move()`; `fallback_to_pass` | |
| `gameplay.py:353` `termination = "double_pass" if is_game_over else "max_moves"` | end = two passes; `max_moves` cap matters |
| `self_play.py:613` `max_moves = 2*size²` | |
| `dataset.py:206-210`, `model.py:227-232` pass target index | |
| `train.py:104-116` augmentation keeps `pass_B` aside, rotates the `N×N` rest | |
| `inference/batched_engine.py:249-256` `pass_idx_padded` slicing | |

### 5.3 Scoring, komi, value semantics
| where | assumption |
|---|---|
| `go.py:289` komi 6.5 hard-coded; `go_game.h` `KOMI=7.5`; `self_play.py --komi 7.5`; `GameRecord.komi`; NPZ `komi` | komi exists |
| `gameplay.py:294-302, 354-361`; `self_play.parse_score_from_result` | result string `"B+x"/"W+x"/"Draw"`, winner from sign of `score()` |
| `GoState.get_reward`, `mcts.cpp:88-99,262-268` | outcome ∈ {0, ½, 1}; value head = BCE win prob only |
| `nn_mcts.py:944-951` resign on `1 - root.Q` | |
| `update_league.py` winner==1 counting | winner codes 1/2 |

### 5.4 Board geometry and action encoding
| where | assumption |
|---|---|
| `agents/base.py:39` `select_move(board: alpha_go_cpp.GoBoard, seed) -> (row, col)` | moves are 2-D coords on a square grid |
| `GoBoard.row_col/flat_index`, `MCTSTree` action ints | `action = r*size + c` |
| `model.py:372,628-630` one-hot(3) of `{0,1,2}`; `Conv2d(3, ...)` | 3 cell states, no scalar inputs |
| `inference/batched_engine.py:126-134,225-232` | square `(H,W)` arrays of `{0,1,2}` |
| `gameplay.py:253` `n_actions = bs*bs+1` for `MoveMetric` arrays; `:417` | |
| `self_play.py:524` `choices=[9,13,19]` | |
| `dataset.py:58` `"{n}x{n}"` dirs; `:194` `bs*bs+1` | |
| `train.py:100-116` D4 symmetry on `(B,N,N)` policy map | square board, cell-wise policy |
| `gameplay.render_illegal_move_debug`, `analysis/plotting` | Go notation, star points |

### 5.5 Colour normalisation
Every evaluator and `GoDataset` swap colours so the side to move is `1`
(`nn_mcts.py:116,237,325,407`, `nn_agent.py:163,219,312`, `dataset.py:177`). Boxes is
impartial — no colour channel — but needs a *score-margin* input instead.

### 5.6 History-dependent state
`GoBoard` carries `seen_hashes_` (superko) and `ko_point_`, so a position is not a pure
function of the array; `set_from_array` resets history. Boxes state is fully described by
`(edge mask, margin for side to move)`.

### 5.7 Registry and paths
Registered agents hard-code `/nfs/checkpoints/...` and `board_size=9`
(`nn_mcts.py:974-1030`); `self_play._AGENT_MODEL_CONFIGS` and the `"cpp-mcts-"` prefix
test (`self_play.py:633`) gate `--batched-inference`.

---

## 6. The minimal seam for a `Game` interface

Goal: Go stays as-is (no renames, same `alpha_go_cpp` names, same NPZ output), Boxes is
added in new files, and shared files change by a few lines each.

### 6.1 C++ — a `State` concept and a templated `MCTSTree`

`MCTSTree` only needs five things from a state: copy, `is_game_over()`, the index of the
side to move, applying an int action, and the terminal outcome for a player. Make
`MCTSTree` a class template over `State` (`template <class State> class MCTSTree`), keep
the member definitions in `mcts.cpp` and add explicit instantiations at the bottom:

```cpp
// go_game.h — additive methods on GoBoard
int   player() const { return to_play_ == BLACK ? 0 : 1; }
void  apply(int action) { action == PASS_ACTION ? pass() : play_flat(action); }
float outcome(int player) const;   // 1 / 0.5 / 0 from get_winner()

// mcts.cpp — the only semantic edits
int8_t current_player = nodes_[node_idx].state.player();           // replaces to_play()==BLACK ? 0:1
new_state.apply(action);                                           // replaces pass()/row_col()/play()
U = nodes_[node_idx].state.outcome(player_perspective);            // replaces get_winner() blocks
U = (child.player_at_parent == node.player_at_parent) ? U : 1.0f - U;  // the Boxes sign rule
```

`using GoMCTSTree = MCTSTree<GoBoard>` keeps `py::class_<...>(m, "MCTSTree")` identical;
`BoxesBoard` + `py::class_<MCTSTree<BoxesBoard>>(m, "BoxesMCTSTree")` are additive.
Rollout mixing (`lambda_ > 0`) stays Go-only (it needs a null action); assert it off for
Boxes. For Go the new flip rule is always `1 - U`, so visit counts are bit-identical.

### 6.2 Python — `Game` protocol + adapters

New `src/alpha_go/game.py`:

```python
class Game(Protocol):
    name: str
    def new_board(self, **cfg: Any) -> Any: ...          # Go: alpha_go_cpp.GoBoard(size, komi)
    def num_actions(self, board: Any) -> int: ...        # Go: size*size + 1
    def action_index(self, action: Any) -> int: ...      # Go: flat idx, PASS_ACTION -> size*size
    def encode(self, board: Any) -> np.ndarray: ...      # (C, H, W) float32 planes for the net
    def to_play(self, board: Any) -> int: ...            # 0 / 1
    def outcome(self, board: Any, player: int) -> float: # 1 / 0.5 / 0
    def margin(self, board: Any) -> float: ...           # Go: score() from player 0; Boxes: box diff
    def symmetries(self, planes, policy) -> list[tuple[np.ndarray, np.ndarray]]: ...
GAMES: dict[str, Game]  # {"go": GoGame(), "boxes": BoxesGame()}
```

`games/go_adapter.py` wraps the existing `alpha_go_cpp.GoBoard` and the existing colour-swap
encoding; `go.py`/`FastGoBoard`/`GoState` are untouched. `mcts.py` gets the one-line
mover-aware backup (identical for Go). Boxes supplies `BoxesBoard` (Python reference and
C++), `BoxesState` (Python MCTS protocol), encoder, symmetry maps, baselines, solver, net.

### 6.3 Shared-file touch points (kept to a flag and a few lines)

* `self_play.py`: `--game {go,boxes}` (default `go`), `--board-cols` (default = rows),
  import Boxes agents when `--game boxes`; pass `game` through to `play_game`.
* `gameplay.py`: optional `game` argument (default → current Go code path); `new_board`
  via `game`; `n_actions` and index mapping via `game`; record `to_play` per position
  (new NPZ key, additive); skip the Go-notation debug render for non-Go.
* `agents/base.py`: widen the `select_move` board annotation (no runtime change).
* `dataset.py`: untouched — Boxes gets its own `BoxesDataset` because its NPZ carries
  `to_play`, lattice boards and margins.
* `model.py`: untouched — `BoxesNet` imports `MaskedResBlock` etc.
* `cpp/CMakeLists.txt`, `cpp/bindings/bindings.cpp`: additive targets/bindings.
* `CLAUDE.md`, `.gitignore`: append-only.

---

## 7. Pre-existing defects and caveats (upstream, not caused by the fork)

1. `tests/test_gpu_lease.py` imports `_read_priorities`/`_write_priorities_stub` from
   `infra.remote_exec`, which do not exist → collection `ImportError`. The upstream suite is
   red at HEAD; our "upstream tests keep passing" baseline must exclude or fix this file.
2. `alpha_go.inference_server` is referenced (`tests/test_cpp_mcts.py:15`, `nn_mcts.py:1140`)
   but not in the repo; the NN parity tests are therefore always skipped.
3. `go.py::FastGoBoard.score()` hard-codes komi 6.5; C++ uses 7.5. `FastGoBoard` has no
   positional superko. Parity tests never compare scores.
4. `cpp/mcts/mcts_test.cpp` is not a CMake target.
5. `scripts/build_cpp.sh` hard-codes `libpython3.10.so`; harmless for the extension (pybind
   modules don't link libpython) but wrong on other Python versions.
6. `agents/nn_agent.py:107` references an undefined `GoResNet` (dead `model_type` branch).
7. `play.py` imports `katago` for the assist feature (not a dependency).
8. Registered MCTS agents hard-code `/nfs/checkpoints/...`; `GAME_DATA_DIR` defaults to
   `/nfs/game_data_root`; `infra.remote_exec.CLUSTER_TOML = /workspace/cluster.toml`.
9. `train.py` overrides the dataset's `is_teacher` with `winner` (documented as a bug in
   `experiments/2026-04-28_00-38-fastlearn/report.md`; fixed there, not in the reference loop).
10. `pyproject.toml` does not declare `matplotlib` (imported by `gameplay.py`; present
    only transitively today) nor `fastapi`/`uvicorn`/`pydantic` (imported by `play.py`;
    **absent** in a fresh `uv sync`, so `python -m alpha_go` / the web UI do not start).
