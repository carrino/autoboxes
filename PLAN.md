# PLAN.md — turning `autoboxes` into a Dots and Boxes AlphaZero trainer

Companion to `ARCHITECTURE.md`. Each phase is a sequence of small commits; every commit
leaves `uv run -m pytest tests/` green (with the upstream-broken
`tests/test_gpu_lease.py` excluded, see ARCHITECTURE §7.1) and keeps Go behaviour
byte-identical.

Naming: "Boxes" everywhere in code (`boxes`, `BoxesBoard`, `--game boxes`).
Player indices are `0` (moves first) and `1`. Score margin is always *for the side to
move*. Edges, boxes and dots live on a `(2R+1) × (2C+1)` lattice.

---

## 0. Ground rules for the fork

* Go code is never deleted, renamed or re-flowed. Go stays the default
  (`--game go`, `play_game(game=None)`, `MCTSTree` keeps its Python name).
* Boxes lives in new files: `src/alpha_go/boxes/…`, `src/alpha_go/cpp/boxes/…`,
  `tests/test_boxes_*.py`, `experiments/<date>-boxes-*/…`.
* Shared files change by the smallest additive diff that lets a flag select the game
  (full list in §5). Each such change is its own commit with "shared:" in the message so
  upstream merges are easy to review.
* `upstream` remote → `https://github.com/ericjang/autogo` (done). Merge upstream with
  `git fetch upstream && git merge upstream/main` on `main` before rebasing feature work.
* Single machine first: nothing in the Boxes loop may require `cluster.toml`, SSH, `/nfs`,
  or Docker. The cluster path stays untouched and usable.
* Correctness before speed: Python reference rules + property tests land before C++;
  C++ must match the reference under random play before any agent uses it.

---

## 1. Phase 0 — reconnaissance (this commit)

Deliverables: `ARCHITECTURE.md`, `PLAN.md`, `CLAUDE.md` (append-only edit), `upstream`
remote. Verified on this checkout (Python 3.13 via `uv`, no GPU):
`uv sync` works; `scripts/build_cpp.sh` fails on a non-3.10 interpreter (hard-coded
`libpython3.10.so`) but a manual cmake with `-DPython3_LIBRARY=libpython3.13.so` builds;
Catch2 `go_game_test`: 143 assertions in 18 cases pass; `uv run -m pytest tests/
--ignore=tests/test_gpu_lease.py`: **101 passed, 24 skipped** (gnugo and NN-checkpoint
tests skip); `tests/test_gpu_lease.py` fails at import; `uv run -m mypy src/` reports 157
errors upstream (strict mode is not clean). These are the numbers every later commit
must keep.

**Stop here for review.** Open decisions are listed in §6.

---

## 2. Phase 1 — playable, testable, trainable (3×3 then 5×5)

### 2.1 Commit plan

| # | commit | new files | shared files touched |
|---|---|---|---|
| 1 | `game.py`: `Game` protocol + registry; `games/go_adapter.py` (`GoGame`) | `src/alpha_go/game.py`, `src/alpha_go/games/__init__.py`, `src/alpha_go/games/go_adapter.py`, `tests/test_game_go_adapter.py` | — |
| 2 | shared: mover-aware backup in Python MCTS (`U = child_value if same mover else 1 - child_value`) + test with a toy game that has extra moves | `tests/test_mcts_extra_move.py` | `src/alpha_go/mcts.py` (1 line + 1 line at root) |
| 3 | Boxes Python reference rules: `BoxesBoard` (edge mask, box owners, scores, to_play, undo-free), `BoxesState` (MCTS protocol), edge/lattice index tables, ASCII render | `src/alpha_go/boxes/__init__.py`, `boxes/rules.py`, `tests/test_boxes_rules.py` | — |
| 4 | Symmetries: lattice D4/D2 transforms for planes, edge permutations, round-trip tests | `boxes/symmetry.py`, `tests/test_boxes_symmetry.py` | — |
| 5 | Exhaustive minimax oracle (negamax over edge subsets with memo) for 1×1, 1×2, 2×2, 2×3; Python MCTS agrees with it; **explicit sign-flip test on a hand-built capture position** | `boxes/oracle.py`, `tests/test_boxes_oracle.py`, `tests/test_boxes_mcts_sign.py` | — |
| 6 | C++ `BoxesBoard` (`uint64_t` edge mask when `E ≤ 64`, `std::bitset` fallback) + pybind; Python-vs-C++ equivalence under random play; perft counts | `cpp/boxes/boxes_game.h/.cpp`, `cpp/boxes/boxes_game_test.cpp`, `tests/test_boxes_cpp_parity.py` | `cpp/CMakeLists.txt`, `cpp/bindings/bindings.cpp` (additive) |
| 7 | shared: template `MCTSTree<State>` with `player()/apply()/outcome()` + mover-aware backup; `BoxesMCTSTree` binding; C++ sign-flip test; C++ MCTS vs oracle; Go MCTS tests unchanged | `tests/test_boxes_cpp_mcts.py` | `cpp/go/go_game.h` (3 additive methods), `cpp/mcts/mcts.h/.cpp` (template + 4 semantic edits), `bindings.cpp`, `CMakeLists.txt` |
| 8 | Baselines: `boxes-random`, `boxes-greedy` (take captures, avoid giving third sides), `boxes-ab-d{N}` (depth-limited alpha-beta, TT on edge mask) + arena CLI (alternating first player, win rate + mean margin ± CI) | `boxes/agents.py`, `boxes/alphabeta.py`, `boxes/arena.py`, `tests/test_boxes_agents.py` | — |
| 9 | Encoding + net: lattice planes, `BoxesNet` (masked ResNet trunk reused from `model.py`, policy over lattice cells gathered at edges, margin-distribution value head + derived scalar win prob) + symmetry-equivariance tests | `boxes/encode.py`, `boxes/model.py`, `tests/test_boxes_model.py` | — |
| 10 | shared: `--game boxes` in `self_play.py`, `game=` in `play_game`, `to_play`/`game` keys in NPZ, `BoxesDataset` | `boxes/dataset.py`, `tests/test_boxes_dataset.py` | `self_play.py`, `gameplay.py`, `agents/base.py` (annotation) |
| 11 | Batched inference for plane inputs (`PlaneBatchedEngine`, fp16 autocast, prints memory footprint), leaf-batched `BoxesNNEvaluator`, `BoxesMCTSAgent` (uses `CppMCTSAgent` logic through `BoxesMCTSTree`) | `boxes/inference.py`, `boxes/nn_agent.py`, `tests/test_boxes_inference.py` | — |
| 12 | Local loop: `experiments/<date>-boxes-3x3-loop/{README.md, run_iteration_local.sh, pre_collect.py, run_games.py, train.py, arena_promote.py, analyze.py}`; runs end to end on 3×3 in minutes | experiment dir | `.gitignore` (`*.pt`, `*.npz`, `results.tsv`) |
| 13 | 5×5 config + overnight run + report | `experiments/<date>-boxes-5x5-loop/` | — |

### 2.2 Rules and data layout (what the tests pin down)

```
R×C boxes, (R+1)×(C+1) dots
horizontal edge h(r,c): r∈[0,R], c∈[0,C)   index  r*C + c
vertical   edge v(r,c): r∈[0,R), c∈[0,C]   index  (R+1)*C + r*(C+1) + c
E = (R+1)*C + R*(C+1)                        (5×5: 60, 3×3: 24, 2×2: 12, 1×1: 4)
box (r,c) sides: h(r,c), h(r+1,c), v(r,c), v(r,c+1)
lattice (2R+1)×(2C+1): dot (2r,2c) · h-edge (2r,2c+1) · v-edge (2r+1,2c) · box (2r+1,2c+1)
```

State = `edges: uint64 mask`, `owner[R*C] ∈ {-1,0,1}`, `score[2]`, `to_play`,
`move_count`. `apply(e)`: set bit; for each box adjacent to `e` whose 4 sides are now set,
assign it to `to_play` and bump `score`; if nothing was captured, `to_play ^= 1`.
Terminal iff `edges == (1<<E)-1`. `outcome(p)` = 1/½/0 by score comparison;
`margin()` = `score[to_play] - score[1-to_play]`.

Tests (Phase 1 commit 3 + 5 + 6): capture gives an extra move; one edge completing two
boxes gives +2 and one extra move; a move completing nothing switches player; terminal
when all edges drawn; perft(1×1)=4!, perft counts for 1×2/2×2 at depths 1–4 recorded as
golden values; Python == C++ for 1000 random games on 1×1…5×5 and 3×4 (edge order, scores,
`to_play`, legal masks, terminal, hash); minimax values: 1×1 = +1 for the second player
… (whatever the oracle says, asserted once computed and reviewed); MCTS with an exact
evaluator reaches the oracle move on every 2×2 / 2×3 root with ≥ 2000 sims.

Sign-flip test (commit 5 and 7): build a position where the side to move can take a box
and then must move again into a lost ending vs. decline the box and win. Run MCTS with a
perfect terminal evaluator (few sims so every path hits terminal). Assert `Q` of the
capturing child equals the oracle value *without* a sign flip relative to the parent and
that the search prefers the oracle move. The same test is run through the Python MCTS and
the C++ `BoxesMCTSTree`.

### 2.3 Network and encoding

Input planes on the lattice, all float32, shape `(K, 2R+1, 2C+1)`:

| plane | content |
|---|---|
| 0 | edge drawn (1 at drawn edge cells) |
| 1 | box captured by side to move |
| 2 | box captured by opponent |
| 3–7 | box side count one-hot 0..4 (at box cells) |
| 8 | is-edge-cell constant mask |
| 9 | is-box-cell constant mask |
| 10 | score margin for side to move / (R·C), broadcast |

Policy: 1×1 conv → logits on every lattice cell; gather at the `E` edge cells (fixed
table) → `(B, E)`; non-edge cells never appear in the action space. Because the action
set *is* a subset of lattice cells, a board symmetry applied to the planes induces the
edge permutation for free (`symmetry.py` derives the permutation from the same transform
applied to an index grid). Value: masked global pool + margin scalar → FC →
`(B, 2RC+1)` logits over margins `-RC..+RC` (CE loss vs final margin for side to move);
`win_prob = P(margin > 0) + ½·P(margin = 0)`; the evaluator hands `win_prob` to MCTS so
the C++ search keeps its `[0,1]` Q convention and upstream's scalar path stays valid
(`value_head="scalar"` config trains BCE like upstream for A/B).

Model sizes: 3×3 → 64ch × 6 blocks; 5×5 → 128ch × 10 blocks (same as upstream default).
fp16 autocast for inference, bf16 autocast + GradScaler for training (RTX 3070 = Ampere,
both supported); fail loudly at startup if `torch.cuda.is_available()` is false unless
`--cpu` is passed (tests use CPU).

### 2.4 Self-play / training / arena on one box

* Collection: `self_play.py --game boxes --board_size 3 --black boxes-mcts --white
  boxes-mcts --num_workers 8 --collect-metrics` (threads share one `PlaneBatchedEngine`;
  each thread runs leaf-batched C++ MCTS with virtual loss). NPZ gains `to_play (n,) int8`,
  `game`, `rows/cols`, `margins (n,) int16`; `boards` is the lattice int8 array per
  position (self-describing, renderable).
* Training: `train.py` forked from upstream, swaps `GoDataset→BoxesDataset`,
  `SizeInvariantGoResNet→BoxesNet`, D4/D2 augmentation via `symmetry.py`, value target =
  final margin from side to move, policy target = MCTS visit distribution (every position,
  no winner masking — the upstream `is_teacher = winner` bug is not inherited).
* Arena/promotion: `arena_promote.py` plays candidate vs champion, N games, alternating
  who moves first, reports win rate, mean margin, 95% CI; promotes at ≥ 55% (configurable)
  into `league_state.json` with the same schema as upstream's `update_league.py`. Also
  reports candidate vs `boxes-random`, `boxes-greedy`, `boxes-ab-d6` every iteration so
  progress is visible against fixed opponents.
* Output paths: `GAME_DATA_DIR` (default `~/autoboxes-data/game_data_root`, never `/mnt/c`),
  checkpoints under `experiments/<exp>/checkpoints/` (gitignored). Replay buffer size
  (positions) and TT entries are CLI flags; their byte footprint is printed at startup.
* Budget targets: 3×3 iteration (200 games × 200 sims + 2 min train + 100 arena games)
  ≈ 5 min; 5×5 overnight = 10–20 iterations of 400 games × 400 sims on the 3070.

---

## 3. Baseline engines (optional, after Phase 1 is green)

1. `boxes/protocol.py`: newline-delimited text protocol
   (`new_game R C`, `play <edge>`, `genmove` → `<edge>`, `quit`) + `boxes/engine_agent.py`
   that wraps any stdio/TCP engine as an arena `Agent`, and `python -m alpha_go.boxes.engine`
   that exposes our own agents through it (round-trip test: our agent vs itself over a pipe).
2. Investigate Dabble and PRsBoxes on the Windows host: command-line flags, position file
   formats, UI automation hooks. Write `BASELINES.md` with what is actually scriptable.
3. Only if a reliable hook exists: `tools/win_bridge/` (PowerShell or Python on the Windows
   side) speaking the protocol over a TCP socket to WSL. Arena-only; never imported by
   training code. Otherwise document the dead end and stop.

---

## 4. Phase 2 — domain knowledge

1. **Exact endgame solver** `boxes/solver.py` + `cpp/boxes/solver.{h,cpp}`: alpha-beta on
   remaining-box margin with TT keyed on edge mask (the margin-so-far is additive and does
   not change the optimal line, so keying on the mask alone gives more hits; the
   `(mask, margin)` key is kept as an option for the depth-limited heuristic search),
   configurable `N` undrawn edges, used as the MCTS leaf evaluator when
   `popcount(~mask) ≤ N` (returns exact win prob / margin). Benchmark solve time vs N on
   5×5 and record a table.
2. **Chain/loop features** via union-find on the strings-and-coins dual (boxes = coins,
   ground node = outside): per-edge planes (safe move, opens chain/loop, chain-length bucket,
   chain vs loop, ends at ground vs junction) and global scalars (chain counts by length,
   loop count, long-chain parity) appended to the planes and the value-head input.
   Property tests against brute-force enumeration on small boards.
3. **Solver-labelled pretraining**: generate positions by greedy-then-random playouts,
   label with the exact solver where applicable, train the value head (and policy on the
   solver's best moves) before self-play; measure arena strength vs `boxes-ab-d*`.

---

## 5. Shared files that will change, and why

| file | change | why |
|---|---|---|
| `src/alpha_go/mcts.py` | backup flips only when the mover changes; root `player_at_parent` via the state | extra-move games; Go identical |
| `src/alpha_go/cpp/mcts/mcts.h`, `mcts.cpp` | `template <class State> class MCTSTree`; use `state.player()/apply()/outcome()`; mover-aware flip in both backup paths; explicit instantiations for `GoBoard` and `BoxesBoard` | one search implementation for both games |
| `src/alpha_go/cpp/go/go_game.h` | add `player()`, `apply(int)`, `outcome(int)` (inline, additive) | satisfy the `State` concept |
| `src/alpha_go/cpp/bindings/bindings.cpp` | `BoxesBoard`, `BoxesMCTSTree` bindings; a small template helper to bind both trees | expose Boxes |
| `src/alpha_go/cpp/CMakeLists.txt` | add `boxes_game` library, link into `mcts` and the module, add `boxes_game_test` (and wire the orphaned `mcts_test`) | build |
| `src/alpha_go/gameplay.py` | optional `game` parameter; `new_board`/`num_actions`/index mapping via `game`; record `to_play`; skip Go debug render when not Go | one play loop for both games |
| `src/alpha_go/self_play.py` | `--game`, `--board-cols`, import Boxes agents on demand | flag selects the game |
| `src/alpha_go/agents/base.py` | `select_move(board: Any, …)` annotation | Boxes boards |
| `scripts/build_cpp.sh` | derive `libpython` from `sysconfig` instead of hard-coding 3.10 | builds on any venv Python (bug fix, upstreamable) |
| `CLAUDE.md`, `.gitignore` | append-only | conventions, artifacts (`pyproject.toml` is left alone; the missing `fastapi`/`uvicorn` only affect upstream's web UI) |

Not touched: `go.py`, `model.py`, `dataset.py`, `engine.py`, `play.py`, `inference/`,
`agents/nn_agent.py`, `agents/nn_mcts.py`, `infra/`, existing experiments.

---

## 6. Decisions I want confirmed before Phase 1

1. **Template vs copy for the C++ MCTS.** Recommended: template in place (§5). The
   alternative — a verbatim copy under `cpp/boxes/` — has zero merge-conflict risk but
   forks 670 lines that then drift from upstream fixes.
2. **Policy head over lattice cells gathered at edges** (recommended, makes symmetries
   trivial and reuses the fully-convolutional trunk) vs a flat `E`-way FC head. The action
   space exposed to MCTS, NPZ and the text protocol is the edge index either way.
3. **NPZ for Boxes**: separate `BoxesDataset` reading `to_play` from the file (recommended)
   vs teaching `GoDataset` about `to_play`. Go NPZs stay unchanged apart from the two new
   keys the shared writer adds.
4. **Terminal/outcome semantics for MCTS**: keep upstream's `[0,1]` win-prob Q (recommended,
   zero search changes beyond the flip rule) vs switching Q to expected margin.
5. **Python version**: devcontainer pins 3.10; this box got 3.13 from `uv`. Pin with
   `uv python pin 3.10` + `requires-python` untouched, or make the build script
   version-agnostic (recommended, also fixes the upstream script).
6. **`tests/test_gpu_lease.py`**: exclude via `--ignore` in the documented test command, or
   add the two missing helpers to `infra/remote_exec.py`. I'd rather not touch infra;
   recommend the ignore until upstream fixes it.
