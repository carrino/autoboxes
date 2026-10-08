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

### Status (2026-10-08)

Landed on `claude/awesome-fermi-ygaeuj`: commits 1-4 (Game protocol + Go adapter,
mover-aware Python backup, reference rules, symmetries), 5 (oracle + sign tests), 6 (C++
board), 7 (templated C++ search, Go byte-identical), 8 (baselines + arena), 9 (encoder +
net), 10 (play loop / CLI / NPZ `to_play` / dataset), 11 (NN MCTS agent + batched
engine), 12 (3x3 loop, smoke-tested on CPU). Open: 5b forced-move collapse, 13 (5x5 run
on the GPU box), the text engine protocol (§3) and Phase 2.

### 2.1 Commit plan

| # | commit | new files | shared files touched |
|---|---|---|---|
| 1 | `game.py`: `Game` protocol + registry; `games/go_adapter.py` (`GoGame`) | `src/alpha_go/game.py`, `src/alpha_go/games/__init__.py`, `src/alpha_go/games/go_adapter.py`, `tests/test_game_go_adapter.py` | — |
| 2 | shared: mover-aware backup in Python MCTS (`U = child_value if same mover else 1 - child_value`) + test with a toy game that has extra moves | `tests/test_mcts_extra_move.py` | `src/alpha_go/mcts.py` (1 line + 1 line at root) |
| 3 | Boxes Python reference rules: `BoxesBoard` (edge mask, box owners, scores, to_play, undo-free), `BoxesState` (MCTS protocol), edge/lattice index tables, ASCII render | `src/alpha_go/boxes/__init__.py`, `boxes/rules.py`, `tests/test_boxes_rules.py` | — |
| 4 | Symmetries: lattice D4/D2 transforms for planes, edge permutations, round-trip tests | `boxes/symmetry.py`, `tests/test_boxes_symmetry.py` | — |
| 5 | Exhaustive minimax oracle (negamax over edge subsets with memo) for 1×1, 1×2, 2×2, 2×3; Python MCTS agrees with it; **explicit sign-flip test on a hand-built capture position** | `boxes/oracle.py`, `tests/test_boxes_oracle.py`, `tests/test_boxes_mcts_sign.py` | — |
| 5b | Forced-move collapse (search side, not the rules): detect opened chains/loops, auto-capture an opened chain down to 2 boxes (a loop down to 4), then expose `take all` vs `keep control` (double-deal) as the only two continuations; `BoxesSearchState` wraps `BoxesBoard` for the search; tests: the collapse never changes the oracle's minimax value on 2×2 / 2×3, and macro-actions map back to their first edge | `boxes/chains.py`, `boxes/forced.py`, `tests/test_boxes_forced.py` | — |
| 6 | C++ `BoxesBoard` (`uint64_t` edge mask when `E ≤ 64`, `std::bitset` fallback) + pybind; Python-vs-C++ equivalence under random play; perft counts | `cpp/boxes/boxes_game.h/.cpp`, `cpp/boxes/boxes_game_test.cpp`, `tests/test_boxes_cpp_parity.py` | `cpp/CMakeLists.txt`, `cpp/bindings/bindings.cpp` (additive) |
| 7 | shared: template `MCTSTree<State>` with `player()/apply()/outcome()` + mover-aware backup; `BoxesMCTSTree` binding over the C++ search state (forced-move collapse ported, parity-tested against `boxes/forced.py`); C++ sign-flip test; C++ MCTS vs oracle; Go MCTS tests unchanged | `tests/test_boxes_cpp_mcts.py` | `cpp/go/go_game.h` (3 additive methods), `cpp/mcts/mcts.h/.cpp` (template + 4 semantic edits), `bindings.cpp`, `CMakeLists.txt` |
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

### 2.5 Forced-move collapse (search side)

BoxesZero's ablation credits chain-loop pruning with the largest single gain, so it is a
Phase 1 item. The raw rules stay one edge per move (arena, text protocol, NPZ and the
external-engine bridge all speak single edges). The collapse lives in the state the
*search* expands (`BoxesSearchState` in Python, the same logic in the C++ search state):

* `boxes/chains.py`: union-find on the strings-and-coins dual (boxes = coins, one ground
  node for the outside) to label chains and loops and to find *opened* components
  (a component containing a box with three drawn sides).
* `boxes/forced.py`: when the side to move faces an opened chain (loop), the forced
  captures are applied automatically down to the last 2 boxes (4 for a loop); the search
  then sees exactly two continuations, `take all` (capture the rest, then move again) and
  `keep control` (the double-dealing edge that hands the opponent those 2/4 boxes and the
  move). With no opened component the action set is the plain undrawn edges, with
  *safe* edges (no third side created) listed before *loony* ones.
* Interface to the rest of the system: each macro-action is identified by its first
  edge, so priors come from the policy head's logit for that edge, visit counts are
  credited to that edge in the training target, and the move actually played is that
  edge (the remaining forced edges are played out by the same agent on its next turns;
  it re-derives the same collapse). This keeps the policy head, NPZ schema and
  `self_play` unchanged.
* Tests: for every position reachable on 2×2 and 2×3, the collapsed game tree has the
  same minimax value as the uncollapsed oracle; the C++ port matches the Python one
  under random play.

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

1. **Exact endgame solver** `boxes/solver.py` + `cpp/boxes/solver.{h,cpp}`, used as the MCTS
   leaf evaluator when `popcount(~mask) ≤ N`, default `N = 28` undrawn edges; benchmark
   `N = 24..36` on 5×5 and record the results before raising the default. The solver's
   margin arithmetic is exact and independent of the net: it returns the remaining margin
   for the side to move, the leaf value is `sign(board.margin() + remaining)` → 1 / ½ / 0
   (BoxesZero §4.5), and the margin distribution head is only a training target and the
   source of `P(margin > 0)` for non-solved leaves.
   * **Transposition table** keyed on the symmetry-canonical edge mask (the minimum over
     the 8 lattice transforms, 4 for rectangular boards; `symmetry.edge_permutation`
     gives the bit permutations), storing the remaining margin for the side to move
     (margin so far is additive and never changes the optimal line). Entries and bytes
     are a CLI flag and are printed at startup.
   * **Move generation**: captures first and forced (reuse the Phase 1 forced-move
     collapse); all edges of one chain or loop are equivalent and generate a single move;
     safe moves (no third side) are ordered before loony ones.
   * **Solver leaf**: when the position is pure chains-and-loops, return `v(G)` in closed
     form from Allcock 2021 (`c(G)`, terminal bonus, Theorems 1–3) extended with
     BoxesZero's 1-chain / 2-chain rules (Theorems 4–5). Property-test the closed form
     against the alpha-beta on every chains-and-loops position reachable on 2×3 and 3×3.
   * **Benchmark**: report the solve-time distribution (p50 / p99 / max), not the mean,
     for `N = 24..36` on positions sampled from real self-play games; the MCTS leaf budget
     is set by the p99.
   * **Validation**: once a Dabble / PRsBoxes bridge exists, sample ~2000 late positions,
     let the engine play them out from both sides against our solver, and assert the
     solver's claimed margin is never beaten. Any disagreement is a solver bug until
     proven otherwise.
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

## 6. Decisions (resolved with the author, 2026-10-08)

1. **C++ MCTS**: `template <class State> class MCTSTree` in place, explicit instantiations
   for `GoBoard` and `BoxesBoard`; `MCTSTree` stays the Go binding name, `BoxesMCTSTree`
   is added. (A macro needs two compilations with clashing symbols; an interface changes
   the evaluator callback signature and heap-allocates every node's state.)
2. **Policy head**: 1×1 conv over lattice cells, gathered at the `E` edge cells. Symmetries
   (8 for square boards, 4 for rectangular) act on the lattice; the edge permutation is
   derived by applying the same transform to an index grid. Two constant planes mark edge
   cells and box cells so an undrawn edge and a dot are distinguishable.
3. **Data**: the shared NPZ writer always records `to_play`; `GoDataset` is untouched;
   `BoxesDataset` reads `to_play`, lattice boards and margins.
4. **Search value**: Q stays a win probability in `[0, 1]`; PUCT, `c_puct`, Dirichlet
   noise, temperature and resign logic are unchanged, only the mover-aware sign rule is
   added. The margin enters as (a) an input plane: score so far for the side to move
   (BoxesZero's third channel, Niu et al. 2025, Entropy 27(3):285), (b) the trained
   value-head distribution over final margins whose `P(margin > 0)` feeds the search,
   (c) terminal values from the final box count, and (d) in Phase 2 the exact endgame
   solver's remaining margin added to the current margin, whose sign is the leaf value
   (BoxesZero §4.5: `score_s = current − opponent − v(s)` → `Q = ±1`). The solver's margin
   arithmetic is exact and independent of the net; the margin-distribution head is a
   training target and the source of `P(margin > 0)` only. Search flag
   `--margin_utility_lambda` (default `0.0`): when nonzero the Boxes evaluator backs up
   `u = win_prob + lambda * tanh(expected_margin / k)` with `--margin_utility_k`
   (default `6`); the `1 − u` perspective flip still holds because `tanh` is odd. Default
   behaviour is unchanged; this is for a later arena comparison (KataGo score utility),
   not Phase 1. Optional training flags derived from BoxesZero: value target
   `0.75·z + 0.25·Q` (upstream already saves the root Q per position) and backward
   training as an alternative to solver-labelled pretraining.
5. **Python version**: not pinned. `scripts/build_cpp.sh` derives `libpython` from the
   venv's `sysconfig` instead of hard-coding 3.10 (shared bug fix).
6. **Broken upstream tests**: excluded (`--ignore=tests/test_gpu_lease.py`); infra stays
   untouched.
