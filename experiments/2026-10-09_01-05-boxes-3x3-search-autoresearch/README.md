# boxes-3x3-search-autoresearch

Autoresearch loop (the `autoresearch` skill's test-time-compute track) over the MCTS search
hyperparameters of `BoxesMCTSAgent`, run on CPU against the **exact oracle** instead of a
head-to-head win rate: a frozen 3x3 checkpoint searches a fixed set of 600 late positions
(6..14 undrawn edges, sampled from the `3x3-cpu` loop's own games) and the metric is the
share of oracle-optimal moves it plays. That is exact and has a binomial standard error of
about 1.2 points at 600 positions, so differences of 3 points or more are real; a 128-game
win rate would need 4 points to say the same.

- `eval_search.py` — the edit target: `VAR` is the only thing changed between runs
  (`CKPT`, positions and seed stay fixed); `--var '<json>'` overrides it for one run; prints
  a `===RESULT===` line with `search_optimal` (the target), `root_q_sign` (the searched root
  value's sign against the oracle's final margin), `sec_per_move`.
- `run_loop.py` — the keep / discard driver: each candidate is the running best plus one
  change; a run is kept when it beats the best on `search_optimal`. Appends every run to
  `results.tsv` (untracked) and `runs.jsonl` (tracked: full `VAR` and result, so any run is
  `eval_search.py --var '<var>'`), then calls `analyze_runs.py`.
- `analyze_runs.py` — `progress.png` and `report.md` from `results.tsv`.

The checkpoint comes from running the 3x3 loop on this container's CPU
(`TAG=3x3-cpu ... run_iteration_local.sh 0 5 --cpu` with the GPU budgets; champion iter5),
so the whole folder is reproducible without a GPU:

```bash
export GAME_DATA_DIR=<root holding experiments/2026-10-08_14-42-boxes-3x3-loop/3x3-cpu>
uv run experiments/2026-10-09_01-05-boxes-3x3-search-autoresearch/run_loop.py   # ~1 min/run
```

Deviation from the skill text: the skill commits each run and reverts the script on a
discard; here nothing is committed per run because only `claude/awesome-fermi-ygaeuj` may
be pushed, and `runs.jsonl` plays the role of the commit history.

Knobs explored: simulations (at fixed budget), `c_puct`, leaf batch size, policy
temperature, the margin-utility term (`margin_utility_lambda`, PLAN §6.4), the endgame
solver at small N (partial solving below the positions' own depth), and the equivalent-edge
merge.

## Findings (rounds 1 and 2, 25 runs; `report.md` has every run)

Baseline 0.823 -> best 0.917 oracle-optimal searched moves (+9.3 points, about seven
standard errors) with `c_puct=2.5 leaf_batch_size=4 policy_temperature=0.7
margin_utility_lambda=1.0 (k=6) num_simulations=200`.

- **The margin-utility term is the dominant knob** (PLAN §6.4): lambda 0 -> 0.25 -> 0.5 ->
  0.75 -> 1.0 gave 0.853 -> 0.860 -> 0.872 -> 0.898 -> 0.917, monotone and not saturated;
  `k=10` instead of 6 lost 1.8 points. The net's expected margin carries more of the
  decision than its P(win) on these positions, which fits a game whose strategy is about
  control and so about margin, not just the sign.
- **Sharper priors**: `policy_temperature=0.7` +2.0 points; 0.5 lost 2.3 against 0.7; 1.5
  lost 1.5. The policy head is under-confident at this stage of training.
- **Leaf batch size** mattered once the values were decisive: 4 beat 16 by 3.8 points on
  top of the best (half the time at 16), only 0.5 in round 1.
- **Simulations**: 200 vs 100 +0.7 at 2.4x the time; 50 lost 3.2. `c_puct` is noise-level
  (2.5 vs 1.5: +0.3 in round 1, +1.0 on top of the best).
- **Equivalent-edge merge**: no change (0.860 either way); the positions here have few
  independent chains.
- **The solver at small N hurts with the margin utility on**: `solver_max_undrawn=10` tied
  the round-1 best (0.872) at a third of the time with lambda=0.5 but lost 3.5 points on
  top of the lambda=1.0 best. Solved and game-over positions back up 1 / 0.5 / 0 while net
  leaves back up `P(win) + lambda * tanh(E[margin] / k)`, so a solved subtree looks worse
  than an unsolved one and the search steers away from exact knowledge. Follow-up: shape
  terminal outcomes with the exact margin on the same scale (the search state has
  `solved_margin()` and the board its box difference), then re-run the solver candidates.

Caveats before carrying these into training: everything here is test-time behaviour of
one 3x3 net on late positions; self-play adds Dirichlet noise and needs diversity, so
sharper priors may cost exploration there. Check transfer first with
`oracle_eval.py --tag 5x5-solver --policy_temperature 0.7 --margin_utility_lambda 1.0`
(no solver in that agent), then try `SEARCH_ARGS` on an arena match and a training arm;
do not combine `margin_utility_lambda` with `SOLVER_N` until the terminal shaping above
lands.
