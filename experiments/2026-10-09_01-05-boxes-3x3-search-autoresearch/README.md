# boxes-3x3-search-autoresearch

Autoresearch loop (the `autoresearch` skill's test-time-compute track) over the MCTS search
hyperparameters of `BoxesMCTSAgent`, run on CPU against the **exact oracle** instead of a
head-to-head win rate: a frozen 3x3 checkpoint searches a fixed set of 600 late positions
(6..14 undrawn edges, sampled from the `3x3-cpu` loop's own games) and the metric is the
share of oracle-optimal moves it plays. That is exact and has a binomial standard error of
about 1.2 points at 600 positions, so differences of 3 points or more are real; a 128-game
win rate would need 4 points to say the same.

- `eval_search.py` — the edit target: `VAR` is the only thing changed between runs
  (`CKPT`, positions and seed stay fixed); prints a `===RESULT===` line with
  `search_optimal` (the target), `root_q_sign` (the searched root value's sign against the
  oracle's final margin), `sec_per_move`.
- `results.tsv` — one line per run (untracked); `progress.png` / `report.md` — analysis.

The checkpoint comes from running the 3x3 loop on this container's CPU
(`TAG=3x3-cpu ... run_iteration_local.sh 0 4 --cpu` with the GPU budgets), so the whole
folder is reproducible without a GPU. Deviation from the skill text: runs are committed on a
local `autoresearch/...` branch and the folder is brought to the shared branch in one commit,
because only `claude/awesome-fermi-ygaeuj` may be pushed.

Knobs explored: simulations (at fixed budget), `c_puct`, leaf batch size, policy
temperature, the margin-utility term (`margin_utility_lambda`, PLAN §6.4), the endgame
solver at small N (partial solving below the positions' own depth), and the equivalent-edge
merge.
