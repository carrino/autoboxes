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
