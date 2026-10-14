# Learning experiments above the solver threshold

These opt-in tools separate policy, value, search objective, search budget, training
targets and opponent diversity. They do not change the normal loop's defaults or
promote checkpoints. The code builds on main `bb565359`; BoxesZero remains a separate
adapter in PR #13. Run from the repository root with the existing Python environment
and current C++ extension. No C++ changes or rebuild are required by this experiment.

The motivating arena had 57/100 wins for the frozen reference, 54/100 for the outcome
control and 44/100 for the four-epoch exact candidate. Exact replay of all 300 games
found every observed move at <=30 remaining edges optimal; much of the gap developed
at 35..38. These numbers motivate diagnostics, not a conclusion that exact targets
cannot work. None of the new options has demonstrated a 5x5 strength improvement yet.

Use one GPU job at a time. Solver labels run on CPU; concurrent labelling changes the
speed of search and the strength of wall-clock-limited opponents such as classic.
Outputs require fresh directories. Failed runs retain their partial diagnostics;
reuse their frozen positions and completed exact labels in a new output directory.

## 1. Cross old/new policy and value without retraining

Set paths to the completed A/B checkpoints. This deliberately uses the four-epoch
candidate from that experiment, not the separately trained 12-epoch checkpoint.

```bash
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
REF="$EXP/logs/exact-ab-20261013-140131/reference.pt"
CAND="$EXP/checkpoints/exact-ab-20261013-140131-exact/iter1.pt"
TAG="angles-$(TZ=America/Los_Angeles date +%Y-%m-%d_%H-%M-%S)"
RUN="$PWD/$EXP/logs/$TAG"
DEV="$GAME_DATA_DIR/$EXP/5x5-s1600/selfplay-it20"

uv run python -u -m alpha_go.boxes.search_probe \
  --reference "$REF" --candidate "$CAND" --games "$DEV" \
  --rows 5 --min-undrawn 35 --max-undrawn 38 --positions 100 \
  --simulations 400 --lambdas 0.5 \
  --solver 30 --solver-budget 50000 --label-budget 20000000 \
  --seed 20261014 --output "$RUN/mix"
```

Four cases use the same positions: reference policy/value, candidate policy/value,
reference policy with candidate value, and candidate policy with reference value.
Each network keeps its complete trunk and readouts, including its own feature encoding.
Mixed cases perform two network forwards; this diagnoses components at equal search
counts and is not an equal-time strength comparison.

`summary.json` reports results on the **common exactly known subset across every
requested case**. `positions.csv` contains each chosen move, exact final margin after
that move, timing and source family. `positions.jsonl` freezes the sampled prefixes;
`labels.json` stores completed exact solves. Unknown labels stay unknown and coverage
is reported. Roots and only the selected policy/search children are solved, avoiding
the cost of solving every possible move just to evaluate a choice. Budgets are finite
per solve, not wall-clock deadlines. The search solver must be below the sampled band.

These source games were training data: this is a development diagnostic. It cannot
establish generalisation. After choosing settings, use fresh paired arena openings
against frozen opponents and a separate final position set. Do not use `5x5-mid` or
the uploaded arena records to generate new training labels.

## 2. Compare margin shaping and additional search on identical positions

Reuse the first probe's positions and completed labels. Only the reference model is
needed to isolate search effects; supply both checkpoint arguments as usual.

```bash
uv run python -u -m alpha_go.boxes.search_probe \
  --reference "$REF" --candidate "$CAND" \
  --positions-file "$RUN/mix/positions.jsonl" --label-cache "$RUN/mix/labels.json" \
  --rows 5 --min-undrawn 35 --max-undrawn 38 \
  --mixes reference --simulations 400 1600 6400 --lambdas 0 0.5 \
  --solver 30 --solver-budget 50000 --label-budget 20000000 \
  --seed 20261014 --output "$RUN/search"
```

This is six cases; start with `--simulations 400` for just the objective comparison.
Measure optimal moves, exact margin lost, winning positions preserved and search time.
Margin loss is in final score-margin points: transferring one box between players
changes the margin by two. C++ MCTS seeds its own RNG; frozen inputs/settings are
reproducible but complete search trajectories are not promised to be bit-identical.
Changing lambda changes both neural leaf utility and shaped exact-terminal utility.

## 3. Exact policy targets at consequential training mistakes

Correcting a root's value does not correct its old search-generated policy. This
labeller prioritises winning-to-losing mistakes, then exact margin regret, and attempts
at most two positions per source game. A policy is saved only if **all legal children**
are solved, giving a uniform distribution over the complete optimal-move set. Incomplete
policies are skipped, with coverage reported. Existing exact arrays are not trusted:
bounded line solves recompute the values, sharing a table within each game.

```bash
POLICY_DATA="$GAME_DATA_DIR/$EXP/$TAG-policy"
uv run python -u -m alpha_go.boxes.policy_labels \
  --games "$GAME_DATA_DIR/$EXP/5x5-s1600/selfplay-it20" \
  --output "$POLICY_DATA" --rows 5 \
  --min-undrawn 35 --max-undrawn 38 --max-per-game 2 --min-regret 2 \
  --workers 4 --node-budget 20000000 --child-node-budget 20000000
```

Use `--manifest PATH` instead of `--games` to process a whole training window. Relative
manifest entries resolve under `$GAME_DATA_DIR`; input directories must exist. The
known `5x5-mid` collection is rejected, including symlinks. Other paths must also be
training data; filenames cannot prove their provenance. The output is a standalone
training collection compatible with `EXTRA_DATA` and `train.py` manifests.

Source files remain unchanged. Each output position has exact value and policy targets,
source hashes and its original replay-family assignment. A new `source_replay_group`
field carries the original identifier so the derived labels cannot cross their source's
stable training/validation boundary. This does not eliminate transpositions or historical
warm-start leakage. Keep the existing replay when adding this small targeted collection;
first compare unchanged replay against the same replay plus these labels at natural weight.

## 4. Remaining-margin value head, with a matched reset control

The new head predicts the remaining margin and adds the current score exactly. Its
support includes only margins feasible for the unclaimed boxes, including parity.
Search/inference still receives final-margin logits, so the arena and existing checkpoint
loader work normally. Legacy checkpoints remain final-margin models.

Changing target semantics requires `--reset-value-head`: only the final value readout is
zeroed; the shared trunk, policy and first value layer are retained. Reset **both** arms
for a fair comparison. This experiment tests target parameterisation plus feasibility
masking, not score subtraction in isolation. It is separate from adding policy labels.

```bash
(
set -euo pipefail
MANIFEST="$PWD/$EXP/logs/exact-ab-20261013-140131/exact-manifest.txt"
test -f "$MANIFEST"
for TARGET in final remaining; do
  test ! -e "$EXP/checkpoints/$TAG-$TARGET"
  uv run "$EXP/train.py" \
    --rows 5 --iteration 1 --tag "$TAG-$TARGET" --dataset-txt "$MANIFEST" \
    --resume-from "$REF" --value-target "$TARGET" --reset-value-head \
    --features chains --min-undrawn 24 --time-budget 0 --max-epochs 4 \
    --seed 20261014 --split-seed 0 --q-mix 0 --exact-weight 1
done
)
```

For later iterations, resume normally without `--reset-value-head`; the target setting
is stored in the checkpoint and inherited when `--value-target` is omitted. Compare these
two candidates using the frozen-position probe, then complete games against the unchanged
reference and classic with one arena worker. Never promote from replay loss alone.

## 5. Additional replay against varied opponents

This optional collector uses the normal shared gameplay loop, a frozen learner, self,
older checkpoints and classic. With 100 games and weights 3:1:1, it collects 60 self-play
games, 20 against the old checkpoint and 20 against classic. Each opponent gets paired
openings and swapped learner colours. Start with fewer games to measure local throughput.

```bash
MIXED_DATA="$GAME_DATA_DIR/$EXP/$TAG-mixed"
uv run python -u -m alpha_go.boxes.mixed_collect \
  --checkpoint "$CAND" --opponent self --opponent "ckpt:$REF" \
  --opponent boxes-classic --weights 3 1 1 \
  --num-games 100 --rows 5 --opening-moves 4 \
  --sims 1600 --solver 30 --solver-budget 50000 \
  --prove_terminals 1 --shape_terminals 1 --margin_utility_lambda 0.5 \
  --seed 20261014 --output "$MIXED_DATA"

uv run "$EXP/relabel_exact.py" --games "$MIXED_DATA" \
  --rows 5 --max-undrawn 38 --num-workers 4
```

The collector runs one game at a time with at most two resident networks. This favours
simple, bounded GPU memory use over maximum self-play throughput. `collection.json`
records quotas, settings, hashes, source code identity and every game. The plan is
deterministic; wall-clock classic search and C++ search randomness can change games.

Complete trajectories stay in each NPZ for exact replay. Boolean `train_mask` selects
only searched learner turns, using actual movers so extra turns are correct. Self-play
selects both seats; opponent and unsearched forced-move policies are excluded. Dataset
filtering, exact-target counts and `exact_fit.py` respect these non-contiguous indices.
Pair families stay together during stable validation. Add this collection to existing
replay, and compare against an equal-sized fresh self-play control before crediting diversity.

Keep `--q-mix 0` with shaped search: recorded root Q is utility, not a win probability.
The collector marks that explicitly and training rejects positive q-mixing for such files.
The standard iteration script is not modified; collecting here does not alter its league.

## Validation

The implementation report records the final checks. Small-board tests cover complete
game collection, extra turns, full optimal-policy sets against the exhaustive oracle,
unknown-label handling, checkpoint compatibility, score-shift gradients and validation
lineage. CPU smoke tests use synthetic small networks and measure integration only.
The real 3070 checkpoints, CUDA execution and 5x5 strength comparisons remain to run.
