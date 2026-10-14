# BoxesZero author-repository opponent

Goal: compare autoboxes with a traceable external neural/search engine on 5x5 boxes.
This adapter runs the author's published code and checkpoint in a separate process.
It does not establish that the checkpoint is the final model evaluated in the paper.
See [report.md](report.md) for the source audit, compatibility decisions and validation.

Run commands from the autoboxes repository root, after the current A/B arena finishes.
No training data or autoboxes checkpoints are modified. A working autoboxes installation
(`uv sync`, `bash scripts/build_cpp.sh`) is required. One GPU worker is sufficient.

## 1. Download the pinned artifacts and smoke-test the real weights

The installer downloads 15 reviewed Python files plus the 12.4 MB author-repository
checkpoint into an external directory. It verifies their Git blob identifiers and
records the checkpoint SHA256. Repeating it reuses verified files. Use its minimal
installation, rather than an upstream full clone: unverified Python/bytecode is rejected.

```bash
uv run python -m alpha_go.boxes.boxeszero_install \
  --destination work/boxeszero --device cpu --simulations 8

mkdir -p work/boxeszero/results
OUT=$(mktemp -d work/boxeszero/results/smoke-XXXXXX)
uv run python -u -m alpha_go.boxes.arena \
  --rows 5 --cols 5 --a boxes-greedy \
  --b "boxeszero:$PWD/work/boxeszero/arena-cpu-s8.json" \
  --num_games 2 --num_workers 1 --opening_moves 4 --seed 20261014 \
  --save_games "$OUT/games.jsonl" 2>&1 | tee "$OUT/arena.log"
```

Eight simulations is only a compatibility smoke test. Check for two completed games,
verified provenance, the inferred input-channel count, and the final `===RESULT===`.
Any checkpoint mismatch, unsupported weight layout, invalid move or worker failure is
an error; the adapter never substitutes random weights or a different search engine.

## 2. Measure a small pilot on the RTX 3070

This configuration uses the same checkpoint with 800 upstream search iterations per
decision. That is a reproducible starting budget, not an equal-time or paper-reproduction
claim. Start with 20 games per opening regime to measure speed and check stability.

```bash
(
set -euo pipefail
uv run python -m alpha_go.boxes.boxeszero_install \
  --destination work/boxeszero --device cuda --simulations 800

EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
T=exact-ab-20261013-140131
CKPT="$EXP/logs/$T/reference.pt"
test -f "$CKPT"
OUT=$(mktemp -d work/boxeszero/results/pilot-XXXXXX)

for OPENING in 0 4; do
  uv run python -u -m alpha_go.boxes.arena \
    --rows 5 --cols 5 --a "ckpt:$CKPT" \
    --b "boxeszero:$PWD/work/boxeszero/arena-cuda-s800.json" \
    --num_games 20 --num_workers 1 --seed 20261014 --opening_moves "$OPENING" \
    --sims 400 --solver 30 --solver_budget 50000 \
    --policy_temperature 1.0 --prove_terminals 1 --shape_terminals 1 \
    --margin_utility_lambda 0.5 \
    --save_games "$OUT/opening-$OPENING.jsonl" \
    2>&1 | tee "$OUT/opening-$OPENING.log"
done
echo "Results: $OUT"
)
```

The reference checkpoint is deliberately named; repeat with the exact-label candidate
only after its current A/B results have been assessed. Keep the same opponent config,
seeds and opening sets. For an independent confirmation, use new seeds and frozen settings.

## Interpreting matches

- Use **game wins** as the primary metric. Upstream stops once a player owns 13 boxes;
  the adapter then draws legal edges to finish our 60-edge game. These completion moves
  cannot change the winner but need not preserve the best possible final margin. The
  arena's margin-based pair score is therefore unsuitable as the success criterion here.
- Keep empty-board matches separate from four-edge random-opening matches. Empty-board
  repetitions revisit a very small set of trajectories and do not provide broad evidence.
- The existing arena's game-win Wilson interval treats games independently. For a final
  random-opening comparison, resample whole two-game opening pairs. The helper below
  computes that interval from saved JSONL. A 20-game pilot is not a strength conclusion.
- Report simulations and both seconds-per-move columns. Worker/model startup is excluded
  from move time; IPC and search are included. Completion after 13 boxes is included in
  the mean and is cheap. Equal simulations are not equal compute. Calibrate timing before
  claiming equal-time superiority; inspect decision times before the winner is fixed.
- If the pilot works, use at least 1,000 games on a fresh paired random-opening seed,
  plus a separate empty-board report. Retain all logs, configs, hashes and move records.
  A win proves superiority over this documented snapshot/settings only.

Pair-aware win-rate interval, for one completed even-length JSONL file:

```bash
OUT=work/boxeszero/results/pilot-XXXXXX  # replace with the printed Results directory
uv run python - "$OUT/opening-4.jsonl" <<'PY'
import json, sys
import numpy as np
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows) >= 4 and len(rows) % 2 == 0
assert [r['index'] for r in rows] == list(range(len(rows)))
for first, second in zip(rows[::2], rows[1::2]):
    assert first['a_first'] and not second['a_first']
    n = first['opening']
    assert n == second['opening'] and first['moves'][:n] == second['moves'][:n]
margins = np.array([r['a_margin'] for r in rows])
pair_wins = ((margins > 0) + 0.5 * (margins == 0)).reshape(-1, 2).mean(axis=1)
rng = np.random.default_rng(20261014)
reads = rng.choice(pair_wins, size=(10000, len(pair_wins)), replace=True).mean(axis=1)
print({'games': len(rows), 'pairs': len(pair_wins), 'a_win_rate': float(pair_wins.mean()),
       'pair_bootstrap_ci95': np.quantile(reads, [0.025, 0.975]).tolist()})
PY
```

## Reproducing local engineering checks

```bash
uv run python -m pytest tests/test_boxes_boxeszero*.py -q
uv run ruff check src/alpha_go/boxes/boxeszero*.py tests/test_boxes_boxeszero*.py
uv run python -m mypy --follow-imports=silent src/alpha_go/boxes/boxeszero*.py

# Optional source-level parity and synthetic-weight tests after installation:
BOXESZERO_ROOT="$PWD/work/boxeszero/upstream" \
  uv run python -B -m pytest tests/test_boxes_boxeszero_worker.py -q
```

The optional tests create synthetic weights in pytest's temporary directory. They do
not replace the downloaded checkpoint and do not measure playing strength.
