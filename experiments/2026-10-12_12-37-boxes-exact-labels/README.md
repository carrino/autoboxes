# Exact 31–36-edge labels (ASTRA task 2)

Generate exact outcomes and uniform-optimal policies exclusively from the uploaded
`5x5-chains` games. `5x5-mid` is never opened by this experiment. Base solver:
`70558b73a181c3b29a246979d507f335e913e8c1`; no task-1 candidate is used unless it has
landed separately. This branch does not merge any other branch.

Machine: Intel Xeon Platinum 8573C, `nproc` 5, four-core CPU quota, 16 GiB memory
limit; Python 3.12.14; C++ Release build. Baseline: 351 passed, 24 skipped, all three
Catch2 binaries passed. Each labeller uses the existing 2^22-entry solver table
(64 MiB), one core, and approximately 0.7 GiB including Python/torch imports.

```bash
git clone https://github.com/carrino/autoboxes
cd autoboxes
git checkout -b astra/exact-labels 70558b73a181c3b29a246979d507f335e913e8c1
uv sync && bash scripts/build_cpp.sh
export GAME_DATA_DIR=/workspace/work/game_data_root
mkdir -p "$GAME_DATA_DIR"
tar xzf boxes-games-slim.tgz -C "$GAME_DATA_DIR"
uv run -m pytest tests/ --ignore=tests/test_gpu_lease.py -q
TASK=experiments/2026-10-12_12-37-boxes-exact-labels
uv run "$TASK/run_labels.py" --game-data "$GAME_DATA_DIR" --workers 4 --full-workers 2 --hours 12 --archive /codex/browser/projectless/0002-files-mentioned-by-the-user-uploaded-file-pointe/output/5x5-oracle-31-36.tgz
uv run ruff check "$TASK/run_labels.py"
MYPYPATH=src uv run -m mypy "$TASK/run_labels.py" --follow-imports=silent
uv run "$TASK/report.py"  # requires the completed manifest and verified archive
```

The cloud checkout was reconstructed from GitHub through the connector and its tree
and commit hashes verified, because direct Git transport was unavailable. Dependencies
were installed from the image's cache using `uv sync --frozen --offline --python python3.12`.
The already verified baseline extension was copied into this worktree's separate venv.
The upload was extracted into `work/game_data_root`, not the home directory.

The script runs four disjoint 100-position pilot shards with seed 0 and a 20M-node
budget. If over 20% are skipped, it repeats once at 40M. It scales measured throughput
to two production workers and about 12 hours (60,000-position maximum), using seed 1.
Two workers leave two CPU cores available for tasks 1 and 3; the pilot itself uses
all four. Every subprocess command, PID, timing, count, and completion is recorded
in `logs/<stage>/manifest.json`. `data/sizing.json` records the budget calculation.

Validation checks per-file and per-shard totals against logs; every visit row must
have a nonempty set of legal optimal edges, and its normalized policy must sum to 1.
It also checks the chosen move, undrawn range, ordering, player values, and root outcome
against the stored final margin. `data/*-undrawn.csv`, `*-margins.csv`, and `*-files.csv`
record distributions, counts, and per-file SHA256 digests. The final manifest records
archive size and SHA256. All NPZ files and the archive stay outside Git.

The script refuses to mix outputs with an existing stage directory. To inspect a
long run, read its per-shard logs and stage manifest; a `running` state is not a
completed or validated dataset.
