"""Assemble the report from captured measurements and verified label summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def table(headers: list[str], rows: list[list]) -> str:
    """Render a Markdown table without changing the stored measurements."""
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def csv_table(name: str) -> str:
    with (ROOT / "data" / name).open() as stream:
        rows = list(csv.reader(stream))
    return table(rows[0], rows[1:])


machine = json.loads((ROOT / "data/resumed-machine.json").read_text())
inputs = json.loads((ROOT / "data/input-manifest.json").read_text())
summaries = {}
for phase in ["pilot20", "pilot40", "full"]:
    path = ROOT / "data" / f"{phase}-verification.json"
    if path.exists():
        summaries[phase] = json.loads(path.read_text())
label_rows = []
for phase, summary in summaries.items():
    label_rows.append([phase, summary["node_budget"], summary["sampled"], summary["positions"],
                       summary["skipped"], f'{100 * summary["skipped"] / summary["sampled"]:.2f}%',
                       f'{summary["wall_seconds"]:.3f}', summary["files"],
                       f'{summary["maximum_policy_sum_error"]:.3g}'])
label_table = (table(["Phase", "Node budget", "Sampled", "Labelled", "Skipped", "Skipped %",
                      "Wall seconds", "NPZ files", "Max policy-sum error"], label_rows)
               if label_rows else "Pilot and full-run verification results are pending.")
undrawn_table = (table(["Undrawn", *summaries],
                       [[n, *(s["positions_per_undrawn"].get(str(n), 0)
                               for s in summaries.values())] for n in range(31, 37)])
                 if summaries else "Pending.")
margins = sorted({int(m) for s in summaries.values() for m in s["final_margins_player_1"]})
margin_table = (table(["Final margin for player 1", *summaries],
                      [[m, *(s["final_margins_player_1"].get(str(m), 0)
                              for s in summaries.values())] for m in margins])
                if summaries else "Pending.")
mover_margins = sorted({int(m) for s in summaries.values() for m in s["final_margins_side_to_move"]})
mover_table = (table(["Final margin for side to move", *summaries],
                     [[m, *(s["final_margins_side_to_move"].get(str(m), 0)
                             for s in summaries.values())] for m in mover_margins])
               if summaries else "Pending.")
plan_path = ROOT / "data/full-plan.json"
sizing = ("```json\n" + plan_path.read_text().rstrip() + "\n```" if plan_path.exists()
          else "The full run will be sized from the completed pilot's measured rate.")
archive_path = ROOT / "data/archive.json"
archive = ("```json\n" + archive_path.read_text().rstrip() + "\n```" if archive_path.exists()
           else "No full-run archive has been created yet.")
events = [json.loads(line) for line in (ROOT / "logs/resumed-commands.jsonl").read_text().splitlines()]
timing_rows = [[e["name"], f'{e["wall_seconds"]:.3f}', e["exit_code"]]
               for e in events if e["event"] == "finished" and "name" in e]
timings = table(["Measurement", "Wall seconds", "Exit code"], timing_rows)
status = (f'The full run produced **{summaries["full"]["positions"]:,} verified exact labels**, '
          f'with {summaries["full"]["skipped"]:,} positions skipped on the node budget.'
          if "full" in summaries else
          "Job 1 is complete. Labelling is in progress; only completed, verified phases are tabulated below.")

text = f"""# CPU solver work on 5x5 Dots and Boxes

**Yes: at a fixed number of undrawn edges, the current solver usually visits fewer
nodes when the largest independent region is smaller.** For a largest region of
20 edges versus an unsplit board, median nodes were 4,166 versus 20,481 at 30
undrawn (4.9x fewer), and 10,205 versus 88,469 at 32 undrawn (8.7x fewer).

{status}

## Machine, checkout, and inputs

- CPU: Intel Xeon Platinum 8573C; `nproc=5`; CPU cgroup quota is four CPU equivalents.
- RAM: 18,932,244,480 bytes reported by `/proc/meminfo`; cgroup limit 16 GiB.
- At job start: {machine['memory_bytes']['MemFree']:,} free bytes,
  {machine['memory_bytes']['MemAvailable']:,} available bytes; conservative free memory
  after the cgroup limit was {machine['conservative_free_GB']:.3f} GB (decimal).
- `N=min(nproc-1, floor(free GB/0.6))=min(4,{machine['memory_process_cap']})=4`.
  Pilot workers observed around 0.7 GB RSS each, still comfortably within RAM.
- No GPU device nodes; all work used the CPU. Python 3.12.14, GNU C++ 14.2.0,
  CMake 3.31. Raw resource records are in `data/machine.json` and `data/resumed-machine.json`.
- Branch `astra/cpu-solver-work`, based on `claude/awesome-fermi-ygaeuj` at
  `b6a2034c07e4e1f0659ebfce6fbbb393abd775cd`, which supplies `--num-shards`.
  Neither `main` nor the source branch is a push destination.
- Read `CLAUDE.md`, `EXPERIMENTS.md` sections 5–10, and `GETTING_STARTED.md` section 8
  before work; also read `ARCHITECTURE.md` and `PLAN.md`.
- Input upload: `boxes-games-slim.tgz`, 4,628,707 bytes,
  SHA-256 `4ca95b315317fb71fe7b5457665cbd06d556bc087262be7a50d80092f736d132`.
- Extracted **{inputs['counts']['5x5-chains']:,} chains games** and
  **{inputs['counts']['5x5-mid']:,} evaluation games**. All archive members were checked
  to be regular NPZ files under exactly those two tag directories.
- `GAME_DATA_DIR={machine['GAME_DATA_DIR']}`. The requested `$HOME/autoboxes-data`
  location was read-only even with an additional filesystem grant, so the configured
  workspace data root was used with the same directory layout.
- `5x5-mid` was used only for measurements. Every labeller command uses
  `--positions-tag 5x5-chains`; no training command was run. The saved mid masks
  are measurement artifacts, separate from the labelled NPZ datasets.

## Build and correctness gate

`uv sync` passed. The first build stopped because Python's relocated installation
still reported `/install/lib/libpython3.12.so`. After the user authorized a local
fix, `scripts/build_cpp.sh` was changed locally to fall back to
`sys.base_prefix/lib/LDLIBRARY` when the configured file is absent. Compilation
then exposed a missing `<algorithm>` include in the existing C++ MCTS test;
`CXXFLAGS='-include algorithm'` supplied it without editing `src/` or `tests/`.

With `PATH=/workspace/autoboxes-tools/bin:$PATH`, the build passed in **104.121 s**.
The required `uv run -m pytest tests/test_boxes_*.py -q` passed:
**243 passed in 63.43 s** (67.203 s shell wall time). No solver work ran before that gate.

**The build edit is intentionally uncommitted, as requested.** No source, tests,
or existing loop scripts were edited. Complete failure output and successful
build/test output are preserved in `logs/setup.txt`, `logs/resume-before-fix.txt`,
`logs/build-missing-algorithm.txt`, and `logs/build-and-tests.txt`.

## Job 1: region sizes and solver cost

{timings}

The first region CSV was copied before the second command overwrote the script's
fixed output filename. Both samples and the final benchmark CSV are preserved
under this report's `data/`. The original requested outputs remain under the loop
folder's `data/` as well. All 1,500 masks are in
`data/positions-5x5-mid-30-32.txt`.

### Region sizes, 4,000 positions at 29–36 undrawn

{csv_table('region_stats-5x5-mid-29-36.csv')}

### Region sizes, the 1,500-position solver-cost sample

{csv_table('region_stats-5x5-mid-30-32.csv')}

### Nodes by largest region, holding undrawn count fixed

![Solver nodes by largest region](figures/solver-nodes-by-region.png)

{csv_table('solver_nodes_by_region.csv')}

The association is strong but not monotonic: geometry and interactions between
regions still matter. At 32 undrawn, the largest-region-30 bin has only three
positions and one reaches the two-million-node cap; its tail is censored.
Sparse small-region bins should not be treated as precise population estimates.
These measurements establish an association under the current solver; they do
not measure the additional speedup a Nimstring decomposition implementation would deliver.

The region measurement starts with a fresh 2^20-entry table for every position
and caps root solves at two million nodes. `solver_bench.py` instead uses a
2^22-entry table (67.1 MB), fresh for each undrawn count and reused across that
count's 200 positions, with uncapped root solves. Its node quantiles therefore
should not be compared as if they came from the same cache protocol or sample.
Region sampling uses all stored games and excludes forced-capture positions;
the benchmark samples self-play games only and does not apply that exclusion.

### Self-play-only timing benchmark

{csv_table('solver_bench-5x5-mid.csv')}

Raw printed tables are retained in `logs/region-29-36.txt`,
`logs/region-nodes-30-32.txt`, and `logs/solver-bench.txt`. The node CSV transcribes
the printed bins; its capped fractions retain the script's two-decimal precision.

## Job 2: exact labels from chains games only

The pilot requests 400 positions total with seed 0, split into four disjoint
shards of 100, initially at a 20-million-node budget. The budget is doubled once
only if more than 20% are skipped. The full run uses seed 1, the same sharding,
and at most 60,000 requested positions, sized for approximately 12 hours.

{label_table}

Wall time is elapsed time for the entire concurrent shard group, including
sampling and output. Per-shard times, requested/sample counts, log paths and exit
statuses are in `data/<phase>-status.json`; each shard has its own complete log.
The node budget applies to individual root/child solver calls, as implemented
by the unchanged labeller; a position is skipped if its root or any child times out.

### Full-run sizing

{sizing}

### Verified positions by undrawn count

{undrawn_table}

### Exact final margins for player 1

{margin_table}

### Exact final margins for the side to move

{mover_table}

The NPZ format stores one visit per optimal edge in `mcts_visits`, rather than
an already-normalized policy. `verify_labels.py` loads every array from every
output file with NumPy, reconciles per-file and per-shard counts with the logs,
requires nonempty finite visit rows, and checks every normalized policy row sums
to one within absolute tolerance 1e-6. It also checks the 31–36 band, legal policy
support, exact root values, valid final margins and distinct sampled positions.
Per-file counts and hashes are in `data/<phase>-files.csv`; all verification
results are in `data/<phase>-verification.json`.

### Full-run archive

{archive}

The NPZ datasets and oracle archive remain on this machine and are excluded from
git. Only the report, logs, measurements, figures and report helper scripts are
intended for the PR.

## Commands, failures, and reproducibility

- `logs/commands.txt` records the original setup/report attempt and publication.
- `logs/resume-setup-commands.txt` records the resumed build, local workarounds,
  input extraction, and job launch environment.
- `logs/resumed-commands.jsonl` records every underlying measurement and label
  command, data root, log path, UTC timestamp, exit code and elapsed wall time.
- `run_cpu_jobs.py` runs the unchanged experiment commands and preserves their
  outputs. `verify_labels.py` verifies the NPZs; `plot_job1.py` produces the figure;
  `write_report.py` assembles this README from the captured evidence.
- Input checksums and counts are in `data/input-manifest.json`. The full per-file
  input manifest is retained at `/workspace/work/cpu-solver/input-manifest-full.json`.
- The initial clone's proxy connection failed in the default sandbox and succeeded
  with the supported network permission. The initial optional chat-output copy
  failed because `/codex` is read-only; the report is available through GitHub.
- The repository ignores `experiments/**/logs/`, so report logs are explicitly
  staged. Verbatim compiler logs retain their original whitespace.
- The chosen source branch had six preexisting file changes relative to `main`
  at clone time. They are inherited; this task's commits contain only this report
  directory. The local build edit must not be staged or committed.
"""
(ROOT / "README.md").write_text(text)
print(ROOT / "README.md")
