# Dabble / PRsBoxes frozen-position comparison

Compare the genuine external engines with autoboxes on the same 100 frozen 5x5
positions at 35–38 undrawn edges. BoxesZero is an algorithmic reference for now;
the downloaded checkpoints have not established the published player's strength.

**Status: both engines passed the three-root native Windows smoke on the user's
machine (eoW2mg). The full 100-position launcher is ready; its results are pending.**

## Successful Windows smoke: eoW2mg

The user reported both workers exiting zero after the header and process-handle
fixes. Each chose edges 45, 37 and 55 on frozen IDs 1, 3 and 6: all exact-optimal,
with final margins +5, +1 and +3. PRsBoxes reported a verified native board and
ownership roundtrip for every decision. Its measured move/save times were 0.500,
0.750 and 3.032 seconds; Dabble's process times were 1.875, 2.406 and 6.797 seconds.
Both requested five seconds. Their timers include different overhead and neither
measurement establishes equal-time strength. The three roots are adapter checks.

This evidence is the pasted console transcript; the raw archive has not yet been
independently audited. Foreground GUI windows were not observed, but the successful
native roundtrip and returned decisions do not require foreground visibility.
PRsBoxes' seed after loading is unverified; Dabble exposes no seed control.

## First Windows smoke: Hgt9uU

User-reported console output and error logs: Dabble returned edges 45, 37 and 55
on frozen IDs 1, 3 and 6, respectively. All three attain the exact root margin
(+5, +1, +3). Elapsed times were 1.968, 2.625 and 6.766 seconds at a requested
five-second budget. The reference autoboxes search preserved one of these three
wins. This tiny, unequal-budget adapter sample does not establish relative strength.

PRsBoxes displayed "Expected Horizontal line character" during File/Open. Its
loader at PE addresses 0x406c26–0x406cab only commits the first-row width when it
hits a non-edge separator after the last dot. Our valid-looking row without that
separator left the constructor's 12-dot width unchanged. The exporter now emits
`#The first row determines width` immediately after the final dot, matching the
native writer at 0x406a48. A regression test characterizes the external parser's
width rule; the previous encoder/decoder roundtrip tests did not cover that rule.

Dabble printed all three results with an empty worker error log, but the Windows
launcher reported failure. The launcher previously converted a null Process.ExitCode
to 1. It now retains the process handle before waiting, and reports an unavailable
exit code explicitly instead of attributing it to the engine. The subsequent eoW2mg
run verified both fixes on Windows. Focused checks at that fix: 35 tests and Ruff passed.

## What is measured

The committed tables contain exact final margins for every legal child (3,640 in
total). Replaying each prefix reconstructs the mover, captured boxes and score.
No inference, solving, training data download or GPU is needed to score a choice.

Report each bot's selected edge, exact-optimal rate, final score-margin regret,
and how often it preserves a winning position. A bot optimizing wins can choose
a smaller winning margin deliberately, so regret alone is not a strength ranking.
One box transferred between players changes the final margin by two points.

Report 35–36 and 37–38 separately, plus paired deltas with a bootstrap over source
families. All completed decisions remain in the denominator; a crashed engine is
an adapter failure, never a substituted random move. Each decision starts a fresh
process. Record executable hashes, time budgets, wall time and engine timing.

These are development positions from iteration 21's self-play, inside iter22's
training window. They are useful for diagnosing mistakes, not an untouched final
strength test. The existing autoboxes decisions used 400 simulations on the RTX
3070; external CPU timings are not an equal-time comparison. After validating the
adapters, run both bots at documented budgets on the user's machine, then use
full paired games through `alpha_go.gameplay` for overall strength.

## Existing autoboxes choices

| Checkpoint | Exact-optimal moves | Mean margin regret | Wins preserved |
| --- | ---: | ---: | ---: |
| iter0 / reference | 32 / 100 | 2.68 | 37 / 73 |
| iter22 / candidate | 25 / 100 | 3.70 | 33 / 73 |

These are the original `astra-probe-PCcE6W` decisions rescored against the later
complete child table. The original report excluded two unsolved roots; the full
child table now establishes both values, so these denominators are 100 and 73,
rather than 98 and 71. No search was rerun.

Reproduce the baseline files:

```bash
uv run experiments/2026-10-14_19-52-boxes-external-probe/prepare_baselines.py
```

`data/probe-positions.csv` and `probe-summary.json` preserve the original choices,
checkpoint hashes, code hashes and search settings. `child-diagnostic-*` are the
complete exact tables from the policy/value audit in PR #17; their byte hashes
are recorded in each baseline and engine run.

## Engine provenance and current blockers

**Dabble:** J. P. Grossman's Dabble 1.3, using the hash-enabled command-line build
archived by KSquares. The pinned repository is
`ofenrohr/ksquares@3376c69a2b95cbc44241c65da224411967b32801`; its README documents
`dabble.exe SAVEGAME TIMEOUT_SECONDS`. This is a modified CLI build, not an
authenticated copy of every competition configuration. It needs the Visual C++
2013 x86 runtime. Its executable hash and source references accompany the worker.

An attempted Linux port of the repository's unfinished nohash source is excluded.
It replayed the 100 frozen roots but failed broader tiny-board tests, including a
phantom capture on 2x2. Historical hash source is also incomplete. Fixing those
search implementations would create a different bot; we do not use them as the
strong opponent benchmark.

**PRsBoxes:** Paul R. Stevens' 2007 Windows executable from the pinned mirror
`yimmon/dots-and-boxes@57c6a5fd849022bf3754c4aa530a2ab2ee7f300c/other/PRsBoxes.exe`.
SHA256: `2f603782446e0df640b832966ac7b19e5ad9e8781aa6bc0c279fa415901b9d55`.
The bridge uses normal File/Open → Computer Move → File/Save. The executable's
`-tFILE` test-creation mode has a different analysis path and is excluded.
The adapter checks native load/save roundtrip, one legal added edge, and updated
ownership. It sets captured boxes relative to the requested mover, who is
Computer. It requires an interactive Windows desktop and MFC42 x86.

Cloud execution tried portable Wine/WoW64: 64-bit utilities ran, but both 32-bit
engines failed during runtime startup before executing their search. This tells
us nothing about their move quality. The subsequent user-run Windows smoke succeeded;
broader native validation remains part of the 100-position run.

## Run the Windows smoke from WSL

After updating the Windows Git mirror and pulling `astra/external-position-probe`
in WSL through the usual workflow, run from the repository:

```bash
bash experiments/2026-10-14_19-52-boxes-external-probe/windows/run_windows_smoke.sh
```

Keep the Windows desktop available; the bots briefly open windows. The launcher
stages an official embeddable CPython and pinned engine executables in a fresh
Windows temporary directory. It does not require Windows Python to be installed.
It verifies download hashes and checks the native x86 runtime DLLs. Missing DLLs
are reported without installing system software; the other engine still gets
its own attempt.

The smoke selects frozen IDs 1, 3 and 6: an uncaptured root, an asymmetric score
with player 1 moving, and an asymmetric score with player 2 moving. Each engine
gets a requested five seconds per move and a separate five-minute process-tree
watchdog. The result is adapter validation data, not a three-game strength claim.

The command prints a `work/boxes-windows-smoke-*/results.tgz` path. Send that
archive back, including on failure: it contains per-engine status, exact input
prefixes, hashes, native saved boards, raw moves and errors. The GPU is unused.
The script leaves its temporary runtime directory for diagnosis and prints the
path. The default smoke remains bounded to three roots.

Cloud checks cover Python compilation, Bash syntax and state conversion. The
PowerShell orchestration and real Windows GUI run on the user's machine. The
CPython ZIP hash is pinned from the manifest linked in `windows/findings.json`;
the launcher checks downloaded bytes before extraction/execution.

## Run all 100 positions

After updating the Windows mirror and pulling `astra/external-position-probe` in
WSL, run:

```bash
bash experiments/2026-10-14_19-52-boxes-external-probe/windows/run_windows_smoke.sh --full
```

Each bot visits all 100 frozen roots at a requested five seconds per decision.
Progress is printed every 30 seconds. A completed decision gets an atomic receipt
with its request, response and end-to-end wall time; completed receipts are copied
to WSL at each progress update. Each engine stops on its first error, preserving
earlier decisions, and the other engine still gets its own attempt. The normal
exit path copies all native diagnostics, including failed roots. Windows staging
is retained even if the launcher is interrupted. This wrapper starts a fresh run;
it does not automatically resume one that stopped.

WSL then scores saved choices using the committed exact tables, with no GPU or
new solver work. `results/summary.json` contains overall and per-band metrics,
paired family-bootstrap deltas against both autoboxes checkpoints, raw decisions,
engine provenance and missing-position lists. It rejects illegal edges, changed
frozen inputs, wrong engine hashes, wrong mover/scores and changed budgets. Missing
decisions are reported separately; comparisons use common completed roots only.

Send the printed `work/boxes-windows-full-*/results.tgz` file, including on failure.
The full Windows orchestration still needs this first user run. Linux checks cover
the scorers, both encoders, 100-request dispatch and persistence after an injected
worker failure; they do not simulate native search or GUI interaction.

## General diagnostic worker protocol

`alpha_go.boxes.external_probe` runs a fresh worker for each root. The worker reads
one JSON request from stdin with `rows`, `cols`, `prefix`, `to_play`, `scores`,
`board`, `seconds`, `seed` and `position`; it returns one JSON object containing
`edge`, plus optional timing/details. It must use the real engine, preserve state,
and reject unsupported inputs. Seed support must be reported honestly.

The engine config has exactly `name`, `command` (argv list), `artifacts` (path to
SHA256 mapping), and nonempty `provenance`. Paths are relative to the config;
`command[0]` must be an explicit executable path. Example invocation after an
adapter has passed its smoke test:

```bash
EXP=experiments/2026-10-14_19-52-boxes-external-probe
uv run python -u -m alpha_go.boxes.external_probe \
  --engine-config /path/to/validated-engine.json \
  --tables "$EXP/data" --output "$EXP/logs/engine-5s" --seconds 5 \
  --baseline "$EXP/data/autoboxes-reference-s400.json"
```

Completed roots are saved atomically. Reusing an output directory resumes only
when engine hashes, code, inputs and settings match. Worker timeouts terminate
the process group on Linux/WSL. Windows adapters need their own bounded cleanup.

## Validation

The core tests use independently solved tiny boards to check captures/extra turns,
win versus margin metrics, common-position family bootstraps, real subprocess
exchange, illegal results, resume integrity and worker timeout cleanup. Format
tests validate external encodings; they do not certify the executables or GUI.

Local verification: 585 tests passed, 29 skipped (the documented upstream GPU
lease test excluded); Ruff and strict mypy for the core and Windows scorer pass.
Both encoders also agree with the reference rules on all 100 frozen roots.
