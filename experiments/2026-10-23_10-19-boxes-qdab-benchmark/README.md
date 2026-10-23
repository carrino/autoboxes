# Full-game Qdab benchmark

Completed result: **17/100 paired-opening wins**, relative Elo **−275** against
this Qdab configuration; empty-board panel **1/20**. The independent verification,
margin distribution, timing limits and next experiment are in [result.md](result.md).

The latest ordinary four-cycle learner won 271–129 against its parent. That
establishes improvement inside our player family; it does not establish strength
against an external opponent or robustness when that opponent chooses the opening.

From the repository root on WSL:

```bash
bash experiments/2026-10-23_10-19-boxes-qdab-benchmark/run_benchmark.sh
```

This freezes the verified `ordinary-four-aqowx5f8` cycle-4 checkpoint, installs the
checksum-pinned Linux Qdab release, and runs the following fixed sample:

| Panel | Games | Starts | Purpose |
|---|---:|---|---|
| Primary | 100 | 50 fresh four-edge openings, sides swapped | Relative Elo estimate |
| Empty board | 20 | Ten side-swapped repeats | Opening robustness; descriptive only |

Autoboxes uses 400 simulations, solver34/50k, prior temperature 1, proof and terminal
shaping enabled, margin utility lambda 0.5/k6. Qdab uses `quctann`, five seconds
requested per decision, with a 30-second external RPC deadline. Its exact presearch
can exceed the requested five seconds; actual time is recorded. These are specified
engine settings, **not an equal-time comparison**. Qdab's random seed is not exposed.
One serial worker is required because Qdab owns a fixed port. Do not run another
Qdab collector concurrently on the same machine.

Allow roughly 4–6 hours initially; full-game timing has not yet been measured on
the 3070 host. Every finished game is saved and independently replayed through all
60 edges, including capture turns. The printed `resume.sh` resumes the same frozen
sample. A timeout or invalid move stops the run and remains recorded; it is never
counted as a forfeit or silently skipped. Three failed attempts at a game require
inspection. Partial panels report counts, without a final Elo estimate. Send the
printed `results.tgz`; weights stay local.

The primary score is `(wins + draws/2)/games`. Relative Elo is
`400 log10(score/(1-score))`, with this exact Qdab configuration assigned zero.
The interval resamples whole opening pairs. A conservative Hoeffding bound is
also reported and used when the bootstrap degenerates. Zero/all wins have an
unbounded estimate, not fictitious finite precision. This is not the published
BoxesZero Elo scale, nor a rating against Dabble or PRsBoxes, whose current adapters
test individual positions rather than full matches.

## Using the result to guide backward training

Current full-board self-play plus exact corrections at 35–37 is not a strict
backward-start curriculum. The intended curriculum starts near the end and moves
the starting frontier earlier as strength improves. It should not be trained or
judged only on our own opening distribution.

Keep these benchmark games evaluation-only. After seeing where Qdab steers us,
collect **separate** games with fresh seeds against Qdab and a mixture of current
and older agents. Use their legal replay prefixes as curriculum starts, grouped
by source game so descendants cannot cross train/evaluation splits. Mix such starts
with full-board games to retain opening competence. Check held-out performance by
opponent and edge band as well as full games; increase the starting range when the
new band improves without losing earlier gains. A failure against Qdab is a reason
to diagnose its trajectories, not automatically move the curriculum farther out.

No new model is trained by this benchmark.

## Validation

The suite passed 947 tests with 29 skips, excluding the known upstream GPU-lease
test. The additional SIGTERM packaging test passed in the ten-test runner suite.
Ruff, strict mypy for the two new source modules, and shell syntax checks passed.

Four full games against the real pinned Qdab binary passed independent replay:
240 edges, 100 Qdab decisions, including four multi-edge capture bundles. Both
paired-opening and empty-board paths were exercised. A completed resume created
no new attempts; an interrupted 400-simulation run packaged its incomplete result.
This smoke used a tiny random CPU network, eight simulations and Qdab's 0.05-second
requested budget. It validates the integration, not playing strength or 3070
runtime. See [validation.json](validation.json). The latest learner's production
match must run on the user's machine, which holds its weights.
